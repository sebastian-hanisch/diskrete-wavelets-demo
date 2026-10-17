"""Presets: Vollständigkeit, gültige Werte, Standardwerte, Parameterschlüssel."""

import dwt_constants as C
import dwt_evaluation as ev
import dwt_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) | {"params"}, name
        assert C.PRESET_HELP[name] and len(preset["params"]) == 3


def test_preset_values_are_inside_the_choices_and_bounds():
    for name, p in C.PRESETS.items():
        assert p["n"] in C.N_CHOICES and p["signal"] in C.SIGNALS and p["noise"] in C.NOISE_MODES and p["p"] in C.WAVELET_P and p["transform"] in C.TRANSFORM_MODES and p["boundary"] in C.BOUNDARY_MODES, name
        assert C.LEVEL_MIN <= p["levels"] <= min(C.MAX_LEVELS, p["n"].bit_length() - 2) and C.SHIFT_MIN <= p["shift"] <= C.SHIFT_MAX and C.SNR_MIN <= p["snr"] <= C.SNR_MAX and 0 <= p["seed"] <= C.SEED_MAX
        for (label, lo, hi, step, default, help_text), v in zip(C.PARAMS[p["signal"]], p["params"]):
            assert lo <= v <= hi, (name, label)


def test_default_settings_are_valid_choices():
    s = ev.Settings()
    assert s.n in C.N_CHOICES and s.signal in C.SIGNALS and s.noise in C.NOISE_MODES and s.p in C.WAVELET_P and s.transform in C.TRANSFORM_MODES and s.boundary in C.BOUNDARY_MODES
    assert C.LEVEL_MIN <= s.levels <= C.MAX_LEVELS and s.levels <= s.n.bit_length() - 2
    for (label, lo, hi, step, default, help_text), v in zip(C.PARAMS[s.signal], (s.p1, s.p2)):
        assert lo <= v <= hi and v == default


def test_every_signal_type_has_one_to_three_parameters_with_valid_ranges():
    assert set(C.PARAMS) == set(C.SIGNALS) and set(C.NOISY_SIGNALS) <= set(C.SIGNALS)
    for kind, specs in C.PARAMS.items():
        assert 1 <= len(specs) <= 3
        for label, lo, hi, step, default, help_text in specs:
            assert lo < hi and lo <= default <= hi and step > 0 and label and help_text


def test_setting_specs_cover_every_preset_key_and_every_parameter():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)
    expected = {P.param_key(ti, slot) for ti, kind in enumerate(C.SIGNALS) for slot in range(1, len(C.PARAMS[kind]) + 1)}
    assert expected <= set(P.SETTING_SPECS)
    s = ev.Settings()
    defaults = {"n": s.n, "signal": s.signal, "p": s.p, "levels": s.levels, "transform": s.transform, "boundary": s.boundary, "shift": s.shift, "noise": s.noise, "snr": s.snr, "seed": s.seed}
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == defaults[key], key


def test_url_params_are_unique_and_not_password_like():
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params)) and max(len(n) for n in C.PRESETS) <= 70
    assert P.param_key(1, 3) == "p13_kept" and P.SETTING_SPECS["p13_kept"].url_param == "q13"
    assert not {"pw", "pass", "password", "pwd", "token", "key"} & set(params)


def test_bounds_and_the_constant_relations():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("shift_slider") == (C.SHIFT_MIN, C.SHIFT_MAX) and P.bounds("n_select") == (None, None)
    assert C.DEFAULT_P in C.WAVELET_P and C.DEFAULT_N in C.N_CHOICES and C.DEFAULT_TRANSFORM in C.TRANSFORM_MODES and C.DEFAULT_BOUNDARY in C.BOUNDARY_MODES
    assert all(n & (n - 1) == 0 for n in C.N_CHOICES)
