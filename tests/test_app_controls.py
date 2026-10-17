"""AppTest: Auswahlfelder, Permalink, bedingte Regler, PDF (die Rauchtests in test_app.py decken nur Knöpfe und Slider ab)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import dwt_constants as C
from dwt_presets import param_key

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
SNR = "Rauschabstand (dB)"


def _run(params=None):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    if params:
        at.query_params.update(params)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _labels(at):
    return [x.label for x in at.slider]


@pytest.mark.parametrize("kind", list(C.SIGNALS))
def test_every_signal_type_runs_and_shows_its_parameter_sliders(kind):
    at = _run()
    at.selectbox(key="signal_select").set_value(kind).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    labels = _labels(at)
    assert all(spec[0] in labels for spec in C.PARAMS[kind])
    others = {spec[0] for k, specs in C.PARAMS.items() if k != kind for spec in specs} - {spec[0] for spec in C.PARAMS[kind]}
    assert not (others & set(labels))


def test_the_noise_controls_exist_only_for_the_noisy_signal_types_and_keep_the_snr():
    at = _run()
    assert all(r.key != "noise_select" for r in at.radio)
    at.selectbox(key="signal_select").set_value("Ton").run(timeout=TIMEOUT)
    assert any(r.key == "noise_select" for r in at.radio) and SNR not in _labels(at)
    at.radio(key="noise_select").set_value("an").run(timeout=TIMEOUT)
    assert not at.exception and SNR in _labels(at)
    at2 = _run({"s": "Ton", "g": "an", "snr": "7"})
    at2.radio(key="noise_select").set_value("aus").run(timeout=TIMEOUT)
    at2.radio(key="noise_select").set_value("an").run(timeout=TIMEOUT)
    assert {x.label: x for x in at2.slider}[SNR].value == 7


def test_parameters_are_kept_per_type_when_switching():
    at = _run({"q01": "0.8"})
    assert {x.label: x for x in at.slider}["Exponent α"].value == 0.8
    at.selectbox(key="signal_select").set_value("Ton").run(timeout=TIMEOUT)
    at.selectbox(key="signal_select").set_value("Potenzgesetz-Rauschen").run(timeout=TIMEOUT)
    assert {x.label: x for x in at.slider}["Exponent α"].value == 0.8


def test_the_level_choices_follow_the_record_length():
    at = _run({"n": "256", "j": "9"})
    assert at.select_slider(key="levels_select").value == 7
    at2 = _run({"n": "4096", "j": "9"})
    assert at2.select_slider(key="levels_select").value == 9
    at2.select_slider(key="n_select").set_value(256).run(timeout=TIMEOUT)
    assert not at2.exception and at2.select_slider(key="levels_select").value == 7


def test_changing_wavelet_transform_boundary_and_shift_runs():
    at = _run()
    at.select_slider(key="p_select").set_value(4).run(timeout=TIMEOUT)
    at.radio(key="transform_select").set_value("DWT").run(timeout=TIMEOUT)
    at.radio(key="boundary_select").set_value("Randkoeffizienten verwerfen").run(timeout=TIMEOUT)
    at.slider(key="shift_slider").set_value(5).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.select_slider(key="p_select").value == 4 and at.radio(key="transform_select").value == "DWT" and at.slider(key="shift_slider").value == 5


def test_permalink_is_applied_and_invalid_values_are_ignored():
    at = _run({"n": "1024", "s": "Sprung im glatten Signal", "q11": "5.0", "q12": "0.7", "q13": "9", "w": "4", "j": "5", "tr": "DWT", "rd": "Randkoeffizienten verwerfen", "sh": "12", "g": "an", "snr": "-99", "seed": "42"})
    assert at.select_slider(key="n_select").value == 1024 and at.selectbox(key="signal_select").value == "Sprung im glatten Signal" and at.radio(key="noise_select").value == "an"
    assert at.select_slider(key="p_select").value == 4 and at.select_slider(key="levels_select").value == 5 and at.radio(key="transform_select").value == "DWT" and at.number_input(key="seed_input").value == 42
    assert at.radio(key="boundary_select").value == "Randkoeffizienten verwerfen" and at.slider(key="shift_slider").value == 12
    by = {x.label: x for x in at.slider}
    assert by["Perioden des Sinus"].value == 5.0 and by["Lage des Sprungs (Anteil der Länge)"].value == 0.7 and by["Höhe des Sprungs"].value == 3.0 and by[SNR].value == C.SNR_MIN
    at2 = _run({"n": "3000", "s": "unsinn", "w": "5", "j": "99", "tr": "FFT", "rd": "x", "sh": "999"})
    assert at2.select_slider(key="n_select").value == C.DEFAULT_N and at2.selectbox(key="signal_select").value == C.DEFAULT_SIGNAL and at2.select_slider(key="p_select").value == C.DEFAULT_P
    assert at2.select_slider(key="levels_select").value == C.DEFAULT_LEVELS and at2.radio(key="transform_select").value == C.DEFAULT_TRANSFORM and at2.radio(key="boundary_select").value == C.DEFAULT_BOUNDARY
    assert at2.slider(key="shift_slider").value == C.SHIFT_MAX


def test_preset_button_sets_the_widgets():
    at = _run()
    at.button(key="preset_Impuls verschieben: die DWT ändert sich, die MODWT nicht").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert at.selectbox(key="signal_select").value == "Impuls" and at.select_slider(key="n_select").value == 256 and at.radio(key="transform_select").value == "DWT" and at.slider(key="shift_slider").value == 1
    at.button(key="preset_Trend plus Rauschen: Haar versagt, D4 nicht").click().run(timeout=TIMEOUT)
    assert at.selectbox(key="signal_select").value == "Trend plus Rauschen" and at.select_slider(key="p_select").value == 1 and at.radio(key="boundary_select").value == "Randkoeffizienten verwerfen"
    assert {x.label: x for x in at.slider}["Trend (Standardabweichungen über die Aufzeichnung)"].value == 100.0


def test_pdf_button_builds_a_download():
    at = _run()
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any(d.proto.label.startswith("⬇️") for d in at.get("download_button"))


def test_the_dice_button_changes_the_seed():
    at = _run()
    before = at.number_input(key="seed_input").value
    [b for b in at.button if "würfeln" in b.label][0].click().run(timeout=TIMEOUT)
    assert not at.exception and at.number_input(key="seed_input").value != before
    assert param_key(0, 1) == "p01_kept"
