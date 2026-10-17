"""Randfälle: alle Signaltypen, Längen, Wavelets, Stufen, Transformationen, Parametergrenzen, Rauschen, Abbildungen und PDF."""

import math

import numpy as np
import pytest

import dwt_constants as C
import dwt_evaluation as ev
import dwt_visualization as V
from dwt_pdf_export import build_pdf
from sweeps_cache import get


def settings(kind, params, **kw):
    p = list(params) + [0.0] * (3 - len(params))
    return ev.Settings(signal=kind, p1=p[0], p2=p[1], p3=p[2], **kw)


@pytest.mark.parametrize("kind", list(C.SIGNALS))
def test_every_type_runs_for_the_shortest_and_the_longest_record_wavelet_and_transform(kind):
    defaults = [spec[4] for spec in C.PARAMS[kind]]
    for n in (min(C.N_CHOICES), max(C.N_CHOICES)):
        for p in (C.WAVELET_P[0], C.WAVELET_P[-1]):
            for transform in C.TRANSFORM_MODES:
                levels = min(C.MAX_LEVELS, int(math.log2(n)) - 1)
                a = ev.analyse(settings(kind, defaults, n=n, p=p, levels=levels, transform=transform, noise="an", snr=5))
                assert a.recon_error < 1e-9 and a.mra_error < 1e-9 and len(a.details) == levels and abs(a.fractions.sum() - 1.0) < 1e-9


@pytest.mark.parametrize("kind", list(C.SIGNALS))
def test_every_parameter_at_its_limits_runs_with_every_level_count_and_boundary(kind):
    specs = C.PARAMS[kind]
    for pick in ("lo", "hi"):
        params = [spec[1] if pick == "lo" else spec[2] for spec in specs]
        for levels in (1, 7):
            for boundary in C.BOUNDARY_MODES:
                a = ev.analyse(settings(kind, params, n=256, levels=levels, boundary=boundary, shift=C.SHIFT_MAX, transform="MODWT"))
                assert np.isfinite(a.recon_error) and a.recon_error < 1e-9
    verdict = ev.verdict_main(ev.analyse(settings(kind, [spec[2] for spec in specs], n=256, levels=7, p=8)))
    assert verdict[0] in ("success", "warning", "info") and verdict[1]


def test_extreme_noise_levels_and_seed_limits_run():
    for snr in (C.SNR_MIN, C.SNR_MAX):
        for seed in (0, C.SEED_MAX):
            a = ev.analyse(settings("Ton", (0.07,), n=256, levels=5, noise="an", snr=snr, seed=seed))
            assert np.isfinite(a.recon_error) and a.band_level == 3


def test_a_silent_signal_has_zero_fractions_and_no_crash():
    a = ev.analyse(settings("Sprung im glatten Signal", (1.0, 0.5, 0.1), n=256, levels=4))
    assert a.fractions.sum() == pytest.approx(1.0)
    zero = ev.analyse(settings("Impuls", (0.5,), n=256, levels=4))
    assert zero.recon_error < 1e-12
    from dwt_resolution import energy_fractions
    import dwt_transform as T
    import dwt_filters as F
    assert energy_fractions(*T.dwt(np.zeros(8), F.daubechies(2), 2)).tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("kind", list(C.SIGNALS))
def test_figures_and_pdf_build_for_every_type(kind):
    for transform in C.TRANSFORM_MODES:
        a = ev.analyse(settings(kind, [spec[4] for spec in C.PARAMS[kind]], n=512, levels=5, transform=transform, noise="an", snr=10))
        for fig in (V.build_signal(a), V.build_coefficients(a), V.build_levels(a), V.build_mra(a), V.build_variance(a)):
            assert len(fig.data) >= 1
        assert build_pdf(a)[:4] == b"%PDF"


def test_variance_figure_layers_and_coefficient_map_shapes():
    a = ev.analyse(settings("Potenzgesetz-Rauschen", (0.5,), n=1024, levels=6))
    assert len(V.build_variance(a).data) == 3 and V.coefficient_map(a).shape == (6, 1024)
    d = ev.analyse(settings("Potenzgesetz-Rauschen", (0.5,), n=1024, levels=6, transform="DWT"))
    assert V.coefficient_map(d).shape == (6, 1024) and V.level_labels(3) == ["Stufe 1", "Stufe 2", "Stufe 3", "Glättung"]
    short = ev.analyse(settings("Potenzgesetz-Rauschen", (0.5,), n=256, levels=6))
    assert len(V.build_variance(short).data) == 1
    imp = ev.analyse(settings("Impuls", (0.5,), n=1024, levels=4))
    assert len(V.build_variance(imp).data) == 2


def test_figures_for_the_experiments():
    assert len(V.build_moments(get("moments")).data) == 5 and len(V.build_sparsity(get("moments")).data) == 1 and len(V.build_shift(get("shift")).data) == 6
    assert len(V.build_alpha(get("alpha")).data) == 7 and len(V.build_trend(get("alpha")).data) == 2 and len(V.build_bands(get("bands")).data) == 3 and len(V.build_variance_spread(get("variance")).data) == 2


def test_pdf_for_every_kennzahl_branch():
    for kind, params, extra in (("Potenzgesetz-Rauschen", (0.5,), {"n": 1024}), ("Potenzgesetz-Rauschen", (0.5,), {"n": 256}), ("Sprung im glatten Signal", (3.0, 0.3, 1.0), {}), ("Impuls", (0.05,), {"shift": 3}),
                                ("Ton", (0.07,), {"levels": 6}), ("Ton", (0.4,), {"levels": 6}), ("Trend plus Rauschen", (0.5, 100.0), {"boundary": "Randkoeffizienten verwerfen"})):
        assert build_pdf(ev.analyse(settings(kind, params, **extra)))[:4] == b"%PDF"
