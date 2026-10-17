"""Kennzahlen der Zerlegung: Energieanteile, Sparsität, Varianzschätzer und Exponent, Bandlage, Verschiebungsempfindlichkeit."""

import math

import numpy as np
import pytest

import dwt_filters as F
import dwt_resolution as R
import dwt_signals as S
import dwt_transform as T
import signal_generator as G


def test_decompose_selects_the_transform_and_rejects_unknown(mini_x):
    h = F.daubechies(1)
    d, a = R.decompose(mini_x, h, 2, "DWT")
    assert [len(v) for v in d] == [2, 1] and len(a) == 1
    w, v = R.decompose(mini_x, h, 2, "MODWT")
    assert [len(x) for x in w] == [4, 4] and len(v) == 4
    with pytest.raises(ValueError):
        R.decompose(mini_x, h, 2, "FFT")


def test_energy_fractions_by_hand_and_for_zero_energy(mini_x):
    d, a = T.dwt(mini_x, F.daubechies(1), 2)
    assert R.energy_fractions(d, a).tolist() == pytest.approx([1 / 30, 4 / 30, 25 / 30]) and R.energy_fractions(d, a).sum() == pytest.approx(1.0)
    assert R.energy_fractions(*T.dwt(np.zeros(4), F.daubechies(1), 2)).tolist() == [0.0, 0.0, 0.0]


def test_variance_by_level_modwt_dwt_and_boundary():
    x = G.SplitMix64(2).normal(512)
    h = F.daubechies(2)
    assert R.variance_by_level(x, h, 4, "DWT").tolist() == T.dwt_variance(T.dwt(x, h, 4)[0]).tolist()
    assert R.variance_by_level(x, h, 4, "MODWT").tolist() == T.modwt_variance(T.modwt(x, h, 4)[0], 4).tolist()
    assert R.variance_by_level(x, h, 4, "MODWT", True).tolist() == T.modwt_variance(T.modwt(x, h, 4)[0], 4, True).tolist()
    assert R.variance_by_level(x, h, 4, "DWT", True).tolist() == R.variance_by_level(x, h, 4, "DWT").tolist()


def test_alpha_estimate_recovers_a_noiseless_power_law_and_reports_missing_levels():
    n, h = 4096, F.daubechies(2)
    alphas = [R.alpha_estimate(S.power_law(n, 0.5, G.SplitMix64(400 + r)), h, 9, "MODWT")[0] for r in range(30)]
    assert np.mean(alphas) == pytest.approx(0.5, abs=0.05) and np.std(alphas) < 0.1
    a, lev, var, fit = R.alpha_estimate(S.power_law(n, 0.5, G.SplitMix64(400)), h, 9, "MODWT")
    assert lev == [2, 3, 4, 5, 6, 7] and len(var) == 9 and fit[0] + 1.0 == a
    assert R.alpha_estimate(np.zeros(256), h, 6, "DWT")[0] is None and R.alpha_estimate(G.SplitMix64(1).normal(256), h, 6, "DWT")[0] is None      # zu wenige Stufen
    assert R.alpha_estimate(G.SplitMix64(1).normal(1024), h, 1, "MODWT")[1:2] == ([],)


def test_vanishing_moments_protect_the_exponent_against_a_trend():
    n = 4096
    x = S.trend_noise(n, 0.5, 100.0, G.SplitMix64(9)).x
    haar = R.alpha_estimate(x, F.daubechies(1), 9, "MODWT", True)[0]
    d4 = R.alpha_estimate(x, F.daubechies(2), 9, "MODWT", True)[0]
    circ = R.alpha_estimate(x, F.daubechies(2), 9, "DWT")[0]
    assert haar > 1.0 and abs(d4 - 0.5) < 0.2 and circ > 1.5


def test_count_above_and_top_energy_share_by_hand():
    v = np.array([4.0, -2.0, 1.0, 0.5, 0.0])
    assert R.count_above(v, 0.4) == 2 and R.count_above(v, 0.5) == 1 and R.count_above(v, 0.0) == 4 and R.count_above(np.zeros(3), 0.1) == 0 and R.count_above(np.array([]), 0.1) == 0
    assert R.top_energy_share(v, 1) == pytest.approx(16 / 21.25) and R.top_energy_share(v, 2) == pytest.approx(20 / 21.25) and R.top_energy_share(v, 10) == pytest.approx(1.0) and R.top_energy_share(np.zeros(3), 2) == 0.0


def test_band_level_by_hand():
    assert R.band_level(0.4) == 1 and R.band_level(0.5 - 1e-9) == 1 and R.band_level(0.25) == 2 and R.band_level(0.2) == 2 and R.band_level(0.07) == 3 and R.band_level(0.09375) == 3 and R.band_level(0.0625) == 4
    assert R.band_level(0.125) == 3 and R.band_level(0.01) == 6


def test_interior_max_ignores_the_boundary_coefficients_of_a_polynomial():
    h = F.daubechies(2)
    x = (np.arange(64) / 64) ** 1
    d, _ = T.dwt(x, h, 1)
    assert R.interior_max(d, 4) < 1e-14 and np.abs(d[0]) .max() > 0.1        # der Rand trägt den Sprung der periodischen Fortsetzung
    x2 = (np.arange(64) / 64) ** 2
    assert R.interior_max(T.dwt(x2, h, 1)[0], 4) > 1e-6


def test_shift_change_dwt_versus_modwt_for_an_impulse():
    h = F.daubechies(2)
    imp = S.impulse(256, 10).x
    assert R.shift_change(imp, h, 5, "DWT", 1) == pytest.approx(0.43301270189221924) and R.shift_change(imp, h, 5, "MODWT", 1) < 1e-12
    assert R.shift_change(imp, h, 5, "DWT", 0) == 0.0 and R.shift_change(imp, h, 5, "DWT", 256) < 1e-12
    assert R.shift_change(imp, h, 5, "DWT", 32) < 1e-12 and R.shift_change(imp, h, 5, "DWT", 2) > 0.1


def test_count_above_and_top_share_for_small_values():
    assert R.count_above(np.array([0.5, 0.1]), 0.1) == 2 and R.count_above(np.array([0.5, 0.04]), 0.1) == 1
    assert R.top_energy_share(np.array([0.5, 0.5]), 1) == pytest.approx(0.5) and R.top_energy_share(np.array([0.5, 0.5]), 2) == pytest.approx(1.0)
