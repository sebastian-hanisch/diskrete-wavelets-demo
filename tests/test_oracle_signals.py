"""Testsignale: Handrechnung, Spektrum des Potenzgesetz-Rauschens gegen die Theorie, Wahrheit."""

import math

import numpy as np
import pytest

import dwt_signals as S
import signal_generator as G

TWO_PI = 2 * math.pi


def test_power_law_is_real_zero_mean_deterministic_and_has_the_right_spectrum():
    n = 512
    x = S.power_law(n, 1.0, G.SplitMix64(5))
    assert x.shape == (n,) and abs(x.mean()) < 1e-12 and x.tolist() == S.power_law(n, 1.0, G.SplitMix64(5)).tolist() and x.tolist() != S.power_law(n, 1.0, G.SplitMix64(6)).tolist()
    for alpha in (0.0, 1.0, 2.0):
        spec = np.mean([np.abs(np.fft.rfft(S.power_law(n, alpha, G.SplitMix64(100 + r)))[1:n // 2]) ** 2 for r in range(300)], axis=0)
        k = np.arange(1, n // 2)
        slope = np.polyfit(np.log(k / n), np.log(spec), 1)[0]
        assert slope == pytest.approx(-alpha, abs=0.06)


def test_power_law_white_noise_has_variance_two():
    x = np.concatenate([S.power_law(256, 0.0, G.SplitMix64(300 + r)) for r in range(200)])
    assert x.var() == pytest.approx(2.0 * (1 - 1.5 / 256), rel=0.03)         # Normierung der Spektralsynthese: (2 n - 3) / n


def test_power_law_signal_and_trend_noise_carry_the_truth():
    s = S.power_law_signal(256, 0.7, G.SplitMix64(1))
    assert s.alpha == 0.7 and s.events == () and s.x.tolist() == S.power_law(256, 0.7, G.SplitMix64(1)).tolist()
    t = S.trend_noise(256, 0.5, 100.0, G.SplitMix64(1))
    base = S.power_law(256, 0.5, G.SplitMix64(1))
    assert t.alpha == 0.5 and t.x == pytest.approx(base + 100.0 * base.std() * (np.arange(256) / 256 - 0.5))
    assert S.trend_noise(256, 0.5, 0.0, G.SplitMix64(1)).x.tolist() == base.tolist()
    assert (t.x[-1] - t.x[0]) > 90 * base.std()


def test_smooth_jump_by_hand():
    s = S.smooth_jump(8, 1.0, 4, 2.0)
    t = np.arange(8)
    assert s.x == pytest.approx(np.sin(TWO_PI * t / 8) + 2.0 * (t >= 4)) and s.events == (4.0,) and s.alpha is None
    assert s.x[3] == pytest.approx(math.sin(TWO_PI * 3 / 8)) and s.x[4] == pytest.approx(0.0 + 2.0, abs=1e-12)


def test_impulse_and_tone_by_hand():
    i = S.impulse(16, 5)
    assert np.flatnonzero(i.x).tolist() == [5] and i.x.sum() == 1.0 and i.events == (5.0,) and i.alpha is None
    t = S.tone(8, 0.25, 0.0)
    assert t.x == pytest.approx([1, 0, -1, 0, 1, 0, -1, 0], abs=1e-12) and t.events == () and S.tone(8, 0.25, math.pi / 2).x[0] == pytest.approx(0.0, abs=1e-12)
    assert S.tone(8, 0.125, 1.0).x[1] == pytest.approx(math.cos(TWO_PI * 0.125 + 1.0))
