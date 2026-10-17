"""DWT, MODWT, Mehrskalenanalyse und Wavelet-Varianz gegen Handrechnung, Definition als Summe und Invarianzen."""

import math

import numpy as np
import pytest

import dwt_filters as F
import dwt_transform as T
import signal_generator as G


def test_haar_mini_instance_by_hand(mini_x):
    h = F.daubechies(1)
    d, a = T.dwt(mini_x, h, 1)
    assert d[0].tolist() == pytest.approx([-1 / math.sqrt(2)] * 2) and a.tolist() == pytest.approx([3 / math.sqrt(2), 7 / math.sqrt(2)])
    d, a = T.dwt(mini_x, h, 2)
    assert d[1].tolist() == pytest.approx([-2.0]) and a.tolist() == pytest.approx([5.0])
    assert T.energy_by_level(d, a).tolist() == pytest.approx([1.0, 4.0, 25.0]) and T.idwt(d, a, h) == pytest.approx(mini_x)


def test_dwt_equals_the_explicit_sum_definition():
    h = F.daubechies(2)
    g = F.qmf(h)
    x = np.random.default_rng(1).normal(size=16)
    d, a = T.dwt(x, h, 2)
    c1 = np.array([sum(h[l] * x[(2 * k + l) % 16] for l in range(4)) for k in range(8)])
    d1 = np.array([sum(g[l] * x[(2 * k + l) % 16] for l in range(4)) for k in range(8)])
    d2 = np.array([sum(g[l] * c1[(2 * k + l) % 8] for l in range(4)) for k in range(4)])
    assert d[0] == pytest.approx(d1) and d[1] == pytest.approx(d2)


def test_dwt_is_an_orthogonal_matrix():
    h = F.daubechies(2)
    n, J = 16, 3
    cols = []
    for i in range(n):
        d, a = T.dwt(np.eye(1, n, i)[0], h, J)
        cols.append(np.concatenate(d + [a]))
    W = np.array(cols).T
    assert W @ W.T == pytest.approx(np.eye(n), abs=1e-12)


@pytest.mark.parametrize("p", [1, 2, 4, 8])
@pytest.mark.parametrize("n,J", [(64, 1), (256, 5), (512, 9)])
def test_perfect_reconstruction_and_energy_for_dwt_and_modwt(p, n, J):
    x = np.random.default_rng(2).normal(size=n)
    h = F.daubechies(p)
    d, a = T.dwt(x, h, J)
    assert T.idwt(d, a, h) == pytest.approx(x, abs=1e-12) and T.energy_by_level(d, a).sum() == pytest.approx((x ** 2).sum(), rel=1e-12)
    w, v = T.modwt(x, h, J)
    assert T.imodwt(w, v, h) == pytest.approx(x, abs=1e-12) and T.energy_by_level(w, v).sum() == pytest.approx((x ** 2).sum(), rel=1e-12)


def test_modwt_works_for_a_non_dyadic_length_and_dwt_rejects_it():
    x = np.random.default_rng(3).normal(size=300)
    h = F.daubechies(4)
    w, v = T.modwt(x, h, 4)
    assert T.imodwt(w, v, h) == pytest.approx(x, abs=1e-12) and len(w) == 4 and all(len(wj) == 300 for wj in w)
    with pytest.raises(ValueError):
        T.dwt(x, h, 3)
    for bad in (0, -1):
        with pytest.raises(ValueError):
            T.dwt(np.zeros(8), h, bad)
        with pytest.raises(ValueError):
            T.modwt(np.zeros(8), h, bad)


def test_modwt_equals_the_time_domain_pyramid():
    h = F.daubechies(2)
    ht, gt = h / math.sqrt(2), F.qmf(h) / math.sqrt(2)
    n = 40
    x = np.random.default_rng(4).normal(size=n)
    v = x.copy()
    w, vj = T.modwt(x, h, 3)
    for j in (1, 2, 3):
        step = 2 ** (j - 1)
        wj = np.array([sum(gt[l] * v[(t - step * l) % n] for l in range(4)) for t in range(n)])
        v = np.array([sum(ht[l] * v[(t - step * l) % n] for l in range(4)) for t in range(n)])
        assert w[j - 1] == pytest.approx(wj, abs=1e-12)
    assert vj == pytest.approx(v, abs=1e-12)


def test_modwt_is_exactly_shift_invariant_and_the_dwt_is_not():
    h = F.daubechies(2)
    x = np.random.default_rng(5).normal(size=64)
    w, v = T.modwt(x, h, 4)
    w2, v2 = T.modwt(np.roll(x, 7), h, 4)
    for a, b in zip(w, w2):
        assert b == pytest.approx(np.roll(a, 7), abs=1e-12)
    assert v2 == pytest.approx(np.roll(v, 7), abs=1e-12)
    imp = np.eye(1, 64, 10)[0]
    e0 = T.energy_by_level(*T.dwt(imp, h, 4))
    e1 = T.energy_by_level(*T.dwt(np.roll(imp, 1), h, 4))
    assert np.abs(e0 - e1).max() > 0.1


def test_mra_is_additive_and_equals_the_single_level_reconstruction():
    h = F.daubechies(3)
    x = np.random.default_rng(6).normal(size=100)
    D, S = T.mra(x, h, 4)
    assert sum(D) + S == pytest.approx(x, abs=1e-12)
    w, v = T.modwt(x, h, 4)
    zeros = [np.zeros(100) for _ in range(4)]
    for j in range(4):
        only = [w[i] if i == j else zeros[i] for i in range(4)]
        assert D[j] == pytest.approx(T.imodwt(only, np.zeros(100), h), abs=1e-12)
    assert S == pytest.approx(T.imodwt(zeros, v, h), abs=1e-12)


def test_stage_filters_are_power_complementary_and_stage_response_matches():
    h = F.daubechies(2)
    bands, low = T.modwt_filters(64, h, 4)
    assert sum(np.abs(g) ** 2 for g in bands) + np.abs(low) ** 2 == pytest.approx(np.ones(64), abs=1e-12)
    f = np.fft.fftfreq(64)
    for j in (1, 2, 3):
        for k in (1, 5, 9):
            assert T.stage_response(h, j, f[k]) == pytest.approx(abs(bands[j - 1][k]) ** 2, abs=1e-12)
    assert T.stage_response(h, 3, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_boundary_length_by_hand_and_the_boundary_property():
    assert T.boundary_length(1, 4) == 4 and T.boundary_length(3, 4) == 22 and T.boundary_length(2, 2) == 4 and T.boundary_length(5, 8) == 218
    h = F.daubechies(2)
    x = np.random.default_rng(7).normal(size=256)
    y = x.copy()
    y[-1] += 5.0
    w, _ = T.modwt(x, h, 4)
    w2, _ = T.modwt(y, h, 4)
    for j in range(1, 5):
        start = T.boundary_length(j, 4) - 1
        assert w[j - 1][start:255] == pytest.approx(w2[j - 1][start:255], abs=1e-12)
        assert np.abs(w[j - 1][:start] - w2[j - 1][:start]).max() > 1e-6 or j == 1


def test_energy_by_level_and_variances_by_hand():
    assert T.energy_by_level([np.array([1.0, 2.0]), np.array([3.0])], np.array([4.0])).tolist() == [5.0, 9.0, 16.0]
    w = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 2.0, 2.0, 2.0])]
    assert T.modwt_variance(w, 2).tolist() == [7.5, 4.0]
    assert T.modwt_variance(w, 2, boundary=True)[0] == pytest.approx(np.mean([2.0, 3.0, 4.0]) ** 0 * np.mean(np.array([2.0, 3.0, 4.0]) ** 2))      # L_1 = 2: t >= 1
    assert T.modwt_variance(w, 2, boundary=True)[1] == 4.0                                                          # L_2 = 4: nur t = 3 bleibt
    assert math.isnan(T.modwt_variance([np.array([1.0, 2.0, 3.0])], 4, boundary=True)[0])                          # L_1 = 4: kein Koeffizient übrig
    assert T.dwt_variance([np.array([1.0, 3.0]), np.array([2.0])]).tolist() == [5.0 / 2, 4.0 / 4]


def test_white_noise_wavelet_variance_is_two_to_the_minus_j():
    x = G.SplitMix64(11).normal(200 * 256).reshape(200, 256)
    h = F.daubechies(2)
    dv = np.mean([T.dwt_variance(T.dwt(r, h, 4)[0]) for r in x], axis=0)
    mv = np.mean([T.modwt_variance(T.modwt(r, h, 4)[0], 4) for r in x], axis=0)
    assert dv == pytest.approx([0.5, 0.25, 0.125, 0.0625], rel=0.05) and mv == pytest.approx([0.5, 0.25, 0.125, 0.0625], rel=0.05)


def test_fit_levels_and_slope_fit():
    assert T.fit_levels(4096, 9) == [2, 3, 4, 5, 6, 7] and T.fit_levels(256, 6) == [2, 3] and T.fit_levels(1024, 1) == [] and T.fit_levels(4096, 9, count_min=16) == [2, 3, 4, 5, 6, 7, 8]
    lev = [2, 3, 4, 5]
    s = T.slope_fit(lev, [2.0 ** (0.5 * j + 1.0) for j in lev])
    assert s == pytest.approx((0.5, 1.0))
    assert T.slope_fit([2, 3], [1.0, 2.0]) is None and T.slope_fit([2, 3, 4, 5], [1.0, float("nan"), 4.0, -1.0]) is None
    assert T.slope_fit([2, 3, 4, 5], [1.0, float("nan"), 4.0, 8.0]) == pytest.approx((1.0, -2.0), abs=1e-9)


def test_gph_alpha_recovers_an_exact_power_law_spectrum():
    n, alpha = 512, 0.8
    k = np.arange(1, n // 2)
    spec = np.zeros(n, dtype=complex)
    ph = np.exp(2j * np.pi * np.random.default_rng(8).uniform(size=len(k)))
    spec[1:n // 2] = (k / n) ** (-alpha / 2) * ph
    spec[n // 2 + 1:] = np.conj(spec[1:n // 2][::-1])
    x = np.real(np.fft.ifft(spec))
    assert T.gph_alpha(x, 40) == pytest.approx(alpha, abs=1e-9)


def test_slope_fit_skips_zero_and_negative_variances():
    assert T.slope_fit([2, 3, 4, 5], [1.0, 0.0, 4.0, 8.0]) == pytest.approx((1.0, -2.0), abs=1e-9)
    assert T.slope_fit([2, 3, 4, 5], [1.0, -0.5, 4.0, 8.0]) == pytest.approx((1.0, -2.0), abs=1e-9)
