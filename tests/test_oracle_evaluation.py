"""Auswertung: Einstellungen, Messung, Analyse je Signaltyp und die Experimentfunktionen (kleine Instanzen von Hand, feste Werte)."""

import math
import pickle

import numpy as np
import pytest

import dwt_constants as C
import dwt_evaluation as ev
import dwt_filters as F
import dwt_signals as S
import dwt_transform as T
import signal_generator as G


def test_settings_flags_and_names():
    s = ev.Settings()
    assert s.has_noise is False and s.discard_boundary is False and (s.n, s.signal, s.p, s.levels, s.transform, s.boundary) == (2048, "Potenzgesetz-Rauschen", 2, 6, "MODWT", "periodisch")
    assert ev.Settings(signal="Ton", noise="an").has_noise is True and ev.Settings(signal="Potenzgesetz-Rauschen", noise="an").has_noise is False
    assert ev.Settings(boundary="Randkoeffizienten verwerfen").discard_boundary is True and ev.Settings(boundary="Randkoeffizienten verwerfen", transform="DWT").discard_boundary is False
    assert ev.wavelet_name(1) == "Haar (p = 1)" and ev.wavelet_name(4) == "D8 (p = 4)" and ev.wavelet_name(8) == "D16 (p = 8)"


def test_noise_sigma_by_hand_and_the_german_decimal_comma():
    assert ev.noise_sigma(0) == pytest.approx(math.sqrt(0.5)) and ev.noise_sigma(10) == pytest.approx(math.sqrt(0.05)) and ev.noise_sigma(20) == pytest.approx(math.sqrt(0.005))
    assert ev.de(0.12345, 3) == "0,123" and ev.de(2.0) == "2,0" and ev.de(-0.5, 2) == "-0,50"


def test_make_signal_for_each_type():
    n = 256
    assert ev.make_signal(ev.Settings(n=n, p1=0.5, seed=3)).x.tolist() == S.power_law(n, 0.5, G.SplitMix64(3)).tolist()
    assert ev.make_signal(ev.Settings(n=n, signal="Sprung im glatten Signal", p1=3.0, p2=0.3, p3=1.0)).x == pytest.approx(S.smooth_jump(n, 3.0, 77, 1.0).x)
    assert ev.make_signal(ev.Settings(n=n, signal="Sprung im glatten Signal", p1=3.0, p2=0.3, p3=1.0)).events == (77.0,)
    assert np.flatnonzero(ev.make_signal(ev.Settings(n=n, signal="Impuls", p1=0.25)).x).tolist() == [64]
    assert np.flatnonzero(ev.make_signal(ev.Settings(n=n, signal="Impuls", p1=1.0)).x).tolist() == [255]
    assert ev.make_signal(ev.Settings(n=n, signal="Ton", p1=0.07)).x == pytest.approx(S.tone(n, 0.07, 0.3).x)
    t = ev.make_signal(ev.Settings(n=n, signal="Trend plus Rauschen", p1=0.5, p2=100.0, seed=3))
    assert t.x.tolist() == S.trend_noise(n, 0.5, 100.0, G.SplitMix64(3)).x.tolist() and t.alpha == 0.5
    with pytest.raises(ValueError):
        ev.make_signal(ev.Settings(signal="Rechteck"))


def test_measurement_adds_noise_only_for_the_noisy_signal_types():
    s = ev.Settings(n=256, signal="Ton", p1=0.07, noise="an", snr=10, seed=5)
    sig, x = ev.build_measurement(s)
    assert x - sig.x == pytest.approx(ev.noise_sigma(10) * G.SplitMix64(5).normal(256)) and (x - sig.x).std() == pytest.approx(ev.noise_sigma(10), rel=0.2)
    assert ev.build_measurement(ev.Settings(n=256, signal="Ton", p1=0.07, seed=5))[1].tolist() == sig.x.tolist()
    sig2, x2 = ev.build_measurement(ev.Settings(n=256, noise="an", snr=0, seed=5))
    assert x2.tolist() == sig2.x.tolist()


def test_analyse_default_golden_and_invariants():
    a = ev.analyse(ev.Settings())
    assert a.alpha_hat == pytest.approx(0.5529357, abs=1e-6) and a.fit_levels == [2, 3, 4, 5, 6] and a.gph == pytest.approx(0.3421498, abs=1e-6) and a.alpha_true == 0.5
    assert a.coef_count == 7 * 2048 and a.recon_error < 1e-12 and a.energy_error < 1e-12 and a.mra_error < 1e-12 and a.fractions.sum() == pytest.approx(1.0) and a.shift_change == 0.0
    assert len(a.details) == 6 and len(a.variance) == 6 and len(a.mra_details) == 6 and a.band_level is None and a.band_fraction is None and a.fit == pytest.approx((-0.4470643, 1.1676647), abs=1e-6)
    d = ev.analyse(ev.Settings(transform="DWT"))
    assert d.coef_count == 2048 and d.recon_error < 1e-12 and d.alpha_hat is not None and abs(d.alpha_hat - a.alpha_hat) < 0.2
    assert ev.analyse(ev.Settings(n=256, levels=6)).alpha_hat is None and ev.analyse(ev.Settings(n=256, levels=6)).fit is None


def test_analyse_the_five_presets_golden():
    a = ev.analyse(ev.Settings(n=1024, signal="Sprung im glatten Signal", p1=3.0, p2=0.3, p3=1.0, p=4, levels=6, transform="DWT"))
    assert a.sparse_count == 73 and a.sparse_top == pytest.approx(0.9932133, abs=1e-6) and a.alpha_true is None and a.gph is None and a.recon_error < 1e-12
    b = ev.analyse(ev.Settings(n=256, signal="Impuls", p1=0.04, p=2, levels=5, transform="DWT", shift=1))
    assert b.shift_change == pytest.approx(0.4330127, abs=1e-6) and ev.analyse(ev.Settings(n=256, signal="Impuls", p1=0.04, p=2, levels=5, transform="MODWT", shift=1)).shift_change < 1e-12
    c = ev.analyse(ev.Settings(n=4096, p1=0.5, p=2, levels=9))
    assert c.alpha_hat == pytest.approx(0.4390636, abs=1e-6) and c.gph == pytest.approx(0.5519088, abs=1e-6)
    d = ev.analyse(ev.Settings(n=4096, signal="Trend plus Rauschen", p1=0.5, p2=100.0, p=1, levels=9, boundary="Randkoeffizienten verwerfen"))
    assert d.alpha_hat == pytest.approx(1.2542438, abs=1e-6) and d.gph == pytest.approx(2.0098055, abs=1e-6)
    d2 = ev.analyse(ev.Settings(n=4096, signal="Trend plus Rauschen", p1=0.5, p2=100.0, p=2, levels=9, boundary="Randkoeffizienten verwerfen"))
    assert d2.alpha_hat == pytest.approx(0.4366553, abs=1e-6) and ev.analyse(ev.Settings(n=4096, signal="Trend plus Rauschen", p1=0.5, p2=100.0, p=2, levels=9)).alpha_hat > 1.5
    e = ev.analyse(ev.Settings(n=1024, signal="Ton", p1=0.09375, levels=6))
    assert e.band_level == 3 and e.band_fraction == pytest.approx(0.7134837, abs=1e-6)


def test_boundary_option_only_changes_the_modwt_variance():
    base = dict(n=1024, signal="Trend plus Rauschen", p1=0.5, p2=100.0, p=2, levels=7)
    a = ev.analyse(ev.Settings(**base, transform="MODWT"))
    b = ev.analyse(ev.Settings(**base, transform="MODWT", boundary="Randkoeffizienten verwerfen"))
    assert a.alpha_hat != b.alpha_hat and a.variance.tolist() != b.variance.tolist()
    d1 = ev.analyse(ev.Settings(**base, transform="DWT"))
    d2 = ev.analyse(ev.Settings(**base, transform="DWT", boundary="Randkoeffizienten verwerfen"))
    assert d1.alpha_hat == d2.alpha_hat


def test_sparsity_does_not_depend_on_the_selected_transform_and_shift_moves_the_signal():
    kw = dict(n=1024, signal="Sprung im glatten Signal", p1=3.0, p2=0.3, p3=1.0, p=4, levels=6)
    assert ev.analyse(ev.Settings(**kw, transform="DWT")).sparse_count == ev.analyse(ev.Settings(**kw, transform="MODWT")).sparse_count
    a = ev.analyse(ev.Settings(**kw, shift=10))
    assert a.x.tolist() == np.roll(ev.make_signal(ev.Settings(**kw)).x, 10).tolist()


def test_band_level_only_for_tones_inside_the_chosen_levels():
    assert ev.analyse(ev.Settings(n=1024, signal="Ton", p1=0.07, levels=6)).band_level == 3
    assert ev.analyse(ev.Settings(n=1024, signal="Ton", p1=0.4, levels=6)).band_level == 1
    assert ev.analyse(ev.Settings(n=1024, signal="Ton", p1=0.01, levels=4)).band_level is None and ev.analyse(ev.Settings(n=1024, signal="Ton", p1=0.01, levels=4)).band_fraction is None
    assert ev.analyse(ev.Settings(n=1024, signal="Impuls", p1=0.1)).band_level is None


def test_analysis_is_picklable():
    a = ev.analyse(ev.Settings(n=256, signal="Ton", p1=0.07, levels=4, noise="an", snr=10))
    b = pickle.loads(pickle.dumps(a))
    assert b.recon_error == a.recon_error and b.settings == a.settings and (b.x == a.x).all()


def test_sweep_moments_small_instance():
    r = ev.sweep_moments(ps=(1, 2), n=64, jump_ps=(1, 4), jump_n=256, jump_levels=4)
    assert [d["p"] for d in r["degrees"]] == [1, 2] and r["degrees"][0]["max"][0] == 0.0 and r["degrees"][0]["max"][1] == pytest.approx(0.011048543, abs=1e-9)
    assert r["degrees"][1]["max"][1] < 1e-15 and r["degrees"][1]["max"][2] == pytest.approx(0.00029900998, abs=1e-10) and r["degrees"][1]["max"][4] == pytest.approx(0.0015658687, abs=1e-9)
    assert [(j["p"], j["count"], j["total"]) for j in r["jump"]] == [(1, 242, 256), (4, 57, 256)] and [j["top"] for j in r["jump"]] == pytest.approx([0.95281066, 0.99257124], abs=1e-8)


def test_sweep_shift_small_instance():
    rows = ev.sweep_shift(p=1, n=64, levels=3, count=4)
    assert [r["name"] for r in rows] == ["Impuls", "Ton f = 0,07", "Ton f = 0,1"] and all(len(r["dwt"]) == 4 == len(r["modwt"]) for r in rows)
    assert rows[1]["dwt"] == pytest.approx([0.057783, 0.090083, 0.081263, 0.083777], abs=1e-6) and rows[2]["dwt"] == pytest.approx([0.024259, 0.024259, 0.009813, 0.009813], abs=1e-6)
    assert all(max(r["modwt"]) < 1e-12 for r in rows) and max(ev.sweep_shift(p=2, n=256, levels=5, count=16)[0]["dwt"]) == pytest.approx(0.5257912, abs=1e-6)


def test_sweep_alpha_small_instance():
    r = ev.sweep_alpha(values=(0.5,), ps=(1, 2), n=1024, levels=7, runs=6, trend_ps=(1, 2), m=32)
    by = {(x["estimator"], x["p"]): x for x in r["stationary"]}
    assert set(by) == {("DWT", 1), ("MODWT", 1), ("DWT", 2), ("MODWT", 2), ("GPH", 0)} and by[("MODWT", 2)]["mean"] == pytest.approx(0.501057, abs=1e-5) and by[("GPH", 0)]["sd"] == pytest.approx(0.441589, abs=1e-5)
    assert [t["p"] for t in r["trend"]] == [1, 2] and r["trend"][0]["inner_mean"] == pytest.approx(1.536099, abs=1e-5) and r["trend"][1]["dwt_mean"] == pytest.approx(1.933861, abs=1e-5)
    assert r["trend"][1]["inner_mean"] == pytest.approx(0.464082, abs=1e-5) and r["trend_gph"][0] == pytest.approx(2.008817, abs=1e-5) and r["runs"] == 6 and r["n"] == 1024
    assert [(x["p"], round(x["mean"], 5)) for x in r["plain_inner"]] == [(1, 0.39718), (2, 0.41289)]


def test_sweep_bands_small_instance():
    rows = ev.sweep_bands(ps=(1, 2), levels_list=(2, 3), n=256)
    assert rows[0]["fractions"][2] == pytest.approx(0.5900971, abs=1e-6) and rows[1]["fractions"][3] == pytest.approx(0.7134837, abs=1e-6)
    assert rows[1]["response"] == pytest.approx([0.4689761, 0.7134837, 0.4709709, 0.0224396, 0.0], abs=1e-6) and rows[1]["response"][1] == pytest.approx(rows[1]["fractions"][3], abs=1e-9)


def test_sweep_variance_small_instance():
    r = ev.sweep_variance(p=1, n=64, levels=3, runs=20)
    assert r["dwt_rel"] == pytest.approx([0.233703, 0.34655, 0.422074], abs=1e-6) and r["modwt_rel"] == pytest.approx([0.207261, 0.242427, 0.267279], abs=1e-6)
    assert r["ratio"] == pytest.approx([1.127575, 1.4295, 1.579152], abs=1e-6) and r["dwt_count"] == 64 and r["modwt_count"] == 256 and r["dwt_mean"][0] == pytest.approx(0.485934, abs=1e-6)
    assert all(abs(m - 2.0 ** -j) < 0.05 for j, m in enumerate(r["modwt_mean"], start=1))


def test_band_level_keeps_the_highest_level_and_drops_frequencies_above_a_half():
    assert ev.analyse(ev.Settings(n=1024, signal="Ton", p1=0.07, levels=3)).band_level == 3
    assert ev.analyse(ev.Settings(n=1024, signal="Ton", p1=0.07, levels=2)).band_level is None
    a = ev.analyse(ev.Settings(n=1024, signal="Ton", p1=0.6, levels=4))
    assert a.band_level is None and a.band_fraction is None
