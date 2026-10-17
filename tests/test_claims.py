"""Jede Zahl und Aussage in den Texten der App (Presets, Einleitung, Experiment-Überschriften) und in der README ist hier an Rechnung oder Messung gebunden."""

from pathlib import Path

import numpy as np
import pytest

import dwt_constants as C
import dwt_evaluation as ev
import dwt_filters as F
import dwt_transform as T
from sweeps_cache import get

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")


def preset_settings(name, **over):
    p = C.PRESETS[name]
    base = dict(n=p["n"], signal=p["signal"], p1=p["params"][0], p2=p["params"][1], p3=p["params"][2], p=p["p"], levels=p["levels"], transform=p["transform"], boundary=p["boundary"], shift=p["shift"],
                noise=p["noise"], snr=p["snr"], seed=p["seed"])
    return ev.Settings(**{**base, **over})


def A(name, **over):
    return ev.analyse(preset_settings(name, **over))


def de(x, k=3):
    return f"{x:.{k}f}".replace(".", ",")


def sci(x):
    return f"{x:.1e}".replace(".", ",")


# --- Presets ----------------------------------------------------------------------------------------------------------------------------------


def test_preset_jump_has_few_large_coefficients():
    name = "Sprung im glatten Signal: wenige große Koeffizienten"
    a = A(name)
    assert a.settings.p == 4 and a.settings.levels == 6 and a.sparse_count == 73 and a.sparse_top > 0.99 and "D8 (p = 4)" in C.PRESET_HELP[name] and "sechs Stufen" in C.PRESET_HELP[name]


def test_preset_impulse_changes_with_the_dwt_and_not_with_the_modwt():
    name = "Impuls verschieben: die DWT ändert sich, die MODWT nicht"
    assert A(name).shift_change > 0.4 and A(name, transform="MODWT").shift_change < 1e-12 and A(name).settings.shift == 1
    assert "um einen Abtastwert verschoben" in C.PRESET_HELP[name] and "MODWT" in C.PRESET_HELP[name]


def test_preset_power_law_estimates_the_exponent_within_the_tolerance():
    name = "Potenzgesetz-Rauschen: der Exponent aus der Wavelet-Varianz"
    a = A(name)
    assert a.settings.n == 4096 and a.settings.p == 2 and a.settings.transform == "MODWT" and abs(a.alpha_hat - 0.5) < C.OK_ALPHA and a.alpha_true == 0.5
    assert "f^−0,5" in C.PRESET_HELP[name] and "n = 4096, D4, MODWT" in C.PRESET_HELP[name]


def test_preset_trend_haar_fails_and_d4_does_not():
    name = "Trend plus Rauschen: Haar versagt, D4 nicht"
    haar = A(name)
    assert haar.settings.p == 1 and de(haar.alpha_hat, 2) == "1,25" and "etwa 1,25" in C.PRESET_HELP[name] and haar.settings.boundary == "Randkoeffizienten verwerfen"
    for p in (2, 3, 4):
        assert abs(A(name, p=p).alpha_hat - 0.5) < 0.1
    assert "100 Standardabweichungen" in C.PRESET_HELP[name] and "in der Nähe von 0,5" in C.PRESET_HELP[name] and A(name, p=2, boundary="periodisch").alpha_hat > 1.5


def test_preset_tone_in_the_band_middle_puts_71_percent_in_the_stage():
    name = "Ton in der Bandmitte: die Stufen sind keine scharfen Bänder"
    a = A(name)
    assert a.settings.p1 == 0.09375 == 0.75 / 8 and a.band_level == 3 and de(100 * a.band_fraction, 0) == "71" and "Stufe 3" in C.PRESET_HELP[name] and "etwa 71 %" in C.PRESET_HELP[name] and "0,09375" in C.PRESET_HELP[name]


# --- Verschwindende Momente, Sparsität ---------------------------------------------------------------------------------------------------------


def test_moment_numbers_in_the_readme():
    res = get("moments")
    by = {r["p"]: r["max"] for r in res["degrees"]}
    assert max(max(by[p][:p]) for p in by) < 2e-15 and "höchstens 1,9e-15" in README and sci(max(max(by[p][:p]) for p in by)) == "1,9e-15"
    assert " / ".join(sci(v) for v in by[1][1:]) == "2,8e-03 / 5,4e-03 / 8,1e-03 / 1,1e-02" and "2,8e-3 / 5,4e-3 / 8,1e-3 / 1,1e-2" in README
    assert " / ".join(sci(v) for v in by[2][2:]) == "1,9e-05 / 5,5e-05 / 1,1e-04" and "1,9e-5 / 5,5e-5 / 1,1e-4" in README
    assert " / ".join(sci(v) for v in by[3][3:]) == "2,0e-07 / 7,8e-07" and "2,0e-7 / 7,8e-7" in README and sci(by[4][4]) == "2,9e-09" and "2,9e-9 bei Grad 4" in README
    assert all(by[p][p] > 1e-9 for p in (1, 2, 3, 4)) and (C.EXP_MOM_N, C.EXP_JUMP_N, C.EXP_JUMP_J) == (256, 1024, 6)


def test_sparsity_numbers_in_the_readme():
    jump = {r["p"]: r for r in get("moments")["jump"]}
    assert [jump[p]["count"] for p in (1, 2, 4, 8)] == [429, 127, 73, 91] and "429 / 127 / 73 / 91 von 1024" in README
    assert " / ".join(de(100 * jump[p]["top"], 1) for p in (1, 2, 4, 8)) == "95,4 / 98,7 / 99,3 / 99,3" and "95,4 / 98,7 / 99,3 / 99,3 % der Energie" in README and C.SPARSE_TOP == 16


# --- Verschiebungsvarianz ----------------------------------------------------------------------------------------------------------------------


def test_shift_numbers_in_the_readme():
    rows = {r["name"]: r for r in get("shift")}
    imp = rows["Impuls"]["dwt"]
    assert " / ".join(de(v) for v in imp) == "0,433 / 0,526 / 0,301 / 0,174 / 0,091 / 0,025" and "0,433 / 0,526 / 0,301 / 0,174 / 0,091 / 0,025" in README and de(max(imp), 2) == "0,53" and "bis zu 0,53" in README
    assert de(max(rows["Ton f = 0,07"]["dwt"])) == "0,036" and de(max(rows["Ton f = 0,1"]["dwt"])) == "0,044" and "um bis zu 0,036" in README and "um bis zu 0,044" in README
    assert all(max(r["modwt"]) < 1e-12 for r in rows.values()) and "unter 1e-12" in README and (C.EXP_SHIFT_N, C.EXP_SHIFT_J, C.EXP_SHIFT_COUNT, C.EXP_SHIFT_P) == (256, 5, 16, 2)
    assert "bei einem Impuls bis über 0,5" in APP and max(imp) > 0.5


# --- Exponent aus der Wavelet-Varianz ---------------------------------------------------------------------------------------------------------


def test_alpha_numbers_in_the_readme():
    res = get("alpha")
    by = {(r["alpha"], r["estimator"], r["p"]): r for r in res["stationary"]}

    def row(est, p):
        return " / ".join(f"{de(by[(a, est, p)]['mean'])} ± {de(by[(a, est, p)]['sd'])}" for a in (0.0, 0.5, 1.0))
    assert row("MODWT", 2) == "0,002 ± 0,051 / 0,492 ± 0,048 / 0,995 ± 0,049" and "0,002 ± 0,051 / 0,492 ± 0,048 / 0,995 ± 0,049" in README
    assert row("DWT", 2) == "-0,002 ± 0,063 / 0,490 ± 0,058 / 0,996 ± 0,062" and "−0,002 ± 0,063 / 0,490 ± 0,058 / 0,996 ± 0,062" in README
    assert row("GPH", 0) == "-0,008 ± 0,177 / 0,518 ± 0,184 / 1,026 ± 0,181" and "−0,008 ± 0,177 / 0,518 ± 0,184 / 1,026 ± 0,181" in README
    assert round(by[(0.5, "GPH", 0)]["sd"] / by[(0.5, "MODWT", 2)]["sd"]) == 4 and "etwa 4-mal kleiner" in README and "etwa 4-mal kleiner" in ev.verdict_alpha(res)[1]
    assert (res["n"], res["levels"], res["runs"], C.EXP_GPH_M, C.EXP_SEED) == (4096, 9, 200, 64, 960000) and "Seed 960000" in README and "200 Läufe bei der Exponentenschätzung, 500 bei der Streuung" in README
    assert all(abs(r["mean"] - r["alpha"]) < 0.05 for r in res["stationary"] if r["estimator"] != "GPH")


def test_trend_numbers_in_the_readme():
    res = get("alpha")
    tr = {r["p"]: r for r in res["trend"]}
    assert f"Haar {de(tr[1]['inner_mean'])} ± {de(tr[1]['inner_sd'])}" == "Haar 1,264 ± 0,014" and "Haar 1,264 ± 0,014" in README
    assert " / ".join(de(tr[p]["inner_mean"]) for p in (2, 3, 4)) == "0,483 / 0,483 / 0,482" and " / ".join(de(tr[p]["inner_sd"]) for p in (2, 3, 4)) == "0,052 / 0,055 / 0,057"
    assert "0,483 / 0,483 / 0,482 (± 0,052 / 0,055 / 0,057)" in README
    assert " / ".join(de(tr[p]["dwt_mean"]) for p in (1, 2, 3, 4)) == "1,263 / 1,910 / 1,908 / 1,872" and "1,263 / 1,910 / 1,908 / 1,872 bei p = 1 bis 4" in README
    assert f"{de(res['trend_gph'][0])} ± {de(res['trend_gph'][1])}" == "2,000 ± 0,010" and "das Periodogramm: 2,000 ± 0,010" in README
    assert " / ".join(de(r["mean"]) for r in res["plain_inner"]) == "0,484 / 0,496 / 0,498" and "ohne Trend 0,484 / 0,496 / 0,498 bei p = 1 / 2 / 3" in README
    assert (res["trend_sd"], res["trend_alpha"]) == (100.0, 0.5) and "1,3 bis 1,9 statt 0,5 bei 100 Standardabweichungen" in README and "1,3 bis 1,9 statt 0,5" in APP
    assert round(tr[1]["dwt_mean"], 1) == 1.3 and round(tr[2]["dwt_mean"], 1) == 1.9


# --- Frequenzselektivität, Streuung -----------------------------------------------------------------------------------------------------------


def test_band_numbers_in_the_readme_and_the_app():
    rows = {r["p"]: r for r in get("bands")}
    for p, line in ((1, "59,0 / 54,0 / 52,9"), (2, "72,8 / 71,3 / 71,3"), (4, "85,3 / 85,2 / 85,2")):
        assert " / ".join(de(100 * rows[p]["fractions"][j], 1) for j in (2, 3, 4)) == line and line in README
    assert de(100 * rows[8]["fractions"][2], 1) == de(100 * rows[8]["fractions"][4], 1) == "94,1" and "bei D16 94,1 % in allen drei" in README
    resp = {p: [de(v, 2) for v in (rows[p]["response"][0], rows[p]["response"][2])] for p in rows}
    assert resp == {1: ["0,41", "0,43"], 2: ["0,47", "0,47"], 4: ["0,49", "0,49"], 8: ["0,50", "0,50"]}
    assert "0,41 / 0,43 (Haar), 0,47 / 0,47 (D4), 0,49 / 0,49 (D8) und 0,50 (D16)" in README
    assert de(100 * rows[1]["fractions"][3], 0) == "54" and de(100 * rows[8]["fractions"][3], 0) == "94" and "54 % (Haar) bis 94 % (D16)" in APP


def test_variance_numbers_in_the_readme():
    res = get("variance")
    assert " / ".join(de(v) for v in res["dwt_rel"]) == "0,064 / 0,087 / 0,128 / 0,175 / 0,229 / 0,350" and "DWT 0,064 / 0,087 / 0,128 / 0,175 / 0,229 / 0,350" in README
    assert " / ".join(de(v) for v in res["modwt_rel"]) == "0,057 / 0,064 / 0,089 / 0,125 / 0,166 / 0,236" and "MODWT 0,057 / 0,064 / 0,089 / 0,125 / 0,166 / 0,236" in README
    assert de(min(res["ratio"]), 2) == "1,11" and de(max(res["ratio"]), 2) == "1,49" and "das 1,11- bis 1,49-Fache" in README
    assert " / ".join(de(v) for v in res["modwt_mean"]) == "0,500 / 0,249 / 0,125 / 0,062 / 0,031 / 0,016" and "0,500 / 0,249 / 0,125 / 0,062 / 0,031 / 0,016" in README
    assert (res["modwt_count"], res["dwt_count"], res["n"], res["runs"]) == (7168, 1024, 1024, 500) and "7168 statt 1024 Koeffizienten" in README
    assert all(abs(m - 2.0 ** -j) < 0.003 for j, m in enumerate(res["modwt_mean"], start=1))
    rel = [np.sqrt(2 ** (j + 1) / 1024) for j in range(1, 7)]
    assert res["dwt_rel"] == pytest.approx(rel, rel=0.1)


# --- Zerlegung, Konstanten, Mini-Instanz ------------------------------------------------------------------------------------------------------


def test_reconstruction_claim_and_the_mini_instance():
    for p in (1, 2, 4, 8):
        for n, J in ((64, 1), (256, 5), (512, 9)):
            x = np.random.default_rng(2).normal(size=n)
            h = F.daubechies(p)
            assert np.abs(T.idwt(*T.dwt(x, h, J), h) - x).max() < 1e-12 and np.abs(T.imodwt(*T.modwt(x, h, J), h) - x).max() < 1e-12
    assert "auf 1e-12 und besser (n = 64 bis 512, 1 bis 9 Stufen, p = 1, 2, 4, 8)" in README
    d, a = T.dwt(np.array([1.0, 2.0, 3.0, 4.0]), F.daubechies(1), 2)
    assert d[1][0] == pytest.approx(-2.0) and a[0] == pytest.approx(5.0) and "d = −1/√2 je Wert, a = (3/√2, 7/√2), zweite Stufe d = −2, a = 5" in README


def test_method_numbers_in_the_readme_and_the_app_match_the_constants():
    assert (min(C.N_CHOICES), max(C.N_CHOICES)) == (256, 4096) and "n = 256 bis 4096, p = 1, 2, 3, 4, 6, 8 (Haar bis D16)" in README and C.WAVELET_P == (1, 2, 3, 4, 6, 8)
    assert C.OK_ALPHA == 0.15 and C.SPARSE_REL == 1e-3 and "1e-3-Fachen" in README and "mindestens 32" in README and "mindestens 32 ist" in APP and T.fit_levels(4096, 9)[-1] == 7
    assert "Spektrum ∝ f^−α" in README and "α < 2p + 1" in README and "(Haar: α < 3)" in README and C.PARAMS["Potenzgesetz-Rauschen"][0][2] == 2.5 and "α bis 2,5" in README


def test_wavelet_variance_condition_warning_only_when_violated():
    assert "α ≥ 2p + 1 = 3 verletzt die Bedingung" in ev.verdict_main(ev.analyse(ev.Settings(signal="Potenzgesetz-Rauschen", p1=3.2, p=1)))[1]
    assert "verletzt die Bedingung" not in ev.verdict_main(ev.analyse(ev.Settings(signal="Potenzgesetz-Rauschen", p1=2.5, p=1)))[1]
    assert "verletzt die Bedingung" not in ev.verdict_main(ev.analyse(ev.Settings(signal="Potenzgesetz-Rauschen", p1=3.2, p=2)))[1]
