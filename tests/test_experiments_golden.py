"""Festwerte der Experimente (feste Zufallsströme): Das Ergebnis ist bei jedem Lauf dasselbe; Abweichungen bedeuten eine geänderte Rechnung."""

import pytest

from sweeps_cache import get

DEGREES = {1: [0.0, 0.00276214, 0.00544874, 0.00806139, 0.0106016], 2: [1.11022e-16, 2.22045e-16, 1.86881e-05, 5.51421e-05, 0.000108471], 3: [5.55112e-17, 1.66533e-16, 1.11022e-16, 1.9992e-07, 7.84162e-07], 4: [2.22045e-16, 2.22045e-16, 2.22045e-16, 3.33067e-16, 2.922e-09], 8: [1.77636e-15, 1.88738e-15, 1.66533e-15, 1.66533e-15, 1.55431e-15]}
JUMP = {1: (429, 0.953928), 2: (127, 0.987226), 4: (73, 0.993213), 8: (91, 0.993308)}
SHIFT_DWT = {'Impuls': [0.433013, 0.525791, 0.301309, 0.173768, 0.091366, 0.025262], 'Ton f = 0,07': [0.000179, 0.001779, 0.035932, 0.035904, 0.001196, 0.001045], 'Ton f = 0,1': [0.005902, 0.035702, 0.044176, 0.009285, 0.002931, 0.000651]}
ALPHA = {(0.0, 'DWT', 1): (-0.008469, 0.057055), (0.0, 'MODWT', 1): (0.002524, 0.043582), (0.0, 'DWT', 2): (-0.00158, 0.062912), (0.0, 'MODWT', 2): (0.002357, 0.050688), (0.0, 'DWT', 4): (-0.000296, 0.060015), (0.0, 'MODWT', 4): (0.001674, 0.0547), (0.0, 'GPH', 0): (-0.008022, 0.177124), (0.5, 'DWT', 1): (0.470773, 0.059696), (0.5, 'MODWT', 1): (0.478538, 0.043239), (0.5, 'DWT', 2): (0.489779, 0.058357), (0.5, 'MODWT', 2): (0.492436, 0.048313), (0.5, 'DWT', 4): (0.490296, 0.057152), (0.5, 'MODWT', 4): (0.495022, 0.051739), (0.5, 'GPH', 0): (0.518228, 0.18421), (1.0, 'DWT', 1): (0.975331, 0.056092), (1.0, 'MODWT', 1): (0.979513, 0.04527), (1.0, 'DWT', 2): (0.995885, 0.062366), (1.0, 'MODWT', 2): (0.994702, 0.049277), (1.0, 'DWT', 4): (0.990343, 0.059739), (1.0, 'MODWT', 4): (0.997522, 0.052019), (1.0, 'GPH', 0): (1.025901, 0.181475)}
TREND = {1: (1.264485, 0.014351, 1.263316, 0.025214), 2: (0.482984, 0.052248, 1.909512, 0.008678), 3: (0.483027, 0.055173, 1.908095, 0.00817), 4: (0.482163, 0.057094, 1.872401, 0.009938)}
TREND_GPH = (1.999814, 0.010208)
PLAIN = {1: (0.483507, 0.043077), 2: (0.496256, 0.049207), 3: (0.498038, 0.052133)}
BANDS = {1: ([0.590097, 0.540372, 0.528738], [0.410533, 0.540372, 0.426777, 0.050622, 0.0]), 2: ([0.728123, 0.713484, 0.712506], [0.468976, 0.713484, 0.470971, 0.02244, 0.0]), 4: ([0.852887, 0.851665, 0.851659], [0.494416, 0.851665, 0.494449, 0.004788, 0.0]), 8: ([0.941002, 0.940993, 0.940993], [0.499742, 0.940993, 0.499742, 0.000243, 0.0])}
VAR_DWT = [0.063627, 0.087031, 0.128168, 0.175183, 0.229062, 0.350115]
VAR_MODWT = [0.0574, 0.063914, 0.088803, 0.125065, 0.166309, 0.235699]
VAR_MEAN_DWT = [0.499693, 0.24891, 0.125954, 0.06249, 0.030843, 0.016022]
VAR_MEAN_MODWT = [0.500186, 0.248728, 0.125479, 0.062187, 0.031381, 0.015855]


def test_moments_golden():
    res = get("moments")
    assert [r["p"] for r in res["degrees"]] == list(DEGREES)
    for r in res["degrees"]:
        assert r["max"] == pytest.approx(DEGREES[r["p"]], abs=1e-12, rel=1e-4)
    assert [r["p"] for r in res["jump"]] == list(JUMP)
    for r in res["jump"]:
        assert (r["count"], r["total"]) == (JUMP[r["p"]][0], 1024) and r["top"] == pytest.approx(JUMP[r["p"]][1], abs=1e-6)


def test_shift_golden():
    rows = get("shift")
    assert [r["name"] for r in rows] == list(SHIFT_DWT)
    for r in rows:
        assert r["dwt"] == pytest.approx(SHIFT_DWT[r["name"]], abs=1e-6) and max(r["modwt"]) < 1e-12


def test_alpha_golden():
    res = get("alpha")
    assert len(res["stationary"]) == len(ALPHA) == 21 and res["runs"] == 200
    for r in res["stationary"]:
        assert (r["mean"], r["sd"]) == pytest.approx(ALPHA[(r["alpha"], r["estimator"], r["p"])], abs=1e-6)
    assert [t["p"] for t in res["trend"]] == list(TREND)
    for t in res["trend"]:
        assert (t["inner_mean"], t["inner_sd"], t["dwt_mean"], t["dwt_sd"]) == pytest.approx(TREND[t["p"]], abs=1e-6)
    assert res["trend_gph"] == pytest.approx(TREND_GPH, abs=1e-6)
    for r in res["plain_inner"]:
        assert (r["mean"], r["sd"]) == pytest.approx(PLAIN[r["p"]], abs=1e-6)


def test_bands_golden():
    rows = get("bands")
    assert [r["p"] for r in rows] == list(BANDS)
    for r in rows:
        fr, resp = BANDS[r["p"]]
        assert [r["fractions"][j] for j in sorted(r["fractions"])] == pytest.approx(fr, abs=1e-6) and r["response"] == pytest.approx(resp, abs=1e-6)


def test_variance_golden():
    res = get("variance")
    assert res["dwt_rel"] == pytest.approx(VAR_DWT, abs=1e-6) and res["modwt_rel"] == pytest.approx(VAR_MODWT, abs=1e-6)
    assert res["dwt_mean"] == pytest.approx(VAR_MEAN_DWT, abs=1e-6) and res["modwt_mean"] == pytest.approx(VAR_MEAN_MODWT, abs=1e-6)
