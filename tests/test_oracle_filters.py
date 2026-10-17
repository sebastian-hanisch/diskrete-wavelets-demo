"""Daubechies-Filter gegen geschlossene Formel, Tabellen und die definierenden Bedingungen."""

import math

import numpy as np
import pytest

import dwt_filters as F

DB3 = [0.3326705529500825, 0.8068915093110924, 0.4598775021184914, -0.13501102001025458, -0.08544127388202666, 0.03522629188570953]
DB4 = [0.23037781330885523, 0.7148465705525415, 0.6308807679295904, -0.02798376941698385, -0.18703481171888114, 0.030841381835986965, 0.032883011666982945, -0.010597401784997278]


def test_haar_by_hand():
    assert F.daubechies(1).tolist() == pytest.approx([1 / math.sqrt(2), 1 / math.sqrt(2)])
    assert F.qmf(F.daubechies(1)).tolist() == pytest.approx([1 / math.sqrt(2), -1 / math.sqrt(2)])


def test_d4_equals_the_closed_form():
    ref = np.array([1 + math.sqrt(3), 3 + math.sqrt(3), 3 - math.sqrt(3), 1 - math.sqrt(3)]) / (4 * math.sqrt(2))
    assert F.daubechies(2) == pytest.approx(ref, abs=1e-14)


def test_db3_and_db4_equal_the_tables():
    assert F.daubechies(3) == pytest.approx(DB3, abs=1e-12) and F.daubechies(4) == pytest.approx(DB4, abs=1e-10)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 6, 8])
def test_filters_satisfy_the_defining_conditions(p):
    h = F.daubechies(p)
    g = F.qmf(h)
    assert len(h) == 2 * p and h.sum() == pytest.approx(math.sqrt(2), abs=1e-12) and (h ** 2).sum() == pytest.approx(1.0, abs=1e-12)
    for k in range(1, p):                                      # Orthogonalität zur Verschiebung um 2k
        assert np.dot(h[2 * k:], h[:len(h) - 2 * k]) == pytest.approx(0.0, abs=1e-10)
    assert np.dot(h, g) == pytest.approx(0.0, abs=1e-12) and (g ** 2).sum() == pytest.approx(1.0, abs=1e-12)
    for m in range(p):                                         # p verschwindende Momente des Wavelet-Filters
        assert abs(np.sum(g * np.arange(len(g)) ** m)) < 1e-6 * max(1.0, (2 * p) ** m)
    assert abs(np.sum(g * np.arange(len(g)) ** p)) > 0.1       # das p-te Moment verschwindet nicht


def test_extremal_phase_orientation_and_bad_input():
    for p in (2, 3, 4, 6):
        h = F.daubechies(p)
        assert abs(h[0]) >= abs(h[-1])                          # Energie am Anfang gesammelt
    with pytest.raises(ValueError):
        F.daubechies(0)


def test_qmf_by_hand():
    assert F.qmf([1.0, 2.0, 3.0, 4.0]).tolist() == [4.0, -3.0, 2.0, -1.0]
