"""Kennzahlen der Zerlegung: Energieanteile, Sparsität, Wavelet-Varianz und Exponentenschätzung, Bandlage eines Tons, Verschiebungsempfindlichkeit. Reine Funktionen, nur numpy."""

import math

import numpy as np

import dwt_transform as T

TRANSFORMS = ("DWT", "MODWT")


def decompose(x, h, levels: int, transform: str):
    """(Detailkoeffizienten je Stufe, Glättung) der gewählten Transformation."""
    if transform == "DWT":
        return T.dwt(x, h, levels)
    if transform == "MODWT":
        return T.modwt(x, h, levels)
    raise ValueError(f"unbekannte Transformation: {transform}")


def energy_fractions(details, smooth) -> np.ndarray:
    """Anteil der Gesamtenergie je Stufe, zuletzt die Glättung; Summe 1 (bei Energie null: alle null)."""
    e = T.energy_by_level(details, smooth)
    total = e.sum()
    return e / total if total > 0 else np.zeros_like(e)


def variance_by_level(x, h, levels: int, transform: str, boundary: bool = False) -> np.ndarray:
    """Wavelet-Varianz je Stufe 1 .. J: MODWT (mit oder ohne Randkoeffizienten) oder DWT (immer periodisch)."""
    if transform == "DWT":
        return T.dwt_variance(T.dwt(x, h, levels)[0])
    w, _ = T.modwt(x, h, levels)
    return T.modwt_variance(w, len(h), boundary)


def alpha_estimate(x, h, levels: int, transform: str, boundary: bool = False):
    """Exponent α aus der Steigung der Wavelet-Varianz (α = Steigung + 1) über die Stufen von fit_levels: (α̂ oder None, Stufen, Varianz, (Steigung, Achsenabschnitt) oder None)."""
    var = variance_by_level(x, h, levels, transform, boundary)
    lev = T.fit_levels(len(x), levels)
    fit = T.slope_fit(lev, var[np.array(lev) - 1]) if lev else None
    return (fit[0] + 1.0 if fit else None), lev, var, fit


def count_above(values, rel: float) -> int:
    """Zahl der Werte, deren Betrag über rel mal dem größten Betrag liegt."""
    v = np.abs(np.asarray(values, dtype=float))
    return int((v > rel * v.max()).sum()) if v.size and v.max() > 0 else 0


def top_energy_share(values, k: int) -> float:
    """Anteil der Energie, den die k betragsgrößten Werte tragen."""
    e = np.sort(np.asarray(values, dtype=float) ** 2)[::-1]
    total = e.sum()
    return float(e[:k].sum() / total) if total > 0 else 0.0


def band_level(f: float) -> int:
    """Stufe j, deren Frequenzband [1/2^(j+1), 1/2^j] die Frequenz f enthält: floor(log2(1/f))."""
    return int(math.floor(math.log2(1.0 / f)))


def interior_max(details, length: int) -> float:
    """Größter Betrag der Detailkoeffizienten der ersten Stufe ohne die Randwerte (p - 1 = length/2 - 1 Koeffizienten an jedem Ende)."""
    d = np.asarray(details[0])
    margin = length // 2
    return float(np.abs(d[margin:len(d) - margin]).max())


def shift_change(x, h, levels: int, transform: str, shift: int) -> float:
    """Größte Änderung des Energieanteils einer Stufe, wenn x zirkulär um `shift` verschoben wird (Glättung eingeschlossen)."""
    a = energy_fractions(*decompose(x, h, levels, transform))
    b = energy_fractions(*decompose(np.roll(x, shift), h, levels, transform))
    return float(np.abs(a - b).max())
