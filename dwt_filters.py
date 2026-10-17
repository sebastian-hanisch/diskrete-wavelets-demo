"""Daubechies-Filter per Spektralfaktorisierung und das Quadraturspiegelfilter. Reine Funktionen, nur numpy."""

import math

import numpy as np


def daubechies(p: int) -> np.ndarray:
    """Skalierungsfilter h der Länge 2p (Summe sqrt 2, Summe der Quadrate 1, Extremalphase) der Daubechies-Wavelet mit p verschwindenden Momenten (p = 1: Haar).

    Die Wurzeln y_k von P(y) = sum_{k<p} C(p-1+k, k) y^k werden über y = (2 - z - 1/z)/4 auf Nullstellen z innerhalb des Einheitskreises abgebildet; h ist das Produkt aus (1 + z)^p und den Faktoren (z - z_k)."""
    if p < 1:
        raise ValueError("p muss mindestens 1 sein")
    if p == 1:
        return np.array([1.0, 1.0]) / math.sqrt(2.0)
    coef = [math.comb(p - 1 + k, k) for k in range(p)]
    zs = []
    for y in np.roots(coef[::-1]):
        r = np.roots([1.0, -(2.0 - 4.0 * y), 1.0])
        zs.append(r[np.argmin(np.abs(r))])
    poly = np.array([1.0 + 0j])
    for _ in range(p):
        poly = np.convolve(poly, [1.0, 1.0])
    for z in zs:
        poly = np.convolve(poly, [1.0, -z])
    h = np.real(poly)
    h = h / h.sum() * math.sqrt(2.0)
    return h[::-1] if abs(h[0]) < abs(h[-1]) else h


def qmf(h) -> np.ndarray:
    """Wavelet-Filter g[l] = (-1)^l h[L - 1 - l] (Quadraturspiegelfilter) zum Skalierungsfilter h."""
    h = np.asarray(h, dtype=float)
    length = len(h)
    return np.array([(-1) ** l * h[length - 1 - l] for l in range(length)])
