"""Diskrete Wavelet-Transformation (DWT, Pyramide, periodisch) und überlappende MODWT (per FFT, periodisch), Mehrskalenanalyse und Wavelet-Varianz.

Reine Funktionen, kein Zufall, kein Streamlit; nur numpy. Abtastrate 1. Die DWT braucht die Länge n = 2^J-fach teilbar; die MODWT jede Länge.
"""

import math
from functools import lru_cache

import numpy as np

from dwt_filters import qmf


def _check_levels(n: int, levels: int, dyadic: bool) -> None:
    if levels < 1:
        raise ValueError("Stufenzahl muss mindestens 1 sein")
    if dyadic and n % (2 ** levels) != 0:
        raise ValueError("Die Länge muss durch 2^J teilbar sein")


def dwt(x, h, levels: int):
    """Periodische DWT per Pyramide: ([d_1, ..., d_J], a_J) mit c[k] = sum_l h[l] a[(2k + l) mod N] und d[k] = sum_l g[l] a[(2k + l) mod N]."""
    a = np.asarray(x, dtype=float).copy()
    _check_levels(len(a), levels, True)
    h = np.asarray(h, dtype=float)
    g = qmf(h)
    length = len(h)
    details = []
    for _ in range(levels):
        size = len(a)
        idx = (2 * np.arange(size // 2)[:, None] + np.arange(length)[None, :]) % size
        window = a[idx]
        details.append(window @ g)
        a = window @ h
    return details, a


def idwt(details, a, h) -> np.ndarray:
    """Inverse der periodischen DWT (perfekte Rekonstruktion)."""
    h = np.asarray(h, dtype=float)
    g = qmf(h)
    length = len(h)
    a = np.asarray(a, dtype=float)
    for d in reversed(details):
        m = len(d)
        out = np.zeros(2 * m)
        idx = (2 * np.arange(m)[:, None] + np.arange(length)[None, :]) % (2 * m)
        np.add.at(out, idx.ravel(), (a[:, None] * h + np.asarray(d)[:, None] * g).ravel())
        a = out
    return a


@lru_cache(maxsize=512)
def _response(n: int, taps: tuple, scale: int) -> np.ndarray:
    """Komplexer Frequenzgang sum_l c[l] exp(-2 pi i f l scale) an den DFT-Frequenzen f = k / n."""
    c = np.array(taps)
    f = np.fft.fftfreq(n) * scale
    return np.exp(-2j * np.pi * np.outer(f, np.arange(len(c)))) @ c


def modwt_filters(n: int, h, levels: int):
    """Frequenzgänge der MODWT-Stufenfilter: ([G_1, ..., G_J], H_J) mit G_j = G(2^(j-1) f) prod_{l<j-1} H(2^l f), H_J = prod_{l<J} H(2^l f); Filter h/sqrt 2 und g/sqrt 2."""
    _check_levels(n, levels, False)
    ht = tuple(np.asarray(h, dtype=float) / math.sqrt(2.0))
    gt = tuple(qmf(h) / math.sqrt(2.0))
    low = np.ones(n, dtype=complex)
    bands = []
    for j in range(1, levels + 1):
        bands.append(low * _response(n, gt, 2 ** (j - 1)))
        low = low * _response(n, ht, 2 ** (j - 1))
    return bands, low


def modwt(x, h, levels: int):
    """MODWT (periodisch, jede Länge): ([W_1, ..., W_J], V_J), Energie sum x^2 = sum_j |W_j|^2 + |V_J|^2."""
    x = np.asarray(x, dtype=float)
    bands, low = modwt_filters(len(x), h, levels)
    spec = np.fft.fft(x)
    return [np.real(np.fft.ifft(spec * g)) for g in bands], np.real(np.fft.ifft(spec * low))


def imodwt(w, v, h) -> np.ndarray:
    """Inverse der MODWT (perfekte Rekonstruktion)."""
    v = np.asarray(v, dtype=float)
    bands, low = modwt_filters(len(v), h, len(w))
    total = np.fft.fft(v) * np.conj(low)
    for wj, g in zip(w, bands):
        total = total + np.fft.fft(wj) * np.conj(g)
    return np.real(np.fft.ifft(total))


def mra(x, h, levels: int):
    """Mehrskalenanalyse (MODWT): ([D_1, ..., D_J], S_J) mit x = S_J + sum_j D_j; D_j = IFFT(X |G_j|^2), S_J = IFFT(X |H_J|^2)."""
    x = np.asarray(x, dtype=float)
    bands, low = modwt_filters(len(x), h, levels)
    spec = np.fft.fft(x)
    return [np.real(np.fft.ifft(spec * np.abs(g) ** 2)) for g in bands], np.real(np.fft.ifft(spec * np.abs(low) ** 2))


def stage_response(h, level: int, f: float) -> float:
    """Leistungsgang |G_j(f)|^2 des MODWT-Stufenfilters j bei der Frequenz f (direkte Summe, keine DFT-Frequenz nötig)."""
    ht = np.asarray(h, dtype=float) / math.sqrt(2.0)
    gt = qmf(h) / math.sqrt(2.0)

    def resp(c, scale):
        return np.sum(c * np.exp(-2j * np.pi * f * scale * np.arange(len(c))))
    out = resp(gt, 2 ** (level - 1))
    for l in range(level - 1):
        out = out * resp(ht, 2 ** l)
    return float(abs(out) ** 2)


def boundary_length(level: int, length: int) -> int:
    """Länge L_j = (2^j - 1)(L - 1) + 1 des MODWT-Stufenfilters; die ersten L_j - 1 Koeffizienten der Stufe j hängen vom periodischen Rand ab."""
    return (2 ** level - 1) * (length - 1) + 1


def energy_by_level(coefs, smooth) -> np.ndarray:
    """Energie je Stufe (Summe der Quadrate), zuletzt die Glättung."""
    return np.array([float(np.sum(np.asarray(c) ** 2)) for c in coefs] + [float(np.sum(np.asarray(smooth) ** 2))])


def modwt_variance(w, filter_length: int, boundary: bool = False) -> np.ndarray:
    """Wavelet-Varianz je Stufe: Mittel von W_j^2; mit boundary=True nur die vom Rand unbeeinflussten Koeffizienten t >= L_j - 1 (nan, wenn keiner übrig bleibt)."""
    out = []
    for j, wj in enumerate(w, start=1):
        start = boundary_length(j, filter_length) - 1 if boundary else 0
        part = np.asarray(wj)[start:]
        out.append(float(np.mean(part ** 2)) if len(part) else float("nan"))
    return np.array(out)


def dwt_variance(details) -> np.ndarray:
    """Wavelet-Varianz je Stufe aus den DWT-Koeffizienten: Mittel von d_j^2 geteilt durch 2^j (für weißes Rauschen der Varianz 1: 2^-j)."""
    return np.array([float(np.mean(np.asarray(d) ** 2)) / 2 ** j for j, d in enumerate(details, start=1)])


def fit_levels(n: int, levels: int, count_min: int = 32) -> list[int]:
    """Stufen für die Geradenanpassung: 2 bis J, solange die DWT mindestens `count_min` Koeffizienten je Stufe hat (n / 2^j >= count_min)."""
    return [j for j in range(2, levels + 1) if n // 2 ** j >= count_min]


def slope_fit(levels, variance) -> tuple[float, float] | None:
    """Gerade log2 ν² = slope · j + intercept über die Stufen mit positiver, endlicher Varianz; None bei weniger als drei Stufen."""
    lev = np.asarray(levels, dtype=float)
    var = np.asarray(variance, dtype=float)
    ok = np.isfinite(var) & (var > 0)
    if ok.sum() < 3:
        return None
    slope, intercept = np.polyfit(lev[ok], np.log2(var[ok]), 1)
    return float(slope), float(intercept)


def gph_alpha(x, m: int) -> float:
    """Exponent α aus dem Periodogramm (GPH): negative Steigung von log I(f) gegen log f über die Frequenzen 1/n bis m/n."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    spec = np.abs(np.fft.rfft(x)) ** 2 / n
    f = np.arange(1, m + 1) / n
    return float(-np.polyfit(np.log(f), np.log(spec[1:m + 1]), 1)[0])
