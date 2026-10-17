"""Testsignale mit exakter Wahrheit: Potenzgesetz-Rauschen, Sprung im glatten Signal, Impuls, Ton, Trend plus Rauschen.

Zufall nur über einen übergebenen SplitMix64. Zeit t = 0 .. n - 1, Abtastrate 1. `alpha` ist der wahre Exponent (Spektrum ∝ f^-α) oder None, `events` die Zeitpunkte von Sprung und Impuls.
"""

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class TestSignal:
    x: np.ndarray
    alpha: float | None
    events: tuple


def power_law(n: int, alpha: float, rng) -> np.ndarray:
    """Zirkulär stationärer Gauß-Prozess mit Spektrum ∝ f^-α (Spektralsynthese über die positiven Frequenzen k / n), Mittelwert 0; n gerade."""
    z = rng.normal(2 * (n // 2)).reshape(2, -1)
    spec = np.zeros(n, dtype=complex)
    k = np.arange(1, n // 2)
    spec[1:n // 2] = (k / n) ** (-alpha / 2.0) * (z[0, :n // 2 - 1] + 1j * z[1, :n // 2 - 1])
    spec[n // 2 + 1:] = np.conj(spec[1:n // 2][::-1])
    spec[n // 2] = (0.5) ** (-alpha / 2.0) * z[0, -1]
    return np.real(np.fft.ifft(spec)) * math.sqrt(n)


def power_law_signal(n: int, alpha: float, rng) -> TestSignal:
    return TestSignal(power_law(n, alpha, rng), float(alpha), ())


def smooth_jump(n: int, periods: float, t_jump: int, height: float) -> TestSignal:
    """Glatter Sinus mit `periods` Perioden über die Aufzeichnung plus ein Sprung der Höhe `height` ab t_jump."""
    t = np.arange(n)
    x = np.sin(TWO_PI * periods * t / n) + height * (t >= t_jump)
    return TestSignal(x, None, (float(t_jump),))


def impulse(n: int, t0: int) -> TestSignal:
    """Einzelimpuls der Höhe 1 bei t0."""
    x = np.zeros(n)
    x[t0] = 1.0
    return TestSignal(x, None, (float(t0),))


def tone(n: int, f: float, phase: float = 0.0) -> TestSignal:
    """Ton cos(2 pi f t + phase)."""
    return TestSignal(np.cos(TWO_PI * f * np.arange(n) + phase), None, ())


def trend_noise(n: int, alpha: float, trend_sd: float, rng) -> TestSignal:
    """Potenzgesetz-Rauschen plus linearer Trend, der über die Aufzeichnung `trend_sd` Standardabweichungen des Rauschens durchläuft (von -trend_sd/2 bis +trend_sd/2)."""
    base = power_law(n, alpha, rng)
    return TestSignal(base + trend_sd * base.std() * (np.arange(n) / n - 0.5), float(alpha), ())
