"""Analyse, Kennzahlen, Experimente (feste Ströme) und Verdikte der Demo. Reine Funktionen; Zufall nur über SplitMix64."""

import math
from dataclasses import dataclass

import numpy as np

import dwt_constants as C
import dwt_filters as F
import dwt_resolution as R
import dwt_signals as S
import dwt_transform as T
import signal_generator as G


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def noise_sigma(snr_db: float) -> float:
    """Standardabweichung des Rauschens gegen eine Linie der Amplitude 1 (Leistung 1/2): sqrt(0,5 / 10^(SNR/10))."""
    return math.sqrt(0.5 / 10.0 ** (snr_db / 10.0))


def wavelet_name(p: int) -> str:
    """Daubechies-Bezeichnung: Haar für p = 1, sonst D(2p)."""
    return "Haar (p = 1)" if p == 1 else f"D{2 * p} (p = {p})"


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    signal: str = C.DEFAULT_SIGNAL
    p1: float = C.PARAMS[C.DEFAULT_SIGNAL][0][4]
    p2: float = 0.0
    p3: float = 0.0
    p: int = C.DEFAULT_P
    levels: int = C.DEFAULT_LEVELS
    transform: str = C.DEFAULT_TRANSFORM
    boundary: str = C.DEFAULT_BOUNDARY
    shift: int = C.DEFAULT_SHIFT
    noise: str = C.DEFAULT_NOISE
    snr: int = C.DEFAULT_SNR
    seed: int = C.DEFAULT_SEED

    @property
    def has_noise(self) -> bool:
        return self.noise == "an" and self.signal in C.NOISY_SIGNALS

    @property
    def discard_boundary(self) -> bool:
        return self.boundary == "Randkoeffizienten verwerfen" and self.transform == "MODWT"


def make_signal(s: Settings) -> S.TestSignal:
    """Das saubere Signal des gewählten Typs (Tonphase 0,3)."""
    if s.signal == "Potenzgesetz-Rauschen":
        return S.power_law_signal(s.n, s.p1, G.SplitMix64(s.seed))
    if s.signal == "Sprung im glatten Signal":
        return S.smooth_jump(s.n, s.p1, int(round(s.p2 * s.n)), s.p3)
    if s.signal == "Impuls":
        return S.impulse(s.n, min(s.n - 1, int(round(s.p1 * s.n))))
    if s.signal == "Ton":
        return S.tone(s.n, s.p1, 0.3)
    if s.signal == "Trend plus Rauschen":
        return S.trend_noise(s.n, s.p1, s.p2, G.SplitMix64(s.seed))
    raise ValueError(f"unbekanntes Signal: {s.signal}")


def build_measurement(s: Settings) -> tuple[S.TestSignal, np.ndarray]:
    """(sauberes Signal, Messung ohne Verschiebung); Rauschen nur bei den Signaltypen mit Rauschregler."""
    sig = make_signal(s)
    x = sig.x.copy()
    if s.has_noise:
        x = x + noise_sigma(s.snr) * G.SplitMix64(s.seed).normal(s.n)
    return sig, x


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    signal: S.TestSignal
    x: np.ndarray                   # Messung nach der Verschiebung
    details: list                   # Detailkoeffizienten je Stufe der gewählten Transformation
    smooth: np.ndarray              # Glättung (DWT: a_J, MODWT: V_J)
    fractions: np.ndarray           # Energieanteile je Stufe, zuletzt die Glättung
    recon_error: float              # größter Betrag der Rekonstruktionsabweichung
    energy_error: float             # relative Abweichung der Energiebilanz
    mra_details: list               # Mehrskalenanalyse: Detailanteile D_j (MODWT)
    mra_smooth: np.ndarray
    mra_error: float
    variance: np.ndarray            # Wavelet-Varianz je Stufe (nan wo nicht auswertbar)
    fit_levels: list
    fit: tuple | None               # (Steigung, Achsenabschnitt) von log2 ν² gegen die Stufe
    alpha_hat: float | None
    alpha_true: float | None
    gph: float | None
    coef_count: int                 # Zahl der Koeffizienten der gewählten Transformation
    sparse_count: int               # DWT-Koeffizienten über SPARSE_REL des größten
    sparse_top: float               # Energieanteil der 16 größten DWT-Koeffizienten
    shift_change: float             # größte Änderung eines Energieanteils bei der gewählten Verschiebung
    band_level: int | None          # Stufe des Frequenzbandes eines Tons
    band_fraction: float | None     # Energieanteil in dieser Stufe


def analyse(s: Settings) -> Analysis:
    sig, x0 = build_measurement(s)
    x = np.roll(x0, s.shift)
    h = F.daubechies(s.p)
    levels = s.levels
    details, smooth = R.decompose(x, h, levels, s.transform)
    rec = T.idwt(details, smooth, h) if s.transform == "DWT" else T.imodwt(details, smooth, h)
    fractions = R.energy_fractions(details, smooth)
    total = float(np.sum(x ** 2))
    energy = float(T.energy_by_level(details, smooth).sum())
    md, ms = T.mra(x, h, levels)
    alpha_hat, lev, var, fit = R.alpha_estimate(x, h, levels, s.transform, s.discard_boundary)
    dwt_details, dwt_smooth = (details, smooth) if s.transform == "DWT" else T.dwt(x, h, levels)
    flat = np.concatenate([np.asarray(d) for d in dwt_details] + [np.asarray(dwt_smooth)])
    band = R.band_level(s.p1) if s.signal == "Ton" and s.p1 < 1.0 else None
    band = band if band is not None and 1 <= band <= levels else None
    return Analysis(s, sig, x, details, smooth, fractions, float(np.abs(rec - x).max()), abs(energy - total) / total if total > 0 else 0.0, md, ms, float(np.abs(sum(md) + ms - x).max()), var, lev, fit,
                    alpha_hat, sig.alpha, _gph(x) if sig.alpha is not None else None, len(np.concatenate([np.asarray(d) for d in details])) + len(smooth), R.count_above(flat, C.SPARSE_REL),
                    R.top_energy_share(flat, C.SPARSE_TOP), R.shift_change(x0, h, levels, s.transform, s.shift) if s.shift else 0.0, band, float(fractions[band - 1]) if band else None)


def _gph(x) -> float:
    """Exponent aus dem Periodogramm (GPH) mit m = round(sqrt(n)) Frequenzen."""
    return T.gph_alpha(x, int(round(math.sqrt(len(x)))))


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------


def sweep_moments(ps=C.EXP_MOM_P, n: int = C.EXP_MOM_N, jump_ps=C.EXP_JUMP_P, jump_n: int = C.EXP_JUMP_N, jump_levels: int = C.EXP_JUMP_J) -> dict:
    """Verschwindende Momente: größter innerer Detailkoeffizient der Stufe 1 bei Polynomen vom Grad 0 bis 4 und Sparsität eines Sinus mit Sprung (Zahl der Koeffizienten über 1e-3 des größten,
    Energieanteil der 16 größten)."""
    t = np.arange(n) / n
    degrees = []
    for p in ps:
        h = F.daubechies(p)
        degrees.append({"p": p, "max": [R.interior_max(T.dwt(t ** deg, h, 1)[0], len(h)) for deg in range(5)]})
    sig = S.smooth_jump(jump_n, 3.0, int(round(0.3 * jump_n)), 1.0).x
    jump = []
    for p in jump_ps:
        d, a = T.dwt(sig, F.daubechies(p), jump_levels)
        flat = np.concatenate(d + [a])
        jump.append({"p": p, "count": R.count_above(flat, C.SPARSE_REL), "total": len(flat), "top": R.top_energy_share(flat, C.SPARSE_TOP)})
    return {"degrees": degrees, "jump": jump}


def sweep_shift(p: int = C.EXP_SHIFT_P, n: int = C.EXP_SHIFT_N, levels: int = C.EXP_SHIFT_J, count: int = C.EXP_SHIFT_COUNT) -> list[dict]:
    """Verschiebungsvarianz: Spanne (größter minus kleinster Wert) des Energieanteils je Stufe über `count` zirkuläre Verschiebungen, DWT gegen MODWT."""
    h = F.daubechies(p)
    t = np.arange(n)
    bases = (("Impuls", np.eye(1, n, 10)[0]), ("Ton f = 0,07", np.cos(2 * math.pi * 0.07 * t + 0.3)), ("Ton f = 0,1", np.cos(2 * math.pi * 0.1 * t)))
    rows = []
    for name, base in bases:
        spans = {}
        for transform in C.TRANSFORM_MODES:
            fr = np.array([R.energy_fractions(*R.decompose(np.roll(base, k), h, levels, transform)) for k in range(count)])
            spans[transform] = (fr.max(axis=0) - fr.min(axis=0)).tolist()
        rows.append({"name": name, "dwt": spans["DWT"], "modwt": spans["MODWT"]})
    return rows


def sweep_alpha(values=C.EXP_ALPHA_VALUES, ps=C.EXP_ALPHA_P, n: int = C.EXP_ALPHA_N, levels: int = C.EXP_ALPHA_J, runs: int = C.EXP_ALPHA_RUNS, trend_sd: float = C.EXP_TREND_SD,
                trend_alpha: float = C.EXP_TREND_ALPHA, trend_ps=C.EXP_TREND_P, m: int = C.EXP_GPH_M) -> dict:
    """Exponent aus der Wavelet-Varianz (Stufen 2 bis n/2^j >= 32): DWT und MODWT für mehrere p gegen das Periodogramm (GPH) bei α-Werten; dazu ein linearer Trend (MODWT ohne die vom Rand
    betroffenen Koeffizienten gegen die zirkuläre DWT und GPH) und die MODWT ohne Randkoeffizienten ohne Trend."""
    stationary = []
    for alpha in values:
        rng = G.SplitMix64(C.EXP_SEED + int(round(10 * alpha)))
        xs = [S.power_law(n, alpha, rng) for _ in range(runs)]
        est = {}
        for p in ps:
            h = F.daubechies(p)
            for transform in C.TRANSFORM_MODES:
                est[(transform, p)] = [R.alpha_estimate(x, h, levels, transform)[0] for x in xs]
        est[("GPH", 0)] = [T.gph_alpha(x, m) for x in xs]
        for (name, p), v in est.items():
            stationary.append({"alpha": alpha, "estimator": name, "p": p, "mean": float(np.mean(v)), "sd": float(np.std(v))})
    rng = G.SplitMix64(C.EXP_SEED + 100)
    xs = [S.trend_noise(n, trend_alpha, trend_sd, rng).x for _ in range(runs)]
    trend = []
    for p in trend_ps:
        h = F.daubechies(p)
        inner = [R.alpha_estimate(x, h, levels, "MODWT", True)[0] for x in xs]
        circ = [R.alpha_estimate(x, h, levels, "DWT")[0] for x in xs]
        trend.append({"p": p, "inner_mean": float(np.mean(inner)), "inner_sd": float(np.std(inner)), "dwt_mean": float(np.mean(circ)), "dwt_sd": float(np.std(circ))})
    gph = [T.gph_alpha(x, m) for x in xs]
    rng = G.SplitMix64(C.EXP_SEED + 110)
    xs = [S.power_law(n, trend_alpha, rng) for _ in range(runs)]
    plain = []
    for p in trend_ps[:3]:
        v = [R.alpha_estimate(x, F.daubechies(p), levels, "MODWT", True)[0] for x in xs]
        plain.append({"p": p, "mean": float(np.mean(v)), "sd": float(np.std(v))})
    return {"stationary": stationary, "trend": trend, "trend_gph": (float(np.mean(gph)), float(np.std(gph))), "plain_inner": plain, "runs": runs, "n": n, "levels": levels, "trend_sd": trend_sd,
            "trend_alpha": trend_alpha}


def sweep_bands(ps=C.EXP_BAND_P, levels_list=C.EXP_BAND_LEVELS, n: int = C.EXP_BAND_N) -> list[dict]:
    """Frequenzselektivität: Energieanteil eines Tons in der Mitte des Bandes der Stufe j (f = 0,75 / 2^j) in dieser Stufe (MODWT), dazu |G_3(f)|^2 bei f = 1/16, 3/32, 1/8, 3/16, 1/4."""
    t = np.arange(n)
    freqs = (1 / 16, 3 / 32, 1 / 8, 3 / 16, 1 / 4)
    rows = []
    for p in ps:
        h = F.daubechies(p)
        frac = {}
        for j in levels_list:
            x = np.cos(2 * math.pi * 0.75 / 2 ** j * t + 0.4)
            frac[j] = float(R.energy_fractions(*T.modwt(x, h, 6))[j - 1])
        rows.append({"p": p, "fractions": frac, "response": [float(T.stage_response(h, 3, f)) for f in freqs]})
    return rows


def sweep_variance(p: int = C.EXP_VAR_P, n: int = C.EXP_VAR_N, levels: int = C.EXP_VAR_J, runs: int = C.EXP_VAR_RUNS) -> dict:
    """Streuung der geschätzten Wavelet-Varianz bei weißem Rauschen: relative Standardabweichung je Stufe, DWT gegen MODWT."""
    h = F.daubechies(p)
    xs = G.SplitMix64(C.EXP_SEED + 400).normal(runs * n).reshape(runs, n)
    vd = np.array([T.dwt_variance(T.dwt(x, h, levels)[0]) for x in xs])
    vm = np.array([T.modwt_variance(T.modwt(x, h, levels)[0], len(h)) for x in xs])
    rel_d, rel_m = vd.std(axis=0) / vd.mean(axis=0), vm.std(axis=0) / vm.mean(axis=0)
    return {"dwt_rel": rel_d.tolist(), "modwt_rel": rel_m.tolist(), "ratio": (rel_d / rel_m).tolist(), "dwt_mean": vd.mean(axis=0).tolist(), "modwt_mean": vm.mean(axis=0).tolist(),
            "n": n, "levels": levels, "runs": runs, "dwt_count": n, "modwt_count": (levels + 1) * n, "p": p}


# --- Verdikte (reine Funktionen: (Art, Text)) ----------------------------------------------------------------------------------------------


def verdict_main(a: Analysis) -> tuple[str, str]:
    s = a.settings
    head = f"{wavelet_name(s.p)}, {s.transform}: "
    if a.alpha_true is not None:
        if a.alpha_hat is None:
            return "info", head + "Für die Geradenanpassung bleiben weniger als drei Stufen (ab Stufe 2, solange n/2^j mindestens 32 ist); wählen Sie ein längeres n oder mehr Stufen."
        lo, hi = a.fit_levels[0], a.fit_levels[-1]
        ok = abs(a.alpha_hat - a.alpha_true) < C.OK_ALPHA
        text = head + f"Der Exponent aus der Wavelet-Varianz (Stufen {lo} bis {hi}) ist α̂ = {de(a.alpha_hat, 2)} gegen wahr {de(a.alpha_true, 2)}; das Periodogramm (GPH) schätzt {de(a.gph, 2)}."
        if a.alpha_true >= 2 * s.p + 1:
            text += f" Hinweis: α ≥ 2p + 1 = {2 * s.p + 1} verletzt die Bedingung der Wavelet-Varianz."
        if s.signal == "Trend plus Rauschen":
            if s.p == 1 and s.p2 > 20:
                text += " Haar hat nur ein verschwindendes Moment, ein linearer Trend braucht mindestens p = 2."
            if s.boundary == "periodisch":
                text += " Die periodische Fortsetzung macht aus dem Trend einen Sprung am Rand; verwerfen Sie die Randkoeffizienten (MODWT)."
        return ("success" if ok else "warning"), text
    if s.signal == "Sprung im glatten Signal":
        return "info", head + f"{a.sparse_count} von {a.settings.n} DWT-Koeffizienten liegen über dem 1e-3-Fachen des größten; die {C.SPARSE_TOP} größten tragen {de(100 * a.sparse_top, 1)} % der Energie."
    shift_text = ""
    if s.shift:
        if a.shift_change < 1e-9:
            shift_text = f" Verschiebung um {s.shift}: Die Energieanteile bleiben gleich (Änderung {de(a.shift_change, 3)})."
        else:
            shift_text = f" Verschiebung um {s.shift}: Der Energieanteil einer Stufe ändert sich um {de(a.shift_change, 3)}; die MODWT bliebe exakt gleich."
    if s.signal == "Ton":
        if a.band_level is None:
            return "info", head + "Die Frequenz liegt außerhalb der Bänder der gewählten Stufen." + shift_text
        return "info", head + f"Stufe {a.band_level} hat das Band {de(1 / 2 ** (a.band_level + 1), 3)} bis {de(1 / 2 ** a.band_level, 3)}; sie trägt {de(100 * a.band_fraction, 1)} % der Energie des Tons, der Rest liegt in den Nachbarstufen." + shift_text
    return "info", head + "Ein Einzelimpuls verteilt seine Energie auf alle Stufen." + (shift_text or " Verschieben Sie ihn, um die Verschiebungsempfindlichkeit zu sehen.")


def verdict_moments(res: dict) -> tuple[str, str]:
    by = {r["p"]: r for r in res["degrees"]}
    jump = {r["p"]: r for r in res["jump"]}
    below = max(max(by[p]["max"][:p]) for p in by)
    grows = f"{by[2]['max'][2]:.1e}".replace(".", ",")
    bound = "unter 1e-14" if below < 1e-14 else f"höchstens {below:.0e}"
    return "info", (f"Polynome vom Grad kleiner als p ergeben im Inneren Detailkoeffizienten von {bound}; ab Grad p wachsen sie (bei p = 2 und Grad 2: {grows}). "
                    f"Bei einem Sinus mit Sprung tragen die {C.SPARSE_TOP} größten Koeffizienten {de(100 * jump[1]['top'], 1)} % der Energie bei Haar und {de(100 * jump[4]['top'], 1)} % bei p = 4. "
                    "**Mehr verschwindende Momente: glatte Teile werden null, nur der Sprung bleibt.**")


def verdict_shift(rows: list[dict]) -> tuple[str, str]:
    imp = rows[0]
    return "warning", (f"Verschiebt man einen Impuls, ändert sich der Energieanteil einer Stufe bei der DWT um bis zu {de(max(imp['dwt']), 2)}, bei der MODWT um {de(max(imp['modwt']), 3)}. "
                       f"Bei einem Ton sind es bis zu {de(max(max(r['dwt']) for r in rows[1:]), 3)} gegen {de(max(max(r['modwt']) for r in rows[1:]), 3)}. **Die DWT ist nicht verschiebungsinvariant, die MODWT schon.**")


def verdict_alpha(res: dict) -> tuple[str, str]:
    def get(alpha, name, p):
        return [r for r in res["stationary"] if r["alpha"] == alpha and r["estimator"] == name and r["p"] == p][0]
    mod, gph = get(0.5, "MODWT", 2), get(0.5, "GPH", 0)
    trend = {r["p"]: r for r in res["trend"]}
    return "info", (f"Bei α = 0,5 schätzt die MODWT (D4) {de(mod['mean'], 2)} ± {de(mod['sd'], 2)}, das Periodogramm {de(gph['mean'], 2)} ± {de(gph['sd'], 2)}; die Streuung ist mit der Wavelet-Varianz etwa "
                    f"{de(gph['sd'] / mod['sd'], 0)}-mal kleiner. Mit einem linearen Trend von {de(res['trend_sd'], 0)} Standardabweichungen bleibt die MODWT mit p = 2 bei {de(trend[2]['inner_mean'], 2)}, Haar steigt auf "
                    f"{de(trend[1]['inner_mean'], 2)} und das Periodogramm auf {de(res['trend_gph'][0], 2)}. **Verschwindende Momente schützen den Exponenten vor dem Trend.**")


def verdict_bands(rows: list[dict]) -> tuple[str, str]:
    by = {r["p"]: r for r in rows}
    return "warning", (f"Ein Ton in der Mitte des Bandes der Stufe 3 liefert nur {de(100 * by[1]['fractions'][3], 0)} % (Haar), {de(100 * by[2]['fractions'][3], 0)} % (D4), {de(100 * by[4]['fractions'][3], 0)} % (D8) und "
                       f"{de(100 * by[8]['fractions'][3], 0)} % (D16) der Energie in dieser Stufe. **Die Stufen sind keine scharfen Frequenzbänder; mehr verschwindende Momente machen sie schärfer.**")


def verdict_variance(res: dict) -> tuple[str, str]:
    r = res["ratio"]
    return "info", (f"Bei weißem Rauschen streut der Schätzer der Wavelet-Varianz bei der DWT auf Stufe 1 bis {res['levels']} relativ um {de(res['dwt_rel'][0], 3)} bis {de(res['dwt_rel'][-1], 3)}, bei der MODWT um "
                    f"{de(res['modwt_rel'][0], 3)} bis {de(res['modwt_rel'][-1], 3)}: das {de(min(r), 2)}- bis {de(max(r), 2)}-Fache. Dafür hat sie {res['modwt_count']} statt {res['dwt_count']} Koeffizienten. "
                    f"**Die MODWT schätzt genauer, aber nicht umsonst.**")
