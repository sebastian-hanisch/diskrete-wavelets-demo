"""Plotly-Figuren. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dwt_constants as C
from dwt_evaluation import Analysis


def _fixed(fig: go.Figure, height: int = 340, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def level_labels(levels: int) -> list[str]:
    return [f"Stufe {j}" for j in range(1, levels + 1)] + ["Glättung"]


def coefficient_map(a: Analysis) -> np.ndarray:
    """Koeffizientenquadrate je Stufe auf die Zeitachse gelegt: DWT-Koeffizienten der Stufe j werden 2^j-mal wiederholt, MODWT-Koeffizienten sind schon zeitgleich; Zeilen Stufe 1 .. J."""
    n = len(a.x)
    rows = []
    for j, d in enumerate(a.details, start=1):
        d = np.asarray(d) ** 2
        rows.append(np.repeat(d, 2 ** j) if a.settings.transform == "DWT" else d)
    return np.array([r[:n] for r in rows])


def build_signal(a: Analysis) -> go.Figure:
    """Messsignal; bei Rauschen zusätzlich das saubere Signal."""
    t = np.arange(len(a.x))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a.x, mode="lines", line=dict(color=C.COLOR_SIGNAL, width=1), name="Messsignal", hovertemplate="t = %{x}: %{y:.3f}<extra></extra>"))
    if a.settings.has_noise:
        fig.add_trace(go.Scatter(x=t, y=np.roll(a.signal.x, a.settings.shift), mode="lines", line=dict(color=C.COLOR_TRUE, width=1, dash="dot"), name="sauberes Signal",
                                 hovertemplate="t = %{x}: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Zeit (Abtastwerte)", range=[0, len(a.x) - 1])
    fig.update_yaxes(title_text="Signal")
    return _fixed(fig, height=240)


def build_coefficients(a: Analysis) -> go.Figure:
    """Koeffizientenquadrate je Stufe über der Zeit (dB gegen das Maximum, bei −60 dB abgeschnitten)."""
    z = coefficient_map(a)
    top = float(z.max())
    db = np.maximum(10.0 * np.log10(np.maximum(z, 1e-300) / top), -60.0) if top > 0 else np.full(z.shape, -60.0)
    fig = go.Figure(go.Heatmap(x=np.arange(z.shape[1]), y=[f"Stufe {j}" for j in range(1, z.shape[0] + 1)], z=db, colorscale="Viridis", zmin=-60.0, zmax=0.0, colorbar=dict(title="dB", thickness=12),
                               hovertemplate="t = %{x}, %{y}: %{z:.1f} dB<extra></extra>"))
    fig.update_xaxes(title_text="Zeit (Abtastwerte); bei der DWT trägt ein Koeffizient der Stufe j 2^j Abtastwerte")
    fig.update_yaxes(title_text="Stufe (1 fein, J grob)")
    return _fixed(fig, height=300)


def build_levels(a: Analysis) -> go.Figure:
    """Energieanteile je Stufe und der Glättung."""
    labels = level_labels(len(a.details))
    fig = go.Figure(go.Bar(x=labels, y=100 * a.fractions, marker_color=C.COLOR_EST, hovertemplate="%{x}: %{y:.2f} %<extra></extra>"))
    fig.update_yaxes(title_text="Energieanteil (%)", rangemode="tozero")
    fig.update_layout(showlegend=False)
    return _fixed(fig, height=260)


def build_mra(a: Analysis) -> go.Figure:
    """Mehrskalenanalyse: Glättung und Detailanteile untereinander (je mit eigenem Versatz), x ist ihre Summe."""
    comps = [("Glättung", a.mra_smooth)] + [(f"D{j}", d) for j, d in reversed(list(enumerate(a.mra_details, start=1)))]
    spread = max(float(np.abs(c).max()) for _, c in comps) * 2.2 or 1.0
    t = np.arange(len(a.x))
    fig = go.Figure()
    for i, (name, c) in enumerate(comps):
        fig.add_trace(go.Scatter(x=t, y=np.asarray(c) - i * spread, mode="lines", line=dict(width=1), name=name, hovertemplate=name + ", t = %{x}: %{customdata:.3f}<extra></extra>", customdata=c))
    fig.update_xaxes(title_text="Zeit (Abtastwerte)", range=[0, len(a.x) - 1])
    fig.update_yaxes(showticklabels=False, title_text="Glättung oben, Stufe 1 unten")
    return _fixed(fig, height=60 * len(comps) + 120)


def build_variance(a: Analysis) -> go.Figure:
    """Wavelet-Varianz gegen die Stufe (log2), die angepasste Gerade und, wenn bekannt, die Gerade mit dem wahren Anstieg α − 1 durch den Schwerpunkt der Anpassung."""
    lev = np.arange(1, len(a.variance) + 1)
    ok = np.isfinite(a.variance) & (a.variance > 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lev[ok], y=np.log2(a.variance[ok]), mode="markers", marker=dict(color=C.COLOR_EST, size=8), name="Wavelet-Varianz", hovertemplate="Stufe %{x}: log2 = %{y:.2f}<extra></extra>"))
    if a.fit is not None:
        used = np.array(a.fit_levels)
        fig.add_trace(go.Scatter(x=used, y=a.fit[0] * used + a.fit[1], mode="lines", line=dict(color=C.COLOR_RIDGE, width=2), name=f"angepasst: Steigung {a.fit[0]:.2f}".replace(".", ","),
                                 hoverinfo="skip"))
        if a.alpha_true is not None:
            cx = float(used.mean())
            cy = a.fit[0] * cx + a.fit[1]
            fig.add_trace(go.Scatter(x=used, y=cy + (a.alpha_true - 1.0) * (used - cx), mode="lines", line=dict(color=C.COLOR_TRUE, width=1.6, dash="dash"),
                                     name=f"wahr: Steigung {a.alpha_true - 1.0:.2f}".replace(".", ","), hoverinfo="skip"))
    fig.update_xaxes(title_text="Stufe j", dtick=1)
    fig.update_yaxes(title_text="log₂ der Wavelet-Varianz")
    return _fixed(fig, height=320)


def build_moments(res) -> go.Figure:
    fig = go.Figure()
    for r in res["degrees"]:
        fig.add_trace(go.Scatter(x=list(range(5)), y=[max(v, 1e-17) for v in r["max"]], mode="lines+markers", name=f"p = {r['p']}", hovertemplate=f"p = {r['p']}, Grad %{{x}}: %{{y:.1e}}<extra></extra>"))
    fig.update_xaxes(title_text="Grad des Polynoms", dtick=1)
    fig.update_yaxes(title_text="größter innerer Detailkoeffizient (bis 1e-17 abgeschnitten)", type="log")
    return _fixed(fig, height=340)


def build_sparsity(res) -> go.Figure:
    rows = res["jump"]
    fig = go.Figure(go.Bar(x=[f"p = {r['p']}" for r in rows], y=[100 * r["top"] for r in rows], marker_color=C.COLOR_EST, hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.update_yaxes(title_text=f"Energie der {C.SPARSE_TOP} größten Koeffizienten (%)", range=[90, 100])
    fig.update_layout(showlegend=False)
    return _fixed(fig, height=300)


def build_shift(rows) -> go.Figure:
    levels = len(rows[0]["dwt"]) - 1
    labels = level_labels(levels)
    fig = make_subplots(rows=1, cols=len(rows), subplot_titles=[r["name"] for r in rows], shared_yaxes=False)
    for i, r in enumerate(rows, start=1):
        fig.add_trace(go.Bar(x=labels, y=r["dwt"], name="DWT", marker_color=C.COLOR_EST, showlegend=i == 1, hovertemplate="%{x}: %{y:.3f}<extra>DWT</extra>"), row=1, col=i)
        fig.add_trace(go.Bar(x=labels, y=[max(v, 1e-6) for v in r["modwt"]], name="MODWT", marker_color=C.COLOR_RIDGE, showlegend=i == 1, hovertemplate="%{x}: %{y:.1e}<extra>MODWT</extra>"), row=1, col=i)
    fig.update_yaxes(title_text="Spanne des Energieanteils", row=1, col=1)
    fig.update_layout(barmode="group")
    return _fixed(fig, height=340)


def build_alpha(res) -> go.Figure:
    fig = go.Figure()
    names = {("DWT", 1): "DWT Haar", ("MODWT", 1): "MODWT Haar", ("DWT", 2): "DWT D4", ("MODWT", 2): "MODWT D4", ("DWT", 4): "DWT D8", ("MODWT", 4): "MODWT D8", ("GPH", 0): "Periodogramm (GPH)"}
    for key, label in names.items():
        sub = [r for r in res["stationary"] if (r["estimator"], r["p"]) == key]
        fig.add_trace(go.Bar(x=[f"α = {r['alpha']:g}".replace(".", ",") for r in sub], y=[r["sd"] for r in sub], name=label, hovertemplate=label + ", %{x}: Std %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text="Standardabweichung von α̂", rangemode="tozero")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=340)


def build_trend(res) -> go.Figure:
    tr = res["trend"]
    labels = [f"p = {r['p']}" for r in tr]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["inner_mean"] for r in tr], error_y=dict(type="data", array=[r["inner_sd"] for r in tr]), name="MODWT ohne Randkoeffizienten", marker_color=C.COLOR_RIDGE,
                         hovertemplate="%{x}: %{y:.3f}<extra>MODWT</extra>"))
    fig.add_trace(go.Bar(x=labels, y=[r["dwt_mean"] for r in tr], error_y=dict(type="data", array=[r["dwt_sd"] for r in tr]), name="DWT periodisch", marker_color=C.COLOR_EST, hovertemplate="%{x}: %{y:.3f}<extra>DWT</extra>"))
    fig.add_hline(y=res["trend_alpha"], line=dict(color=C.COLOR_TRUE, width=1.6, dash="dash"), annotation_text="wahr", annotation_position="top left")
    fig.update_yaxes(title_text="geschätzter Exponent α̂ (Trend)", rangemode="tozero")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=340)


def build_bands(rows) -> go.Figure:
    fig = go.Figure()
    for j in sorted(rows[0]["fractions"]):
        fig.add_trace(go.Bar(x=[f"p = {r['p']}" for r in rows], y=[100 * r["fractions"][j] for r in rows], name=f"Stufe {j}", hovertemplate=f"Stufe {j}, %{{x}}: %{{y:.1f}} %<extra></extra>"))
    fig.update_yaxes(title_text="Energie des Tons in der richtigen Stufe (%)", range=[0, 100])
    fig.update_layout(barmode="group")
    return _fixed(fig, height=320)


def build_variance_spread(res) -> go.Figure:
    levels = list(range(1, len(res["dwt_rel"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=levels, y=res["dwt_rel"], mode="lines+markers", name="DWT", hovertemplate="Stufe %{x}: %{y:.3f}<extra>DWT</extra>"))
    fig.add_trace(go.Scatter(x=levels, y=res["modwt_rel"], mode="lines+markers", name="MODWT", hovertemplate="Stufe %{x}: %{y:.3f}<extra>MODWT</extra>"))
    fig.update_xaxes(title_text="Stufe j", dtick=1)
    fig.update_yaxes(title_text="relative Standardabweichung von ν̂²", rangemode="tozero")
    return _fixed(fig, height=320)
