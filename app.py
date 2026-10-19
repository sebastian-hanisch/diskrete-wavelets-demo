"""Diskrete Wavelets und Mehrskalenanalyse: exakt zerlegt, aber verschiebungsvariant – und die Wavelet-Varianz als Skalenspektrum - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 12 der Reihe „Spektral- und Wavelet-Analyse“. Fünf Testsignale mit exakter Wahrheit (Potenzgesetz-Rauschen, Sprung im glatten Signal, Impuls, Ton, Trend plus Rauschen) werden mit der
diskreten Wavelet-Transformation (DWT, Daubechies) und der überlappenden MODWT zerlegt; gezeigt werden Koeffizienten, Mehrskalenanalyse und Wavelet-Varianz, dazu fünf Experimente zu verschwindenden
Momenten, Verschiebungsvarianz, Exponentenschätzung, Frequenzselektivität der Stufen und Streuung des Varianzschätzers. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import dwt_constants as C
import dwt_evaluation as ev
from dwt_evaluation import Settings, analyse, de, wavelet_name
from dwt_pdf_export import build_pdf
from dwt_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    param_key,
    randomize_seed,
    sync_query_params,
)
from dwt_visualization import (
    build_alpha,
    build_bands,
    build_coefficients,
    build_levels,
    build_moments,
    build_mra,
    build_shift,
    build_signal,
    build_sparsity,
    build_trend,
    build_variance,
    build_variance_spread,
)

st.set_page_config(page_title="Diskrete Wavelets – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Rechne die Experimente …")
def _experiment(name):
    return getattr(ev, "sweep_" + name)()


def _short(p: int) -> str:
    return "Haar" if p == 1 else f"D{2 * p}"


st.title("🪜 Diskrete Wavelets: exakt zerlegt, aber verschiebungsvariant")
st.markdown(
    """
Das kontinuierliche Skalogramm der vorigen Demo ist redundant: Benachbarte Zellen sind stark korreliert. Die **diskrete Wavelet-Transformation (DWT)** wählt nur Zweierpotenz-Skalen und ein Raster und zerlegt eine Reihe der
Länge $2^J$ **exakt und ohne Redundanz**: Ein Paar Filter ($h$ glättet, $g$ nimmt die Differenz) wird auf die Reihe angewandt und halbiert ihre Länge, dann auf die Glättung, und so weiter (die **Pyramide**). Daubechies-Filter mit $p$
verschwindenden Momenten machen Polynome vom Grad kleiner als $p$ zu null und glatte Teile nahezu null. Der Preis: Die DWT ist **nicht verschiebungsinvariant**; die **MODWT** lässt das Halbieren weg und ist es. Die **Wavelet-Varianz** (mittlere Energie je Stufe) ist das
Skalenspektrum: Aus ihrer Steigung liest man den Exponenten von Potenzgesetz-Rauschen ab.

Dies ist die zwölfte Demo der Reihe „Spektral- und Wavelet-Analyse“, auf das **Skalogramm** aufbauend; sie liefert die Koeffizienten, auf denen die Wavelet-Schrumpfung danach Schwellwerte setzt. Alles ist **simuliert**: Signale
und Exponenten sind bekannt. Weiter unten: **🔬 Experimente**, **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So entsteht die Zerlegung", expanded=True):
    st.markdown(
        """
1. **Filterpaar:** Das Glättungsfilter $h$ (Länge $2p$) hat die Summe $\\sqrt 2$ und die Summe der Quadrate 1; das Wavelet-Filter ist $g[l] = (-1)^l h[2p - 1 - l]$ und hat $p$ verschwindende Momente.
2. **Eine Stufe:** $c[k] = \\sum_l h[l]\\,a[2k + l]$ (Glättung) und $d[k] = \\sum_l g[l]\\,a[2k + l]$ (Detail), je mit der halben Länge (periodisch fortgesetzt). Die nächste Stufe arbeitet auf $c$.
3. **Exakt:** Energie und Reihe sind ohne Verlust rekonstruierbar (Fehler höchstens etwa $10^{-14}$). Die **MODWT** behält alle $n$ Werte je Stufe und braucht $(J + 1)\\,n$ Koeffizienten.
4. **Mehrskalenanalyse:** $x = S_J + \\sum_j D_j$: Jede Stufe liefert einen Anteil in der Zeitdomäne, der gesamte Satz addiert sich zum Signal.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Signal**")
    n = st.select_slider("Länge n der Aufzeichnung", options=list(C.N_CHOICES), key="n_select", help="Zahl der Abtastwerte (Zweierpotenz, Abtastrate 1).")
    kind = st.selectbox("Signaltyp", list(C.SIGNALS), key="signal_select",
                        help="Potenzgesetz-Rauschen: Spektrum ∝ f^−α. Sprung: glatter Sinus mit Sprung. Impuls: ein Einzelwert. Ton: eine feste Frequenz. Trend plus Rauschen: Rauschen und ein linearer Trend.")
    ti = C.SIGNALS.index(kind)
    values = []
    for slot, (label, lo, hi, step, default, help_text) in enumerate(C.PARAMS[kind], start=1):
        v = st.slider(label, lo, hi, float(st.session_state[param_key(ti, slot)]), step=step, format="%.3f" if step < 0.01 else "%.2f", help=help_text)
        st.session_state[param_key(ti, slot)] = float(v)
        values.append(float(v))
    while len(values) < 3:
        values.append(0.0)
    st.markdown("**Wavelet und Transformation**")
    p = st.select_slider("Wavelet (Daubechies)", options=list(C.WAVELET_P), key="p_select", format_func=_short,
                         help="p verschwindende Momente, Filterlänge 2p. Haar (p = 1) ist die einfachste; mehr Momente: glatter, besser in der Frequenz, länger.")
    max_levels = min(C.MAX_LEVELS, int(math.log2(n)) - 1)
    if st.session_state["levels_select"] > max_levels:
        st.session_state["levels_select"] = max_levels
    levels = st.select_slider("Stufen J", options=list(range(C.LEVEL_MIN, max_levels + 1)), key="levels_select", help="Zahl der Zerlegungsstufen; höchstens log₂(n) − 1.")
    transform = st.radio("Transformation", list(C.TRANSFORM_MODES), key="transform_select", horizontal=True,
                         help="DWT: dezimiert, nicht redundant, nicht verschiebungsinvariant. MODWT: nicht dezimiert, (J + 1)n Koeffizienten, verschiebungsinvariant, jede Länge.")
    boundary = st.radio("Rand (Wavelet-Varianz)", list(C.BOUNDARY_MODES), key="boundary_select", help="Nur MODWT: periodisch nutzt alle Koeffizienten; „Randkoeffizienten verwerfen“ lässt die von der periodischen "
                        "Fortsetzung betroffenen aus (nötig bei Trend). Die DWT ist immer periodisch.")
    shift = st.slider("Verschiebung (Abtastwerte, zirkulär)", C.SHIFT_MIN, C.SHIFT_MAX, int(st.session_state["shift_slider"]), key="shift_slider",
                      help="Verschiebt die Messung vor der Zerlegung; zeigt die Verschiebungsempfindlichkeit der DWT.")
    if kind in C.NOISY_SIGNALS:
        st.markdown("**Messung**")
        noise = st.radio("Rauschen", list(C.NOISE_MODES), key="noise_select", horizontal=True, help="Weißes Gaußrauschen zum Signal addiert.")
        if noise == "an":
            snr = st.slider("Rauschabstand (dB)", C.SNR_MIN, C.SNR_MAX, int(st.session_state["snr_kept"]), help="Leistung einer Linie der Amplitude 1 (also 1/2) gegen die Rauschleistung.")
            st.session_state["snr_kept"] = int(snr)
        else:
            snr = int(st.session_state["snr_kept"])
    else:
        noise, snr = "aus", int(st.session_state["snr_kept"])
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_select": int(n), "signal_select": kind, "p_select": int(p), "levels_select": int(levels), "transform_select": transform, "boundary_select": boundary, "shift_slider": int(shift),
                   "noise_select": noise, "snr_kept": int(snr), "seed_input": int(seed), **{param_key(ti, slot): v for slot, v in enumerate(values[:len(C.PARAMS[kind])], start=1)}})

settings = Settings(int(n), kind, values[0], values[1], values[2], int(p), int(levels), transform, boundary, int(shift), noise, int(snr), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rekonstruktionsfehler", f"{a.recon_error:.1e}".replace(".", ","), delta=f"Energiebilanz {a.energy_error:.0e}".replace(".", ","), delta_color="off",
          help="Größte Abweichung der aus den Koeffizienten zurückgerechneten Reihe von der Messung; die Zerlegung ist exakt.")
if a.alpha_true is not None:
    m2.metric("Exponent α̂", de(a.alpha_hat, 2) if a.alpha_hat is not None else "–", delta=f"wahr {de(a.alpha_true, 2)}; Periodogramm {de(a.gph, 2)}", delta_color="off",
              help="α̂ = Steigung der Geraden durch log₂ der Wavelet-Varianz gegen die Stufe, plus 1; das Periodogramm (GPH) zum Vergleich.")
elif kind == "Ton" and a.band_fraction is not None:
    m2.metric(f"Energie in Stufe {a.band_level}", f"{de(100 * a.band_fraction, 1)} %", delta=f"Band {de(1 / 2 ** (a.band_level + 1), 3)} bis {de(1 / 2 ** a.band_level, 3)}", delta_color="off",
              help="Anteil der Energie des Tons in der Stufe, deren Band seine Frequenz enthält.")
else:
    m2.metric("Koeffizienten über 1e-3", f"{a.sparse_count}", delta=f"16 größte: {de(100 * a.sparse_top, 1)} % der Energie", delta_color="off", help="DWT-Koeffizienten über dem 1e-3-Fachen des größten (n insgesamt).")
m3.metric("Koeffizienten", f"{a.coef_count}", delta=f"{'nicht redundant' if transform == 'DWT' else f'{levels + 1}-fach redundant'}", delta_color="off", help="Zahl der Werte der gewählten Transformation: n bei der DWT, (J + 1) n bei der MODWT.")
m4.metric("Änderung bei Verschiebung", de(a.shift_change, 3), delta=f"Verschiebung {settings.shift}", delta_color="off",
          help="Größte Änderung eines Energieanteils (Stufe oder Glättung), wenn die Messung zirkulär verschoben wird; bei der MODWT null.")
st.plotly_chart(build_signal(a), width="stretch", key="signal")
st.plotly_chart(build_coefficients(a), width="stretch", key="coefficients")
st.caption("Koeffizientenquadrate je Stufe über der Zeit, in dB gegen das Maximum. Bei der DWT gilt ein Koeffizient der Stufe j für 2^j Abtastwerte; bei der MODWT hat jede Stufe n Werte.")
st.plotly_chart(build_levels(a), width="stretch", key="levels")
st.caption("Anteil der Gesamtenergie je Stufe und in der Glättung (Summe 100 %).")
st.plotly_chart(build_mra(a), width="stretch", key="mra")
st.caption(f"Mehrskalenanalyse (MODWT): Das Signal ist die Summe aus Glättung und Detailanteilen D1 bis D{levels}; die Summe weicht um {a.mra_error:.0e} vom Signal ab.".replace(".", ","))
st.plotly_chart(build_variance(a), width="stretch", key="variance")
st.caption("Wavelet-Varianz je Stufe (log₂). Bei Potenzgesetz-Rauschen mit Spektrum ∝ f^−α ist die Steigung α − 1; angepasst wird ab Stufe 2, solange n/2^j mindestens 32 ist.")

if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = build_pdf(a)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and "pdf_bytes" in st.session_state:
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="diskrete-wavelets.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was verrät die Zerlegung, und was verändert eine Verschiebung?")
kind_v, text = ev.verdict_main(a)
getattr(st, kind_v)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
st.caption("Feste Zufallsströme, unabhängig von den Reglern. Abtastrate 1; Daubechies-Filter D2 bis D16 (Haar = p = 1); DWT periodisch, MODWT periodisch (wenn nicht anders genannt).")

st.markdown("#### Verschwindende Momente und Sparsität")
mom = _experiment("moments")
st.caption(f"Polynome vom Grad 0 bis 4 (n = {C.EXP_MOM_N}): größter Detailkoeffizient der Stufe 1 im Inneren (ohne die Randwerte). Darunter ein Sinus mit 3 Perioden und einem Sprung der Höhe 1 bei 0,3 n "
           f"(n = {C.EXP_JUMP_N}, {C.EXP_JUMP_J} Stufen): Koeffizienten über dem 1e-3-Fachen des größten und Energieanteil der {C.SPARSE_TOP} größten.")
st.dataframe(pd.DataFrame([{"p": r["p"], **{f"Grad {d}": f"{v:.1e}".replace(".", ",") for d, v in enumerate(r["max"])}} for r in mom["degrees"]]), hide_index=True, width="stretch")
st.plotly_chart(build_moments(mom), width="stretch", key="exp_moments")
st.dataframe(pd.DataFrame([{"p": r["p"], "Koeffizienten über 1e-3": f"{r['count']} von {r['total']}", f"Energie der {C.SPARSE_TOP} größten": f"{de(100 * r['top'], 1)} %"} for r in mom["jump"]]), hide_index=True, width="stretch")
st.plotly_chart(build_sparsity(mom), width="stretch", key="exp_sparsity")
kind_v, text = ev.verdict_moments(mom)
getattr(st, kind_v)(text)

st.markdown("#### Verschiebungsvarianz: DWT gegen MODWT")
shift_rows = _experiment("shift")
st.caption(f"n = {C.EXP_SHIFT_N}, {C.EXP_SHIFT_J} Stufen, D4: Spanne (größter minus kleinster Wert) des Energieanteils je Stufe über {C.EXP_SHIFT_COUNT} zirkuläre Verschiebungen (0 bis {C.EXP_SHIFT_COUNT - 1}); "
           "Impuls bei 10, Töne bei f = 0,07 und 0,1.")
st.dataframe(pd.DataFrame([{"Signal": r["name"], "Transformation": t, **{lab: de(v, 3) for lab, v in zip([f"Stufe {j}" for j in range(1, C.EXP_SHIFT_J + 1)] + ["Glättung"], r[key])}}
                           for r in shift_rows for t, key in (("DWT", "dwt"), ("MODWT", "modwt"))]), hide_index=True, width="stretch")
st.plotly_chart(build_shift(shift_rows), width="stretch", key="exp_shift")
kind_v, text = ev.verdict_shift(shift_rows)
getattr(st, kind_v)(text)

st.markdown("#### Der Exponent aus der Wavelet-Varianz")
alpha = _experiment("alpha")
st.caption(f"Potenzgesetz-Rauschen (Spektrum ∝ f^−α, zirkulär stationär), n = {alpha['n']}, {alpha['levels']} Stufen, {alpha['runs']} Läufe; α̂ = Steigung von log₂ ν̂² gegen die Stufe plus 1 (Stufen 2 bis 7). "
           f"Periodogramm: GPH mit m = {C.EXP_GPH_M} Frequenzen. Darunter ein linearer Trend von {de(alpha['trend_sd'], 0)} Standardabweichungen (α = {de(alpha['trend_alpha'], 1)}): MODWT ohne die vom Rand betroffenen Koeffizienten "
           "gegen die periodische DWT.")
names = {("DWT", 1): "DWT Haar", ("MODWT", 1): "MODWT Haar", ("DWT", 2): "DWT D4", ("MODWT", 2): "MODWT D4", ("DWT", 4): "DWT D8", ("MODWT", 4): "MODWT D8", ("GPH", 0): "Periodogramm (GPH)"}
st.dataframe(pd.DataFrame([{"Schätzer": label, **{f"α = {al:g}".replace(".", ","): f"{de(r['mean'], 3)} ± {de(r['sd'], 3)}" for al in C.EXP_ALPHA_VALUES for r in alpha["stationary"]
                                                 if r["alpha"] == al and (r["estimator"], r["p"]) == key}} for key, label in names.items()]), hide_index=True, width="stretch")
st.plotly_chart(build_alpha(alpha), width="stretch", key="exp_alpha")
st.dataframe(pd.DataFrame([{"p": r["p"], "MODWT ohne Randkoeffizienten": f"{de(r['inner_mean'], 3)} ± {de(r['inner_sd'], 3)}", "DWT periodisch": f"{de(r['dwt_mean'], 3)} ± {de(r['dwt_sd'], 3)}"} for r in alpha["trend"]]
                          + [{"p": "GPH", "MODWT ohne Randkoeffizienten": f"{de(alpha['trend_gph'][0], 3)} ± {de(alpha['trend_gph'][1], 3)}", "DWT periodisch": ""}]), hide_index=True, width="stretch")
st.plotly_chart(build_trend(alpha), width="stretch", key="exp_trend")
kind_v, text = ev.verdict_alpha(alpha)
getattr(st, kind_v)(text)

st.markdown("#### Frequenzselektivität der Stufen")
bands = _experiment("bands")
st.caption(f"Ton in der Mitte des Bandes der Stufe j (f = 0,75/2^j), MODWT, n = {C.EXP_BAND_N}: Anteil der Energie in dieser Stufe; dazu der Leistungsgang |G₃(f)|² der Stufe 3 bei f = 1/16, 3/32, 1/8, 3/16, 1/4.")
st.dataframe(pd.DataFrame([{"Wavelet": _short(r["p"]), **{f"Stufe {j}": f"{de(100 * r['fractions'][j], 1)} %" for j in C.EXP_BAND_LEVELS}, **{f"|G₃|² bei {lab}": de(v, 3) for lab, v in zip(("1/16", "3/32", "1/8", "3/16", "1/4"), r["response"])}}
                           for r in bands]), hide_index=True, width="stretch")
st.plotly_chart(build_bands(bands), width="stretch", key="exp_bands")
kind_v, text = ev.verdict_bands(bands)
getattr(st, kind_v)(text)

st.markdown("#### Streuung der Wavelet-Varianz: MODWT gegen DWT")
spread = _experiment("variance")
st.caption(f"Weißes Rauschen (Varianz 1), n = {spread['n']}, {spread['levels']} Stufen, {spread['runs']} Läufe, D4: relative Standardabweichung des Schätzers ν̂²_j; Erwartung beider Schätzer 2^−j (im Mittel unverzerrt).")
st.dataframe(pd.DataFrame([{"Stufe": j, "DWT": de(d, 3), "MODWT": de(m, 3), "Verhältnis": de(r, 2), "Erwartung (DWT)": de(e1, 4), "Erwartung (MODWT)": de(e2, 4)}
                           for j, (d, m, r, e1, e2) in enumerate(zip(spread["dwt_rel"], spread["modwt_rel"], spread["ratio"], spread["dwt_mean"], spread["modwt_mean"]), start=1)]), hide_index=True, width="stretch")
st.plotly_chart(build_variance_spread(spread), width="stretch", key="exp_spread")
kind_v, text = ev.verdict_variance(spread)
getattr(st, kind_v)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Zerlegung ist verschiebungsinvariant** | Die DWT ändert ihre Energieanteile je Stufe bei fast jeder Verschiebung (nicht bei Vielfachen von 2^J); bei einem Impuls bis über 0,5; die MODWT bleibt gleich, braucht aber (J + 1)-mal so viele Koeffizienten. | MODWT, Cycle Spinning |
| **Die Reihe ist zirkulär** | Die periodische Fortsetzung macht aus einem Trend einen Sprung am Rand; die Exponentenschätzung wird stark verzerrt (1,3 bis 1,9 statt 0,5). | Randkoeffizienten verwerfen, Spiegeln (nicht gebaut) |
| **Ein verschwindendes Moment genügt** | Haar nullt nur Konstanten: Ein starker linearer Trend verfälscht die Wavelet-Varianz; ab p = 2 bleibt sie unverändert, wenn die vom periodischen Rand betroffenen Koeffizienten verworfen werden (MODWT). | größeres p |
| **Das Potenzgesetz ist stationär** | Die Wavelet-Varianz gilt für Spektren ∝ f^−α nur mit α < 2p + 1; sonst ist sie verfälscht. | größeres p |
| **Stufen sind Frequenzbänder** | Die Filter sind nicht scharf: Ein Ton in der Bandmitte liefert nur 54 % (Haar) bis 94 % (D16) der Energie in der Stufe, der Rest liegt in den Nachbarstufen. | Wavelet-Pakete (nicht gebaut) |
| **Die Länge ist 2^J** | Die DWT braucht eine durch 2^J teilbare Länge; die MODWT hat diese Bindung nicht. | MODWT |
""",
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Filter.** $h$ hat Länge $2p$, $\sum h = \sqrt 2$, $\sum h^2 = 1$ und $\sum_l h[l]h[l+2k] = 0$ für $k \ne 0$; $g[l] = (-1)^l h[2p-1-l]$ mit $\sum_l l^m g[l] = 0$ für $m < p$ (verschwindende Momente). Die Daubechies-Filter entstehen aus der
Spektralfaktorisierung von $|H(\omega)|^2 = \cos^{2p}(\omega/2)\,P(\sin^2(\omega/2))$ mit $P(y) = \sum_{k<p}\binom{p-1+k}{k} y^k$.

**DWT.** $c_j[k] = \sum_l h[l]\,a_{j-1}[(2k + l) \bmod N_{j-1}]$, $d_j[k] = \sum_l g[l]\,a_{j-1}[(2k + l) \bmod N_{j-1}]$, $a_0 = x$; $\|x\|^2 = \sum_j \|d_j\|^2 + \|a_J\|^2$ (Orthogonalität), Rekonstruktion mit den transponierten Filtern.

**MODWT.** Mit den Filtern $h/\sqrt 2$ und $g/\sqrt 2$ ohne Dezimieren: $\tilde W_j = \tilde G_j X$ im Frequenzbereich, $\tilde G_j(f) = \tilde G(2^{j-1} f) \prod_{l<j-1}\tilde H(2^l f)$; $\sum_j |\tilde G_j|^2 + |\tilde H_J|^2 = 1$, also
$x = \sum_j D_j + S_J$ mit $D_j = \mathrm{IFFT}(X |\tilde G_j|^2)$ (Mehrskalenanalyse).

**Wavelet-Varianz.** $\hat\nu^2_j = \frac1N \sum_t \tilde W_j[t]^2$ (MODWT) bzw. $\overline{d_j^2}/2^j$ mit dem Mittel über die $N/2^j$ Koeffizienten der Stufe (DWT); für weißes Rauschen der Varianz 1 ist $\nu^2_j = 2^{-j}$. Bei einem Spektrum
$\propto f^{-\alpha}$ ist $\nu^2_j \propto 2^{j(\alpha - 1)}$ (für $-1 < \alpha < 2p + 1$), also $\hat\alpha = 1 + $ Steigung von $\log_2 \hat\nu^2_j$ gegen $j$.

**Randlänge.** Das MODWT-Stufenfilter hat die Länge $L_j = (2^j - 1)(2p - 1) + 1$; die ersten $L_j - 1$ Koeffizienten der Stufe $j$ hängen von der periodischen Fortsetzung ab.

**Quellen.** Lehrbuchwissen und Aufsätze (Daubechies 1988; Mallat 1989; Percival und Walden 2000; Abry und Veitch 1998). Formeln hier per Simulation, Handrechnung und gegen geschlossene Formeln geprüft; die Quellen selbst wurden nicht gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Spektren und Wavelets: von Fourier zum Vergleich](https://sebastianhanisch.net/konzepte-spektral-wavelet-analyse.html)."
)
