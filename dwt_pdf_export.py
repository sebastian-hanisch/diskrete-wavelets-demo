"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import dwt_constants as C
from dwt_evaluation import Analysis, de, wavelet_name

_REPLACE = {"–": "-", "—": "-", "−": "-", "²": "2", "₂": "2", "€": "EUR", "→": "->", "σ": "sigma", "Σ": "Summe", "±": "+-", "√": "Wurzel aus ", "×": "x", "é": "e", "„": '"', "“": '"', "β": "beta", "π": "pi",
            "≥": ">=", "α̂": "alpha_hat", "α": "alpha", "ν": "nu"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Diskrete Wavelets: Einstellung und Kennzahlen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    params = ", ".join(f"{label} {de(v, 3)}" for (label, *_rest), v in zip(C.PARAMS[s.signal], (s.p1, s.p2, s.p3)))
    noise = f"Rauschen {s.snr} dB gegen eine Linie der Amplitude 1" if s.has_noise else "ohne Rauschen"
    pdf.multi_cell(0, 5, _s(f"{s.signal} ({params}), n = {s.n}, {wavelet_name(s.p)}, {s.levels} Stufen, {s.transform}, Rand: {s.boundary}, Verschiebung {s.shift}, {noise}, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    out = [f"Koeffizienten: {a.coef_count}; Rekonstruktionsfehler {a.recon_error:.1e}, Energiebilanz {a.energy_error:.1e}, Mehrskalenanalyse (Summe der Anteile) {a.mra_error:.1e}.".replace(".", ","),
           "Energieanteile je Stufe (Stufe 1 bis J, Glättung): " + ", ".join(de(100 * f, 1) + " %" for f in a.fractions) + ".",
           f"DWT-Koeffizienten über dem {C.SPARSE_REL:.0e}-Fachen des größten: {a.sparse_count}; die {C.SPARSE_TOP} größten tragen {de(100 * a.sparse_top, 1)} % der Energie."]
    if a.alpha_true is not None and a.alpha_hat is not None:
        out.append(f"Exponent aus der Wavelet-Varianz (Stufen {a.fit_levels[0]} bis {a.fit_levels[-1]}): alpha_hat = {de(a.alpha_hat, 3)}, wahr {de(a.alpha_true, 3)}, Periodogramm (GPH) {de(a.gph, 3)}.")
    if s.shift:
        out.append(f"Verschiebung um {s.shift}: größte Änderung eines Energieanteils {de(a.shift_change, 4)}.")
    if a.band_level is not None:
        out.append(f"Stufe {a.band_level} (Band {de(1 / 2 ** (a.band_level + 1), 3)} bis {de(1 / 2 ** a.band_level, 3)}): {de(100 * a.band_fraction, 1)} % der Energie des Tons.")
    for line in out:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: diskrete-wavelets-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
