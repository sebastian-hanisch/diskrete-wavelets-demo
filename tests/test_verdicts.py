"""Verdikte: reine Funktionen an handgesetzten Zeilen; Art, Schwellen und jede Zahl im Text."""

import numpy as np

import dwt_evaluation as ev
import dwt_signals as S


def make_analysis(signal="Potenzgesetz-Rauschen", p1=0.5, p2=0.0, p=2, transform="MODWT", boundary="periodisch", shift=0, n=4096, levels=9, alpha_hat=0.49, alpha_true=0.5, gph=0.52, fit_levels=(2, 3, 4, 5, 6, 7),
                  sparse_count=73, sparse_top=0.9932, shift_change=0.0, band_level=None, band_fraction=None):
    z = np.zeros(1)
    s = ev.Settings(n=n, signal=signal, p1=p1, p2=p2, p=p, transform=transform, boundary=boundary, shift=shift, levels=levels)
    return ev.Analysis(s, S.TestSignal(z, alpha_true, ()), z, [], z, np.zeros(2), 0.0, 0.0, [], z, 0.0, np.zeros(2), list(fit_levels), None, alpha_hat, alpha_true, gph, 100, sparse_count, sparse_top, shift_change,
                       band_level, band_fraction)


def test_main_verdict_power_law_threshold_and_text():
    kind, text = ev.verdict_main(make_analysis())
    assert kind == "success"
    assert text == "D4 (p = 2), MODWT: Der Exponent aus der Wavelet-Varianz (Stufen 2 bis 7) ist α̂ = 0,49 gegen wahr 0,50; das Periodogramm (GPH) schätzt 0,52."
    assert ev.verdict_main(make_analysis(alpha_hat=0.6499))[0] == "success" and ev.verdict_main(make_analysis(alpha_hat=0.65))[0] == "warning" and ev.verdict_main(make_analysis(alpha_hat=0.3501))[0] == "success"
    assert ev.verdict_main(make_analysis(alpha_hat=0.35))[0] == "warning"
    kind, text = ev.verdict_main(make_analysis(alpha_hat=None, fit_levels=()))
    assert kind == "info" and text == "D4 (p = 2), MODWT: Für die Geradenanpassung bleiben weniger als drei Stufen (ab Stufe 2, solange n/2^j mindestens 32 ist); wählen Sie ein längeres n oder mehr Stufen."


def test_main_verdict_names_the_wavelet_and_the_violated_condition():
    assert ev.verdict_main(make_analysis(p=1, transform="DWT"))[1].startswith("Haar (p = 1), DWT: ")
    text = ev.verdict_main(make_analysis(p1=3.0, alpha_true=3.0, alpha_hat=3.0, p=1))[1]
    assert text.endswith(" Hinweis: α ≥ 2p + 1 = 3 verletzt die Bedingung der Wavelet-Varianz.")
    assert "verletzt" not in ev.verdict_main(make_analysis(p1=2.9, alpha_true=2.9, alpha_hat=2.9, p=1))[1] and "α ≥ 2p + 1 = 5" in ev.verdict_main(make_analysis(alpha_true=5.0, alpha_hat=5.0, p=2))[1]


def test_main_verdict_trend_hints():
    base = dict(signal="Trend plus Rauschen", p2=100.0, alpha_hat=1.25)
    kind, text = ev.verdict_main(make_analysis(p=1, boundary="Randkoeffizienten verwerfen", **base))
    assert kind == "warning" and text.endswith(" Haar hat nur ein verschwindendes Moment, ein linearer Trend braucht mindestens p = 2.")
    text = ev.verdict_main(make_analysis(p=2, boundary="periodisch", **base))[1]
    assert text.endswith(" Die periodische Fortsetzung macht aus dem Trend einen Sprung am Rand; verwerfen Sie die Randkoeffizienten (MODWT).") and "Haar hat nur" not in text
    both = ev.verdict_main(make_analysis(p=1, boundary="periodisch", **base))[1]
    assert "Haar hat nur" in both and "periodische Fortsetzung" in both
    assert "Haar hat nur" not in ev.verdict_main(make_analysis(p=1, boundary="Randkoeffizienten verwerfen", signal="Trend plus Rauschen", p2=20.0, alpha_hat=0.5))[1]
    assert "periodische Fortsetzung" not in ev.verdict_main(make_analysis(p=2, boundary="periodisch", signal="Potenzgesetz-Rauschen"))[1]


def test_main_verdict_jump_impulse_and_tone():
    kind, text = ev.verdict_main(make_analysis(signal="Sprung im glatten Signal", alpha_hat=None, alpha_true=None, gph=None, n=1024, p=4, transform="DWT"))
    assert kind == "info" and text == "D8 (p = 4), DWT: 73 von 1024 DWT-Koeffizienten liegen über dem 1e-3-Fachen des größten; die 16 größten tragen 99,3 % der Energie."
    base = dict(alpha_hat=None, alpha_true=None, gph=None, n=256)
    kind, text = ev.verdict_main(make_analysis(signal="Impuls", **base))
    assert kind == "info" and text == "D4 (p = 2), MODWT: Ein Einzelimpuls verteilt seine Energie auf alle Stufen. Verschieben Sie ihn, um die Verschiebungsempfindlichkeit zu sehen."
    text = ev.verdict_main(make_analysis(signal="Impuls", shift=1, shift_change=0.433, transform="DWT", **base))[1]
    assert text == "D4 (p = 2), DWT: Ein Einzelimpuls verteilt seine Energie auf alle Stufen. Verschiebung um 1: Der Energieanteil einer Stufe ändert sich um 0,433; die MODWT bliebe exakt gleich."
    text = ev.verdict_main(make_analysis(signal="Impuls", shift=2, shift_change=1e-12, **base))[1]
    assert text.endswith(" Verschiebung um 2: Die Energieanteile bleiben gleich (Änderung 0,000).") and "Verschieben Sie ihn" not in text
    kind, text = ev.verdict_main(make_analysis(signal="Ton", p1=0.09375, band_level=3, band_fraction=0.7134, **base))
    assert kind == "info" and text == "D4 (p = 2), MODWT: Stufe 3 hat das Band 0,062 bis 0,125; sie trägt 71,3 % der Energie des Tons, der Rest liegt in den Nachbarstufen."
    text = ev.verdict_main(make_analysis(signal="Ton", p1=0.4, band_level=None, shift=3, shift_change=0.05, **base))[1]
    assert text == "D4 (p = 2), MODWT: Die Frequenz liegt außerhalb der Bänder der gewählten Stufen. Verschiebung um 3: Der Energieanteil einer Stufe ändert sich um 0,050; die MODWT bliebe exakt gleich."


def test_verdict_moments_text():
    res = {"degrees": [{"p": 1, "max": [0.0, 0.0028, 0.0054, 0.0081, 0.0106]}, {"p": 2, "max": [1e-16, 2e-16, 1.869e-05, 5.5e-05, 1.1e-04]}, {"p": 8, "max": [1.8e-15] * 5}],
           "jump": [{"p": 1, "count": 429, "total": 1024, "top": 0.9539}, {"p": 4, "count": 73, "total": 1024, "top": 0.9932}]}
    kind, text = ev.verdict_moments(res)
    assert kind == "info"
    assert text == ("Polynome vom Grad kleiner als p ergeben im Inneren Detailkoeffizienten von unter 1e-14; ab Grad p wachsen sie (bei p = 2 und Grad 2: 1,9e-05). "
                    "Bei einem Sinus mit Sprung tragen die 16 größten Koeffizienten 95,4 % der Energie bei Haar und 99,3 % bei p = 4. **Mehr verschwindende Momente: glatte Teile werden (fast) null, es bleiben die beiden Sprünge (der gesetzte und der des periodischen Randes); über p = 4 hinaus gibt es in dieser Messung keinen weiteren Gewinn.**")


def test_verdict_shift_text():
    rows = [{"name": "Impuls", "dwt": [0.433, 0.526, 0.3], "modwt": [1e-16, 2e-16, 0.0]}, {"name": "Ton f = 0,07", "dwt": [0.0002, 0.0359, 0.0], "modwt": [1e-17, 0.0, 0.0]},
            {"name": "Ton f = 0,1", "dwt": [0.0059, 0.0442, 0.0], "modwt": [0.0, 2e-16, 0.0]}]
    kind, text = ev.verdict_shift(rows)
    assert kind == "warning"
    assert text == ("Verschiebt man einen Impuls, ändert sich der Energieanteil einer Stufe bei der DWT um bis zu 0,53, bei der MODWT um 0,000. Bei einem Ton sind es bis zu 0,044 gegen 0,000. "
                    "**Die DWT ist nicht verschiebungsinvariant, die MODWT schon.**")


def test_verdict_alpha_text():
    st = []
    for a in (0.0, 0.5, 1.0):
        st += [{"alpha": a, "estimator": "MODWT", "p": 2, "mean": a - 0.01, "sd": 0.048}, {"alpha": a, "estimator": "GPH", "p": 0, "mean": a + 0.02, "sd": 0.184}]
    res = {"stationary": st, "trend": [{"p": 1, "inner_mean": 1.264}, {"p": 2, "inner_mean": 0.483}], "trend_sd": 100.0, "trend_gph": (2.0, 0.01)}
    kind, text = ev.verdict_alpha(res)
    assert kind == "info"
    assert text == ("Bei α = 0,5 schätzt die MODWT (D4) 0,49 ± 0,05, das Periodogramm 0,52 ± 0,18; die Streuung ist mit der Wavelet-Varianz etwa 4-mal kleiner als beim Periodogramm mit m = 64. Mit einem linearen Trend von 100 Standardabweichungen "
                    "bleibt die MODWT mit p = 2 bei 0,48, Haar steigt auf 1,26 und das Periodogramm auf 2,00. **Verschwindende Momente schützen den Exponenten vor dem Trend, wenn die vom periodischen Rand betroffenen Koeffizienten verworfen werden.**")


def test_verdict_bands_text():
    rows = [{"p": p, "fractions": {2: f, 3: f, 4: f}} for p, f in ((1, 0.5403), (2, 0.7135), (4, 0.8517), (8, 0.9410))]
    kind, text = ev.verdict_bands(rows)
    assert kind == "warning"
    assert text == ("Ein Ton in der Mitte des Bandes der Stufe 3 liefert nur 54 % (Haar), 71 % (D4), 85 % (D8) und 94 % (D16) der Energie in dieser Stufe. "
                    "**Die Stufen sind keine scharfen Frequenzbänder; mehr verschwindende Momente machen sie schärfer.**")


def test_verdict_variance_text():
    res = {"dwt_rel": [0.064, 0.35], "modwt_rel": [0.057, 0.236], "ratio": [1.11, 1.36, 1.49], "levels": 6, "modwt_count": 7168, "dwt_count": 1024}
    kind, text = ev.verdict_variance(res)
    assert kind == "info"
    assert text == ("Bei weißem Rauschen streut der Schätzer der Wavelet-Varianz bei der DWT auf Stufe 1 bis 6 relativ um 0,064 bis 0,350, bei der MODWT um 0,057 bis 0,236: das 1,11- bis 1,49-Fache. "
                    "Dafür hat sie 7168 statt 1024 Koeffizienten. **Die MODWT schätzt genauer, aber nicht umsonst.**")


def test_trend_hint_for_haar_starts_just_above_twenty_standard_deviations():
    base = dict(signal="Trend plus Rauschen", p=1, boundary="Randkoeffizienten verwerfen", alpha_hat=1.0)
    assert "Haar hat nur" in ev.verdict_main(make_analysis(p2=21.0, **base))[1] and "Haar hat nur" not in ev.verdict_main(make_analysis(p2=20.0, **base))[1]


def test_verdict_moments_uses_the_largest_of_the_inner_maxima():
    res = {"degrees": [{"p": 2, "max": [1e-16, 2e-16, 1.869e-05, 5.5e-05, 1.1e-04]}, {"p": 3, "max": [1e-15, 3e-15, 5e-15, 2e-7, 8e-7]}, {"p": 8, "max": [1.8e-15] * 5}],
           "jump": [{"p": 1, "count": 429, "total": 1024, "top": 0.9539}, {"p": 4, "count": 73, "total": 1024, "top": 0.9932}]}
    assert "von unter 1e-14;" in ev.verdict_moments(res)[1]
    res["degrees"][1]["max"] = [1e-15, 3e-15, 5e-13, 2e-7, 8e-7]
    assert "von höchstens 5e-13;" in ev.verdict_moments(res)[1]
