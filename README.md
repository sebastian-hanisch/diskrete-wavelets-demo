# Diskrete Wavelets und Mehrskalenanalyse: exakt zerlegt, aber verschiebungsvariant – und die Wavelet-Varianz als Skalenspektrum – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-diskrete-wavelets-demo.streamlit.app/)** -->

Zwölfte Demo der Reihe **Spektral- und Wavelet-Analyse** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, aufbauend auf der Demo zum **kontinuierlichen
Skalogramm**. Fünf Testsignale mit exakter Wahrheit (Potenzgesetz-Rauschen, Sprung im glatten Signal, Impuls, Ton, Trend plus Rauschen) werden mit der **diskreten Wavelet-Transformation (DWT)** und der überlappenden
**MODWT** zerlegt (Daubechies-Filter mit 1 bis 8 verschwindenden Momenten). Gemessen werden verschwindende Momente und Sparsität, die Verschiebungsvarianz, die Schätzung des Exponenten von Potenzgesetz-Rauschen aus der
**Wavelet-Varianz**, die Frequenzselektivität der Stufen und die Streuung des Varianzschätzers. Die Demo liefert die Koeffizienten, auf denen die Wavelet-Schrumpfung danach Schwellwerte setzt.

## Warum dieses Problem

Das kontinuierliche Skalogramm ist redundant und stark korreliert. Die DWT wählt nur Zweierpotenz-Skalen und ein Raster und zerlegt eine Reihe der Länge 2^J exakt und ohne Redundanz; Filter mit p verschwindenden Momenten
machen Polynome vom Grad kleiner als p zu null und glatte Teile nahezu null. Der Preis ist die Verschiebungsvarianz: Schiebt man die Reihe, ändern sich die Koeffizienten. Die MODWT lässt das Halbieren weg und ist verschiebungsinvariant. Die Wavelet-Varianz (mittlere
Energie je Stufe) ist das Skalenspektrum: Bei Potenzgesetz-Rauschen liest man den Exponenten an ihrer Steigung ab, auch bei einem Trend, wenn das Wavelet genug verschwindende Momente hat und die vom periodischen Rand betroffenen Koeffizienten verworfen werden.

## Modell

Testsignale auf den Zeitpunkten t = 0 … n − 1 (Abtastrate 1): zirkulär stationäres Gauß-Rauschen mit Spektrum ∝ f^−α (Spektralsynthese), ein Sinus mit Sprung, ein Einzelimpuls, ein Ton, Potenzgesetz-Rauschen plus ein
linearer Trend. Daubechies-Skalierungsfilter h der Länge 2p per Spektralfaktorisierung, Wavelet-Filter g[l] = (−1)^l h[2p − 1 − l]. DWT als periodische Pyramide (c[k] = Σ h[l] a[2k + l]), MODWT per FFT (Filter h/√2,
jede Länge), Mehrskalenanalyse x = S_J + Σ D_j, Wavelet-Varianz als Mittel von W_j² (MODWT, auf Wunsch ohne die vom periodischen Rand betroffenen Koeffizienten) bzw. von d_j²/2^j (DWT). Der Exponent ist die Steigung von
log₂ ν̂² gegen die Stufe plus 1, angepasst über die Stufen 2 bis J, solange n/2^j mindestens 32 ist. Alles selbst in numpy; die Gegenproben in den Tests sind Summendefinition, Matrixform und geschlossene Formeln (scipy wird nicht gebraucht).

## Methodik

n = 256 bis 4096, p = 1, 2, 3, 4, 6, 8 (Haar bis D16). Experimente über feste Zufallsströme (Seed 960000): 200 Läufe bei der Exponentenschätzung, 500 bei der Streuung; die übrigen Experimente sind ohne Zufall.
Das Rauschen der App hat die Varianz 0,5/10^(SNR/10), also gegen eine Linie der Amplitude 1. Die Exponentenschätzung gilt für zirkulär erzeugte Daten und stationäre Spektren (α < 2p + 1).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` und `tests/test_experiments_golden.py`.

| Frage | Befund |
|---|---|
| Wie genau ist die Zerlegung? | Rekonstruktion und Energieerhaltung gelten für DWT und MODWT auf 1e-12 und besser (n = 64 bis 512, 1 bis 9 Stufen, p = 1, 2, 4, 8), die MODWT auch bei n = 300. Mini-Instanz Haar, x = (1, 2, 3, 4): d = −1/√2 je Wert, a = (3/√2, 7/√2), zweite Stufe d = −2, a = 5. |
| Was bringen verschwindende Momente (n = 256, Stufe 1, innere Detailkoeffizienten)? | Polynome vom Grad kleiner als p ergeben unter 1e-14; ab dem Grad p wachsen sie: p = 1: 2,8e-3 / 5,4e-3 / 8,1e-3 / 1,1e-2 bei Grad 1 bis 4; p = 2: 1,9e-5 / 5,5e-5 / 1,1e-4 bei Grad 2 bis 4; p = 3: 2,0e-7 / 7,8e-7 bei Grad 3 und 4; p = 4: 2,9e-9 bei Grad 4. |
| Wie sparsam ist ein Sinus mit Sprung (n = 1024, 6 Stufen)? | Koeffizienten über dem 1e-3-Fachen des größten bei p = 1 / 2 / 4 / 8: 429 / 127 / 73 / 91 von 1024; die 16 größten tragen 95,4 / 98,7 / 99,3 / 99,3 % der Energie; es bleiben die beiden Sprünge (der gesetzte und der des periodischen Randes), über p = 4 hinaus gibt es hier keinen weiteren Gewinn. |
| Wie verschiebungsvariant ist die DWT (n = 256, 5 Stufen, D4, 16 Verschiebungen)? | Der Energieanteil einer Stufe ändert sich bei einem Impuls um bis zu 0,53 (Stufe 1 bis 5 und Glättung: 0,433 / 0,526 / 0,301 / 0,174 / 0,091 / 0,025), bei einem Ton der Frequenz 0,07 um bis zu 0,036 und bei 0,1 um bis zu 0,044. Bei der MODWT bleiben alle Anteile gleich (Änderung unter 1e-12). |
| Wie gut ist der Exponent aus der Wavelet-Varianz (n = 4096, 9 Stufen, 200 Läufe)? | α̂ bei α = 0 / 0,5 / 1 mit MODWT D4: 0,002 ± 0,051 / 0,492 ± 0,048 / 0,995 ± 0,049; DWT D4: −0,002 ± 0,063 / 0,490 ± 0,058 / 0,996 ± 0,062; Periodogramm (GPH, m = 64): −0,008 ± 0,177 / 0,518 ± 0,184 / 1,026 ± 0,181. Die Streuung ist mit der Wavelet-Varianz etwa 4-mal kleiner als beim Periodogramm mit m = 64; bei viel größerem m (ab etwa 1000) und exaktem Potenzgesetz ist die Streuung des Periodogramms kleiner als die der Wavelet-Varianz. |
| Was macht ein linearer Trend von 100 Standardabweichungen (α = 0,5)? | MODWT ohne die vom Rand betroffenen Koeffizienten: Haar 1,264 ± 0,014, p = 2 / 3 / 4: 0,483 / 0,483 / 0,482 (± 0,052 / 0,055 / 0,057); ohne Trend 0,484 / 0,496 / 0,498 bei p = 1 / 2 / 3. Die periodische DWT sieht den Sprung am Rand: 1,263 / 1,910 / 1,908 / 1,872 bei p = 1 bis 4; das Periodogramm: 2,000 ± 0,010. |
| Sind die Stufen scharfe Frequenzbänder (MODWT, Ton in der Bandmitte)? | Nein: Der Anteil der Energie in der Stufe 2 / 3 / 4 beträgt bei Haar 59,0 / 54,0 / 52,9 %, bei D4 72,8 / 71,3 / 71,3 %, bei D8 85,3 / 85,2 / 85,2 %, bei D16 94,1 % in allen drei. An den Bandrändern (f = 1/16 und 1/8) beträgt der Leistungsgang der Stufe 3 0,41 / 0,43 (Haar), 0,47 / 0,47 (D4), 0,49 / 0,49 (D8) und 0,50 (D16). |
| Wie stark streut der Schätzer der Wavelet-Varianz (weißes Rauschen, n = 1024, 500 Läufe, D4)? | Relative Standardabweichung auf Stufe 1 bis 6: DWT 0,064 / 0,087 / 0,128 / 0,175 / 0,229 / 0,350, MODWT 0,057 / 0,064 / 0,089 / 0,125 / 0,166 / 0,236, das 1,11- bis 1,49-Fache; beide Schätzer sind im Mittel unverzerrt (gemessene Mittel 0,500 / 0,249 / 0,125 / 0,062 / 0,031 / 0,016 gegen 2^−j). Die MODWT hat 7168 statt 1024 Koeffizienten. |

## Ehrliche Grenzen

- **Heimvorteil:** Zirkulär stationäre Gauß-Prozesse mit exaktem Potenzgesetz; echte Reihen sind nur näherungsweise Potenzgesetze und nicht zirkulär.
- **Randbehandlung:** Periodisch ist für Reihen mit starkem Trend verfälscht (1,3 bis 1,9 statt 0,5 bei 100 Standardabweichungen); das Verwerfen der Randkoeffizienten kostet Daten, und bei der DWT gibt es die Option nicht.
- **Gültigkeitsbereich der Wavelet-Varianz:** Die Steigung entspricht α − 1 nur für Spektren mit α < 2p + 1; im Regelbereich der App (α bis 2,5) ist das für jedes p erfüllt (Haar: α < 3), darüber ist nichts gemessen.
- **Gemessen für n = 4096 und 9 Stufen:** Bias und Streuung hängen an der Länge und der Stufenzahl; keine allgemeine Formel, nur die gemessenen Werte. Bei kurzen Reihen bleiben weniger als drei Stufen für die Anpassung und die App meldet das.
- **Die Stufen sind keine scharfen Bänder** (siehe oben); die Wavelet-Pakete, die das verbessern würden, sind nicht gebaut.
- **Nur Daubechies:** Symlets, Coiflets und andere Familien sind nicht gebaut; keine Schrumpfung (Schwellwerte), keine 2D-Transformation, keine Konfidenzintervalle der Wavelet-Varianz.
- **Quellen** (Daubechies 1988; Mallat 1989; Percival und Walden 2000; Abry und Veitch 1998): nicht gelesen; Formeln per Simulation, Handrechnung und gegen geschlossene Formeln geprüft.

## Verifikation

- **Kern gegen Handrechnung:** Haar, x = (1, 2, 3, 4): siehe oben; D4-Filter gegen die geschlossene Formel (1 + √3, 3 + √3, 3 − √3, 1 − √3)/(4√2), db3 und db4 gegen die Tabellen.
- **Orakel (`tests/test_oracle_*.py`):** Filterbedingungen (Σh = √2, Σh² = 1, Orthogonalität zur 2er-Verschiebung, verschwindende Momente), DWT gegen die Summendefinition und als orthogonale Matrix, MODWT gegen die
  Zeitbereichs-Pyramide, exakte Verschiebungsinvarianz der MODWT, Randlänge, Mehrskalenanalyse additiv, Wavelet-Varianz weißen Rauschens 2^−j, Spektrum der Spektralsynthese, Exponent gegen den wahren Wert.
- **Generator** (`tests/test_generator.py`): wortgleich aus der ersten Demo.
- **Zahlen:** jede Zahl dieser README in `tests/test_claims.py`; Festwerte der Experimente in `tests/test_experiments_golden.py`; Verdikte in `tests/test_verdicts.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen), Auswahlfelder, Permalink, Stufen je n, bedingte Regler, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `signal_generator.py` | Gemeinsamer Generator der Reihe (wortgleich; hier nur der Zufallsstrom SplitMix64) |
| `dwt_filters.py` | Daubechies-Filter per Spektralfaktorisierung, Quadraturspiegelfilter |
| `dwt_transform.py` | DWT, MODWT, Mehrskalenanalyse, Wavelet-Varianz, Randlänge, Anpassung, GPH |
| `dwt_signals.py` | die fünf Testsignale mit exakter Wahrheit |
| `dwt_resolution.py` | Energieanteile, Sparsität, Exponentenschätzung, Bandlage, Verschiebungsempfindlichkeit |
| `dwt_constants.py` | Regler-Grenzen, Signaltypen, feste Annahmen, Experimente, Presets |
| `dwt_evaluation.py` | Analyse, Experimente, Verdikte |
| `dwt_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `dwt_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `dwt_pdf_export.py` | PDF der aktuellen Einstellung (erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (von den Tests derzeit nicht gebraucht) |
| `tests/` | Kern, Orakel, Signale, Zahlen, Verdikte, Presets, Randfälle, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Wavelet-Schrumpfung (Schwellwerte, Cycle Spinning):** Thema der Demo danach.
- **Wavelet-Pakete, 2D-Wavelets, Symlets und Coiflets:** nicht enthalten.
- **Konfidenzintervalle der Wavelet-Varianz:** nur die Streuung ist gemessen.
- **Gemessene Daten:** alles ist simuliert, damit die Wahrheit bekannt ist.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Python 3.12 oder neuer. Die Experimente brauchen beim ersten Aufruf wenige Sekunden (danach zwischengespeichert).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
