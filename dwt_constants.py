"""Konstanten: Regler-Grenzen, Signaltypen, feste Annahmen, Experimente, Farben und Presets (Kern: dwt_filters.py, dwt_transform.py, dwt_signals.py, dwt_resolution.py)."""

N_CHOICES = (256, 512, 1024, 2048, 4096)
DEFAULT_N = 2048
SIGNALS = ("Potenzgesetz-Rauschen", "Sprung im glatten Signal", "Impuls", "Ton", "Trend plus Rauschen")
DEFAULT_SIGNAL = "Potenzgesetz-Rauschen"
# Je Signaltyp bis zu drei Parameter: (Beschriftung, Minimum, Maximum, Schritt, Standardwert, Hilfetext)
PARAMS = {
    "Potenzgesetz-Rauschen": (("Exponent α", -0.5, 2.5, 0.1, 0.5, "Das Spektrum fällt wie f^−α: 0 weißes Rauschen, 1 rosa Rauschen, 2 Brownsche Bewegung."),),
    "Sprung im glatten Signal": (("Perioden des Sinus", 1.0, 8.0, 0.5, 3.0, "Zahl der Perioden über die Aufzeichnung."), ("Lage des Sprungs (Anteil der Länge)", 0.1, 0.9, 0.05, 0.3, "Zeitpunkt des Sprungs."),
                                 ("Höhe des Sprungs", 0.1, 3.0, 0.1, 1.0, "Höhe des Sprungs (der Sinus hat Amplitude 1).")),
    "Impuls": (("Lage des Impulses (Anteil der Länge)", 0.02, 0.95, 0.01, 0.05, "Zeitpunkt des Einzelimpulses."),),
    "Ton": (("Frequenz", 0.01, 0.45, 0.005, 0.07, "Frequenz des Tons (Zyklen je Abtastwert)."),),
    "Trend plus Rauschen": (("Exponent α", -0.5, 2.5, 0.1, 0.5, "Das Spektrum des Rauschens fällt wie f^−α."), ("Trend (Standardabweichungen über die Aufzeichnung)", 0.0, 400.0, 10.0, 100.0,
                                                                                                                  "Der lineare Trend durchläuft so viele Standardabweichungen des Rauschens.")),
}
NOISY_SIGNALS = ("Sprung im glatten Signal", "Impuls", "Ton")                 # nur hier gibt es das Rauschen als Regler
WAVELET_P = (1, 2, 3, 4, 6, 8)
DEFAULT_P = 2
LEVEL_MIN, DEFAULT_LEVELS = 1, 6
TRANSFORM_MODES = ("DWT", "MODWT")
DEFAULT_TRANSFORM = "MODWT"
BOUNDARY_MODES = ("periodisch", "Randkoeffizienten verwerfen")
DEFAULT_BOUNDARY = "periodisch"
SHIFT_MIN, SHIFT_MAX, DEFAULT_SHIFT = 0, 64, 0
NOISE_MODES = ("aus", "an")
DEFAULT_NOISE = "aus"
SNR_MIN, SNR_MAX, DEFAULT_SNR = -10, 40, 20                                    # Linie der Amplitude 1 (Leistung 1/2) gegen die Rauschleistung, in dB
SEED_MAX, DEFAULT_SEED = 999999, 39
OK_ALPHA = 0.15                                                                # Abweichung des Exponenten, unter der die Schätzung „trifft“ (etwa drei Standardabweichungen)
SPARSE_REL = 1e-3                                                              # Schwelle für „großen“ Koeffizienten (Bruchteil des größten)
SPARSE_TOP = 16
MAX_LEVELS = 10

# Experimente (feste Ströme)
EXP_SEED = 960000
EXP_MOM_N, EXP_MOM_P = 256, (1, 2, 3, 4, 8)
EXP_JUMP_N, EXP_JUMP_J, EXP_JUMP_P = 1024, 6, (1, 2, 4, 8)
EXP_SHIFT_N, EXP_SHIFT_J, EXP_SHIFT_P, EXP_SHIFT_COUNT = 256, 5, 2, 16
EXP_ALPHA_N, EXP_ALPHA_J, EXP_ALPHA_RUNS = 4096, 9, 200
EXP_ALPHA_VALUES = (0.0, 0.5, 1.0)
EXP_ALPHA_P = (1, 2, 4)
EXP_TREND_ALPHA, EXP_TREND_SD, EXP_TREND_P = 0.5, 100.0, (1, 2, 3, 4)
EXP_GPH_M = 64
EXP_BAND_N, EXP_BAND_P, EXP_BAND_LEVELS = 1024, (1, 2, 4, 8), (2, 3, 4)
EXP_VAR_N, EXP_VAR_J, EXP_VAR_RUNS, EXP_VAR_P = 1024, 6, 500, 2

COLOR_TRUE = "#B03A2E"
COLOR_RIDGE = "#4C9F70"
COLOR_SIGNAL = "rgba(120,120,120,0.9)"
COLOR_EST = "rgba(76,120,168,0.95)"
COLOR_AUX = "#E8833A"

PRESETS = {
    "Sprung im glatten Signal: wenige große Koeffizienten": {"n": 1024, "signal": "Sprung im glatten Signal", "params": (3.0, 0.3, 1.0), "p": 4, "levels": 6, "transform": "DWT", "boundary": "periodisch",
                                                              "shift": 0, "noise": "aus", "snr": 20, "seed": DEFAULT_SEED},
    "Impuls verschieben: die DWT ändert sich, die MODWT nicht": {"n": 256, "signal": "Impuls", "params": (0.04, 0.0, 0.0), "p": 2, "levels": 5, "transform": "DWT", "boundary": "periodisch", "shift": 1,
                                                                "noise": "aus", "snr": 20, "seed": DEFAULT_SEED},
    "Potenzgesetz-Rauschen: der Exponent aus der Wavelet-Varianz": {"n": 4096, "signal": "Potenzgesetz-Rauschen", "params": (0.5, 0.0, 0.0), "p": 2, "levels": 9, "transform": "MODWT", "boundary": "periodisch",
                                                                   "shift": 0, "noise": "aus", "snr": 20, "seed": DEFAULT_SEED},
    "Trend plus Rauschen: Haar versagt, D4 nicht": {"n": 4096, "signal": "Trend plus Rauschen", "params": (0.5, 100.0, 0.0), "p": 1, "levels": 9, "transform": "MODWT", "boundary": "Randkoeffizienten verwerfen",
                                                    "shift": 0, "noise": "aus", "snr": 20, "seed": DEFAULT_SEED},
    "Ton in der Bandmitte: die Stufen sind keine scharfen Bänder": {"n": 1024, "signal": "Ton", "params": (0.09375, 0.0, 0.0), "p": 2, "levels": 6, "transform": "MODWT", "boundary": "periodisch",
                                                                   "shift": 0, "noise": "aus", "snr": 20, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Sprung im glatten Signal: wenige große Koeffizienten": "Ein Sinus mit einem Sprung, D8 (p = 4) und sechs Stufen: Wenige Koeffizienten tragen fast die ganze Energie; der glatte Teil ergibt Detailkoeffizienten nahe null.",
    "Impuls verschieben: die DWT ändert sich, die MODWT nicht": "Ein Einzelimpuls, um einen Abtastwert verschoben: Bei der DWT ändert sich der Energieanteil der Stufen; schalten Sie auf MODWT, dann bleibt er gleich.",
    "Potenzgesetz-Rauschen: der Exponent aus der Wavelet-Varianz": "Rauschen mit Spektrum ∝ f^−0,5 (n = 4096, D4, MODWT): Die Steigung der Wavelet-Varianz gegen die Stufe plus 1 ergibt den Exponenten.",
    "Trend plus Rauschen: Haar versagt, D4 nicht": "Derselbe Exponent 0,5 plus ein linearer Trend von 100 Standardabweichungen: Mit Haar (p = 1) wird der Exponent zu hoch geschätzt (etwa 1,25); mit D4 und mehr verschwindenden Momenten bleibt er in der Nähe von 0,5.",
    "Ton in der Bandmitte: die Stufen sind keine scharfen Bänder": "Ein Ton bei 0,09375, der Mitte des Bandes der Stufe 3: Mit D4 liegen nur etwa 71 % der Energie in dieser Stufe, der Rest in den Nachbarstufen.",
}
