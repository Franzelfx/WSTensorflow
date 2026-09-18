# WSTensorflow – TensorFlow / Keras Workshops

> ✅ **Du befindest dich auf dem Branch `loesung`** – hier liegen zusätzlich die
> Musterlösungen (`loesung.py`) und Lösungsbesprechungen (`docs/loesung.md`)
> je Workshop. Die zu bearbeitende Aufgabe liegt auf dem Branch `main`
> (`git checkout main`).

Workshop-Material zu TensorFlow / Keras: Aufgabenstellungen, Starter-Code,
Musterlösungen und Beispieldaten. Jeder Workshop ist eine in sich geschlossene
Einheit: klonen, Aufgabenblatt durcharbeiten, `workshop.py` vervollständigen,
trainieren, experimentieren.

**Repository:** <https://github.com/Franzelfx/WSTensorflow>

## Branches

| Branch    | Inhalt |
|-----------|--------|
| `main`    | **Die Aufgabe** – Aufgabenblätter und Starter-Code mit `TODO`-Markierungen |
| `loesung` | **Die Lösung** – zusätzlich `loesung.py` und `docs/loesung.md` je Workshop |

```bash
# Aufgabe auschecken (Standard)
git clone git@github.com:Franzelfx/WSTensorflow.git
cd WSTensorflow

# Lösung ansehen
git checkout loesung
```

> Tipp: Erst selbst versuchen, dann erst in die `loesung` schauen.

## Workshops

| Workshop | Thema | Datensatz |
|----------|-------|-----------|
| [01_cnn_cifar10](01_cnn_cifar10/) | Erstes CNN zur Bilderkennung (Conv2D, Pooling, Dense), Trainingsverlauf plotten | CIFAR-10 (über Keras) |
| [02_cnn_optimierung](02_cnn_optimierung/) | CNN auf > 90 % Accuracy optimieren: BatchNormalization, Dropout, Callbacks, LR-Scheduler | CIFAR-10 (über Keras) |
| [03_lstm_forex](03_lstm_forex/) | LSTM zur Zeitreihen-Vorhersage auf Forex-Marktdaten (64 Zeitschritte rein, 32 raus) | [data/AUDCAD_15.csv](data/AUDCAD_15.csv) |

Der Schwierigkeitsgrad steigt: `01` ist der Einstieg mit einem minimalen CNN,
`02` baut darauf auf und ergänzt Regularisierung, Callbacks und einen
Vorhersagemodus, `03` wechselt zu Sequenzdaten und LSTMs.

## Aufbau des Repositories

```text
WSTensorflow/
├── requirements.txt
├── data/
│   └── AUDCAD_15.csv        # Marktdaten für Workshop 03
├── extras/                  # ergänzende Skripte außerhalb der Workshops
├── 01_cnn_cifar10/
├── 02_cnn_optimierung/
└── 03_lstm_forex/
    ├── README.md            # Kurzüberblick, Ausführen, Dateien
    ├── workshop.py          # Starter-Code mit TODOs
    ├── loesung.py           # Musterlösung        (nur Branch loesung)
    └── docs/
        ├── workshop.md      # Aufgabenblatt
        ├── workshop.pdf     # Aufgabenblatt als PDF
        ├── loesung.md       # Besprechung der Lösung (nur Branch loesung)
        └── loesung.pdf      # Besprechung als PDF    (nur Branch loesung)
```

Alle drei Workshop-Ordner sind gleich aufgebaut.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ausführen

Alle Kommandos werden **aus dem Repository-Root** ausgeführt:

```bash
# Workshop 01 – einfaches CNN
python 01_cnn_cifar10/workshop.py

# Workshop 02 – optimiertes CNN
python 02_cnn_optimierung/workshop.py            # trainieren
python 02_cnn_optimierung/workshop.py --predict  # Vorhersage für ein zufälliges Testbild

# Workshop 03 – LSTM auf Forex-Daten
python 03_lstm_forex/workshop.py data/AUDCAD_15.csv
python 03_lstm_forex/workshop.py data/AUDCAD_15.csv --predict
```

Auf dem Branch `loesung` laufen die Musterlösungen analog mit `loesung.py`
statt `workshop.py`.

Die Skripte schreiben Modelle (`*.h5`), zwischengespeicherte Datensätze (`*.npy`),
TensorBoard-Logs (`logs*/`) und Plots (`plot.png`) in das aktuelle
Arbeitsverzeichnis. Diese Artefakte sind über [.gitignore](.gitignore) vom
Repository ausgeschlossen und werden beim Training neu erzeugt.

TensorBoard starten (Workshop 01, Variante `loesung_tensorboard.py`):

```bash
tensorboard --logdir logs_cifar_01
```

## Konventionen

- Jeder Workshop hat ein Aufgabenblatt in `docs/workshop.md` (auch als PDF)
  und einen Starter `workshop.py`, in dem die zu bearbeitenden Stellen mit
  `TODO` markiert sind.
- Die Musterlösung liegt als `loesung.py` mit Besprechung in `docs/loesung.md`
  auf dem Branch `loesung`. Der Branch enthält `main` vollständig; Änderungen
  an der Aufgabe werden mit `git merge main` nachgezogen.
- PDFs werden aus den Markdown-Dateien erzeugt:
  `pandoc docs/workshop.md -o docs/workshop.pdf --pdf-engine=xelatex`

## Extras

- [extras/cifar10_schrittweise.py](extras/cifar10_schrittweise.py) – kommentierte
  Schritt-für-Schritt-Fassung des CIFAR-10-Trainings für die Live-Demo.
- [extras/regression_tree.py](extras/regression_tree.py) – Regressionsbaum von Hand
  (MSE, Splits) im Vergleich zu `sklearn.tree.DecisionTreeRegressor`.
