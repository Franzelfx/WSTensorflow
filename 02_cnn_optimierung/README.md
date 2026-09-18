# Workshop 02 – CNN auf über 90 % Accuracy optimieren

Aufbauend auf Workshop 01: das einfache CNN wird durch eine tiefere
Architektur, BatchNormalization, Dropout, Callbacks (ModelCheckpoint,
EarlyStopping, LearningRateScheduler) und einen Vorhersagemodus so erweitert,
dass es auf CIFAR-10 über 90 % Accuracy erreicht.

## Ausführen

Aus dem Repository-Root, mit aktivierter virtueller Umgebung (siehe
[Setup](../README.md#setup)):

```bash
python 02_cnn_optimierung/workshop.py             # trainieren
python 02_cnn_optimierung/workshop.py --predict   # Vorhersage (nach Aufgabe 6)
```

Auf dem Branch `loesung` zusätzlich:

```bash
python 02_cnn_optimierung/loesung.py
python 02_cnn_optimierung/loesung.py --predict
```

## Dateien

| Datei | Inhalt |
|-------|--------|
| [docs/workshop.md](docs/workshop.md) | Das Aufgabenblatt |
| [docs/workshop.pdf](docs/workshop.pdf) | Aufgabenblatt als PDF |
| [workshop.py](workshop.py) | Starter-Code (das CNN aus Workshop 01) mit `TODO`-Markierungen |
| `loesung.py` | Musterlösung mit `--predict`-Modus – nur auf Branch `loesung` |
| `docs/loesung.md` | Besprechung der Lösung – nur auf Branch `loesung` |

## Aufgaben in Kurzform

Details im [Aufgabenblatt](docs/workshop.md).

1. Daten laden und normalisieren
2. Tieferes CNN mit BatchNormalization und Dropout
3. Adam mit Lernrate 0.0005
4. Callbacks: ModelCheckpoint, EarlyStopping, LearningRateScheduler
5. Bis zu 50 Epochen, Batch-Größe 64, Trainingsverlauf plotten
6. `--predict`: zufälliges Testbild mit vorhergesagtem Label anzeigen

Erzeugte Artefakte (`*.h5`, `*.npy`, `plot.png`) landen im aktuellen
Arbeitsverzeichnis und sind per `.gitignore` ausgeschlossen.
