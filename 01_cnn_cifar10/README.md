# Workshop 01 – Erstes CNN zur Bilderkennung

Der Einstieg: ein kleines Convolutional Neural Network (Conv2D, Pooling, Dense)
auf CIFAR-10 trainieren, den Trainingsverlauf plotten und auf den Testdaten
evaluieren.

## Ausführen

Aus dem Repository-Root, mit aktivierter virtueller Umgebung (siehe
[Setup](../README.md#setup)):

```bash
python 01_cnn_cifar10/workshop.py
```

Auf dem Branch `loesung` zusätzlich:

```bash
python 01_cnn_cifar10/loesung.py
python 01_cnn_cifar10/loesung_tensorboard.py   # Variante mit TensorBoard-Logging
tensorboard --logdir logs_cifar_01
```

## Dateien

| Datei | Inhalt |
|-------|--------|
| [docs/workshop.md](docs/workshop.md) | Das Aufgabenblatt |
| [docs/workshop.pdf](docs/workshop.pdf) | Aufgabenblatt als PDF |
| [workshop.py](workshop.py) | Starter-Code mit `TODO`-Markierungen |
| `loesung.py` | Musterlösung – nur auf Branch `loesung` |
| `loesung_tensorboard.py` | Musterlösung mit TensorBoard-Callback – nur auf Branch `loesung` |
| `docs/loesung.md` | Besprechung der Lösung – nur auf Branch `loesung` |

## Aufgaben in Kurzform

Details im [Aufgabenblatt](docs/workshop.md).

1. CIFAR-10 laden und normalisieren
2. CNN mit mindestens zwei Conv2D-, zwei Pooling- und zwei Dense-Layern bauen
3. Kompilieren und trainieren
4. Accuracy und Loss über die Epochen plotten
5. Auf den Testdaten evaluieren

Erzeugte Artefakte (`model.h5`, `plot.png`, `logs_*/`) landen im aktuellen
Arbeitsverzeichnis und sind per `.gitignore` ausgeschlossen.
