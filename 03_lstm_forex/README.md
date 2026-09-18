# Workshop 03 – LSTM zur Zeitreihen-Vorhersage auf Forex-Daten

Ein LSTM-Netz lernt aus 64 Zeitschritten AUD/CAD-Marktdaten die nächsten
32 Zeitschritte vorherzusagen. Das Skript liest die CSV ein, normalisiert,
bildet Sequenzen, trainiert mit EarlyStopping/ModelCheckpoint und plottet
Trainingsverlauf und Vorhersagen.

## Ausführen

Aus dem Repository-Root, mit aktivierter virtueller Umgebung (siehe
[Setup](../README.md#setup)):

```bash
python 03_lstm_forex/workshop.py data/AUDCAD_15.csv             # trainieren
python 03_lstm_forex/workshop.py data/AUDCAD_15.csv --predict   # vorhersagen
```

Auf dem Branch `loesung` zusätzlich:

```bash
python 03_lstm_forex/loesung.py data/AUDCAD_15.csv
python 03_lstm_forex/loesung.py data/AUDCAD_15.csv --predict
tensorboard --logdir logs
```

## Dateien

| Datei | Inhalt |
|-------|--------|
| [docs/workshop.md](docs/workshop.md) | Das Aufgabenblatt |
| [docs/workshop.pdf](docs/workshop.pdf) | Aufgabenblatt als PDF |
| [workshop.py](workshop.py) | Starter-Code (Funktionsgerüst) mit `TODO`-Markierungen |
| [../data/AUDCAD_15.csv](../data/AUDCAD_15.csv) | Marktdaten AUD/CAD, 15-Minuten-Kerzen |
| `loesung.py` | Musterlösung mit `--predict`-Modus – nur auf Branch `loesung` |
| `docs/loesung.md` | Besprechung der Lösung – nur auf Branch `loesung` |

## Aufgaben in Kurzform

Details im [Aufgabenblatt](docs/workshop.md).

1. CSV einlesen und jede Spalte normalisieren
2. Trainings-/Testdaten aufteilen
3. Sequenzen `(samples, timesteps, features)` mit 64 rein / 32 raus bilden
4. LSTM-Architektur mit Ausgabe für 32 Zeitschritte
5. Training mit EarlyStopping und ModelCheckpoint, Historie plotten
6. `--predict`: Vorhersagen gegen echte Werte plotten

Erzeugte Artefakte (`best_model.h5`, `logs/`) landen im aktuellen
Arbeitsverzeichnis und sind per `.gitignore` ausgeschlossen.
