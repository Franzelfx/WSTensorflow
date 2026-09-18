"""Workshop 03 – LSTM zur Zeitreihen-Vorhersage auf Forex-Daten.

Starter-Code mit TODOs. Aufgabenblatt: docs/workshop.md

Ausführen (aus dem Repository-Root):
    python 03_lstm_forex/workshop.py data/AUDCAD_15.csv            # trainieren
    python 03_lstm_forex/workshop.py data/AUDCAD_15.csv --predict  # vorhersagen
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

INPUT_TIMESTEPS = 64   # Zeitschritte als Eingabe
OUTPUT_TIMESTEPS = 32  # Zeitschritte, die vorhergesagt werden
MODEL_PATH = "best_model.h5"


# ---------------------------------------------------------------------------
# 1. Daten einlesen und normalisieren
# ---------------------------------------------------------------------------
def load_data(file_path):
    """Liest die Marktdaten aus der CSV-Datei ein."""
    # TODO: CSV mit Pandas einlesen und DataFrame zurückgeben
    raise NotImplementedError


def normalize_data(data):
    """Normalisiert jede numerische Spalte individuell auf den Bereich 0–1.

    Gibt das normalisierte Array und den Scaler zurück (für spätere Rücktransformation).
    """
    # TODO: nur numerische Spalten auswählen (die Spalte 't' ist ein Datum)
    # TODO: MinMaxScaler anwenden
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2./3. Sequenzen bilden und aufteilen
# ---------------------------------------------------------------------------
def prepare_data(data, input_timesteps, output_timesteps):
    """Erzeugt Eingabe-Sequenzen (samples, input_timesteps, features)
    und Ziel-Sequenzen (samples, output_timesteps, features)."""
    # TODO: mit einem gleitenden Fenster über die Daten laufen
    raise NotImplementedError


def split_data(samples, labels, test_size=0.2):
    """Teilt Samples und Labels in Trainings- und Testdaten auf."""
    # TODO: train_test_split verwenden
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4. Neuronales Netz erstellen
# ---------------------------------------------------------------------------
def build_model(input_timesteps, num_features, output_timesteps):
    """Baut und kompiliert das LSTM-Modell."""
    model = Sequential()
    # TODO: LSTM-Schichten (ggf. mit Dropout) hinzufügen
    # TODO: Ausgabeschicht mit output_timesteps * num_features Neuronen
    # TODO: model.compile(optimizer='adam', loss='mse')
    return model


# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------
def train_model(model, X_train, y_train, epochs=50, batch_size=64):
    """Trainiert das Modell mit EarlyStopping und ModelCheckpoint."""
    # TODO: Callbacks anlegen (EarlyStopping auf val_loss, ModelCheckpoint -> MODEL_PATH)
    # TODO: model.fit(...) mit validation_split aufrufen und history zurückgeben
    raise NotImplementedError


def plot_training_history(history):
    """Plottet loss und val_loss über die Epochen."""
    # TODO: history.history['loss'] und ['val_loss'] plotten
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 6. Vorhersagen und Visualisierung
# ---------------------------------------------------------------------------
def make_predictions(model, X_test):
    """Erzeugt Vorhersagen für die Testdaten."""
    # TODO: model.predict(...)
    raise NotImplementedError


def plot_predictions(y_test, y_pred):
    """Plottet echte Werte gegen Vorhersagen."""
    # TODO: y_test und y_pred (z. B. nur die Close-Spalte) gegeneinander plotten
    raise NotImplementedError


def main(file_path, predict):
    data = load_data(file_path)
    data, scaler = normalize_data(data)

    samples, labels = prepare_data(data, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS)
    # Labels zu einem flachen Vektor je Sample umformen: (samples, 32 * features)
    labels = labels.reshape(labels.shape[0], OUTPUT_TIMESTEPS * samples.shape[2])

    X_train, X_test, y_train, y_test = split_data(samples, labels)
    num_features = X_train.shape[2]

    if predict:
        # TODO: gespeichertes Modell laden, Vorhersagen machen und plotten
        raise NotImplementedError
    else:
        model = build_model(INPUT_TIMESTEPS, num_features, OUTPUT_TIMESTEPS)
        history = train_model(model, X_train, y_train)
        plot_training_history(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LSTM-Training und -Vorhersage auf Forex-Marktdaten."
    )
    parser.add_argument("file_path", help="Pfad zur CSV-Datei mit den Marktdaten")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Gespeichertes Modell laden und Vorhersagen plotten",
    )
    args = parser.parse_args()
    main(args.file_path, args.predict)
