"""Workshop 02 – CNN auf über 90 % Accuracy optimieren.

Starter-Code: das einfache CNN aus Workshop 01. Erweitere es gemäß
docs/workshop.md – die Stellen sind mit TODO markiert.

Ausführen (aus dem Repository-Root):
    python 02_cnn_optimierung/workshop.py            # trainieren
    python 02_cnn_optimierung/workshop.py --predict  # Vorhersage (noch zu implementieren)
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# TODO (6): Kommandozeilen-Parameter --predict ergänzen
parser = argparse.ArgumentParser(description="CIFAR-10 Training / Vorhersage")
args = parser.parse_args()

# Konstanten
EPOCHS = 10        # TODO (5): bis zu 50 Epochen
BATCH_SIZE = 32    # TODO (5): Batch-Größe 64
LEARNING_RATE = 0.001  # TODO (3): initiale Lernrate 0.0005
MODEL_PATH = "cifar10_model.h5"
CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ---------------------------------------------------------------------------
# 1. Daten laden und vorbereiten
# ---------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# ---------------------------------------------------------------------------
# 2. Modell erstellen
# ---------------------------------------------------------------------------
# TODO (2): tieferes Netz – mehr Conv-Blöcke, BatchNormalization nach jedem
#           Conv2D, Dropout gegen Overfitting
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)

# ---------------------------------------------------------------------------
# 3. Modell kompilieren
# ---------------------------------------------------------------------------
# TODO (3): Adam mit learning_rate=LEARNING_RATE verwenden
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ---------------------------------------------------------------------------
# 4. Callbacks
# ---------------------------------------------------------------------------
# TODO (4): ModelCheckpoint (bestes Modell nach val_accuracy speichern)
# TODO (4): EarlyStopping
# TODO (4): LearningRateScheduler – Lernrate alle N Epochen halbieren
callbacks = []

# ---------------------------------------------------------------------------
# 5. Trainieren und Trainingsverlauf visualisieren
# ---------------------------------------------------------------------------
model.summary()
history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
)

# TODO (5): accuracy / val_accuracy plotten und als plot.png speichern

# Modell evaluieren
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc}")

# ---------------------------------------------------------------------------
# 6. Vorhersagemodus
# ---------------------------------------------------------------------------
# TODO (6): bei --predict das gespeicherte Modell laden, ein zufälliges
#           Testbild vorhersagen und mit vorhergesagtem Label anzeigen
