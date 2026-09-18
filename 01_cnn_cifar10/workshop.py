"""Workshop 01 – Erstes CNN zur Bilderkennung auf CIFAR-10.

Starter-Code mit TODOs. Aufgabenblatt: docs/workshop.md

Ausführen (aus dem Repository-Root):
    python 01_cnn_cifar10/workshop.py
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

# ---------------------------------------------------------------------------
# 1. Daten laden und vorbereiten
# ---------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# TODO: Pixelwerte (0–255) auf den Bereich 0–1 normalisieren

# ---------------------------------------------------------------------------
# 2. Modell erstellen
# ---------------------------------------------------------------------------
model = Sequential(
    [
        # TODO: mindestens zwei Conv2D-Layer, jeweils gefolgt von MaxPooling2D
        #       (Eingabeform der Bilder: (32, 32, 3))
        # TODO: Flatten + mindestens zwei Dense-Layer
        #       (letzter Layer: 10 Klassen, activation='softmax')
    ]
)

# ---------------------------------------------------------------------------
# 3. Modell kompilieren und trainieren
# ---------------------------------------------------------------------------
# TODO: model.compile(...) mit Optimizer, Loss und Metrik 'accuracy'

# TODO: history = model.fit(...) mit validation_data=(x_test, y_test)

# ---------------------------------------------------------------------------
# 4. Trainingsverlauf visualisieren
# ---------------------------------------------------------------------------
# TODO: accuracy / val_accuracy und loss / val_loss aus history.history plotten
# TODO: Plot als plot.png speichern (plt.savefig VOR plt.show aufrufen)

# ---------------------------------------------------------------------------
# 5. Evaluierung
# ---------------------------------------------------------------------------
# TODO: model.evaluate(...) auf den Testdaten und Test-Accuracy ausgeben
