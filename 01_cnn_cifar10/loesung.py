"""Workshop 01 – Musterlösung: Erstes CNN zur Bilderkennung auf CIFAR-10.

Ausführen (aus dem Repository-Root):
    python 01_cnn_cifar10/loesung.py
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10  # type: ignore
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore

# ---------------------------------------------------------------------------
# 1. Daten laden und vorbereiten
# ---------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Pixelwerte auf 0–1

# ---------------------------------------------------------------------------
# 2. Modell erstellen
# ---------------------------------------------------------------------------
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),  # -> 30x30x32
        MaxPooling2D(2, 2),                                               # -> 15x15x32
        Conv2D(64, (3, 3), activation="relu"),                            # -> 13x13x64
        MaxPooling2D(2, 2),                                               # -> 6x6x64
        Flatten(),                                                        # -> 2304
        Dense(64, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)
model.summary()

# ---------------------------------------------------------------------------
# 3. Modell kompilieren und trainieren
# ---------------------------------------------------------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# ---------------------------------------------------------------------------
# 4. Trainingsverlauf visualisieren
# ---------------------------------------------------------------------------
fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(10, 4))
ax_acc.plot(history.history["accuracy"], label="train")
ax_acc.plot(history.history["val_accuracy"], label="validation")
ax_acc.set_title("Accuracy")
ax_acc.set_xlabel("Epoch")
ax_acc.legend()
ax_loss.plot(history.history["loss"], label="train")
ax_loss.plot(history.history["val_loss"], label="validation")
ax_loss.set_title("Loss")
ax_loss.set_xlabel("Epoch")
ax_loss.legend()
fig.tight_layout()
fig.savefig("plot.png")  # vor plt.show() speichern, sonst ist die Grafik leer
plt.show()

# Modell speichern
model.save("model.h5")

# ---------------------------------------------------------------------------
# 5. Evaluierung
# ---------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
