"""Workshop 01 – Musterlösung mit TensorBoard-Logging (Bonus).

Identisch zu loesung.py, zusätzlich wird der Trainingsverlauf über den
TensorBoard-Callback nach logs_cifar_01/ geschrieben.

Ausführen (aus dem Repository-Root):
    python 01_cnn_cifar10/loesung_tensorboard.py
    tensorboard --logdir logs_cifar_01
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard  # type: ignore
from tensorflow.keras.datasets import cifar10  # type: ignore
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore

# 1. Daten laden und vorbereiten
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Modell erstellen
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

# 3. Modell kompilieren und trainieren – mit TensorBoard-Callback
tensorboard_callback = TensorBoard(log_dir="logs_cifar_01")
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback],
)

# 4. Trainingsverlauf visualisieren
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(loc="upper left")
plt.savefig("plot.png")
plt.show()

model.save("model.h5")

# 5. Evaluierung
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
