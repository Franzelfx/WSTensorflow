"""Workshop 02 – Musterlösung: CNN auf über 90 % Accuracy optimieren.

Ausführen (aus dem Repository-Root):
    python 02_cnn_optimierung/loesung.py            # trainieren
    python 02_cnn_optimierung/loesung.py --predict  # Vorhersage für ein zufälliges Testbild
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import argparse
import random

# Argument parsing
parser = argparse.ArgumentParser(description="CIFAR-10 Training/Prediction Script")
parser.add_argument(
    "--predict", action="store_true", help="Make a prediction on a random test image"
)
args = parser.parse_args()

# Constants
X_TRAIN_PATH = "x_train.npy"
Y_TRAIN_PATH = "y_train.npy"
X_TEST_PATH = "x_test.npy"
Y_TEST_PATH = "y_test.npy"
MODEL_SAVE_PATH = "cifar10_model.h5"
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EARLY_STOPPING_PATIENCE = 25
DROPOUT = 0.25
CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
STEP_DROP = 5  # Halve the learning rate every 5 epochs


# Learning Rate Scheduler Function
def lr_schedule(epoch, lr):
    """Halve the learning rate every STEP_DROP epochs."""
    if epoch % STEP_DROP == 0 and epoch:
        return lr * 0.5
    return lr

def save_dataset(x_train, y_train, x_test, y_test):
    """Save the dataset to disk.

    Args:
        x_train (_type_): X_train dataset
        y_train (_type_): Y_train dataset
        x_test (_type_): X_test dataset
        y_test (_type_): Y_test dataset
    """
    np.save(X_TRAIN_PATH, x_train)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(X_TEST_PATH, x_test)
    np.save(Y_TEST_PATH, y_test)


def load_dataset():
    x_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    x_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    return (x_train, y_train), (x_test, y_test)

# Data augmentation by rotating the images
def rotate_image(image):
    return tf.image.rot90(image, k=random.randint(0, 3))

# Data augmentation by flipping the images
def flip_image(image):
    return tf.image.random_flip_left_right(image)


if not args.predict:
    # This block will run if the script is not called with the --predict flag
    cifar10 = tf.keras.datasets.cifar10.load_data() # -> Load something different here
    (x_train, y_train), (x_test, y_test) = cifar10
    
    # Data augmentation
    x_train = np.concatenate([x_train, np.array([rotate_image(image) for image in x_train])])
    x_train = np.concatenate([x_train, np.array([flip_image(image) for image in x_train])])

    # Save the data
    if not (
        os.path.exists(X_TRAIN_PATH)
        and os.path.exists(Y_TRAIN_PATH)
        and os.path.exists(X_TEST_PATH)
        and os.path.exists(Y_TEST_PATH)
    ):
        save_dataset(x_train, y_train, x_test, y_test)

    # Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation="relu", input_shape=(32, 32, 3)), # -> Adjust the input shape to new data
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(DROPOUT),
        
        Conv2D(64, (3, 3), padding='same', activation="relu"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(DROPOUT),
        
        Conv2D(128, (3, 3), padding='same', activation="relu"),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(DROPOUT),
        
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(DROPOUT),
        Dense(10, activation="softmax"),
    ])

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, mode="max"
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    # Show summary of the model
    model.summary()
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, lr_scheduler],
    )

    # Plot training history
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.legend(loc="lower right")
    # Save the plot (before plt.show(), otherwise the figure is empty)
    plt.savefig("plot.png")
    plt.show()

else:
    # This block will run if the script is called with the --predict flag
    model = load_model(MODEL_SAVE_PATH)
    (x_train, y_train), (x_test, y_test) = load_dataset()
    x_test = x_test / 255.0

    random_index = random.randint(0, len(x_test) - 1)
    random_image = x_test[random_index]
    random_image_array = np.expand_dims(
        random_image, 0
    )  # Expand dims so the input is (1, 32, 32, 3)
    prediction = model.predict(random_image_array)
    predicted_label = CIFAR10_LABELS[np.argmax(prediction)]

    plt.imshow(random_image)
    plt.title(f"Predicted label: {predicted_label}")
    plt.show()
