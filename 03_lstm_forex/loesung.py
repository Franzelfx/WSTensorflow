"""Workshop 03 – Musterlösung: LSTM zur Zeitreihen-Vorhersage auf Forex-Daten.

Ausführen (aus dem Repository-Root):
    python 03_lstm_forex/loesung.py data/AUDCAD_15.csv            # trainieren
    python 03_lstm_forex/loesung.py data/AUDCAD_15.csv --predict  # vorhersagen
    tensorboard --logdir logs
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
# Tensorboard
from tensorflow.keras.callbacks import TensorBoard

# Read market data from CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


# Normalize the data
def normalize_data(data):
    # Convert date column to datetime and set as index if necessary
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        data.set_index("date", inplace=True)

    # Select only numeric columns for normalization
    numeric_data = data.select_dtypes(include=[np.number])

    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(numeric_data)
    return normalized_data, scaler


# Prepare the data for the LSTM model
def prepare_data(data, input_timesteps, output_timesteps):
    samples = []
    labels = []
    for i in range(len(data) - input_timesteps - output_timesteps + 1):
        samples.append(data[i : i + input_timesteps])
        labels.append(
            data[i + input_timesteps : i + input_timesteps + output_timesteps]
        )
    samples = np.array(samples)
    labels = np.array(labels)
    return samples, labels


# Split the data into training and testing sets
def split_data(samples, labels, test_size=0.2):
    return train_test_split(samples, labels, test_size=test_size, random_state=42)


# Build the LSTM model
def build_model(input_timesteps, num_features, output_timesteps):
    model = Sequential()
    model.add(
        LSTM(
            100,
            activation="relu",
            return_sequences=True,
            input_shape=(input_timesteps, num_features),
        )
    )
    model.add(Dropout(0.2))
    model.add(LSTM(100, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(output_timesteps * num_features))
    model.compile(optimizer="adam", loss="mse")
    return model


# Train the model
def train_model(model, X_train, y_train, epochs=50, batch_size=64):
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    model_checkpoint = ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss", mode="min"
    )
    # Checkpoint for Tensorboard
    tensorboard = TensorBoard(log_dir="logs")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint, tensorboard],
    )
    return history


# Plot training history
def plot_training_history(history):
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.legend()
    plt.show()


# Make predictions
def make_predictions(model, X_test):
    return model.predict(X_test)


# Plot predictions
def plot_predictions(y_test, y_pred):
    plt.plot(y_test.flatten(), label="True")
    plt.plot(y_pred.flatten(), label="Predicted")
    plt.legend()
    plt.show()


# Main script
def main(file_path, predict):
    data = load_data(file_path)
    data, scaler = normalize_data(data)

    input_timesteps = 64
    output_timesteps = 32
    samples, labels = prepare_data(data, input_timesteps, output_timesteps)

    # Ensure the labels have the correct shape
    labels = labels.reshape(labels.shape[0], output_timesteps * samples.shape[2])

    X_train, X_test, y_train, y_test = split_data(samples, labels)

    num_features = X_train.shape[2]
    model = build_model(input_timesteps, num_features, output_timesteps)

    if predict:
        model = load_model("best_model.h5")
        y_pred = make_predictions(model, X_test)
        y_test = y_test.reshape(
            y_test.shape[0], output_timesteps, num_features
        )  # Reshape for plotting
        y_pred = y_pred.reshape(
            y_pred.shape[0], output_timesteps, num_features
        )  # Reshape for plotting
        plot_predictions(y_test, y_pred)
    else:
        history = train_model(model, X_train, y_train)
        plot_training_history(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and predict market data using LSTM."
    )
    parser.add_argument(
        "file_path", type=str, help="Path to the CSV file containing market data."
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Flag to indicate if predictions should be made.",
    )
    args = parser.parse_args()

    main(args.file_path, args.predict)
