import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data = {
    "X": np.array([1, 3, 4, 5, 6, 2, 7, 8, 9, 10, 11, 12, 13, 14]),
    "Y": np.array([1, 1.2, 1.4, 1.1, 1, 5.5, 6.1, 6.7, 6.4, 6, 6, 3, 3.2, 3.1]),
}

# Sort Values aufsteigend nach X - Verwendung von Numpy
sorted_indices = np.argsort(data["X"])
data["X"] = data["X"][sorted_indices]
data["Y"] = data["Y"][sorted_indices]


# Formel zur Berechnung des MSE
def calculate_MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Datensplits erzeugen
def split_data(data, split_x):
    left_part = data["X"] <= split_x
    right_part = data["X"] > split_x

    left_split = {"X": data["X"][left_part], "Y": data["Y"][left_part]}
    right_split = {"X": data["X"][right_part], "Y": data["Y"][right_part]}

    return left_split, right_split


# Define and identify best split_element
def best_split_mse(data):
    best_split_x = None
    best_mse = float("inf")

    for i in range(0, len(data["X"]) - 1):
        interm_split_x = (data["X"][i] + data["X"][i + 1]) / 2
        left_split, right_split = split_data(data, interm_split_x)

        # Berechne Mittelwerte der y-Werte für beide Splits
        mean_left = np.mean(left_split["Y"]) if len(left_split["Y"]) > 0 else np.nan
        mean_right = np.mean(right_split["Y"]) if len(right_split["Y"]) > 0 else np.nan

        # MSE Berechnung für beide Nodes
        mse_left = calculate_MSE(
            left_split["Y"], np.full(len(left_split["Y"]), mean_left)
        )
        mse_right = calculate_MSE(
            right_split["Y"], np.full(len(right_split["Y"]), mean_right)
        )
        interm_mse = mse_left + mse_right

        if interm_mse < best_mse:
            best_mse = interm_mse
            best_split_x = interm_split_x
            
        # Print mse
        print("Split X:", interm_split_x, "MSE:", interm_mse)

    return best_split_x, best_mse


def recursive_split_fixed(data, threshold=0.05):
    # Initialisierung der Warteschlange (queue) mit den ursprünglichen Daten und den entsprechenden Indizes.
    # Die Warteschlange enthält Tupel aus (aktuellen Daten, Indizes im ursprünglichen Array).
    queue = [
        (data, np.arange(len(data["X"])))
    ]  # Tupel aus Daten und entsprechenden Indizes

    # Initialisierung eines Arrays, das die Vorhersagen speichert. Es hat die gleiche Größe wie die Eingabedaten.
    predictions = np.zeros_like(data["X"], dtype=float)

    # Verarbeite jedes Element in der Warteschlange, bis diese leer ist.
    while queue:
        # Entferne das erste Element aus der Warteschlange.
        # `current_data` ist das Dictionary, das den aktuellen Split der Daten enthält.
        # `indices` sind die Indizes des aktuellen Splits im ursprünglichen Datensatz.
        current_data, indices = queue.pop(0)

        # Bestimme den besten Splitpunkt `best_split_x` basierend auf den aktuellen Daten.
        # `best_mse` repräsentiert den MSE für den besten gefundenen Split.
        best_split_x, best_mse = best_split_mse(current_data)
        print("Best Split X:", best_split_x, "Best MSE:", best_mse)

        # Teile die aktuellen Daten in zwei Teile: links und rechts vom besten Splitpunkt.
        left_split, right_split = split_data(current_data, best_split_x)

        # Ermittele die Indizes für den linken und rechten Split.
        # Diese Indizes beziehen sich auf das ursprüngliche Datenarray.
        left_indices = indices[current_data["X"] <= best_split_x]
        right_indices = indices[current_data["X"] > best_split_x]

        # Berechne den Mittelwert der Y-Werte für den linken Split und weise diesen den entsprechenden Vorhersagen zu.
        if (
            len(left_indices) > 0
        ):  # Stelle sicher, dass es Elemente im linken Split gibt.
            predictions[left_indices] = np.mean(left_split["Y"])

        # Ähnlich, berechne den Mittelwert für den rechten Split und weise diesen den Vorhersagen zu.
        if (
            len(right_indices) > 0
        ):  # Stelle sicher, dass es Elemente im rechten Split gibt.
            predictions[right_indices] = np.mean(right_split["Y"])

        # Wenn der linke Split mehr als ein Element hat und der MSE über dem Schwellenwert liegt,
        # füge ihn zur Warteschlange für weitere Splits hinzu.
        if len(left_split["X"]) > 1 and best_mse > threshold:
            queue.append((left_split, left_indices))

        # Ähnlich, wenn der rechte Split mehr als ein Element hat und der MSE über dem Schwellenwert liegt,
        # füge ihn zur Warteschlange für weitere Splits hinzu.
        if len(right_split["X"]) > 1 and best_mse > threshold:
            queue.append((right_split, right_indices))

    # Gib das finale Vorhersage-Array zurück, das die vorhergesagten Werte für jede Eingabe enthält.
    return predictions

# Manuelle Implementierung
manual_predictions_fixed = recursive_split_fixed(data)
print("MSE Manuelle Implementation:", calculate_MSE(data["Y"], manual_predictions_fixed))

# Vergleich mit scikit-learn DecisionTreeRegressor
X = data["X"].reshape(-1, 1)
y = data["Y"]

# Modell trainieren
model = DecisionTreeRegressor(max_depth=2)
model.fit(X, y)

# Vorhersagen auf den Trainingsdaten
sklearn_predictions = model.predict(X)

# Plotten der Ergebnisse
plt.figure(figsize=(14, 8))
plt.scatter(X, y, color="blue", label="Original Data")
plt.plot(
    X,
    manual_predictions_fixed,
    color="green",
    linestyle="--",
    label="Manuelle Implementation",
)
plt.plot(X, sklearn_predictions, color="red", label="sklearn DecisionTreeRegressor")
plt.title("Vergleich Manuelle Implementation vs. sklearn DecisionTreeRegressor")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
