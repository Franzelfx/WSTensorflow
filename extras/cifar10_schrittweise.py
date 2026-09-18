# Wir möchten ein neuronales Netz trainieren, um cifar10 Datensatz- Bilder zu erkennen.

# 1. Importiere die benötigten Bibliotheken
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 2. Lade den cifar10 Datensatz
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 3. Teile tetst in val und test auf
val_images = test_images[:5000]
val_labels = test_labels[:5000]
test_images = test_images[5000:]
test_labels = test_labels[5000:]

# 4. Normalisiere die Pixelwerte auf den Bereich 0-1
train_images, val_images, test_images = train_images / 255.0, val_images / 255.0, test_images / 255.0

# 5. Definiere das Modell, 10% Parametergrösse im Verhältnis zum Datensatz
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) # 32 Filter, 3x3 Kernel
model.add(layers.MaxPooling2D((2, 2))) # MaxPooling 2x2, filtert wichtige Informationen, verkleinert Bild um 50%
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # 64 Filter, 3x3 Kernel
model.add(layers.MaxPooling2D((2, 2))) # MaxPooling 2x2, filtert wichtige Informationen, verkleinert Bild um 50%

# Output layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='linear'))
model.add(layers.Dense(10))

# 6. Kompiliere das Modell
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 7. Trainiere das Modell
history = model.fit(train_images, train_labels, epochs=20, 
                    validation_data=(val_images, val_labels))

# 8. Evaluieren des Modells
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

# 9. Plotte die Genauigkeit und den Loss
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Teste das Modell mit einem Bild aus dem Test- datesatz und gib die Vorhersage aus
classes = ["Flugzeug", "Auto", "Vogel", "Katze", "Hirsch", "Hund", "Frosch", "Pferd", "Schiff", "Lastwagen"]
img = test_images[0]
img = (np.expand_dims(img,0))
predictions = model.predict(img)
print(classes[np.argmax(predictions[0])])

# 10. Speichere das Modell
model.save('cifar10_model.h5')
