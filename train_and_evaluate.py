import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy
from model_config import create_model

# Abilita il Mixed Precision Training solo su piattaforme supportate
if tf.config.experimental.list_physical_devices('GPU'):
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    set_global_policy(policy)

# Carica i dati preprocessati
train_images = joblib.load("processed_train_images.pkl")
train_labels = joblib.load("train_labels.pkl")

# Assicurati che i dati siano numpy array
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Divisione in training e validation set
X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

# Crea il modello
num_classes = len(set(train_labels))  # Numero di classi nel dataset
model = create_model(num_classes)

# Fine-tuning: Rendere più strati del modello base addestrabili
for layer in model.layers[:100]:
    layer.trainable = False
for layer in model.layers[100:]:
    layer.trainable = True

# Callback: Early Stopping, Checkpoint e Riduzione Learning Rate
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_loss")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,  # Aumenta la rotazione
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,  # Aggiungi lo zoom
    shear_range=0.2,  # Aggiungi lo shear
    horizontal_flip=True,
    fill_mode='nearest'
)

# Addestramento del modello con Data Augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),  # Ridotto il batch size per velocizzare
    validation_data=(X_val, y_val),
    epochs=50,  # Aumenta il numero di epoche se necessario
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Salva il modello finale
model.save("final_model.keras")

# Debug: Mostra le metriche di addestramento
print("Addestramento completato.")
print(f"Accuratezza finale: {history.history['accuracy'][-1]}")
print(f"Loss finale: {history.history['loss'][-1]}")

# Carica i dati di test
test_data = joblib.load("test_data.pkl")
test_images, test_labels = test_data

# Assicurati che i dati siano numpy array
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Valutazione del modello
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Visualizzazione dei Risultati
import matplotlib.pyplot as plt

# Grafico dell'accuratezza
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Grafico della perdita
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
