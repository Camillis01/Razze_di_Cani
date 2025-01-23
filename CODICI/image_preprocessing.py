import numpy as np
import cv2
from tqdm import tqdm  # Per una barra di avanzamento
import joblib

# Carica i dati di training dal file salvato
train_data = joblib.load("train_data.pkl")
train_images, train_labels = train_data

def load_images(image_paths):
    """
    Carica le immagini dai percorsi forniti.
    :param image_paths: Lista di percorsi delle immagini.
    :return: Lista di immagini caricate (numpy array).
    """
    loaded_images = []
    for path in tqdm(image_paths, desc="Caricamento immagini"):
        img = cv2.imread(path)
        if img is not None:
            loaded_images.append(img)
        else:
            print(f"Errore nel caricamento dell'immagine: {path}")
    return loaded_images

def preprocess_images_in_batches(images, target_size=(224, 224), batch_size=100):
    """
    Ridimensiona e normalizza le immagini in batch per ridurre l'uso di memoria.
    :param images: Lista di immagini originali.
    :param target_size: Dimensione target (altezza, larghezza).
    :param batch_size: Numero di immagini da preprocessare per batch.
    :return: Lista di immagini preprocessate (numpy array).
    """
    processed_images = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_processed = []
        for img in batch:
            try:
                resized_img = cv2.resize(img, target_size)
                normalized_img = (resized_img / 255.0).astype(np.float32)
                batch_processed.append(normalized_img)
            except Exception as e:
                print(f"Errore durante il preprocessing dell'immagine: {e}")
        processed_images.extend(batch_processed)
    return np.array(processed_images)

if __name__ == "__main__":
    # Verifica che train_images e train_labels esistano
    assert len(train_images) == len(train_labels), "Le immagini e le etichette non corrispondono in numero."
    print(f"Numero di immagini da preprocessare: {len(train_images)}")
    print(f"Numero di etichette: {len(train_labels)}")

    # Carica le immagini dai percorsi
    loaded_train_images = load_images(train_images)

    # Preprocessa le immagini in batch
    processed_train_images = preprocess_images_in_batches(loaded_train_images, batch_size=500)

    if len(processed_train_images) != len(train_labels):
        print("Attenzione: Il numero di immagini preprocessate non corrisponde al numero di etichette.")

    print(f"Forma del primo batch preprocessato: {processed_train_images[0].shape}")

    # Salva le immagini preprocessate
    try:
        joblib.dump(processed_train_images, "processed_train_images.pkl")
        joblib.dump(train_labels, "train_labels.pkl")
        print("Dati preprocessati salvati correttamente.")
    except Exception as e:
        print(f"Errore durante il salvataggio dei dati preprocessati: {e}")
