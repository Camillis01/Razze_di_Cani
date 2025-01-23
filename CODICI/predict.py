import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Carica il modello salvato
model = load_model("final_model.keras")

# Mappa delle classi (aggiorna con i tuoi nomi di classi)
class_labels = ['Abyssinian', 'American Bobtail', 'Basset Hound', 'Beagle', 'Bengal',
                'Birman', 'British Shorthair', 'Chihuahua', 'German Shepherd', 'Great Dane',
                'Havanese', 'Japanese Chin', 'Leonberger', 'Maine Coon', 'Miniature Schnauzer',
                'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard',
                'Samoyed', 'Shiba Inu', 'Sphynx', 'Soft Coated Wheaten Terrier', 'Yorkshire Terrier',
                'Boxer', 'Keeshond', 'Scottish Fold', 'Siamese', 'Abyssinian', 'Newfoundland', 
                'Egyptian Mau', 'Staffordshire Bull Terrier', 'Bombay', 'Yorkshire Terrier']

def preprocess_image(image_path):
    """
    Preprocessa l'immagine per fare previsioni.
    :param image_path: Percorso dell'immagine.
    :return: Immagine preprocessata come numpy array.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def predict_breed(image_path):
    """
    Prevedi la razza del cane dall'immagine fornita.
    :param image_path: Percorso dell'immagine.
    :return: Nome della razza predetta.
    """
    img = preprocess_image(image_path)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    breed = class_labels[class_idx]
    return breed

if __name__ == "__main__":
    # Percorso dell'immagine scaricata dal web
    image_path = "C:/Users/camil/Downloads/prestontesta.jpg"  # Aggiorna questo percorso con il percorso della tua immagine scaricata
    breed = predict_breed(image_path)
    print(f"La razza predetta per l'immagine è: {breed}")
