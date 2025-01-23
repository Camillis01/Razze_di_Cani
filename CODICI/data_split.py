from data_loader import images, labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Codifica delle etichette da stringhe a numeri
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Divisione in training e test set
train_images, test_images, train_labels, test_labels = train_test_split(
    images, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

# Output della divisione
print(f"Immagini nel Training Set: {len(train_images)}")
print(f"Immagini nel Test Set: {len(test_images)}")

# Debug: Visualizza le prime etichette
print(f"Prime etichette nel Training Set: {train_labels[:5]}")
print(f"Prime etichette nel Test Set: {test_labels[:5]}")

# Salva i dati divisi per il preprocessing e l'addestramento del modello
import joblib

try:
    joblib.dump((train_images, train_labels), "train_data.pkl")
    joblib.dump((test_images, test_labels), "test_data.pkl")
    print("Dati di training e test salvati correttamente.")
except Exception as e:
    print(f"Errore durante il salvataggio dei dati di training e test: {e}")


#ho appena diviso il mio traning Set e il mio Test Set: ora inizierò con