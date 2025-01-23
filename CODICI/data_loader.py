import os
from collections import Counter

# Mappa completa dei nomi delle razze (modifica secondo necessità)
complete_breeds = {
    'Abyssinian': 'Abyssinian',
    'american': 'American Bobtail',
    'english': 'English Bulldog',
    'basset': 'Basset Hound',
    'beagle': 'Beagle',
    'Bengal': 'Bengal',
    'Birman': 'Birman',
    'British': 'British Shorthair',
    'chihuahua': 'Chihuahua',
    'german': 'German Shepherd',
    'great': 'Great Dane',
    'havanese': 'Havanese',
    'japanese': 'Japanese Chin',
    'leonberger': 'Leonberger',
    'Maine': 'Maine Coon',
    'miniature': 'Miniature Schnauzer',
    'Persian': 'Persian',
    'pomeranian': 'Pomeranian',
    'pug': 'Pug',
    'Ragdoll': 'Ragdoll',
    'Russian': 'Russian Blue',
    'saint': 'Saint Bernard',
    'samoyed': 'Samoyed',
    'shiba': 'Shiba Inu',
    'Sphynx': 'Sphynx',
    'wheaten': 'Soft Coated Wheaten Terrier',
    'yorkshire': 'Yorkshire Terrier',
    'boxer': 'Boxer',
    'keeshond': 'Keeshond',
    'scottish': 'Scottish Fold',
    'Siamese': 'Siamese',
    'newfoundland': 'Newfoundland',
    'Egyptian': 'Egyptian Mau',
    'staffordshire': 'Staffordshire Bull Terrier',
    'Bombay': 'Bombay'
}

# Percorsi principali
images_path = "./DogBreedDataset/images/images"
annotations_path = "./DogBreedDataset/annotations/annotations/list.txt"

# Controlla che i percorsi esistano
if not os.path.exists(images_path):
    print(f"Errore: La directory delle immagini non esiste: {images_path}")
    exit()

if not os.path.exists(annotations_path):
    print(f"Errore: Il file delle annotazioni non esiste: {annotations_path}")
    exit()

# Elenca i file disponibili nella cartella Images
image_files = set(os.listdir(images_path))  # Usa un set per velocizzare il matching

# Caricamento immagini ed etichette con filtro
images = []
labels = []

with open(annotations_path, "r") as f:
    lines = f.readlines()

for line in lines:
    if line.startswith('#') or not line.strip():  # Ignora commenti e righe vuote
        continue

    parts = line.strip().split()
    image_name = parts[0] + ".jpg"  # Aggiunge estensione .jpg
    
    # Estrai il prefisso del nome della razza
    prefix_name = image_name.split('_')[0]
    
    # Mappa il prefisso del nome della razza al nome completo
    breed_name = complete_breeds.get(prefix_name, "Unknown")
    
    # Considera solo le immagini che esistono in entrambe le liste
    if image_name in image_files:
        image_path = os.path.join(images_path, image_name)
        images.append(image_path)
        labels.append(breed_name)
    else:
        print(f"Immagine non trovata per: {image_name}")

print(f"\nCaricate {len(images)} immagini con etichette.")
if len(labels) > 5:
    print(f"Prime 5 etichette: {labels[:5]}")
else:
    print(f"Etichette caricate: {labels}")

# Mostra la distribuzione delle etichette
label_counts = Counter(labels)
print(f"Distribuzione delle etichette: {label_counts}")



#ogni razza di cane è stata divisa in una classe, grazie a questo codice 
#scopro che alla razza classe2 sono state associate 200 immagini e via dicendo
#noto che sono tutte equamente distribuite