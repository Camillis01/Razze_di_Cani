import os

# Percorsi delle sottocartelle
trimap_path = "./DogBreedDataset/annotations/annotations/trimaps"
xmls_path = "./DogBreedDataset/annotations/annotations/xmls"

# Funzione per elencare i file
def list_files(folder_path, folder_name):
    """
    Elenca i primi due file nella cartella specificata.
    :param folder_path: Percorso della cartella.
    :param folder_name: Nome della cartella per output.
    """
    try:
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            if len(files) > 0:
                print(f"Primi due file nella cartella {folder_name}:")
                print(files[:2])
            else:
                print(f"La cartella {folder_name} è vuota.")
        else:
            print(f"La cartella {folder_name} non esiste.")
    except Exception as e:
        print(f"Errore durante l'elenco dei file nella cartella {folder_name}: {e}")

# Controlla le cartelle
list_files(trimap_path, "trimaps")
list_files(xmls_path, "xmls")
list_file_path = "./DogBreedDataset/annotations/annotations/list.txt"

if os.path.exists(list_file_path):
    try:
        with open(list_file_path, 'r') as f:
            lines = f.readlines()
            print("Prime righe di list.txt:")
            print(lines[:5])  # Stampa le prime 5 righe
    except Exception as e:
        print(f"Errore durante la lettura del file list.txt: {e}")
else:
    print("Il file list.txt non esiste.")


#nelle precedenti righe di codice ho controllato di leggere correttamente i primi due file presenti nelle 
#sottocartelle Trimaps e xmls dato che avevo estratto manualmente i file in un cartella