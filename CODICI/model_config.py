from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def create_model(num_classes):
    """
    Configura il modello ResNet50 pre-addestrato.
    :param num_classes: Numero di classi del dataset.
    :return: Modello compilato.
    """
    # Carica il modello ResNet50 pre-addestrato
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Congela i pesi del modello base
    for layer in base_model.layers:
        layer.trainable = False

    # Aggiungi la testa personalizzata
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout per ridurre overfitting
    predictions = Dense(num_classes, activation='softmax')(x)

    # Crea il modello finale
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Configura un esempio del modello con il numero corretto di classi
    num_classes = 37  # Aggiorna questo numero con il totale delle tue classi
    model = create_model(num_classes)
    model.summary()
