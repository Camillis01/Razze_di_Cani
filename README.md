# Razze_di_Cani
# Dog Breed Classification with Deep Learning

This project focuses on classifying dog breeds using a Convolutional Neural Network (CNN) trained on a preprocessed dataset. It leverages TensorFlow and Keras for model development and training. The project includes several phases from data preprocessing to evaluation.

## Project Structure

- `annotations/`: Contains metadata for the dataset.  
- `images/`: Original image data.  
- `data_loader.py`: Script for loading and preparing the dataset.  
- `data_split.py`: Handles splitting the data into training, validation, and test sets.  
- `image_preprocessing.py`: Resizes, normalizes, and augments the images for the model.  
- `model_config.py`: Defines and configures the CNN model using Transfer Learning.  
- `train_and_evaluate.py`: Trains the model and evaluates its performance on test data.  
- `predict.py`: Allows for predictions on new images.  
- `best_model.keras`: The best model saved during training.  
- `final_model.keras`: The final trained model.  
- `train_data.pkl`, `train_labels.pkl`, `test_data.pkl`: Serialized data used for model training and testing.  
- `processed_train_images.pkl`: Preprocessed training images.  

## Phases

1. **Data Loader**  
   Script: `data_loader.py`  
   Loads the dataset from the source files and prepares it for preprocessing.  
   Ensures the integrity and consistency of image-label pairs.  

2. **Data Split**  
   Script: `data_split.py`  
   Splits the dataset into training, validation, and test sets in a stratified manner.  

3. **Image Preprocessing**  
   Script: `image_preprocessing.py`  
   Resizes images to 224x224 pixels.  
   Normalizes pixel values to the range [0, 1].  
   Includes data augmentation for robustness.  

4. **Model Configuration**  
   Script: `model_config.py`  
   Defines a CNN model based on ResNet50 architecture.  
   Includes dropout layers to reduce overfitting and mixed precision for performance optimization.  

5. **Training and Evaluation**  
   Script: `train_and_evaluate.py`  
   Uses callbacks like Early Stopping and Model Checkpointing.  
   Includes learning rate reduction for efficient optimization.  
   Outputs training metrics like accuracy and loss.  

6. **Prediction**  
   Script: `predict.py`  
   Provides an easy interface for predicting the breed of a dog from an input image.  

## Features

- **Transfer Learning**: Uses ResNet50 pre-trained on ImageNet for feature extraction.  
- **Data Augmentation**: Increases dataset diversity to improve generalization.  
- **Mixed Precision Training**: Reduces computation time on compatible GPUs.  
- **Visualization**: Generates training/validation accuracy and loss graphs.  

## How to Run

1. Clone the repository:  
   ```bash
   git clone <repository-url>
   cd DogBreedClassification


Future Work
Improve model accuracy by experimenting with different architectures.
Explore larger datasets for enhanced performance.
Deploy the model as a web app for real-time predictions.

License
This project is licensed under GNU General Public License v3.0 and is restricted to non-commercial use.
