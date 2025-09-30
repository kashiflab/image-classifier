"""
data_loader.py
Handles downloading the dataset (via kagglehub) and preparing
TensorFlow ImageDataGenerators for training, validation, and testing.
"""

import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 128
BATCH_SIZE = 32

def load_data():
    # Download dataset from Kaggle using kagglehub
    path = kagglehub.dataset_download("chetankv/dogs-cats-images")
    print("Dataset path:", path)

    # Dataset folders inside the downloaded path
    train_dir = f"{path}/dataset/training_set"
    test_dir = f"{path}/dataset/test_set"

    # Data augmentation for training (rescale + random flips/zooms)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )

    # Training generator (80% of training data)
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training"
    )

    # Validation generator (20% of training data)
    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation"
    )

    # Test generator (only rescale, no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    return train_gen, val_gen, test_gen
