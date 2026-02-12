# src/data/augmentation.py
from tensorflow import keras


def get_image_augmentation():
    """
    Return image augmentation pipeline used during training.
    """
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.2),
            keras.layers.RandomContrast(0.3),
        ]
    )
