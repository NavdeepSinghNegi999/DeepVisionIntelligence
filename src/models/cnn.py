# src/models/cnn.py

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet


def get_cnn_model(image_size=(299, 299)):
    """
    Build a CNN encoder using EfficientNetB0.

    The model:
        1. Loads a pretrained EfficientNetB0 backbone.
        2. Removes the classification head.
        3. Freezes all CNN weights.
        4. Reshapes spatial features into a sequence.

    Args:
        image_size (tuple): Input image size (height, width).

    Returns:
        keras.Model: CNN encoder producing image feature sequences.
    """
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    features = base_model.output
    features = layers.Reshape((-1, features.shape[-1]))(features)

    return keras.Model(inputs=base_model.input, outputs=features)
