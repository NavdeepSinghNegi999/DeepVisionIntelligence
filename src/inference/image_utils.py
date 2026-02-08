# src/inference/image_utils.py

import tensorflow as tf

IMAGE_SIZE = (299, 299)

def read_image_inf(image_path: str) -> tf.Tensor:
    """
    Load and preprocess an image for inference.

    This function:
        1. Reads the image from disk.
        2. Decodes it as an RGB JPEG image.
        3. Resizes it to the model's expected input size.
        4. Normalizes pixel values to [0, 1].
        5. Adds a batch dimension.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tf.Tensor: Preprocessed image tensor of shape
                   (1, height, width, 3).
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.expand_dims(img, axis=0)
