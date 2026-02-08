# src/inference/generate.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.inference.image_utils import read_image_inf


def generate_caption(image_path: str, caption_model, tokenizer, seq_length: int, show_image: bool = False) -> str:
    """
    Generate a caption for a given image using a trained captioning model.

    This function:
        1. Loads and preprocesses the image.
        2. Extracts image features using the CNN encoder.
        3. Iteratively predicts the next word using the decoder.
        4. Stops when the end token is generated or max length is reached.

    Args:
        image_path (str): Path to the input image.
        caption_model: Trained image captioning model.
        tokenizer: Text vectorization layer used during training.
        seq_length (int): Maximum caption sequence length.
        show_image (bool): Whether to display the image during inference.

    Returns:
        str: Generated caption without start/end tokens.
    """
    vocab = tokenizer.get_vocabulary()
    index_to_word = dict(enumerate(vocab))
    max_len = seq_length - 1

    image = read_image_inf(image_path)

    if show_image:
        img = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
        plt.imshow(img.numpy())
        plt.axis("off")
        plt.show()

    image_features = caption_model.cnn_model(image)
    encoded_image = caption_model.encoder(image_features, training=False)

    decoded_caption = "startseq"

    for i in range(max_len):
        tokenized = tokenizer([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized, 0)

        predictions = caption_model.decoder(tokenized, encoded_image, training=False, mask=mask)

        next_token_id = np.argmax(predictions[0, i])
        next_word = index_to_word.get(next_token_id)

        if next_word == "endseq":
            break

        decoded_caption += " " + next_word

    return decoded_caption.replace("startseq", "").replace("[UNK]", "").strip()
