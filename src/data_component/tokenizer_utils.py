# src/data/tokenizer_utils.py

import os
import tensorflow as tf


def save_tokenizer(vectorizer, save_dir):
    """
    Save TextVectorization layer as a model.
    """
    os.makedirs(save_dir, exist_ok=True)

    inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    outputs = vectorizer(inputs)
    model = tf.keras.Model(inputs, outputs)

    model.save(os.path.join(save_dir, "tokenizer"), save_format="tf")


def load_tokenizer(save_dir):
    """
    Load saved TextVectorization layer.
    """
    model = tf.keras.models.load_model(os.path.join(save_dir, "tokenizer"), compile=False)
    return model.layers[1]
