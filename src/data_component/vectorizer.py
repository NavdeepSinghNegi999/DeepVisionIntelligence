# src/data/vectorizer.py

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization


def build_text_vectorizer(captions, vocab_size, seq_length):
    """
    Build and adapt a TextVectorization layer on captions.

    Args:
        captions (list or pd.Series): Preprocessed captions.
        vocab_size (int): Maximum vocabulary size.
        seq_length (int): Output sequence length.

    Returns:
        TextVectorization: Adapted vectorization layer.
    """

    vectorizer = TextVectorization(max_tokens=vocab_size, output_mode="int", output_sequence_length=seq_length,)
    vectorizer.adapt(captions)
    return vectorizer
