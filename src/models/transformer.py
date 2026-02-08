# src/models/transformer.py

import tensorflow as tf
from tensorflow.keras import layers


class TransformerEncoderBlock(layers.Layer):
    """
    Transformer encoder block for image feature encoding.
    """

    def __init__(self, embed_dim, dense_dim, num_heads):
        super().__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense = layers.Dense(embed_dim, activation="relu")
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, inputs, training, mask=None):
        x = self.norm1(inputs)
        x = self.dense(x)
        attn = self.attention(query=x, value=x, key=x, training=training, attention_mask=None)
        return self.norm2(x + attn)


class PositionalEmbedding(layers.Layer):
    """
    Token + positional embedding layer.
    """

    def __init__(self, sequence_length, vocab_size, embed_dim):
        super().__init__()
        self.token_embed = layers.Embedding(vocab_size, embed_dim)
        self.pos_embed = layers.Embedding(sequence_length, embed_dim)
        self.scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs):
        positions = tf.range(tf.shape(inputs)[-1])
        return self.token_embed(inputs) * self.scale + self.pos_embed(positions)

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, 0)


class TransformerDecoderBlock(layers.Layer):
    """
    Transformer decoder block for caption generation.
    """

    def __init__(self, embed_dim, ff_dim, num_heads, seq_len, vocab_size):
        super().__init__()
        self.embedding = PositionalEmbedding(seq_len, vocab_size, embed_dim)

        self.attn1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attn2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

        self.ffn1 = layers.Dense(ff_dim, activation="relu")
        self.ffn2 = layers.Dense(embed_dim)

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()

        self.dropout1 = layers.Dropout(0.3)
        self.dropout2 = layers.Dropout(0.5)

        self.classifier = layers.Dense(vocab_size, activation="softmax")
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        x = self.embedding(inputs)
        causal_mask = self._causal_mask(x)

        attn1 = self.attn1(query=x, value=x, key=x, attention_mask=causal_mask, training=training)
        x = self.norm1(x + attn1)

        attn2 = self.attn2(query=x, value=encoder_outputs, key=encoder_outputs, training=training)
        x = self.norm2(x + attn2)

        ffn = self.ffn2(self.dropout1(self.ffn1(x), training=training))
        x = self.norm3(x + ffn)

        return self.classifier(self.dropout2(x, training=training))

    def _causal_mask(self, x):
        seq_len = tf.shape(x)[1]
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return mask[None, :, :]
