#!/usr/bin/env python
# coding: utf-8

# # Tensorflow check

# In[119]:


import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))


# In[32]:


import pandas as pd
from src.data.preprocessing import preprocess_caption
from src.training.setup import split_dataset

caption_path = "../data/raw/captions.txt" 
df = pd.read_csv(caption_path)

# Preprocess captions
df["caption"] = df["caption"].apply(preprocess_caption)

# Split dataset (NO LEAKAGE)
train_df, val_df, test_df = split_dataset(df)

print(len(train_df), len(val_df), len(test_df))


# # Start

# In[121]:


import os
import re
import json
import cv2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

from sklearn.model_selection import train_test_split


seed = 999
np.random.seed(seed)
pd.set_option('display.max_colwidth', None)


# In[122]:


base_dir = r'../'
data_dir = os.path.join(base_dir, 'data/raw')
img_dir = os.path.join(data_dir, 'Images')
caption_dir = os.path.join(data_dir, 'captions.txt')
# models_dir = os.path.join(base_dir, 


# ## Methods

# In[123]:


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


# In[124]:


# src/models/caption_model.py

import tensorflow as tf
from tensorflow import keras


class ImageCaptioningModel(keras.Model):
    """
    Custom Keras model for image captioning with
    CNN encoder + Transformer encoder/decoder.
    """

    def __init__(self, cnn_model, encoder, decoder, image_aug=None):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.image_aug = image_aug

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")

    def call(self, inputs):
        images, training, captions = inputs
        features = self.cnn_model(images)
        encoded = self.encoder(features, training=False)
        return self.decoder(captions, encoded, training=training, mask=None)

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)  
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


    def calculate_accuracy(self, y_true, y_pred, mask):
        acc = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        acc = tf.logical_and(mask, acc)
        acc = tf.cast(acc, tf.float32)
        return tf.reduce_sum(acc) / tf.reduce_sum(tf.cast(mask, tf.float32))

    def _compute_loss_and_acc(self, image_features, captions, training):
        encoded = self.encoder(image_features, training=training)

        seq_inp = captions[:, :-1]
        seq_true = captions[:, 1:]
        mask = tf.math.not_equal(seq_true, 0)

        preds = self.decoder(seq_inp, encoded, training=training, mask=mask)

        loss = self.calculate_loss(seq_true, preds, mask)
        acc = self.calculate_accuracy(seq_true, preds, mask)

        return loss, acc

    def train_step(self, data):
        images, captions = data

        if self.image_aug:
            images = self.image_aug(images)

        image_features = self.cnn_model(images)

        with tf.GradientTape() as tape:
            loss, acc = self._compute_loss_and_acc(image_features, captions, True)

        train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    def test_step(self, data):
        images, captions = data
        image_features = self.cnn_model(images)

        loss, acc = self._compute_loss_and_acc(image_features, captions, False)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]


# In[125]:


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


# In[126]:


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


# In[164]:


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


# In[165]:


# src/inference/load_model.py

import json
import tensorflow as tf
# from src.models.cnn import get_cnn_model
# from src.models.transformer import TransformerEncoderBlock, TransformerDecoderBlock
# from src.models.caption_model import ImageCaptioningModel


def get_inference_model(config_path):
    """
    Load image captioning model for inference from config.
    """
    with open(config_path) as f:
        cfg = json.load(f)

    cnn = get_cnn_model()
    encoder = TransformerEncoderBlock(cfg["EMBED_DIM"], cfg["FF_DIM"], cfg["NUM_HEADS"])
    decoder = TransformerDecoderBlock(
        cfg["EMBED_DIM"], cfg["FF_DIM"], cfg["NUM_HEADS"], cfg["SEQ_LENGTH"], cfg["VOCAB_SIZE"]
    )

    model = ImageCaptioningModel(cnn, encoder, decoder)

    img_input = tf.keras.Input(shape=(299, 299, 3))
    cap_input = tf.keras.Input(shape=(None,))
    model([img_input, False, cap_input])

    return model


# In[ ]:





# ##  utils

# In[166]:


# src/data/preprocessing.py

import re
import string
import contractions
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

def preprocess_caption(caption: str) -> str:
    """
    Preprocess a raw image caption for training an image captioning model.

    This function performs the following steps:
        1. Converts text to lowercase.
        2. Expands English contractions (e.g., "don't" -> "do not").
        3. Removes punctuation characters.
        4. Normalizes whitespace.
        5. Removes English stopwords and single-character tokens.
        6. Adds sequence boundary tokens ("startseq" and "endseq").

    Args:
        caption (str): Raw caption text associated with an image.

    Returns:
        str: Cleaned caption formatted as
             "startseq <processed caption> endseq".
    """

    caption = caption.lower()
    caption = contractions.fix(caption)

    caption = caption.translate(str.maketrans("", "", string.punctuation))
    caption = re.sub(r"\s+", " ", caption).strip()
    caption = " ".join(word for word in caption.split() if word not in STOP_WORDS and len(word) > 1)

    return f"startseq {caption} endseq"


# In[167]:


# src/data/image_utils.py

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


# In[168]:


# src/inference/generate.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from src.data.image_utils import read_image_inf


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


# In[169]:


from sklearn.model_selection import train_test_split


def dataset_split(df, dataset_len=40000, min_token=5, max_token=25, shuffle=True, seed=42,):
    """
    Filter captions by length and split dataset into train/val/test.

    Returns:
        train_df, val_df, test_df
    """

    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df = df[df["caption"].apply(lambda cap: min_token <= len(cap.split()) <= max_token)].iloc[:dataset_len]

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed)

    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)

    return (train_df.reset_index(drop=True),val_df.reset_index(drop=True),test_df.reset_index(drop=True),)


# In[170]:


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


# In[171]:


#📁 src/data/augmentation.py
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


# In[172]:


# src/data/dataset.py
import os
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def load_and_preprocess_image(image_path, image_size):
    """
    Load and preprocess an image from disk.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def decode_image_and_vectorize(caption, image_name, img_dir, vectorizer, image_size):
    """
    Load image and vectorize caption.
    """
    image_path = tf.strings.join([img_dir, image_name], separator=os.path.sep)

    image = load_and_preprocess_image(image_path, image_size)
    caption = vectorizer(caption)

    return image, caption
    

def make_dataset(df, img_dir, vectorizer, image_size, batch_size, shuffle=True):
    """
    Create tf.data.Dataset for image captioning.
    """
    dataset = tf.data.Dataset.from_tensor_slices(
        (df["caption"].values, df["image"].values)
    )

    if shuffle:
        dataset = dataset.shuffle(batch_size * 8)

    dataset = dataset.map(
        lambda cap, img: decode_image_and_vectorize(cap, img, img_dir, vectorizer, image_size),
        num_parallel_calls=AUTOTUNE,
    )

    return dataset.batch(batch_size).prefetch(AUTOTUNE)


# In[ ]:





# In[173]:


# src/train/setup.py
 
# from src.data.augmentation import get_image_augmentation
# from src.models.cnn import get_cnn_model
# from src.models.transformer import (TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptioningModel,)


def build_caption_model(vocab_size, seq_length, embed_dim=512, ff_dim=512, num_heads=6,):
    """
    Build and return the image captioning model.
    """
    cnn_model = get_cnn_model()

    encoder = TransformerEncoderBlock(embed_dim=embed_dim, dense_dim=ff_dim, num_heads=num_heads)

    decoder = TransformerDecoderBlock(embed_dim=embed_dim, ff_dim=ff_dim, num_heads=num_heads, seq_len=seq_length, vocab_size=vocab_size)

    return ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder,decoder=decoder,image_aug=get_image_augmentation())


# In[ ]:





# In[174]:


class WarmupLearningRateSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup learning rate schedule.

    The learning rate increases linearly for `warmup_steps`,
    then stays constant at `base_lr`.
    """

    def __init__(self, base_lr, warmup_steps):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)

        warmup_lr = self.base_lr * (step / warmup_steps)
        return tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: self.base_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
        }



def build_optimizer(train_dataset, epochs: int, base_lr: float = 1e-4, warmup_ratio: int = 15):
    """
    Build Adam optimizer with warmup learning rate schedule.

    Args:
        train_dataset (tf.data.Dataset): Training dataset.
        epochs (int): Total number of epochs.
        base_lr (float): Target learning rate after warmup.
        warmup_ratio (int): Warmup steps = total_steps / warmup_ratio.

    Returns:
        tf.keras.optimizers.Optimizer
    """
    total_steps = len(train_dataset) * epochs
    warmup_steps = total_steps // warmup_ratio

    lr_schedule = WarmupLearningRateSchedule(base_lr=base_lr,warmup_steps=warmup_steps)

    return keras.optimizers.Adam(learning_rate=lr_schedule)


# In[ ]:





# In[175]:


def plot_history():
    """
    Plot training loss over epochs.
    """
    loss = history.history['loss']
    epochs_range = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, loss, label='Training Loss', marker='o', linestyle='-', color='orange')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.show()


# ##  Evaluation

# In[208]:


# src/training/utils.py

import os
import json
import tensorflow as tf

def save_training_artifacts(model, history, config, save_dir, weights_path):
    """
    Save model weights, training history, and config.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save history
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history.history, f)

    # Save model weights
    model.save_weights(weights_path)

    # Save training config
    with open(os.path.join(save_dir, "config_train.json"), "w") as f:
        json.dump(config, f, indent=4)


# In[232]:


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


# In[246]:


# src/data/save_splits.py
import os


def save_splits(df, train_df, val_df, test_df, data_dir):
    os.makedirs(data_dir, exist_ok=True)

    df.to_csv(os.path.join(data_dir, "df.csv"), index=False)
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)


# In[195]:


# src/models/inference.py
import json
# from src.models.model import get_inference_model


def load_inference_model(config_path, weights_path):
    with open(config_path) as f:
        config = json.load(f)

    model = get_inference_model(config_path)
    model.load_weights(weights_path)

    return model, config


# In[240]:


# src/evaluation/utils.py
from collections import defaultdict


def build_image_caption_map(all_df, subset_df):
    """
    Map image_id -> list of ground truth captions.
    """
    image_caption_map = defaultdict(list)
    image_ids = set(subset_df["image"].unique())

    for _, row in all_df.iterrows():
        if row["image"] in image_ids:
            caption = (row["caption"].replace("startseq ", "").replace(" endseq", "").strip())
            image_caption_map[row["image"]].append(caption)

    return dict(image_caption_map)


# In[242]:


# src/evaluation/bleu.py

import os
import numpy as np
from tqdm import tqdm
# from src.inference.generate import generate_caption
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(model, tokenizer, image_caption_map):
    """
    Compute BLEU-1 to BLEU-4 scores.
    """
    bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores = [], [], [], []
    smooth_fn = SmoothingFunction().method1

    for image_id in tqdm(image_caption_map.keys(), desc="Calculating BLEU Scores", unit="image"):
        image_path = os.path.join(img_dir, image_id)

        predicted_caption = generate_caption(image_path, model, tokenizer, model_config["SEQ_LENGTH"], False).strip()
        predicted_caption = predicted_caption.split()

        reference_captions = [caption.split() for caption in image_caption_map[image_id]]

        bleu_1 = sentence_bleu(reference_captions, predicted_caption, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
        bleu_2 = sentence_bleu(reference_captions, predicted_caption, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
        bleu_3 = sentence_bleu(reference_captions, predicted_caption, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn)
        bleu_4 = sentence_bleu(reference_captions, predicted_caption, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)

        bleu_1_scores.append(bleu_1)
        bleu_2_scores.append(bleu_2)
        bleu_3_scores.append(bleu_3)
        bleu_4_scores.append(bleu_4)

    return [np.mean(bleu_1_scores), np.mean(bleu_2_scores), np.mean(bleu_3_scores), np.mean(bleu_4_scores)]



# In[192]:


config_train = {
    "IMAGE_SIZE": IMAGE_SIZE,
    "SEQ_LENGTH": SEQ_LENGTH,
    "EMBED_DIM": EMBED_DIM,
    "NUM_HEADS": NUM_HEADS,
    "FF_DIM": FF_DIM,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "VOCAB_SIZE": VOCAB_SIZE,
}

save_training_artifacts(caption_model, history, config_train, save_dir=save_path, weights_path=weights_path)

save_tokenizer(vectorization, save_path)
save_splits(train_df, valid_df, test_df, data_dir)


# In[ ]:





# ## Start

# In[176]:


df = pd.read_csv(caption_dir)

print(f"Total Number of images {len(df)//5} and total number of captions {len(df)}")

df.head()


# In[177]:


df['caption'] = df['caption'].apply(preprocess_caption)
df.head()


# In[178]:


df = df.head(1000)


# In[179]:


MAX_LEN = max(df['caption'].apply(lambda cap: len(cap.split())))
print("Maximum caption length:", MAX_LEN)


# In[180]:


train_df, valid_df, test_df = dataset_split(df, dataset_len=len(df))
print("Number of training samples: ", len(train_df))
print("Number of validation samples: ", len(valid_df))
print("Number of test samples: ", len(test_df))


# In[181]:


# caption_lengths = train_df["caption"].apply(lambda x: len(x.split()))
# SEQ_LENGTH = int(caption_lengths.quantile(0.95))
# print(caption_lengths.describe())



unique_words = set(" ".join(train_df["caption"]).split())
VOCAB_SIZE = len(unique_words)

SEQ_LENGTH = MAX_LEN
print(VOCAB_SIZE, SEQ_LENGTH)


# In[182]:


vectorizer = build_text_vectorizer(train_df["caption"], VOCAB_SIZE, SEQ_LENGTH)


# In[183]:


vocab = vectorizer.get_vocabulary()

word_to_index = {word: idx for idx, word in enumerate(vocab)}
print("Word to Index Mapping (First 5 words):", list(word_to_index.items())[:5])


# In[184]:


# src/train.py

# from src.data.dataset import make_dataset
# from src.train.setup import build_caption_model

BATCH_SIZE = 32
IMAGE_SIZE = (299, 299)
EPOCHS = 5

train_dataset = make_dataset(train_df, img_dir, vectorizer, IMAGE_SIZE, BATCH_SIZE,shuffle=True)
val_dataset = make_dataset(valid_df, img_dir, vectorizer, IMAGE_SIZE, BATCH_SIZE, shuffle=False)
test_dataset = make_dataset(test_df, img_dir, vectorizer, IMAGE_SIZE, BATCH_SIZE, shuffle=False)

model = build_caption_model(vocab_size=VOCAB_SIZE, seq_length=SEQ_LENGTH)

batch = next(iter(train_dataset))
print(batch[0].shape, batch[1].shape)


# In[185]:


loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="none")
optimizer = build_optimizer(train_dataset, epochs=EPOCHS)

model.compile(optimizer=optimizer, loss=loss)


# In[186]:


history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)


# In[205]:


EMBED_DIM = 512
FF_DIM = 512
NUM_HEADS = 6


# In[209]:


# from src.training.utils import save_training_artifacts

save_dir = os.path.join(base_dir, "artifacts")
weights_path = os.path.join(save_dir, "transformer_weights.h5")

config = {
    "IMAGE_SIZE": IMAGE_SIZE,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "SEQ_LENGTH": SEQ_LENGTH,
    "VOCAB_SIZE": VOCAB_SIZE,
    "EMBED_DIM": EMBED_DIM,
    "FF_DIM": FF_DIM,
    "NUM_HEADS": NUM_HEADS,
}

save_training_artifacts(model=model, history=history, config=config, save_dir=save_dir, weights_path=weights_path)


# In[210]:


save_tokenizer(vectorizer, save_dir)


# In[248]:


data_dir = os.path.join(save_dir, "data")
save_splits(df, train_df, valid_df, test_df, data_dir)


# In[ ]:





# In[234]:


loaded_tokenizer = load_tokenizer(save_dir)


# In[225]:


get_model_config_path = os.path.join(save_dir, 'config_train.json')
get_model_weights_path = os.path.join(save_dir, 'transformer_weights.h5')


# In[236]:


loaded_model, loaded_config = load_inference_model(get_model_config_path, get_model_weights_path)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[250]:


df = pd.read_csv(os.path.join(data_dir, 'df.csv'))
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))


train_image_caption_map = build_image_caption_map(df, train_df)
test_image_caption_map = build_image_caption_map(df, test_df)


# In[252]:


bleu_score = calculate_bleu(model, tokenizer, train_image_caption_map)
print(f"Average BLEU-1 Score on Train Dataset: {bleu_score[0]:.4f}")
print(f"Average BLEU-2 Score on Train Dataset: {bleu_score[1]:.4f}")
print(f"Average BLEU-3 Score on Train Dataset: {bleu_score[2]:.4f}")
print(f"Average BLEU-4 Score on Train Dataset: {bleu_score[3]:.4f}")


# In[253]:


bleu_score = calculate_bleu(model, tokenizer, test_image_caption_map)
print(f"Average BLEU-1 Score on Test Dataset: {bleu_score[0]:.4f}")
print(f"Average BLEU-2 Score on Test Dataset: {bleu_score[1]:.4f}")  
print(f"Average BLEU-3 Score on Test Dataset: {bleu_score[2]:.4f}")
print(f"Average BLEU-4 Score on Test Dataset: {bleu_score[3]:.4f}")


# In[ ]:





# In[ ]:




