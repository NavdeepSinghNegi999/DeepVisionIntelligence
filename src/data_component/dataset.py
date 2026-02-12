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
