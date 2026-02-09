# src/models/inference.py
import json
import tensorflow as tf
from src.models.cnn import get_cnn_model
from src.models.transformer import TransformerEncoderBlock, TransformerDecoderBlock
from src.models.caption_model import ImageCaptioningModel

def get_inference_model(config_path, image_size=(299, 299)):
    """
    Load image captioning model for inference from config.
    """
    with open(config_path) as f:
        cfg = json.load(f)

    cnn = get_cnn_model(image_size)
    encoder = TransformerEncoderBlock(cfg["EMBED_DIM"], cfg["FF_DIM"], cfg["NUM_HEADS"])
    decoder = TransformerDecoderBlock(cfg["EMBED_DIM"], cfg["FF_DIM"], cfg["NUM_HEADS"], cfg["SEQ_LENGTH"], cfg["VOCAB_SIZE"])

    model = ImageCaptioningModel(cnn, encoder, decoder)

    img_input = tf.keras.Input(shape=(*image_size, 3))
    cap_input = tf.keras.Input(shape=(None,))
    model([img_input, False, cap_input])

    return model

def load_inference_model(config_path, weights_path, image_size=(299, 299)):
    with open(config_path) as f:
        config = json.load(f)

    model = get_inference_model(config_path, image_size=image_size)
    model.load_weights(weights_path)

    return model, config
