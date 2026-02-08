# src/evaluation/bleu.py

import os
import numpy as np
from tqdm import tqdm
from src.inference.generate import generate_caption
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def calculate_bleu(model, tokenizer, image_caption_map, img_dir, seq_len):
    """
    Compute BLEU-1 to BLEU-4 scores.
    """
    bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores = [], [], [], []
    smooth_fn = SmoothingFunction().method1

    for image_id in tqdm(image_caption_map.keys(), desc="Calculating BLEU Scores", unit="image"):
        image_path = os.path.join(img_dir, image_id)

        predicted_caption = generate_caption(image_path, model, tokenizer, seq_len, False).strip()
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

