import os
import pandas as pd
from src.evaluation.utils import build_image_caption_map
from src.evaluation.bleu import calculate_bleu
from src.inference.load_model import load_inference_model
from src.data.tokenizer_utils import load_tokenizer

ROOT_DIR = "./"
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
DATA_DIR = os.path.join(ROOT_DIR,"data/processed")
IMG_DIR = os.path.join(ROOT_DIR, "data/raw/Images")

model, config = load_inference_model(
    os.path.join(ARTIFACTS_DIR, "config_train.json"),
    os.path.join(ARTIFACTS_DIR, "transformer_weights.h5"),
)

tokenizer = load_tokenizer(ARTIFACTS_DIR)

df = pd.read_csv(os.path.join(DATA_DIR, "df.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

image_caption_map = build_image_caption_map(df, test_df)

bleu_scores = calculate_bleu(model, tokenizer, image_caption_map, IMG_DIR, seq_len=config["SEQ_LENGTH"])

print("BLEU Scores:")
print("BLEU-1:", bleu_scores[0])
print("BLEU-2:", bleu_scores[1])
print("BLEU-3:", bleu_scores[2])
print("BLEU-4:", bleu_scores[3])
