import os
import json
from src.inference.load_model import load_inference_model
from src.data.tokenizer_utils import load_tokenizer
from src.inference.generate import generate_caption

ROOT_DIR = "./"
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "config_train.json")
WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "transformer_weights.h5")
IMAGE_PATH = "data/raw/Images/3637013_c675de7705.jpg"  

# Load model + config
model, config = load_inference_model(CONFIG_PATH, WEIGHTS_PATH)

# Load tokenizer
tokenizer = load_tokenizer(ARTIFACTS_DIR)

# Generate caption
caption = generate_caption(image_path=IMAGE_PATH, caption_model=model, tokenizer=tokenizer, seq_length=config["SEQ_LENGTH"], show_image=True)

print("Generated Caption:")
print(caption)
