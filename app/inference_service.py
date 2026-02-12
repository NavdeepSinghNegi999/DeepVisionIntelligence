import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import yaml
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.inference.generate import generate_caption
from src.inference.load_model import load_inference_model
from src.data_component.tokenizer_utils import load_tokenizer


# =========================
# LOAD EVERYTHING ONLY ONCE (Important for FastAPI)
# =========================

with open(os.path.join(ROOT_DIR, "config.yaml"), "r") as f:
    CONFIG = yaml.safe_load(f)

IMAGE_SIZE = tuple(CONFIG["data"]["image_size"])

ARTIFACTS_DIR = os.path.join(ROOT_DIR, CONFIG["paths"]["artifacts"])
CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "config_train.json")
WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "transformer_weights.h5")

# Load model & tokenizer once (very important)
model, model_config = load_inference_model(config_path=CONFIG_PATH, weights_path=WEIGHTS_PATH, image_size=IMAGE_SIZE)

tokenizer = load_tokenizer(ARTIFACTS_DIR)

# =========================
# FUNCTION YOU NEED
# =========================
def generate_image_caption(image_path: str) -> str:
    """
    Receives image path and returns generated caption
    """

    caption = generate_caption(
        image_path=image_path,
        caption_model=model,
        tokenizer=tokenizer,
        seq_length=model_config["SEQ_LENGTH"],
        image_size=IMAGE_SIZE,
        show_image=False,  # important for API use
    )

    return caption
