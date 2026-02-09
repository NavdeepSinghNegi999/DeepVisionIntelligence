import os
import yaml

from src.inference.load_model import load_inference_model
from src.inference.generate import generate_caption
from src.data.tokenizer_utils import load_tokenizer
from src.utils.logging_utils import setup_logger


def main():
    # =========================
    # INITIAL SETUP
    # =========================
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    logger = setup_logger(log_name="inference")
    logger.info("Starting inference pipeline")

    # =========================
    # LOAD CONFIG
    # =========================
    logger.info("Loading configuration file")

    with open(os.path.join(ROOT_DIR, "config.yaml"), "r") as f:
        CONFIG = yaml.safe_load(f)

    IMAGE_SIZE = tuple(CONFIG["data"]["image_size"])

    # =========================
    # PATHS
    # =========================
    ARTIFACTS_DIR = os.path.join(ROOT_DIR, CONFIG["paths"]["artifacts"])
    IMG_DIR = os.path.join(ROOT_DIR, CONFIG["paths"]["images"])

    CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "config_train.json")
    WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "transformer_weights.h5")

    # Example image (can be replaced with CLI / API input)
    IMAGE_PATH = os.path.join(IMG_DIR, "3637013_c675de7705.jpg")

    logger.info(f"Using image: {IMAGE_PATH}")

    # =========================
    # LOAD MODEL & TOKENIZER
    # =========================
    logger.info("Loading trained model")

    model, model_config = load_inference_model(
        config_path=CONFIG_PATH,
        weights_path=WEIGHTS_PATH,
        image_size=IMAGE_SIZE,
    )

    logger.info("Loading tokenizer")
    tokenizer = load_tokenizer(ARTIFACTS_DIR)

    # =========================
    # GENERATE CAPTION
    # =========================
    logger.info("Generating caption")

    caption = generate_caption(
        image_path=IMAGE_PATH,
        caption_model=model,
        tokenizer=tokenizer,
        seq_length=model_config["SEQ_LENGTH"],
        image_size=IMAGE_SIZE,
        show_image=True,
    )

    # =========================
    # OUTPUT
    # =========================
    logger.info("Inference completed successfully")

    print("\nGenerated Caption")
    print("-" * 30)
    print(caption)
    print("-" * 30)


if __name__ == "__main__":
    main()
