import os
import yaml
import pandas as pd

from src.evaluation.utils import build_image_caption_map
from src.evaluation.bleu import calculate_bleu
from src.inference.load_model import load_inference_model
from src.data_component.tokenizer_utils import load_tokenizer
from src.utils.logging_utils import setup_logger


def main():
    # =========================
    # INITIAL SETUP
    # =========================
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    logger = setup_logger(log_name="evaluation")
    logger.info("Starting evaluation pipeline")

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
    DATA_DIR = os.path.join(ROOT_DIR, CONFIG["paths"]["processed"])
    IMG_DIR = os.path.join(ROOT_DIR, CONFIG["paths"]["images"])

    CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "config_train.json")
    WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "transformer_weights.h5")

    logger.info("Resolved evaluation paths")

    # =========================
    # LOAD MODEL & TOKENIZER
    # =========================
    logger.info("Loading trained model and tokenizer")

    model, model_config = load_inference_model(
        config_path=CONFIG_PATH,
        weights_path=WEIGHTS_PATH,
        image_size=IMAGE_SIZE,
    )

    tokenizer = load_tokenizer(ARTIFACTS_DIR)

    # =========================
    # LOAD DATA
    # =========================
    logger.info("Loading processed dataset")

    df = pd.read_csv(os.path.join(DATA_DIR, "df.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    logger.info(f"Test samples: {len(test_df)}")

    # =========================
    # BUILD CAPTION MAP
    # =========================
    logger.info("Building image - captions mapping")

    image_caption_map = build_image_caption_map(df, test_df)

    # =========================
    # EVALUATION (BLEU)
    # =========================
    logger.info("Computing BLEU scores")

    bleu_scores = calculate_bleu(
        model=model,
        tokenizer=tokenizer,
        image_caption_map=image_caption_map,
        img_dir=IMG_DIR,
        seq_len=model_config["SEQ_LENGTH"],
        image_size=IMAGE_SIZE,
    )

    # =========================
    # RESULTS
    # =========================
    logger.info(
        f"BLEU Scores | "
        f"BLEU-1={bleu_scores[0]:.4f}, "
        f"BLEU-2={bleu_scores[1]:.4f}, "
        f"BLEU-3={bleu_scores[2]:.4f}, "
        f"BLEU-4={bleu_scores[3]:.4f}"
    )

    print("\nBLEU Evaluation Results")
    print(f"BLEU-1: {bleu_scores[0]:.4f}")
    print(f"BLEU-2: {bleu_scores[1]:.4f}")
    print(f"BLEU-3: {bleu_scores[2]:.4f}")
    print(f"BLEU-4: {bleu_scores[3]:.4f}")

    logger.info("Evaluation pipeline completed successfully...")


if __name__ == "__main__":
    main()
