import os
import yaml
import pandas as pd
import tensorflow as tf

import mlflow
import mlflow.tensorflow

from src.data_component.preprocessing import preprocess_caption
from src.data_component.dataset_split import dataset_split
from src.data_component.vectorizer import build_text_vectorizer
from src.data_component.dataset import make_dataset
from src.data_component.save_splits import save_splits
from src.data_component.tokenizer_utils import save_tokenizer

from src.training.setup import (build_caption_model,build_optimizer, get_loss)
from src.training.utils import save_training_artifacts

from src.utils.logging_utils import setup_logger

from src.evaluation.utils import build_image_caption_map
from src.evaluation.bleu import calculate_bleu

def main():
    # =========================
    # INITIAL SETUP
    # =========================
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    logger = setup_logger(log_name="training")
    logger.info("Starting training pipeline")

    # =========================
    # MLFLOW SETUP
    # =========================
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("DeepVisionIntelligence")

    # =========================
    # LOAD CONFIG
    # =========================
    logger.info("Loading configuration file")
    
    with open(os.path.join(ROOT_DIR, "config.yaml"), "r") as f:
        CONFIG = yaml.safe_load(f)

    # =========================
    # PATHS
    # =========================

    CAPTION_PATH = os.path.join(ROOT_DIR, CONFIG["paths"]["captions"])
    IMG_DIR = os.path.join(ROOT_DIR, CONFIG["paths"]["images"])
    ARTIFACTS_DIR = os.path.join(ROOT_DIR, CONFIG["paths"]["artifacts"])
    WEIGHTS_PATH = os.path.join(ROOT_DIR, CONFIG["paths"]["weights"])
    PROCESSED_DIR = os.path.join(ROOT_DIR, CONFIG["paths"]["processed"])
    
    logger.info("Resolved project paths")

    # =========================
    # DATA CONFIG
    # =========================
    BATCH_SIZE = CONFIG["data"]["batch_size"]
    IMAGE_SIZE = tuple(CONFIG["data"]["image_size"])
    MAX_SAMPLES = CONFIG["data"]["max_samples"]

    # =========================
    # TRAINING CONFIG
    # =========================
    EPOCHS = CONFIG["training"]["epochs"]

    # =========================
    # MODEL CONFIG
    # =========================
    EMBED_DIM = CONFIG["model"]["embed_dim"]
    FF_DIM = CONFIG["model"]["ff_dim"]
    NUM_HEADS = CONFIG["model"]["num_heads"]

    logger.info(f"Training params | epochs={EPOCHS}, batch_size={BATCH_SIZE}, image_size={IMAGE_SIZE}")

    # =========================
    # 1. LOAD & PREPROCESS DATA
    # =========================
    df = pd.read_csv(CAPTION_PATH)
    df["caption"] = df["caption"].apply(preprocess_caption)
    df = df.head(MAX_SAMPLES)                        # remove when testing complete.
    
    logger.info(f"Dataset loaded with {len(df)} samples")

    # =========================
    # 2. SPLIT DATA
    # =========================
    logger.info("Splitting dataset into train/val/test")

    train_df, val_df, test_df = dataset_split(df)

    logger.info(f"Split sizes | train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # =========================
    # 3. TEXT VECTORIZATION
    # =========================
    logger.info("Building text vectorizer")
    SEQ_LENGTH = max(train_df["caption"].str.split().str.len())
    VOCAB_SIZE = len(set(" ".join(train_df["caption"]).split()))

    vectorizer = build_text_vectorizer(train_df["caption"], vocab_size=VOCAB_SIZE, seq_length=SEQ_LENGTH)
    logger.info(f"Vectorizer ready | vocab_size={VOCAB_SIZE}, seq_length={SEQ_LENGTH}" )

    # =========================
    # 4. DATASETS
    # =========================
    logger.info("Creating TensorFlow datasets")
    train_ds = make_dataset(train_df, IMG_DIR, vectorizer, IMAGE_SIZE, BATCH_SIZE)
    val_ds = make_dataset(val_df, IMG_DIR, vectorizer, IMAGE_SIZE, BATCH_SIZE, shuffle=False)

    # =========================
    # 5. MODEL
    # =========================
    logger.info("Building captioning model")
    model = build_caption_model(vocab_size=VOCAB_SIZE, seq_length=SEQ_LENGTH, embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS, image_size=IMAGE_SIZE)

    optimizer = build_optimizer(train_ds, EPOCHS)
    loss = get_loss()

    model.compile(optimizer=optimizer, loss=loss)
    logger.info("Model compiled successfully")

    # =========================
    # MLFLOW RUN
    # =========================
    with mlflow.start_run(run_name=f"epochs={EPOCHS}_bs={BATCH_SIZE}"):

        # -------------------------
        # LOG PARAMETERS
        # -------------------------
        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "ff_dim": FF_DIM,
            "num_heads": NUM_HEADS,
            "vocab_size": VOCAB_SIZE,
            "seq_length": SEQ_LENGTH,
            "image_size": str(IMAGE_SIZE),
            "max_samples": MAX_SAMPLES
        })

        # =========================
        # 6. TRAIN
        # =========================
        logger.info("Starting model training")
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
        logger.info("Model training completed")

        # -------------------------
        # LOG TRAIN METRICS
        # -------------------------
        for epoch in range(EPOCHS):
            mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)

        # =========================
        # 7. SAVE ARTIFACTS
        # =========================
        logger.info("Saving training artifacts")
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

        save_training_artifacts(model=model, history=history, config=config_train, save_dir=ARTIFACTS_DIR, weights_path=WEIGHTS_PATH)
        save_tokenizer(vectorizer, ARTIFACTS_DIR)
        save_splits(df, train_df, val_df, test_df, PROCESSED_DIR)
    
        mlflow.log_artifacts(ARTIFACTS_DIR)

        logger.info("Training pipeline completed successfully")
        print("Training completed successfully")
        


        # =========================
        # 8. EVALUATION (BLEU)
        # =========================
        logger.info("Starting model evaluation (BLEU scores)")
        
        image_caption_map = build_image_caption_map(df, test_df)

        bleu_scores = calculate_bleu(model=model, tokenizer=vectorizer, image_caption_map=image_caption_map, img_dir=IMG_DIR,seq_len=SEQ_LENGTH, image_size=IMAGE_SIZE)

        logger.info(
            f"BLEU scores | "
            f"BLEU-1={bleu_scores[0]:.4f}, "
            f"BLEU-2={bleu_scores[1]:.4f}, "
            f"BLEU-3={bleu_scores[2]:.4f}, "
            f"BLEU-4={bleu_scores[3]:.4f}"
        )
        mlflow.log_metrics({
            "bleu_1": bleu_scores[0],
            "bleu_2": bleu_scores[1],
            "bleu_3": bleu_scores[2],
            "bleu_4": bleu_scores[3],
        })

        print("BLEU Scores:", bleu_scores)
    logger.info("Training pipeline completed successfully...")


if __name__ == "__main__":
    main()
