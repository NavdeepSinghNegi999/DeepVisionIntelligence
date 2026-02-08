import os
import yaml
import pandas as pd
from src.data.dataset import make_dataset
from src.data.save_splits import save_splits
from src.utils.logging_utils import setup_logger
from src.data.dataset_split import dataset_split
from src.data.tokenizer_utils import save_tokenizer
from src.data.preprocessing import preprocess_caption
from src.data.vectorizer import build_text_vectorizer
from src.training.utils import save_training_artifacts
from src.training.setup import build_caption_model, build_optimizer, get_loss

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

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
CKPT_DIR = os.path.join(ROOT_DIR, CONFIG["paths"]["checkpoint"])


# =========================
# DATA
# =========================

BATCH_SIZE = CONFIG["data"]["batch_size"]
IMAGE_SIZE = tuple(CONFIG["data"]["image_size"])
MAX_SAMPLES = CONFIG["data"]["max_samples"]

# =========================
# TRAINING
# =========================

EPOCHS = CONFIG["training"]["epochs"]

# =========================
# MODEL
# =========================

EMBED_DIM = CONFIG["model"]["embed_dim"]
FF_DIM = CONFIG["model"]["ff_dim"]
NUM_HEADS = CONFIG["model"]["num_heads"]









# logger = setup_logger()
# logger.info("Step 1: Loading captions file")
df = pd.read_csv(CAPTION_PATH)
df["caption"] = df["caption"].apply(preprocess_caption)
df = df.head(MAX_SAMPLES)

# logger.info("Step 1 Done : Loading captions file")

# 2. Split
train_df, val_df, test_df = dataset_split(df)

# 3. Vectorizer
SEQ_LENGTH = max(train_df["caption"].str.split().str.len())
VOCAB_SIZE = len(set(" ".join(train_df["caption"]).split()))
vectorizer = build_text_vectorizer(train_df["caption"], VOCAB_SIZE, SEQ_LENGTH)

# 4. Dataset
train_ds = make_dataset(train_df, IMG_DIR, vectorizer, IMAGE_SIZE, BATCH_SIZE)
val_ds = make_dataset(val_df, IMG_DIR, vectorizer, IMAGE_SIZE, BATCH_SIZE, shuffle=False)

# 5. Model
loss = get_loss()
model = build_caption_model(vocab_size=VOCAB_SIZE, seq_length=SEQ_LENGTH, embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS)

optimizer = build_optimizer(train_ds, EPOCHS)
model.compile(optimizer=optimizer, loss=loss)

# 6. Train
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)


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

# 7. Save
save_training_artifacts(model, history, config_train, ARTIFACTS_DIR, WEIGHTS_PATH)
save_tokenizer(vectorizer, ARTIFACTS_DIR)
save_splits(df, train_df, val_df, test_df, PROCESSED_DIR)



