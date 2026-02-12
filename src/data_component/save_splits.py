# src/data/save_splits.py
import os


def save_splits(df, train_df, val_df, test_df, data_dir):
    os.makedirs(data_dir, exist_ok=True)

    df.to_csv(os.path.join(data_dir, "df.csv"), index=False)
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
