from sklearn.model_selection import train_test_split


def dataset_split(df, dataset_len=40000, min_token=5, max_token=25, shuffle=True, seed=42,):
    """
    Filter captions by length and split dataset into train/val/test.

    Returns:
        train_df, val_df, test_df
    """

    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df = df[df["caption"].apply(lambda cap: min_token <= len(cap.split()) <= max_token)].iloc[:dataset_len]

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed)

    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)

    return (train_df.reset_index(drop=True),val_df.reset_index(drop=True),test_df.reset_index(drop=True),)