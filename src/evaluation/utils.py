# src/evaluation/utils.py
from collections import defaultdict

def build_image_caption_map(all_df, subset_df):
    """
    Map image_id -> list of ground truth captions.
    """
    image_caption_map = defaultdict(list)
    image_ids = set(subset_df["image"].unique())

    for _, row in all_df.iterrows():
        if row["image"] in image_ids:
            caption = (row["caption"].replace("startseq ", "").replace(" endseq", "").strip())
            image_caption_map[row["image"]].append(caption)

    return dict(image_caption_map)