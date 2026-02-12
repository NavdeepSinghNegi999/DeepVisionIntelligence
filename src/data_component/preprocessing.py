# src/data/preprocessing.py

import re
import string
import contractions
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

def preprocess_caption(caption: str) -> str:
    """
    Preprocess a raw image caption for training an image captioning model.

    This function performs the following steps:
        1. Converts text to lowercase.
        2. Expands English contractions (e.g., "don't" -> "do not").
        3. Removes punctuation characters.
        4. Normalizes whitespace.
        5. Removes English stopwords and single-character tokens.
        6. Adds sequence boundary tokens ("startseq" and "endseq").

    Args:
        caption (str): Raw caption text associated with an image.

    Returns:
        str: Cleaned caption formatted as
             "startseq <processed caption> endseq".
    """

    caption = caption.lower()
    caption = contractions.fix(caption)

    caption = caption.translate(str.maketrans("", "", string.punctuation))
    caption = re.sub(r"\s+", " ", caption).strip()
    caption = " ".join(word for word in caption.split() if word not in STOP_WORDS and len(word) > 1)

    return f"startseq {caption} endseq"