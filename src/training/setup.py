# src/train/setup.py
 
from src.data_component.augmentation import get_image_augmentation
from src.models.cnn import get_cnn_model
from src.models.transformer import (TransformerEncoderBlock, TransformerDecoderBlock)
from src.models.caption_model import ImageCaptioningModel
from tensorflow import keras
import tensorflow as tf


def build_caption_model(vocab_size, seq_length, embed_dim=512, ff_dim=512, num_heads=6, image_size=(299, 299)):
    """
    Build and return the image captioning model.
    """
    cnn_model = get_cnn_model(image_size)

    encoder = TransformerEncoderBlock(embed_dim=embed_dim, dense_dim=ff_dim, num_heads=num_heads)

    decoder = TransformerDecoderBlock(embed_dim=embed_dim, ff_dim=ff_dim, num_heads=num_heads, seq_len=seq_length, vocab_size=vocab_size)

    return ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder,decoder=decoder,image_aug=get_image_augmentation())

class WarmupLearningRateSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup learning rate schedule.

    The learning rate increases linearly for `warmup_steps`,
    then stays constant at `base_lr`.
    """

    def __init__(self, base_lr, warmup_steps):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)

        warmup_lr = self.base_lr * (step / warmup_steps)
        return tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: self.base_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
        }


def build_optimizer(train_dataset, epochs: int, base_lr: float = 1e-4, warmup_ratio: int = 15):
    """
    Build Adam optimizer with warmup learning rate schedule.

    Args:
        train_dataset (tf.data.Dataset): Training dataset.
        epochs (int): Total number of epochs.
        base_lr (float): Target learning rate after warmup.
        warmup_ratio (int): Warmup steps = total_steps / warmup_ratio.

    Returns:
        tf.keras.optimizers.Optimizer
    """
    total_steps = len(train_dataset) * epochs
    warmup_steps = total_steps // warmup_ratio

    lr_schedule = WarmupLearningRateSchedule(base_lr=base_lr,warmup_steps=warmup_steps)

    return keras.optimizers.Adam(learning_rate=lr_schedule)



def get_loss():
    """
    Return loss function for image captioning.

    Uses SparseCategoricalCrossentropy because:
    - captions are integer token IDs
    - padding tokens are masked manually in the model
    """
    return keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="none")
