# src/training/utils.py

import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt

def save_training_artifacts(model, history, config, save_dir, weights_path):
    """
    Save model weights, training history, and config.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save history
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history.history, f)

    # Save model weights
    model.save_weights(weights_path)

    # Save training config
    with open(os.path.join(save_dir, "config_train.json"), "w") as f:
        json.dump(config, f, indent=4)


def plot_history(history):
    """
    Plot training loss over epochs.
    """
    loss = history.history['loss']
    epochs_range = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, loss, label='Training Loss', marker='o', linestyle='-', color='orange')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.show()
