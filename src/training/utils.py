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

    # -------------------------
    # 1. Save training history
    # -------------------------
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history.history, f)

    # -------------------------
    # 2. Save training config
    # -------------------------
    with open(os.path.join(save_dir, "config_train.json"), "w") as f:
        json.dump(config, f, indent=4)

    # -------------------------
    # 3. Save weights only
    # -------------------------
    model.save_weights(weights_path)

    # -------------------------
    # Saves full model
    # -------------------------
    # model.save("transformer_model")

    # -------------------------
    # 4. Save full model (SavedModel) ✅ recommended
    # -------------------------
    # saved_model_dir = os.path.join(save_dir, "saved_model")
    # model.save(saved_model_dir, include_optimizer=False, save_format="tf")

 


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
