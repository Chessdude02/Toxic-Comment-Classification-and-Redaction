#!/usr/bin/env python3
"""
Quick Overfitting Check
========================

Minimal helper for a fast train/validation gap check after training.
Previously this file existed but was completely empty, even though
`fixed_training_code.py`'s example snippet calls `quick_overfitting_check(...)`
- so anyone following that snippet would get an ImportError/NameError.
"""


def quick_overfitting_check(history_dict, loss_gap_threshold=0.15, acc_gap_threshold=0.1):
    """
    Inspect a Keras `History.history` dict and print a fast overfitting verdict.

    Returns True if no significant overfitting was detected.
    """
    train_loss = history_dict["loss"][-1]
    val_loss = history_dict["val_loss"][-1]
    train_acc = history_dict["accuracy"][-1]
    val_acc = history_dict["val_accuracy"][-1]

    loss_gap = val_loss - train_loss
    acc_gap = train_acc - val_acc

    print("QUICK OVERFITTING CHECK")
    print("-" * 30)
    print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Gap: {loss_gap:+.4f}")
    print(f"Train acc:  {train_acc:.4f} | Val acc:  {val_acc:.4f} | Gap: {acc_gap:+.4f}")

    overfitting = loss_gap > loss_gap_threshold or acc_gap > acc_gap_threshold
    if overfitting:
        print("Overfitting detected: validation metrics trail training metrics by a lot.")
    else:
        print("No significant overfitting detected.")

    return not overfitting


if __name__ == "__main__":
    # Example history showing a healthy fit, for a quick sanity check of this module.
    example_history = {
        "loss": [0.65, 0.32, 0.24, 0.21],
        "val_loss": [0.55, 0.29, 0.23, 0.22],
        "accuracy": [0.66, 0.93, 0.98, 1.0],
        "val_accuracy": [0.75, 0.94, 0.97, 0.98],
    }
    quick_overfitting_check(example_history)
