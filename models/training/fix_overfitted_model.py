#!/usr/bin/env python3
"""
Complete Model Retraining Solution
=================================

Retrains the toxicity classifier with proper regularization, a real
train/validation split that's actually monitored, and a right-sized
architecture - fixing the original model's severe overfitting
(90M+ parameters, no dropout, no validation_data in model.fit()).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score
from tensorflow.keras.preprocessing import sequence, text
import warnings

from common.data import load_dataset, debias_context_traps
from common.model import build_model, get_callbacks, class_weights
from common.paths import MODEL_PATH, TOKENIZER_PATH, CONFIG_PATH, TRAINING_CURVES_PATH, ARTIFACTS_DIR

warnings.filterwarnings("ignore")

# Synthetic fallback data is short templated sentences - small vocab/len is
# plenty. Real Jigsaw comments run much longer, so both need to scale up.
# REAL_SAMPLE_CAP keeps CPU-only training tractable in this environment;
# raise/remove it if training on a machine with more compute/a GPU.
SYNTHETIC_MAX_VOCAB, SYNTHETIC_MAX_LEN = 10000, 64
REAL_MAX_VOCAB, REAL_MAX_LEN = 20000, 120
REAL_SAMPLE_CAP = 60000


def prepare_data(df, test_size=0.2, max_vocab=SYNTHETIC_MAX_VOCAB, max_len=SYNTHETIC_MAX_LEN):
    print("Preparing data...")

    df = df.copy()
    df["clean_comment"] = df["comment_text"].apply(lambda x: str(x).lower().strip())

    # Force plain numpy object/int arrays - pandas' default pyarrow-backed
    # string dtype breaks sklearn's fancy indexing in train_test_split.
    X = np.asarray(df["clean_comment"].tolist(), dtype=object)
    y = np.asarray(df["toxic"].tolist(), dtype=int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )

    print(f"Training: {len(X_train)} samples ({y_train.mean():.1%} toxic)")
    print(f"Validation: {len(X_val)} samples ({y_val.mean():.1%} toxic)")

    tokenizer = text.Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)

    X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=max_len)
    X_val_pad = sequence.pad_sequences(X_val_seq, maxlen=max_len)

    # `num_words` only limits which words texts_to_sequences() uses - it
    # doesn't shrink tokenizer.word_index itself, which keeps every word
    # seen during fit_on_texts. Sizing the embedding layer off the raw
    # word_index (as the original script did) wastes a lot of memory on
    # embedding rows for words that are never actually fed to the model.
    vocab_size = min(len(tokenizer.word_index) + 1, max_vocab)
    print(f"Vocabulary size: {vocab_size:,} (capped at {max_vocab:,}), sequence length: {max_len}")

    return X_train_pad, X_val_pad, y_train, y_val, tokenizer, vocab_size


def train_model(X_train, X_val, y_train, y_val, vocab_size, max_len, epochs=40, batch_size=32):
    print("\nTRAINING")
    print("=" * 30)

    model = build_model(vocab_size, max_len)
    model.summary()

    callbacks = get_callbacks(ARTIFACTS_DIR / "checkpoint_best.keras")
    weights = class_weights(y_train)
    print(f"Class weights: {weights}")
    print(f"Batch size: {batch_size}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=weights,
        verbose=2,
        shuffle=True,
    )
    return model, history


def analyze_training_results(history):
    print("\nTRAINING RESULTS ANALYSIS")
    print("=" * 35)

    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]

    loss_gap = final_val_loss - final_train_loss
    acc_gap = final_train_acc - final_val_acc

    print(f"Final training loss:   {final_train_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}  (gap {loss_gap:+.4f})")
    print(f"Final training acc:    {final_train_acc:.4f}")
    print(f"Final validation acc:  {final_val_acc:.4f}  (gap {acc_gap:+.4f})")

    overfitting = abs(loss_gap) > 0.15 or acc_gap > 0.1
    print("Overfitting detected!" if overfitting else "No significant overfitting detected.")

    epochs = range(1, len(history.history["loss"]) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history.history["loss"], "b-", label="Training Loss")
    plt.plot(epochs, history.history["val_loss"], "r-", label="Validation Loss")
    plt.title("Loss Curves"); plt.xlabel("Epoch"); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history.history["accuracy"], "b-", label="Training Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], "r-", label="Validation Accuracy")
    plt.title("Accuracy Curves"); plt.xlabel("Epoch"); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1, 3, 3)
    gaps = np.array(history.history["val_loss"]) - np.array(history.history["loss"])
    plt.plot(epochs, gaps, "orange")
    plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    plt.title("Validation - Training Loss Gap"); plt.xlabel("Epoch"); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(TRAINING_CURVES_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    return not overfitting


def test_fixed_model(model, tokenizer, max_len, threshold=0.5):
    print("\nTESTING FIXED MODEL")
    print("=" * 40)

    def predict_text(t):
        t = str(t).lower()
        seq = tokenizer.texts_to_sequences([t])
        padded = sequence.pad_sequences(seq, maxlen=max_len)
        return float(model.predict(padded, verbose=0)[0][0])

    test_cases = [
        ("Hello, how are you?", False, "Greeting"),
        ("Thank you for your help", False, "Gratitude"),
        ("I disagree with you", False, "Disagreement"),
        ("You are an idiot", True, "Clear insult"),
        ("Shut up, moron", True, "Rude command"),
        ("Go kill yourself", True, "Threat"),
        ("You're stupid", True, "Insult"),
        ("Fucking idiot", True, "Profanity + insult"),
        ("This is great!", False, "Positive"),
        ("Have a nice day", False, "Polite"),
        ("You killed it in that presentation", False, "Non-toxic 'killed'"),
        ("I hate brussels sprouts", False, "Non-toxic 'hate'"),
    ]

    correct = 0
    for txt, expected_toxic, description in test_cases:
        prob = predict_text(txt)
        predicted_toxic = prob > threshold
        is_correct = predicted_toxic == expected_toxic
        correct += is_correct
        status = "PASS" if is_correct else "FAIL"
        print(f"[{status}] {description:22} | '{txt[:35]:35}' -> {prob:.3f}")

    accuracy = correct / len(test_cases)
    print(f"\nTest accuracy: {correct}/{len(test_cases)} = {accuracy:.1%}")
    return accuracy >= 0.8


def save_artifacts(model, tokenizer, config):
    print("\nSAVING ARTIFACTS")
    print("=" * 35)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {TOKENIZER_PATH}")

    with open(CONFIG_PATH, "wb") as f:
        pickle.dump(config, f)
    print(f"Config saved to {CONFIG_PATH}")


def main():
    print("FIXING OVERFITTED MODEL - COMPLETE RETRAINING")
    print("=" * 55)

    df, source = load_dataset()

    if source == "real":
        max_vocab, max_len, batch_size = REAL_MAX_VOCAB, REAL_MAX_LEN, 256
        if len(df) > REAL_SAMPLE_CAP:
            print(f"Subsampling {len(df)} real rows down to {REAL_SAMPLE_CAP} "
                  f"(stratified) to keep CPU-only training tractable here.")
            frac = REAL_SAMPLE_CAP / len(df)
            parts = [group.sample(frac=frac, random_state=42) for _, group in df.groupby("toxic")]
            df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
        df = debias_context_traps(df)
    else:
        max_vocab, max_len, batch_size = SYNTHETIC_MAX_VOCAB, SYNTHETIC_MAX_LEN, 32

    X_train, X_val, y_train, y_val, tokenizer, vocab_size = prepare_data(
        df, max_vocab=max_vocab, max_len=max_len
    )

    model, history = train_model(X_train, X_val, y_train, y_val, vocab_size, max_len, batch_size=batch_size)

    training_ok = analyze_training_results(history)

    # Pick the decision threshold that maximizes F1 on the validation set,
    # instead of assuming 0.5 is right for an imbalanced dataset (~10% toxic).
    val_probs = model.predict(X_val, verbose=0).ravel()
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
    best_idx = int(np.argmax(f1s[:-1])) if len(thresholds) else 0
    threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
    print(f"\nBest validation-F1 threshold: {threshold:.3f} "
          f"(F1={f1s[best_idx]:.3f} vs {f1_score(y_val, val_probs > 0.5):.3f} at 0.5)")

    test_ok = test_fixed_model(model, tokenizer, max_len, threshold=threshold)

    config = {
        "vocab_size": vocab_size,
        "max_len": max_len,
        "threshold": threshold,
        "label_columns": ["toxic"],
        "dataset_source": source,
    }
    save_artifacts(model, tokenizer, config)

    print("\n" + "=" * 60)
    print("FINAL ASSESSMENT")
    print("=" * 60)
    if training_ok and test_ok:
        print("SUCCESS: model generalizes and passes the sanity-check test cases.")
    else:
        print("PARTIAL: model runs end-to-end but still has gaps - see results above.")

    return model, history


if __name__ == "__main__":
    model, history = main()
