#!/usr/bin/env python3
"""
Evaluate on the real Kaggle held-out test set
==============================================

The Jigsaw "Toxic Comment Classification Challenge" ships `test.csv` (unlabeled
comments) and `test_labels.csv` (labels released after the competition ended).
Rows with label -1 in test_labels.csv were excluded from scoring and must be
dropped. This script reports real, held-out metrics for the trained model -
not just the sanity-check sentences used elsewhere in this repo.

Usage:
    python evaluate_on_kaggle_test.py --test-csv path/to/test.csv \\
        --labels-csv path/to/test_labels.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
)

from common.paths import MODEL_PATH, TOKENIZER_PATH, CONFIG_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="test.csv")
    parser.add_argument("--labels-csv", default="test_labels.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    print("Loading model artifacts...")
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(CONFIG_PATH, "rb") as f:
        config = pickle.load(f)

    print(f"Loading {args.test_csv} / {args.labels_csv}...")
    test_df = pd.read_csv(args.test_csv)
    labels_df = pd.read_csv(args.labels_csv)

    df = test_df.merge(labels_df, on="id")
    before = len(df)
    df = df[df["toxic"] != -1].reset_index(drop=True)
    print(f"{len(df)}/{before} rows have real labels (rest were excluded from Kaggle scoring)")

    texts = df["comment_text"].astype(str).str.lower().str.strip()
    y_true = df["toxic"].to_numpy()

    seqs = tokenizer.texts_to_sequences(texts)
    padded = sequence.pad_sequences(seqs, maxlen=config["max_len"])

    print(f"Running inference on {len(padded)} held-out comments...")
    probs = model.predict(padded, batch_size=args.batch_size, verbose=1).ravel()
    y_pred = (probs > config["threshold"]).astype(int)

    print("\n" + "=" * 55)
    print("HELD-OUT TEST SET RESULTS (real Kaggle labels)")
    print("=" * 55)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"AUC:      {roc_auc_score(y_true, probs):.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["clean", "toxic"]))
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
