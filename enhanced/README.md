# Enhanced Layer — Toxic Comment Classification & Redaction (Rebuild)

This folder is a **separate, later rebuild** of the classifier + redaction system that lives at the
repo root. It is not a continuation of the root notebook's Transformer model — it's an independent,
smaller BiLSTM pipeline built from scratch to fix specific, concrete problems found in an earlier
version of this project (severe overfitting, no working redaction module, an undebiased training set).
Read this before assuming it's a drop-in upgrade to the root system: different architecture
(BiLSTM vs. Transformer), different framework usage patterns, different dataset handling, and it
ships its own training/evaluation scripts rather than one notebook.

## What's genuinely here

- **A real overfitting diagnosis and fix.** An earlier version of this model was a 90M+ parameter
  RNN trained with no dropout and no monitored validation split. `analysis/` contains the tooling
  that diagnosed this (`OverfittingAnalyzer`, training-curve checks); `training/fix_overfitted_model.py`
  is the corrected retraining pipeline (regularization, class weighting, early stopping, a
  validation-F1-tuned decision threshold instead of a flat 0.5).
- **A real held-out evaluation.** `evaluation/evaluate_on_kaggle_test.py` scores the trained model
  against the Jigsaw competition's **official, separately-released test labels** (`test.csv` +
  `test_labels.csv`, dropping the `-1` rows Kaggle excluded from scoring) — not just hand-picked
  sanity-check sentences.
- **A documented bias fix.** `common/data.py::debias_context_traps` oversamples non-toxic,
  idiomatic uses of common trigger words ("you *killed* it out there", "I *hate* Mondays") because
  the Jigsaw corpus is skewed toward literal usage, which otherwise causes a keyword-correlation
  false-positive pattern — a known, documented issue with Jigsaw-trained toxicity models (it's the
  subject of the follow-up "Jigsaw Unintended Bias" Kaggle competition).
- **A working redaction module.** `redaction/redact.py` — lexicon-based span masking, gated by the
  trained classifier so a comment isn't redacted just because it contains a flagged word; the
  classifier has to agree the comment is actually toxic first.
- **A real generated chart.** `saved_models/training_curves.png` is an actual matplotlib output from
  a training run (loss/accuracy curves + the train/val loss gap), not a placeholder image.

## Verified results (real weights shipped)

Unlike most of this repository's other performance claims, these numbers were reproduced end to
end as part of this restructure — not carried over from an earlier, unverified pass. Training and
evaluation were both run against the genuine Kaggle files (`train.csv`, `test.csv`,
`test_labels.csv`), not the synthetic fallback:

| Metric | Value |
|---|---|
| Training data | 48,480 real Jigsaw comments (stratified 60K sample of the 159,571-row training set, 80/20 train/val split) |
| Architecture | BiLSTM (embedding 100d → SpatialDropout1D → Bidirectional LSTM(64) → Dense(32) → Dense(1)), 2,088,641 params |
| Training | Early-stopped at epoch 19/40 (patience 6, monitor `val_loss`), best weights restored from epoch 13 |
| Decision threshold | 0.838 (validation-F1-tuned, vs. F1 0.741 at a flat 0.5) |
| **Held-out test accuracy** (official 63,978-row Kaggle test set) | **90.86%** |
| **Held-out test AUC** | **0.9377** |
| Toxic-class precision / recall / F1 | 0.51 / 0.79 / 0.62 |
| Clean-class precision / recall / F1 | 0.98 / 0.92 / 0.95 |
| Confusion matrix [TN, FP / FN, TP] | [53287, 4601 / 1249, 4841] |

The trained model, tokenizer, and config (`saved_models/toxicity_model.keras`, `tokenizer.pickle`,
`config.pickle`, ~29MB total) are checked into this repo specifically so `enhanced/inference/`
and `enhanced/redaction/` work immediately without a training step — reproduce or retrain with the
commands below if you want to verify this yourself or improve on it. As with any Jigsaw-trained
toxicity model, precision on the toxic class (0.51) is modest relative to recall (0.79) — this
reflects a real trade-off (see `common/data.py`'s debiasing docstring), not a bug: the decision
threshold favors catching more toxic content at the cost of more false positives, which is a
defensible choice for a moderation tool but worth knowing before quoting "90.86% accuracy" as the
whole story.

If you retrain without a real Jigsaw CSV available, the scripts fall back to a small templated
synthetic dataset (see `common/data.py`) purely so the pipeline still runs end-to-end — results from
that fallback are not representative of real-world performance and should never be quoted as such.

## Folder Structure

```
enhanced/
├── common/            # Shared data loading, model architecture, artifact paths
│   ├── data.py         # Real-CSV-or-synthetic-fallback loader + context-trap debiasing
│   ├── model.py        # BiLSTM architecture (AdamW, cosine-decay LR, label smoothing)
│   └── paths.py        # Shared artifact locations (saved_models/)
├── training/           # Model architecture and training scripts
├── analysis/           # Overfitting analysis and diagnosis tools
├── evaluation/         # Model testing and real held-out Kaggle evaluation
├── inference/          # Ready-to-use classifier for new text
├── redaction/          # Classifier-gated lexicon redaction
└── saved_models/       # Generated at training time (weights gitignored, training_curves.png kept)
```

---

## training/

| File | Description |
|------|-------------|
| `fix_overfitted_model.py` | Full retraining pipeline with regularization, early stopping, validation-F1 threshold tuning, and context-trap debiasing |
| `fixed_training_code.py` | Corrected training procedure — shows the key changes needed to prevent overfitting |
| `improved_model_template.py` | Reusable model architecture template with dropout, L2 regularization, and anti-overfitting callbacks |

## analysis/

| File | Description |
|------|-------------|
| `overfitting_analysis.py` | `OverfittingAnalyzer` class — checks model complexity, training history, data leakage, and performance gaps |
| `overfitting_diagnosis.py` | Diagnoses a saved model and generates a prioritized action plan |
| `real_data_overfitting_analysis.py` | Analysis specific to the Jigsaw dataset — evaluates the original 90M-parameter model's overfitting |
| `check_training_curves.py` | Quick visual check of training vs. validation loss/accuracy curves |
| `quick_overfitting_demo.py` | Loads a saved model and demonstrates overfitting with concrete prediction examples |
| `quick_overfitting_check.py` | Minimal helper script for a fast overfitting check after training |

## evaluation/

| File | Description |
|------|-------------|
| `evaluate_on_kaggle_test.py` | Scores the trained model against the real, officially-labeled Jigsaw test set |
| `model_performance_test.py` | `ModelTester` class — basic, edge case, robustness, adversarial, calibration, and length-sensitivity tests |
| `test_fixed_model.py` | Compares the retrained model against the original overfitted model |

## inference/

| File | Description |
|------|-------------|
| `use_fixed_model.py` | `ToxicityClassifier` class with `predict()` and `batch_predict()` methods |

## redaction/

| File | Description |
|------|-------------|
| `redact.py` | `ToxicRedactor` — lexicon-based span masking, gated by the trained classifier so non-toxic uses of a trigger word aren't redacted |

---

## Dependencies

Uses the same stack as the root project (`tensorflow`, `pandas`, `scikit-learn`, `matplotlib`) —
install with the root `requirements.txt`:

```bash
pip install -r ../requirements.txt
```

## Quick Start

All commands below assume you're running from inside `enhanced/`.

### 1. Train the model
Real data (recommended): download `train.csv` from the [Jigsaw Toxic Comment
Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
and either place it in this directory or point `JIGSAW_TRAIN_CSV` at it. Without real data, the
script falls back to a small templated synthetic dataset so it still runs end-to-end (see caveat above).

```bash
export JIGSAW_TRAIN_CSV=/path/to/train.csv
python training/fix_overfitted_model.py
```

### 2. Evaluate on the real, held-out Kaggle test set

```bash
python evaluation/evaluate_on_kaggle_test.py \
    --test-csv /path/to/test.csv --labels-csv /path/to/test_labels.csv
```

### 3. Run the broader test suite (edge cases, robustness, calibration)

```bash
python evaluation/model_performance_test.py
```

### 4. Classify new text

```python
from inference.use_fixed_model import ToxicityClassifier

classifier = ToxicityClassifier()
result = classifier.predict("Your text here")
print(result['classification'], result['probability'])
```

### 5. Redact toxic comments

```python
from redaction.redact import ToxicRedactor

redactor = ToxicRedactor()
result = redactor.redact("You are an idiot and should shut up")
print(result['redacted'])  # "You are an [redacted] and should [redacted]"
```

---

## How this relates to the root project

- Root (`../`) — the original BISAG-N internship deliverable: a custom Transformer classifier
  (see `../Industrial_Grade_Toxic_Comment_Classifier.ipynb`), a Flask redaction web app/API
  (`../src/`), an overfitting-diagnosis toolkit for that model (`../models/`), and a rule-based
  heuristic add-on (`../upgrade2/`). See `../docs/Classifying_Toxic_Comments_Using_Deep_Learning.pdf`
  for the associated research paper.
- `enhanced/` (this folder) — a from-scratch, independent redo focused specifically on getting one
  model (BiLSTM) properly regularized, debiased, and evaluated against real held-out labels end to
  end, plus a genuinely working redaction module.

They are not meant to be imported into each other or run as one pipeline — treat them as two
separate, standalone systems documented in one repository.
