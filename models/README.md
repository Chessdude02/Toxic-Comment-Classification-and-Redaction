# Overfitting Diagnosis & Model Improvement Toolkit

This folder documents diagnosing and fixing severe overfitting in an earlier RNN-based version of
the toxicity classifier (~96% train accuracy, poor generalization — no dropout, no monitored
validation split, an oversized architecture for the data available). It is diagnostic/reference
tooling, not the production classifier — the production model is the Transformer defined in
`../Industrial_Grade_Toxic_Comment_Classifier.ipynb`.

## Folder Structure

```
models/
├── training/     # Regularized model architectures + retraining scripts
├── analysis/     # Overfitting detection and diagnosis tools
├── evaluation/   # Performance/robustness test suites
└── inference/    # Minimal interface for using a fixed model
```

## training/

| File | Description |
|------|-------------|
| `fix_overfitted_model.py` | End-to-end retraining pipeline with regularization, class weighting, and monitored validation |
| `fixed_training_code.py` | `create_overfitting_resistant_model` — a right-sized, regularized architecture (dropout, L2, early stopping) |
| `improved_model_template.py` | Reusable template combining the above patterns |

## analysis/

| File | Description |
|------|-------------|
| `overfitting_analysis.py` | `OverfittingAnalyzer` — checks model complexity, training-history gaps, and data leakage |
| `overfitting_diagnosis.py` | Diagnoses a saved model and produces a prioritized action plan |
| `real_data_overfitting_analysis.py` | Analysis specific to a Jigsaw-derived dataset |
| `check_training_curves.py` | Quick visual check of training vs. validation curves |
| `quick_overfitting_demo.py` | Loads a saved model and demonstrates overfitting with concrete examples |

## evaluation/

| File | Description |
|------|-------------|
| `model_performance_test.py` | `ModelTester` — basic, edge-case, adversarial, calibration, and length-sensitivity tests |
| `test_fixed_model.py` | Compares a retrained model against the original overfitted one |

## inference/

| File | Description |
|------|-------------|
| `use_fixed_model.py` | Minimal `ToxicityClassifier`-style interface for a fixed/retrained model |

## Relationship to `enhanced/`

`enhanced/common/model.py` explicitly builds on and corrects the regularization approach used in
this folder's `fixed_training_code.py` (its docstring says as much) — so `enhanced/` isn't
unrelated background reading, it's a documented next iteration on the ideas diagnosed here, applied
to a different (BiLSTM) architecture.
