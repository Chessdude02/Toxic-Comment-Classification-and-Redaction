# Models

This directory contains all scripts related to training, analyzing, evaluating, and using the toxic comment classification model.

## Folder Structure

```
models/
├── training/          # Model architecture and training scripts
├── analysis/          # Overfitting analysis and diagnosis tools
├── evaluation/        # Model testing and performance evaluation
└── inference/         # Scripts for using the trained model
```

---

## training/

Scripts for building and training the model with overfitting prevention.

| File | Description |
|------|-------------|
| `fix_overfitted_model.py` | Full retraining pipeline with regularization, early stopping, and validation monitoring |
| `fixed_training_code.py` | Corrected training procedure — shows the key changes needed to prevent overfitting |
| `improved_model_template.py` | Reusable model architecture template with dropout, L2 regularization, and anti-overfitting callbacks |

---

## analysis/

Tools to detect and understand overfitting in the model.

| File | Description |
|------|-------------|
| `overfitting_analysis.py` | Comprehensive `OverfittingAnalyzer` class — checks model complexity, training history, data leakage, and performance gaps |
| `overfitting_diagnosis.py` | Diagnoses the current saved model and generates a prioritized action plan |
| `real_data_overfitting_analysis.py` | Analysis specific to the Jigsaw dataset — evaluates the 90M-parameter SimpleRNN model's overfitting |
| `check_training_curves.py` | Quick visual check of training vs. validation loss/accuracy curves |
| `quick_overfitting_demo.py` | Loads the saved model and demonstrates overfitting with concrete prediction examples |
| `quick_overfitting_check.py` | Minimal helper script for a fast overfitting check after training |

---

## evaluation/

Comprehensive test suites for measuring model performance and generalization.

| File | Description |
|------|-------------|
| `model_performance_test.py` | `ModelTester` class — runs basic, edge case, robustness, adversarial, calibration, and length-sensitivity tests |
| `test_fixed_model.py` | Tests the retrained fixed model and compares it against the original overfitted model |

---

## inference/

Ready-to-use interface for running the trained model.

| File | Description |
|------|-------------|
| `use_fixed_model.py` | `ToxicityClassifier` class with `predict()` and `batch_predict()` methods for classifying new text |

---

## Quick Start

### 1. Retrain the model (fixes overfitting)
```bash
python models/training/fix_overfitted_model.py
```

### 2. Analyze an existing model for overfitting
```bash
python models/analysis/overfitting_diagnosis.py
```

### 3. Evaluate model performance
```bash
python models/evaluation/model_performance_test.py
```

### 4. Classify new text
```python
from models.inference.use_fixed_model import ToxicityClassifier

classifier = ToxicityClassifier()
result = classifier.predict("Your text here")
print(result['classification'], result['probability'])
```
