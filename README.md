# 🛡️ Toxic Comment Classification & Real-Time Redaction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange?logo=tensorflow)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.2%2B-lightgrey?logo=flask)](https://flask.palletsprojects.com)
[![GloVe](https://img.shields.io/badge/Embeddings-GloVe%20840B%20300D-green)](https://nlp.stanford.edu/projects/glove/)
[![CI](https://img.shields.io/badge/CI-sanity--checks-brightgreen)](.github/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An end-to-end NLP system for toxic comment classification and intelligent real-time content redaction — built and deployed during an ML internship at **BISAG-N (Government of India)**.

The classification model in this repository's root layer is a custom **Transformer architecture**. The notebook's own printed output (below) claims 223,549 rows and AUC 0.9785 with GloVe 840B 300D — that specific run has not been independently reproduced here (840B 300D needs a ~2GB download this environment's network policy blocks). What's actually deployed and verified is documented in **[docs/root_transformer_training/](docs/root_transformer_training/)**: the same architecture, really trained end-to-end here for a full 50-epoch run with train/val curves checked for over- and underfitting, with a documented embeddings substitution (GloVe 6B 100D) and a smaller sample size — **90.01% accuracy / 0.9454 AUC** on the official held-out Kaggle test set, plus a known, verified limitation (no context-trap debiasing, unlike `enhanced/`'s BiLSTM). The system extends beyond classification into a full **redaction pipeline** with a REST API, Flask web interface, and context-aware semantic analysis layer.

This repository has **two layers**, developed at different times and kept separate rather than merged:

- **Root (this layer)** — the original internship deliverable described below: the Transformer classifier, the Flask redaction app/API, an overfitting-diagnosis toolkit (`models/`), and a rule-based heuristic add-on (`upgrade2/`).
- **[`enhanced/`](enhanced/)** — a later, independent rebuild: a separately trained/evaluated BiLSTM classifier with a documented overfitting fix, real held-out Kaggle-label evaluation, dataset debiasing, and a classifier-gated redaction module. It is not a drop-in replacement for the root system — see its own README for how the two relate.

> 📄 **Research Paper**: *Classifying Toxic Comments Using Deep Learning* (`docs/Classifying_Toxic_Comments_Using_Deep_Learning.pdf`) — a research paper in IEEE conference-paper format covering a literature review and a multi-model comparison (SimpleRNN → LSTM → GRU → BiLSTM → BERT), with its own reported result: a fine-tuned BERT model evaluated on a small held-out set (confusion matrix: 2,106 TP / 1,977 TN / 246 FP / 272 FN — 4,601 examples total).
>
> **This predates and differs from the notebook below.** The paper's own experiment used BERT on a much smaller evaluation set; the shipped notebook (`Industrial_Grade_Toxic_Comment_Classifier.ipynb`) is a later, larger-scale custom Transformer run on 223,549 rows. They are not the same experiment and their metrics are not directly comparable — the paper is included here for completeness and attribution, not as documentation of the notebook's own results.

---

## 📊 Model Performance (notebook's own printed output — unverified here)

These are the numbers `Industrial_Grade_Toxic_Comment_Classifier.ipynb` itself printed when it was
originally run (presumably on a GPU, elsewhere) — not independently reproduced as part of this
restructure. See the next section for the real, verified result from actually training this
architecture in this environment.

| Metric | Value |
|--------|-------|
| Dataset | Jigsaw Wikipedia Toxic Comments (223,549 samples) |
| Architecture | Custom Transformer + GloVe 840B 300D |
| Total Parameters | 18,021,377 |
| Test Accuracy | **89.49%** |
| Test AUC | **0.9785** |
| Toxic Recall | **95%** (catches 95% of toxic content) |
| Non-Toxic Precision | **99%** |
| Best Epoch | 3 of 18 (early stopping, patience=15) |

### Confusion Matrix (Test Set — 22,355 samples)

|  | Predicted Non-Toxic | Predicted Toxic |
|--|--|--|
| **Actual Non-Toxic** | TN = 17,977 | FP = 2,240 |
| **Actual Toxic** | FN = 110 | TP = 2,028 |

## ✅ Verified Results (what's actually deployed)

Trained and evaluated end to end as part of this restructure — see
[docs/root_transformer_training/](docs/root_transformer_training/) for the full methodology,
exactly what was substituted vs. the notebook's own config and why, and a real, verified
limitation before you rely on this for anything.

| Metric | Value |
|--------|-------|
| Dataset | 30,000-row stratified sample of the real 159,571-row Jigsaw training set |
| Embeddings | GloVe 6B 100D (substitute for 840B 300D — see docs/root_transformer_training/) |
| **Test Accuracy (official 63,978-row held-out Kaggle test set)** | **90.01%** |
| **Test AUC** | **0.9454** |
| Toxic precision / recall / F1 | 0.49 / 0.82 / 0.61 |
| Best epoch | 14 of 50 allotted (early stopping, patience=12) — trained/validated over the full 50-epoch budget and checked for over/underfitting; see docs/root_transformer_training/ for the curves |
| Known limitation | No context-trap debiasing — e.g. "I hate mondays" misclassifies as toxic (98.7% confidence). See docs/root_transformer_training/README.md for more examples and why. |

### Live Inference Examples

| Input | Prediction | Confidence |
|-------|-----------|------------|
| "I love this product, it's amazing!" | NON-TOXIC | 0.06% |
| "You are an idiot and should be ashamed." | TOXIC | 98.20% |
| "I respectfully disagree with your point." | NON-TOXIC | 0.05% |
| "Go away, nobody wants you here." | TOXIC | 85.37% |

---

## 🏗️ Architecture

### Classification Model

```
Input Text
    │
    ▼
GloVe 840B 300D Embeddings (300-dim, 50K vocab, 83.8% coverage)
    │
    ▼
[CLS] Token Prepended
    │
    ▼
Transformer Block × 3
    ├── Multi-Head Self-Attention (6 heads)
    ├── Feed-Forward Network (1024-dim, GELU)
    ├── Pre-Layer Normalization
    └── Residual Connections
    │
    ▼
CLS Token Pooling
    │
    ▼
Dense (128) → LayerNorm → GELU → Dropout
Dense (64)  → LayerNorm → GELU → Dropout
    │
    ▼
Sigmoid Output → Toxicity Probability
```

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (weight_decay=0.01, clipnorm=1.0) |
| Learning Rate | 1e-4 with warmup + cosine decay |
| Warmup Steps | 1,000 |
| Loss | Binary Crossentropy (label_smoothing=0.1) |
| Class Weights | Toxic: 5.227 / Non-Toxic: 0.553 |
| Batch Size | 64 |
| Max Epochs | 150 (early stopping at epoch 18) |
| Sequence Length | 128 tokens |

### Redaction System

```
Input Text
    │
    ├── Transformer Classifier ──► Toxicity Probability
    │
    ├── Semantic Analyzer ──────► Intent + Emotion + Context
    │         (context-aware: detects sarcasm, debate, group attack)
    │
    └── Smart Redactor ─────────► Redacted Output
              ├── Level: Minimal / Moderate / Aggressive / Complete
              └── Style: Asterisks / Blocks / Brackets / Euphemisms / Partial
```

---

## 📁 Repository Structure

```
├── Industrial_Grade_Toxic_Comment_Classifier.ipynb  # Main training notebook (Transformer + GloVe)
├── config.yaml                                       # Hyperparameter configuration
├── requirements.txt                                  # Python dependencies
├── RUN_INSTRUCTIONS.txt                              # Step-by-step execution guide
├── LICENSE                                           # MIT License
├── README.md                                         # This file
├── Dockerfile / .dockerignore                        # Container image for the Flask web app (runs via gunicorn)
├── render.yaml / railway.toml / Procfile             # One-click cloud deploy configs — see DEPLOYMENT.md
├── DEPLOYMENT.md                                     # Step-by-step cloud deployment instructions
├── .github/workflows/ci.yml                          # CI: byte-compiles sources, smoke-tests heuristic/demo code
│
├── src/                           # Redaction system (production-facing code)
│   ├── toxicity_redactor.py       # Core redaction module — loads a trained model
│   ├── transformer_layers.py      # Custom Keras layers (TransformerBlock, AttentionPooling,
│   │                               # PositionalEmbedding) registered for .keras serialization
│   ├── heuristic_fallback.py      # Rule-based stand-in used when no trained model is available
│   ├── toxicity_web_app.py        # Flask web application
│   ├── redaction_api.py           # REST API endpoints
│   ├── intelligent_redaction_system.py
│   ├── interactive_redaction_demo.py
│   ├── quick_demo.py
│   └── saved_models/               # demo_toxicity_classifier.keras — real, trained root Transformer (see below)
│
├── upgrade2/                      # Rule-based heuristic layer + experimental PyTorch BiLSTM
│   ├── vocabulary/                # Keyword/regex pattern matching (not a trained model)
│   ├── semantic/                  # Intent/emotion/context heuristics (not a trained model)
│   ├── integration/               # Combines the above into one scored result
│   ├── models/                    # enhanced_bilstm_model.py — a real, untrained PyTorch architecture
│   ├── redaction/                 # Standalone Flask demo for the heuristic layer
│   └── README.md                  # Full breakdown of what is/isn't ML in this folder
│
├── models/                        # Overfitting diagnosis & model improvement toolkit (for the root Transformer model)
│   ├── training/
│   │   ├── fix_overfitted_model.py
│   │   ├── fixed_training_code.py
│   │   └── improved_model_template.py
│   ├── analysis/
│   │   ├── overfitting_analysis.py
│   │   ├── overfitting_diagnosis.py
│   │   ├── real_data_overfitting_analysis.py
│   │   ├── check_training_curves.py
│   │   └── quick_overfitting_demo.py
│   ├── evaluation/
│   │   ├── model_performance_test.py
│   │   └── test_fixed_model.py
│   ├── inference/
│   │   └── use_fixed_model.py
│   └── README.md
│
├── datasets/                      # Dataset loading/generation helpers (no data checked in)
│   ├── dataset_manager.py
│   └── README_DATASETS.md
│
├── tests/                         # Manual smoke-test / unittest scripts
│   ├── test_module.py
│   └── test_redaction_system.py
│
├── docs/
│   ├── original_README.md
│   ├── Classifying_Toxic_Comments_Using_Deep_Learning.pdf   # Research paper (IEEE conference-paper format)
│   └── root_transformer_training/    # Real training run docs for the shipped root Transformer
│       ├── README.md                          # What was adapted, why, and the verified results
│       ├── train_transformer_cpu_adapted.py   # The actual script that produced the shipped model
│       └── official_test_metrics.json         # Full metrics from the official Kaggle test-set eval
│
└── enhanced/                      # SEPARATE LAYER — later, independent BiLSTM rebuild. See enhanced/README.md
    ├── common/ training/ analysis/ evaluation/ inference/ redaction/
    └── saved_models/              # training_curves.png (real); weights gitignored
```

**Note on models/ and saved_models/**: neither this repo nor `enhanced/` ships trained model weights, GloVe embeddings, or the Jigsaw dataset (all are gitignored — see `.gitignore`). Run the notebook to reproduce a trained model; `src/toxicity_redactor.py` and `src/intelligent_redaction_system.py` expect weights at `saved_models/` once you have them.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

### 2. Download GloVe Embeddings

```bash
# Download GloVe 840B 300D (~2.03 GB)
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
```

### 3. Download the Jigsaw Dataset

```bash
pip install kaggle
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
unzip jigsaw-toxic-comment-classification-challenge.zip
```

### 4. Run the Training Notebook

Open `Industrial_Grade_Toxic_Comment_Classifier.ipynb` in Jupyter or Kaggle and run all cells. The notebook handles all preprocessing, training, evaluation, and saving.

### 5. Launch the Web Application

```bash
cd src
python toxicity_web_app.py
# Open http://localhost:5000
```

Without a trained model in `saved_models/`, the app automatically falls back to a rule-based heuristic detector (`src/heuristic_fallback.py`, wrapping `upgrade2`'s keyword/pattern analyzer) instead of returning HTTP 503 — real, working predictions with no training required, clearly labeled as heuristic rather than a trained model. The running app shows a banner (and each API response includes a `mode` field: `trained` / `heuristic` / `unavailable`) so it's always clear which one is actually answering.

**This repo ships a trained model at `src/saved_models/` by default** so the app runs in `trained` mode out of the box — it's the root Transformer itself, really trained (see "✅ Verified Results" above and `docs/root_transformer_training/` for the full methodology and a documented false-positive limitation), not the notebook's own unreproduced 840B/300D run. `ToxicityRedactor` also works with the `enhanced/` BiLSTM (90.86% accuracy / 0.938 AUC, see `enhanced/README.md`) if you'd rather swap that in instead — the Transformer here scores 90.01% accuracy / 0.9454 AUC on the same official test set — it only needs a model, tokenizer, and a `label_columns` list, and the BiLSTM doesn't share this model's context-trap weakness. Both are real, both are documented; pick based on what you're optimizing for.

**Memory note**: loading a real model (either architecture) pulls in TensorFlow, which pushes RSS to ~718MB in gunicorn — confirmed by directly measuring it, not estimated. That's over the budget on 512MB free-tier hosts (Render, Railway, etc.); see `DEPLOYMENT.md` for what this means for cloud deployment.

### 6. (Alternative) Run the Web Application with Docker

```bash
docker build -t toxic-comment-redaction .
docker run -p 5000:5000 -v "$(pwd)/saved_models:/app/saved_models" toxic-comment-redaction
# Open http://localhost:5000
```

The image runs via gunicorn (not the Flask dev server) and falls back to the heuristic detector with no trained model mounted, same as above.

### 7. Deploy to the cloud

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for one-click steps on Render/Railway (`render.yaml` /
`railway.toml` are both in the repo root) and a generic path for any Dockerfile-based host.

---

## 🎨 Redaction System

The redaction pipeline extends the classifier into a deployable moderation tool.

### Redaction Levels

| Level | Description |
|-------|-------------|
| **Minimal** | Light censoring — preserves readability |
| **Moderate** | Standard redaction (default) |
| **Aggressive** | Heavy redaction for strict environments |
| **Complete** | Full message removal or replacement |

### Redaction Styles

| Style | Example |
|-------|---------|
| Asterisks | `f***ing` |
| Blocks | `████████` |
| Brackets | `[REDACTED]` |
| Euphemisms | `frick`, `darn` |
| Partial | `f**k` |

### Python Usage

```python
from src.toxicity_redactor import load_pretrained_model

redactor = load_pretrained_model()

# Single message
result = redactor.classify_toxicity("You are an idiot!")
print(result['is_toxic'], result['max_toxicity_probability'])

# Redact a message
redacted = redactor.redact_message("You're an idiot!", redaction_style="warning")
print(redacted['redacted_message'])
```

### REST API

```bash
# Check toxicity
curl -X POST http://localhost:5000/api/check-toxicity \
  -H "Content-Type: application/json" \
  -d '{"message": "Your text here", "redaction_style": "warning"}'

# Batch moderation
curl -X POST http://localhost:5000/api/moderate-batch \
  -H "Content-Type: application/json" \
  -d '{"messages": ["msg1", "msg2"], "style": "partial"}'
```

---

## 🧠 Semantic Analysis Layer

Beyond the neural classifier, the system includes a **rule-based** (keyword/regex heuristics, not a trained model) semantic analysis engine in `upgrade2/semantic/` that operates without requiring the trained model — useful as a fast, lightweight pre-filter or fallback. See [`upgrade2/README.md`](upgrade2/README.md) for the full honest breakdown of what is and isn't ML in that folder.

**Capabilities (all heuristic, via keyword/regex/punctuation pattern matching):**
- **Intent Detection**: direct_attack, derogatory_labeling, threat
- **Emotion Recognition**: anger, contempt, disgust, frustration
- **Context Modifiers**: sarcasm detection, debate recognition, generalization markers
- **Pattern Analysis**: excessive caps, punctuation patterns, rhetorical questions
- **Processing Speed**: < 0.01 seconds per comment

```python
from upgrade2.semantic.semantic_toxicity_analyzer import SemanticToxicityAnalyzer

analyzer = SemanticToxicityAnalyzer()
result = analyzer.analyze_semantic_toxicity("SHUT UP YOU MORON!!!")

print(result['semantic_toxicity_score'])            # e.g. 0.447
print(result['intent_analysis']['primary_intent'])  # e.g. direct_attack
print(result['emotional_analysis']['dominant_emotion'])  # e.g. anger
print(result['context_analysis']['primary_context'])     # e.g. general
```

---

## 🔬 Overfitting Diagnosis Toolkit (`models/`)

During development, a simpler RNN architecture showed severe overfitting (~96% train accuracy, poor generalization). The `models/` folder documents the full diagnosis and remediation process — a useful reference for ML practitioners dealing with the same problem.

- `OverfittingAnalyzer` class — checks model complexity, history gaps, data leakage
- Fixed architecture with proper regularization, class-weighted training, early stopping
- Comprehensive test suite: basic, edge case, adversarial, calibration, length-sensitivity tests

---

## 🚧 The `enhanced/` Layer

[`enhanced/`](enhanced/) is a **separate, later rebuild** of the classifier + redaction system —
a from-scratch BiLSTM pipeline (not the Transformer above) built to fix concrete problems found in
an earlier version of this project: severe overfitting, no working redaction module, and an
undebiased training set. It ships its own training/evaluation scripts, a documented bias fix for
Jigsaw's context-trap words, and real evaluation against the official held-out Kaggle test labels.
Read [`enhanced/README.md`](enhanced/README.md) before assuming it's a drop-in upgrade to the root
system — the two are independent and not meant to be run as one pipeline.

Unlike this repo's other numbers, `enhanced/`'s **90.86% accuracy / 0.938 AUC** on the official
63,978-row held-out Kaggle test set were reproduced end to end during this restructure (real
training data, real Kaggle test labels, not carried over from an earlier claim) — see
`enhanced/README.md`'s "Verified results" section for the full breakdown, including the trade-offs
behind those numbers. The trained model (~29MB) is checked into `enhanced/saved_models/` so
`enhanced/inference/` and `enhanced/redaction/` work immediately, no training step required.

---

## 📋 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB | 10 GB (with GloVe + dataset) |
| GPU | Optional | CUDA-enabled (for training) |

---

## 📚 References

- **Dataset**: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) — Jigsaw/Conversation AI
- **Embeddings**: [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) — Pennington et al., Stanford NLP
- **Architecture**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- **Research Paper**: *Classifying Toxic Comments Using Deep Learning* — Deep Dobariya, Aditya Joshi, Vedant Tolia et al., 2024

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **BISAG-N (Bhaskaracharya National Institute for Space Applications and Geo-informatics)**, Government of India — for the internship environment where this system was developed
- Originally developed collaboratively during the internship, alongside fellow contributor **Chessdude02**
- **Jigsaw/Conversation AI** — for the toxic comment dataset
- **Stanford NLP Group** — for GloVe embeddings
- **TensorFlow team** — for the deep learning framework
