"""Shared artifact locations so training, inference, and evaluation scripts agree on where things live."""

from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = MODELS_DIR / "saved_models"
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "toxicity_model.keras"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.pickle"
CONFIG_PATH = ARTIFACTS_DIR / "config.pickle"
TRAINING_CURVES_PATH = ARTIFACTS_DIR / "training_curves.png"
