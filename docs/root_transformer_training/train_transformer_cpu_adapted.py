"""
Adapted, CPU-feasible run of Industrial_Grade_Toxic_Comment_Classifier.ipynb.

Extracted verbatim (TransformerBlock, AttentionPooling, PositionalEmbedding,
DataPipeline, ModelArchitectures) from the actual notebook cells, with the
minimum changes needed to make it (a) run outside Colab and (b) finish on a
4-core CPU box in a reasonable time:

  - embedding_dim 300 -> 100 (glove.6B.100d instead of glove.840B.300d -
    840B needed a ~2GB download this environment's network policy blocks;
    100d was obtained from a public GitHub LFS mirror instead, sha256
    verified against its LFS pointer)
  - num_heads 6 -> 4 (100 / 4 = 25, a clean head size for 100-dim; 100 isn't
    evenly divisible by 6)
  - load_glove_embeddings(): the notebook hardcodes `if len(coefs) == 300`,
    which would silently discard every 100-dim vector. Changed to check the
    configured embedding_dim instead - not a design choice, a necessary fix
    for this to work at all with a different embedding file.
  - Training data capped to a stratified 30,000-row sample of the real
    159,571-row Jigsaw train.csv (same reasoning as enhanced/'s 60K cap:
    the original 223K-row run needed a GPU and was documented as taking
    "2-7 hours"; this keeps a CPU run bounded to roughly an hour instead of
    a multi-day one).
  - Max epochs capped at 30 (from 150) with early-stopping patience 6 (from
    15) - both scaled down for the smaller dataset/faster convergence,
    following the same early-stopping logic (monitor val_auc) as the
    original.
  - Dropped the DiagnosticsMonitor/TensorBoard callbacks (cells 14/20) -
    informational only, don't affect the trained result.
  - Cell 20's actual recompile (plain Adam + binary_crossentropy +
    BinaryAccuracy/AUC) is what really ran in the original notebook
    (cell 18's fancier AdamW/label-smoothing/precision/recall/f1 compile
    gets overwritten by it) - replicated that, not the aspirational one.

Everything else (architecture, warmup+cosine-decay schedule, class
weighting, CLS-token pooling) is unchanged from the notebook.
"""

import json
import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Layer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

RUN_DIR = Path(__file__).resolve().parent
CACHE_DIR = RUN_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR = RUN_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

SCRATCH = RUN_DIR.parent
TRAIN_CSV = SCRATCH / "jigsaw_data" / "train.csv"
TEST_CSV = SCRATCH / "jigsaw_data" / "test.csv"
TEST_LABELS_CSV = SCRATCH / "jigsaw_data" / "test_labels.csv"
GLOVE_PATH = SCRATCH / "glove.6B.100d.txt"
SAMPLED_TRAIN_CSV = RUN_DIR / "train_sampled_30k.csv"

SAMPLE_CAP = 30000

CONFIG = {
    "data": {
        "train_path": str(SAMPLED_TRAIN_CSV),
        "embedding_path": str(GLOVE_PATH),
        "val_split": 0.15,
        "test_split": 0.10,
        "random_seed": 42,
        "max_sequence_length": 64,
        "vocab_size": 50000,
        "cache_dir": str(CACHE_DIR),
    },
    "model": {
        "architecture": "transformer",
        "embedding_dim": 100,
        "num_heads": 4,
        "ff_dim": 1024,
        "num_transformer_blocks": 3,
        "dense_units": 128,
        "dropout_rate": 0.2,
        "attention_dropout": 0.1,
        "l2_regularization": 0.00001,
        "use_layer_norm": True,
        "trainable_embeddings": True,
        "use_cls_token": True,
        "pooling_strategy": "cls_token",
    },
    "training": {
        "epochs": 50,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "optimizer": "adamw",
        "warmup_steps": 300,
        "loss": "binary_crossentropy",
        "class_weights": True,
        # Patience raised from 6 (30-epoch run) to 12 so early stopping doesn't
        # cut the run short before all 50 epochs get a chance to run - we want
        # to actually see the overfitting point on the curve, not just the
        # point where the previous, tighter patience gave up.
        "early_stopping": {"enabled": True, "patience": 12, "min_delta": 0.0001},
        "reduce_lr": {"enabled": True, "patience": 7, "factor": 0.5, "min_lr": 0.00001},
        "checkpoint_freq": 5,
    },
}


def build_sampled_csv():
    if SAMPLED_TRAIN_CSV.exists():
        print(f"Using existing sampled CSV: {SAMPLED_TRAIN_CSV}")
        return
    print(f"Building a stratified {SAMPLE_CAP}-row sample from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    frac = SAMPLE_CAP / len(df)
    parts = [group.sample(frac=frac, random_state=42) for _, group in df.groupby("toxic")]
    sampled = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    sampled.to_csv(SAMPLED_TRAIN_CSV, index=False)
    print(f"  Wrote {len(sampled)} rows ({sampled['toxic'].mean():.1%} toxic) to {SAMPLED_TRAIN_CSV}")


# ---------------------------------------------------------------------------
# From notebook cell 8 (verbatim)
# ---------------------------------------------------------------------------
class TransformerBlock(Layer):
    """Optimized Transformer block with pre-norm, GELU, and attention dropout"""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, attention_dropout=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=attention_dropout,
        )

        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False, mask=None):
        x_norm1 = self.layernorm1(inputs)
        attn_output = self.att(x_norm1, x_norm1, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output

        x_norm2 = self.layernorm2(out1)
        ffn_output = self.ffn(x_norm2)
        return out1 + ffn_output

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
            "attention_dropout": self.attention_dropout,
        })
        return config


class AttentionPooling(Layer):
    def __init__(self, embed_dim, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.attention_weights = layers.Dense(1, activation="tanh")

    def call(self, inputs):
        attention_scores = self.attention_weights(inputs)
        attention_scores = tf.nn.softmax(attention_scores, axis=1)
        weighted_sum = tf.reduce_sum(inputs * attention_scores, axis=1)
        return weighted_sum

    def get_config(self):
        config = super(AttentionPooling, self).get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


class PositionalEmbedding(Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, use_cls_token=False, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        pos_length = sequence_length + 1 if use_cls_token else sequence_length
        self.pos_emb = layers.Embedding(input_dim=pos_length, output_dim=embed_dim)

        if use_cls_token:
            self.cls_token = self.add_weight(
                name="cls_token", shape=(1, 1, embed_dim),
                initializer="random_normal", trainable=True,
            )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        if self.use_cls_token:
            cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
            inputs = tf.concat([cls_tokens, inputs], axis=1)
            positions = tf.range(start=0, limit=seq_len + 1, delta=1)
        else:
            positions = tf.range(start=0, limit=seq_len, delta=1)

        position_embeddings = self.pos_emb(positions)
        return inputs + position_embeddings

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "use_cls_token": self.use_cls_token,
        })
        return config


# ---------------------------------------------------------------------------
# From notebook cell 10, with load_glove_embeddings() patched for dynamic dim
# ---------------------------------------------------------------------------
class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.data_config = config["data"]
        self.max_sequence_length = self.data_config["max_sequence_length"]
        self.vocab_size = self.data_config["vocab_size"]
        self.tokenizer = None
        self.embedding_matrix = None
        Path(self.data_config["cache_dir"]).mkdir(parents=True, exist_ok=True)

    def load_data(self):
        print(f"Loading data from {self.data_config['train_path']}...")
        df = pd.read_csv(self.data_config["train_path"])
        print(f"  Loaded {len(df)} samples")
        df = df.dropna(subset=["comment_text", "toxic"])
        df = df.drop_duplicates(subset=["comment_text"])
        toxic_count = df["toxic"].sum()
        print(f"  Toxic: {toxic_count} ({toxic_count/len(df)*100:.1f}%)")
        print(f"  Non-toxic: {len(df) - toxic_count} ({(len(df) - toxic_count)/len(df)*100:.1f}%)")
        return df

    def create_tokenizer(self, texts):
        cache_file = Path(self.data_config["cache_dir"]) / "tokenizer.pkl"
        if cache_file.exists():
            print("  Loading tokenizer from cache...")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        print("  Creating tokenizer...")
        tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>", lower=True)
        tokenizer.fit_on_texts(texts)
        with open(cache_file, "wb") as f:
            pickle.dump(tokenizer, f)
        print(f"  Tokenizer created with {len(tokenizer.word_index)} unique tokens")
        return tokenizer

    def load_glove_embeddings(self):
        cache_file = Path(self.data_config["cache_dir"]) / "glove_embeddings.pkl"
        if cache_file.exists():
            print("  Loading GloVe embeddings from cache...")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        expected_dim = self.config["model"]["embedding_dim"]
        print(f"  Loading GloVe embeddings (dim={expected_dim})...")
        embeddings_index = {}
        with open(self.data_config["embedding_path"], "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype="float32")
                    # NOTEBOOK BUG FIX: original hardcoded `== 300`, which
                    # would silently drop every vector from a non-300d file.
                    if len(coefs) == expected_dim:
                        embeddings_index[word] = coefs
                except ValueError:
                    continue
        print(f"  Loaded {len(embeddings_index)} word vectors")
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings_index, f)
        return embeddings_index

    def create_embedding_matrix(self, tokenizer, embeddings_index):
        print("  Creating embedding matrix...")
        word_index = tokenizer.word_index
        num_words = min(self.vocab_size, len(word_index)) + 1
        embedding_dim = self.config["model"]["embedding_dim"]
        embedding_matrix = np.zeros((num_words, embedding_dim))
        found = 0
        for word, i in word_index.items():
            if i >= num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                found += 1
            else:
                embedding_matrix[i] = np.random.normal(0, 0.1, embedding_dim)
        print(f"  Found embeddings for {found}/{num_words} words ({found/num_words*100:.1f}%)")
        return embedding_matrix

    def prepare_data(self, force_reload=False):
        cache_file = Path(self.data_config["cache_dir"]) / "prepared_data.pkl"
        if cache_file.exists() and not force_reload:
            print("Loading prepared data from cache...")
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
                self.tokenizer = cached["tokenizer"]
                self.embedding_matrix = cached["embedding_matrix"]
                return (cached["X_train"], cached["X_val"], cached["X_test"],
                        cached["y_train"], cached["y_val"], cached["y_test"])

        print("Preparing data from scratch...")
        df = self.load_data()

        train_val_df, test_df = train_test_split(
            df, test_size=self.data_config["test_split"],
            random_state=self.data_config["random_seed"], stratify=df["toxic"],
        )
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.data_config["val_split"] / (1 - self.data_config["test_split"]),
            random_state=self.data_config["random_seed"], stratify=train_val_df["toxic"],
        )
        print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        self.tokenizer = self.create_tokenizer(train_df["comment_text"])

        print("  Converting texts to sequences...")
        X_train = pad_sequences(
            self.tokenizer.texts_to_sequences(train_df["comment_text"]),
            maxlen=self.max_sequence_length, padding="post", truncating="post",
        )
        X_val = pad_sequences(
            self.tokenizer.texts_to_sequences(val_df["comment_text"]),
            maxlen=self.max_sequence_length, padding="post", truncating="post",
        )
        X_test = pad_sequences(
            self.tokenizer.texts_to_sequences(test_df["comment_text"]),
            maxlen=self.max_sequence_length, padding="post", truncating="post",
        )

        y_train = train_df["toxic"].values
        y_val = val_df["toxic"].values
        y_test = test_df["toxic"].values

        embeddings_index = self.load_glove_embeddings()
        self.embedding_matrix = self.create_embedding_matrix(self.tokenizer, embeddings_index)

        print("  Caching prepared data...")
        with open(cache_file, "wb") as f:
            pickle.dump({
                "X_train": X_train, "X_val": X_val, "X_test": X_test,
                "y_train": y_train, "y_val": y_val, "y_test": y_test,
                "tokenizer": self.tokenizer, "embedding_matrix": self.embedding_matrix,
            }, f)

        print("Data preparation complete")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def compute_class_weights(self, y_train):
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        return dict(zip(classes, weights))


# ---------------------------------------------------------------------------
# From notebook cell 12 (verbatim)
# ---------------------------------------------------------------------------
class ModelArchitectures:
    def __init__(self, config):
        self.config = config
        self.model_config = config["model"]
        self.data_config = config["data"]

    def create_model(self, embedding_matrix, architecture=None):
        if architecture is None:
            architecture = self.model_config["architecture"]
        print(f"Creating {architecture} model...")
        if architecture == "transformer":
            return self.create_transformer_model(embedding_matrix)
        raise ValueError(f"Unknown architecture: {architecture}. Use 'transformer'")

    def create_transformer_model(self, embedding_matrix):
        vocab_size, embedding_dim = embedding_matrix.shape
        max_length = self.data_config["max_sequence_length"]
        use_cls = self.model_config.get("use_cls_token", False)
        pooling_strategy = self.model_config.get("pooling_strategy", "mean")

        inputs = Input(shape=(max_length,), name="input_ids")
        x = Embedding(
            vocab_size, embedding_dim, weights=[embedding_matrix],
            input_length=max_length, trainable=self.model_config["trainable_embeddings"],
            name="word_embeddings",
        )(inputs)
        x = PositionalEmbedding(max_length, vocab_size, embedding_dim, use_cls_token=use_cls)(x)
        x = Dropout(self.model_config["dropout_rate"], name="embedding_dropout")(x)

        for i in range(self.model_config["num_transformer_blocks"]):
            x = TransformerBlock(
                embed_dim=embedding_dim, num_heads=self.model_config["num_heads"],
                ff_dim=self.model_config["ff_dim"], dropout_rate=self.model_config["dropout_rate"],
                attention_dropout=self.model_config.get("attention_dropout", 0.1),
                name=f"transformer_block_{i}",
            )(x)

        x = layers.LayerNormalization(epsilon=1e-6, name="final_layer_norm")(x)

        if pooling_strategy == "cls_token" and use_cls:
            pooled = x[:, 0, :]
        elif pooling_strategy == "attention":
            pooled = AttentionPooling(embedding_dim)(x)
        elif pooling_strategy == "max":
            pooled = layers.GlobalMaxPooling1D()(x)
        else:
            pooled = layers.GlobalAveragePooling1D()(x)

        dense1 = Dense(
            self.model_config["dense_units"], activation="gelu",
            kernel_regularizer=regularizers.l2(self.model_config["l2_regularization"]),
            name="dense_1",
        )(pooled)
        if self.model_config.get("use_layer_norm", True):
            dense1 = layers.LayerNormalization(epsilon=1e-6)(dense1)
        dense1 = Dropout(self.model_config["dropout_rate"])(dense1)

        dense2 = Dense(
            self.model_config["dense_units"] // 2, activation="gelu",
            kernel_regularizer=regularizers.l2(self.model_config["l2_regularization"]),
            name="dense_2",
        )(dense1)
        dense2 = Dropout(self.model_config["dropout_rate"] * 0.5)(dense2)

        outputs = Dense(1, activation="sigmoid", name="output")(dense2)
        model = Model(inputs=inputs, outputs=outputs, name="OptimizedTransformer")

        print("  Model configuration:")
        print(f"    - Transformer blocks: {self.model_config['num_transformer_blocks']}")
        print(f"    - Attention heads: {self.model_config['num_heads']}")
        print(f"    - Feed-forward dim: {self.model_config['ff_dim']}")
        print(f"    - Pooling strategy: {pooling_strategy}")
        print(f"    - Trainable embeddings: {self.model_config['trainable_embeddings']}")
        return model


# ---------------------------------------------------------------------------
# From notebook cell 20's warmup scheduler (verbatim)
# ---------------------------------------------------------------------------
class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    def __init__(self, warmup_steps, total_steps, initial_lr, max_lr):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.step = 0

    def on_train_batch_begin(self, batch, logs=None):
        self.step += 1
        if self.step <= self.warmup_steps:
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (self.step / self.warmup_steps)
        else:
            progress = (self.step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        self.model.optimizer.learning_rate.assign(lr)


def main():
    build_sampled_csv()

    print("=" * 80)
    print("PREPARING DATA")
    print("=" * 80)
    data_pipeline = DataPipeline(CONFIG)
    X_train, X_val, X_test, y_train, y_val, y_test = data_pipeline.prepare_data()
    print(f"\nData prepared: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

    class_weights = None
    if CONFIG["training"]["class_weights"]:
        class_weights = data_pipeline.compute_class_weights(y_train)
        print(f"Class weights: {class_weights}")

    print("=" * 80)
    print("BUILDING MODEL")
    print("=" * 80)
    model_factory = ModelArchitectures(CONFIG)
    model = model_factory.create_model(data_pipeline.embedding_matrix)
    model.summary()

    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    optimizer = keras.optimizers.Adam(learning_rate=CONFIG["training"]["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss=CONFIG["training"]["loss"],
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc")],
    )

    callbacks = []

    if CONFIG["training"].get("warmup_steps", 0) > 0:
        batch_size = CONFIG["training"]["batch_size"]
        steps_per_epoch = math.ceil(len(X_train) / batch_size)
        total_steps = steps_per_epoch * CONFIG["training"]["epochs"]
        callbacks.append(WarmUpCosineDecayScheduler(
            warmup_steps=CONFIG["training"]["warmup_steps"], total_steps=total_steps,
            initial_lr=1e-7, max_lr=CONFIG["training"]["learning_rate"],
        ))
        print(f"  Using warmup + cosine decay scheduler "
              f"(warmup: {CONFIG['training']['warmup_steps']} steps, total: {total_steps} steps)")

    if CONFIG["training"]["early_stopping"]["enabled"]:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=CONFIG["training"]["early_stopping"]["patience"],
            min_delta=CONFIG["training"]["early_stopping"]["min_delta"],
            restore_best_weights=True, verbose=1,
        ))

    checkpoint_path = str(ARTIFACTS_DIR / "checkpoint_best.keras")
    callbacks.append(keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1,
    ))

    history_csv_path = str(ARTIFACTS_DIR / "training_history.csv")
    callbacks.append(keras.callbacks.CSVLogger(history_csv_path))

    print(f"Training for up to {CONFIG['training']['epochs']} epochs")
    print(f"Batch size: {CONFIG['training']['batch_size']}")

    history = model.fit(
        X_train, y_train,
        batch_size=CONFIG["training"]["batch_size"],
        epochs=CONFIG["training"]["epochs"],
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    with open(ARTIFACTS_DIR / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    print(f"\nActually trained {len(history.history['loss'])} epochs "
          f"(cap was {CONFIG['training']['epochs']}, early-stop patience "
          f"{CONFIG['training']['early_stopping']['patience']})")

    print("=" * 80)
    print("FINAL EVALUATION ON HELD-OUT TEST SPLIT (from the 30K sample)")
    print("=" * 80)
    test_results = model.evaluate(X_test, y_test, verbose=1)
    test_metrics = dict(zip(model.metrics_names, test_results))
    print(f"Test metrics: {test_metrics}")

    model.save(str(ARTIFACTS_DIR / "transformer_model.keras"))
    with open(ARTIFACTS_DIR / "tokenizer.pickle", "wb") as f:
        pickle.dump(data_pipeline.tokenizer, f)
    with open(ARTIFACTS_DIR / "config.pickle", "wb") as f:
        pickle.dump({
            "label_columns": ["toxic"],
            "threshold": 0.5,
            "max_len": CONFIG["data"]["max_sequence_length"],
        }, f)
    with open(ARTIFACTS_DIR / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\nSaved model/tokenizer/config/test_metrics to", ARTIFACTS_DIR)

    # Real held-out evaluation against the official Kaggle test labels, same
    # as enhanced/evaluation/evaluate_on_kaggle_test.py does for the BiLSTM.
    print("=" * 80)
    print("EVALUATING ON THE OFFICIAL HELD-OUT KAGGLE TEST SET")
    print("=" * 80)
    test_df = pd.read_csv(TEST_CSV)
    labels_df = pd.read_csv(TEST_LABELS_CSV)
    merged = test_df.merge(labels_df, on="id")
    merged = merged[merged["toxic"] != -1].reset_index(drop=True)
    print(f"{len(merged)} rows have real labels")

    seqs = data_pipeline.tokenizer.texts_to_sequences(merged["comment_text"].astype(str))
    padded = pad_sequences(seqs, maxlen=CONFIG["data"]["max_sequence_length"], padding="post", truncating="post")
    probs = model.predict(padded, batch_size=256, verbose=1).ravel()
    y_true = merged["toxic"].to_numpy()
    y_pred = (probs > 0.5).astype(int)

    official_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, probs)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=["clean", "toxic"]),
    }
    print(f"Accuracy: {official_metrics['accuracy']:.4f}")
    print(f"AUC:      {official_metrics['auc']:.4f}")
    print(official_metrics["classification_report"])
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(official_metrics["confusion_matrix"])

    with open(ARTIFACTS_DIR / "official_test_metrics.json", "w") as f:
        json.dump(official_metrics, f, indent=2)


if __name__ == "__main__":
    main()
