"""Model architecture and training utilities shared across scripts.

The original `create_overfitting_resistant_model` over-corrected: spatial
dropout 0.4, recurrent dropout 0.3, dense dropout 0.6, lr 1e-4 flat. On a
small dataset that regularizes so hard the model can't separate classes at
all - predictions all hover around 0.5 instead of confidently classifying.
This version dials regularization back to levels appropriate for the model
size and adds AdamW + a warmup/cosine-decay schedule + label smoothing,
matching the training setup used elsewhere in this project.
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Bidirectional, Dense, Dropout, Embedding, LSTM, SpatialDropout1D,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential


def build_model(vocab_size, max_len=64, embedding_dim=100):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
        Dense(32, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        warmup_target=2e-3,
        warmup_steps=100,
        alpha=0.05,
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.build(input_shape=(None, max_len))
    return model


def get_callbacks(checkpoint_path, patience=6):
    # No ReduceLROnPlateau here: the optimizer already anneals its learning
    # rate via the CosineDecay schedule in build_model(), and Keras won't
    # let a callback mutate the learning rate once it's schedule-driven.
    return [
        EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True,
            verbose=1, mode="min",
        ),
        ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss", save_best_only=True,
            verbose=0, mode="min",
        ),
    ]


def class_weights(y_train):
    """Balance classes if the dataset isn't perfectly 50/50."""
    import numpy as np
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    return {int(c): total / (len(classes) * count) for c, count in zip(classes, counts)}
