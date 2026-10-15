"""
Custom Keras layers for the root Transformer classifier
(Industrial_Grade_Toxic_Comment_Classifier.ipynb, cell 8), extracted verbatim
and registered for serialization so tf.keras.models.load_model() can
resolve them by name alone, without every caller needing to pass
custom_objects explicitly.

Importing this module (which toxicity_redactor.py does) is enough - the
decorator runs its registration as a side effect of import.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

# tf.keras is a lazy-loading proxy in this TensorFlow/Keras version and
# doesn't expose `.saving` - the standalone `keras` package (Keras 3, a
# separate pip package since TF 2.16+) does. Needed only for the
# registration decorator below; everything else keeps using tf.keras.
import keras as _keras_standalone


@_keras_standalone.saving.register_keras_serializable(package="toxic_comment_transformer")
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


@_keras_standalone.saving.register_keras_serializable(package="toxic_comment_transformer")
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


@_keras_standalone.saving.register_keras_serializable(package="toxic_comment_transformer")
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
