
# FIXED TRAINING CODE - PREVENTS OVERFITTING
# ==========================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

def create_non_overfitting_model(vocab_size, max_len=512):
    """
    Create a model that won't overfit to the Jigsaw dataset
    """
    model = Sequential([
        # Much smaller embedding
        Embedding(vocab_size, 100, input_length=max_len),  # Reduced from 300 to 100
        SpatialDropout1D(0.4),  # Add spatial dropout
        
        # Much smaller LSTM
        LSTM(64, dropout=0.3, recurrent_dropout=0.3),  # Reduced from 100 to 64
        
        # Add regularization
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Lower learning rate
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def get_anti_overfitting_callbacks():
    """
    Callbacks to prevent overfitting
    """
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

# CORRECTED TRAINING PROCEDURE:
# =============================

# 1. Load data properly (you already have this)
# train = pd.read_csv('/kaggle/input/train-csv/jigsaw-toxic-comment-train.csv')

# 2. Create proper train/validation split
xtrain, xvalid, ytrain, yvalid = train_test_split(
    train.comment_text.values, 
    train.toxic.values,
    stratify=train.toxic.values,
    test_size=0.2,
    random_state=42
)

# 3. Tokenize with reasonable vocabulary size
token = text.Tokenizer(num_words=50000)  # Limit vocabulary to 50K
token.fit_on_texts(list(xtrain) + list(xvalid))

max_len = 300  # Reduce from 1500 to 300
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)

xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

# 4. Create the improved model
vocab_size = len(token.word_index) + 1
model = create_non_overfitting_model(vocab_size, max_len)

# 5. Train with validation monitoring
callbacks = get_anti_overfitting_callbacks()

history = model.fit(
    xtrain_pad, ytrain,
    validation_data=(xvalid_pad, yvalid),  # CRITICAL: Add validation data!
    epochs=20,  # More epochs but with early stopping
    batch_size=32,  # Smaller batch size
    callbacks=callbacks,  # Anti-overfitting callbacks
    verbose=1
)

# 6. Analyze for overfitting
quick_overfitting_check(history.history)
