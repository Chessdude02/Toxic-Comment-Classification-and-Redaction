
# IMPROVED MODEL ARCHITECTURE - OVERFITTING RESISTANT
# ==================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, SpatialDropout1D, LSTM, Dense, Dropout, 
    BatchNormalization
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_overfitting_resistant_model(vocab_size, max_len, num_classes):
    """
    Create a model designed to resist overfitting
    """
    model = Sequential([
        # Embedding layer
        Embedding(vocab_size, 100, input_length=max_len),
        SpatialDropout1D(0.4),  # Increased dropout
        
        # Reduced LSTM complexity
        LSTM(32, dropout=0.3, recurrent_dropout=0.3),  # Reduced from 64 to 32
        
        # Batch normalization for stability
        BatchNormalization(),
        
        # Smaller dense layer with L2 regularization
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),  # Reduced from 32 to 16
        Dropout(0.5),  # High dropout
        
        # Output layer
        Dense(num_classes, activation='sigmoid')
    ])
    
    # Compile with lower learning rate
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def get_anti_overfitting_callbacks():
    """
    Get callbacks that prevent overfitting
    """
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=3,  # Stop if no improvement for 3 epochs
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Reduce LR by half
            patience=2,  # Wait 2 epochs before reducing
            min_lr=1e-7,
            verbose=1
        )
    ]

# Example usage:
# model = create_overfitting_resistant_model(vocab_size=5000, max_len=128, num_classes=6)
# callbacks = get_anti_overfitting_callbacks()
# history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
#                    epochs=20, callbacks=callbacks)
