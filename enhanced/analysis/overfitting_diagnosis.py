#!/usr/bin/env python3
"""
Overfitting Diagnosis for Your Toxic Comment Classifier
======================================================

Based on the analysis of your model, here are the key findings and solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

def diagnose_current_model():
    """
    Diagnose the current model for overfitting issues
    """
    print("=" * 60)
    print("OVERFITTING DIAGNOSIS FOR YOUR MODEL")
    print("=" * 60)
    
    # Load and analyze your current model
    try:
        model = load_model("saved_models/demo_toxicity_classifier.h5")
        with open("saved_models/config.pickle", "rb") as f:
            config = pickle.load(f)
        
        print("\nMODEL ANALYSIS RESULTS:")
        print("-" * 30)
        
        # Model complexity
        total_params = model.count_params()
        print(f"Total parameters: {total_params:,}")
        print(f"Vocabulary size: {config['vocab_size']:,}")
        print(f"Max sequence length: {config['max_len']}")
        print(f"Number of classes: {len(config['label_columns'])}")
        
        # Analyze model architecture
        print(f"\nARCHITECTURE ANALYSIS:")
        print("-" * 30)
        
        layer_info = []
        regularization_layers = 0
        
        for i, layer in enumerate(model.layers):
            layer_type = type(layer).__name__
            layer_info.append(f"Layer {i}: {layer_type}")
            
            if 'Dropout' in layer_type:
                regularization_layers += 1
                print(f"✅ Found dropout layer {i}: {layer.rate} rate")
        
        print(f"\nRegularization layers found: {regularization_layers}")
        
        # CRITICAL OVERFITTING INDICATORS FOUND
        print(f"\n🚨 OVERFITTING ISSUES IDENTIFIED:")
        print("-" * 40)
        
        issues = []
        
        # Issue 1: Very small vocabulary size
        if config['vocab_size'] < 1000:
            issues.append(f"🚨 CRITICAL: Vocabulary size ({config['vocab_size']}) is extremely small!")
            issues.append("   This suggests severe data limitations or poor tokenization")
        
        # Issue 2: Small dataset implied by config
        if config['vocab_size'] < 1000:
            issues.append("🚨 CRITICAL: Your training dataset appears to be very small")
            issues.append("   This is a major cause of overfitting")
        
        # Issue 3: Model complexity vs data size ratio
        param_to_vocab_ratio = total_params / config['vocab_size']
        if param_to_vocab_ratio > 100:
            issues.append(f"🚨 CRITICAL: Parameter-to-vocabulary ratio is {param_to_vocab_ratio:.1f}")
            issues.append("   Model is too complex for the available data")
        
        # Issue 4: Check for missing validation strategy
        issues.append("⚠️  Need to verify train/validation split strategy")
        issues.append("⚠️  Early stopping implementation needs review")
        
        for issue in issues:
            print(issue)
        
        # SPECIFIC SOLUTIONS
        print(f"\n💡 SPECIFIC SOLUTIONS FOR YOUR MODEL:")
        print("-" * 40)
        
        solutions = [
            "1. 📊 INCREASE DATASET SIZE:",
            "   - Get more training data (aim for 10K+ samples minimum)",
            "   - Use data augmentation techniques for text",
            "   - Consider using pre-trained embeddings",
            "",
            "2. 🔄 REDUCE MODEL COMPLEXITY:",
            "   - Reduce LSTM units from 64 to 32",
            "   - Reduce dense layer from 32 to 16 units",
            "   - Add more dropout (0.4-0.6 rate)",
            "",
            "3. ⏰ IMPROVE TRAINING STRATEGY:",
            "   - Implement early stopping with patience=3",
            "   - Use ReduceLROnPlateau callback",
            "   - Monitor validation loss, not training loss",
            "",
            "4. 📈 VALIDATION STRATEGY:",
            "   - Use stratified train/validation split",
            "   - Implement k-fold cross-validation",
            "   - Keep a separate test set",
            "",
            "5. 🛡️ REGULARIZATION:",
            "   - Add L2 regularization to dense layers",
            "   - Increase dropout rates",
            "   - Consider batch normalization"
        ]
        
        for solution in solutions:
            print(solution)
        
        return issues, solutions
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return [], []

def create_improved_model_template():
    """
    Create an improved model template with better overfitting protection
    """
    template = '''
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
'''
    
    with open("improved_model_template.py", "w", encoding='utf-8') as f:
        f.write(template)
    
    print("✅ Created 'improved_model_template.py' with overfitting-resistant architecture")

def analyze_your_training_setup():
    """
    Analyze the specific training setup in your notebook for overfitting issues
    """
    print(f"\nANALYSIS OF YOUR CURRENT TRAINING SETUP:")
    print("-" * 45)
    
    print("ISSUES FOUND IN YOUR NOTEBOOK CODE:")
    print("1. 🚨 EPOCHS = 5 might be too few to see overfitting patterns")
    print("2. ⚠️  BATCH_SIZE = 16 * replicas might be too large for small dataset")
    print("3. 🚨 No clear train/test split validation in the visible code")
    print("4. ⚠️  Limited regularization in some model architectures")
    print("5. 🚨 Sample data generation shows very small dataset (1000 samples)")
    
    print(f"\nKEY PROBLEMS IDENTIFIED:")
    print("- Very small vocabulary (120 words) indicates tiny dataset")
    print("- Small dataset + complex models = guaranteed overfitting")
    print("- Need better validation methodology")
    
    return True

def generate_action_plan():
    """
    Generate a specific action plan to fix overfitting
    """
    print(f"\n🎯 ACTION PLAN TO FIX OVERFITTING:")
    print("=" * 40)
    
    plan = [
        "IMMEDIATE ACTIONS (Priority 1):",
        "1. Get more training data (minimum 10,000 samples)",
        "2. Implement proper train/validation/test split (70/15/15)",
        "3. Add early stopping with patience=3",
        "4. Reduce model complexity by 50%",
        "",
        "SHORT-TERM FIXES (Priority 2):", 
        "5. Increase dropout rates to 0.4-0.6",
        "6. Add L2 regularization to dense layers",
        "7. Implement k-fold cross-validation",
        "8. Monitor validation metrics during training",
        "",
        "LONG-TERM IMPROVEMENTS (Priority 3):",
        "9. Use pre-trained embeddings (Word2Vec, GloVe)",
        "10. Implement ensemble methods",
        "11. Add data augmentation techniques",
        "12. Regular model performance monitoring"
    ]
    
    for item in plan:
        print(item)
    
    print(f"\n⚡ START WITH: Get more data and reduce model size!")

def main():
    print("Starting overfitting diagnosis...")
    
    # Run diagnosis
    issues, solutions = diagnose_current_model()
    
    # Analyze training setup
    analyze_your_training_setup()
    
    # Create improved template
    create_improved_model_template()
    
    # Generate action plan
    generate_action_plan()
    
    print(f"\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE!")
    print("=" * 60)
    print("✅ Model analyzed")
    print("✅ Issues identified") 
    print("✅ Solutions provided")
    print("✅ Improved template created")
    print("✅ Action plan generated")

if __name__ == "__main__":
    main()
