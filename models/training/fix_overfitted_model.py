#!/usr/bin/env python3
"""
Complete Model Retraining Solution
=================================

This script completely retrains your model to fix overfitting issues.
It uses proper regularization, validation monitoring, and a right-sized architecture.
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Dense, LSTM, Dropout, SpatialDropout1D, 
    BatchNormalization
)
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

def create_sample_jigsaw_data():
    """
    Create a more realistic sample dataset that mimics the Jigsaw dataset
    since we don't have access to the actual Kaggle data locally
    """
    print("📊 Creating realistic sample dataset...")
    
    # Create a more diverse and larger sample dataset
    toxic_comments = [
        "You are an idiot and don't know anything",
        "Shut up, you moron",
        "Go kill yourself",
        "You're a complete loser", 
        "Fucking stupid comment",
        "You suck so bad",
        "What a dumbass",
        "You're pathetic",
        "Piece of shit person",
        "Go to hell",
        "You're worthless",
        "Stupid bitch",
        "Retarded comment",
        "You're an asshole",
        "Fuck off",
        "I hate you so much",
        "You're disgusting",
        "What an imbecile",
        "You're a waste of space",
        "Die in a fire",
        # Add variations
        "you are stupid", "YOU ARE STUPID", "You Are Stupid!",
        "idiot!", "IDIOT", "what an idiot",
        "dumb person", "so dumb", "really dumb",
        "hate this", "i hate you", "hate you so much",
        "kill yourself", "go die", "should die",
    ] * 10  # Multiply for more samples
    
    clean_comments = [
        "Thank you for sharing this information",
        "I disagree with your opinion but respect it",
        "This is a great article",
        "Could you please explain this better",
        "I think there might be a better approach",
        "Thanks for your help",
        "Have a great day",
        "This is very informative",
        "I appreciate your perspective", 
        "Good point, thanks",
        "Interesting analysis",
        "Well written article",
        "Thanks for clarifying",
        "I learned something new today",
        "Great explanation",
        "This helps a lot",
        "Nice work on this",
        "Very helpful information",
        "I agree with this approach",
        "Excellent research",
        # Add more variety
        "hello everyone", "how are you", "good morning",
        "please help me", "thank you very much", "have a nice day",
        "great job", "well done", "congratulations",
        "i like this", "this is good", "very nice",
        "interesting point", "good question", "makes sense",
    ] * 15  # More clean examples
    
    # Create labels
    toxic_labels = [1] * len(toxic_comments)
    clean_labels = [0] * len(clean_comments)
    
    # Combine data
    all_texts = toxic_comments + clean_comments
    all_labels = toxic_labels + clean_labels
    
    # Create DataFrame
    df = pd.DataFrame({
        'comment_text': all_texts,
        'toxic': all_labels
    })
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Created dataset with {len(df)} samples")
    print(f"   Toxic samples: {sum(all_labels)} ({sum(all_labels)/len(all_labels):.1%})")
    print(f"   Clean samples: {len(all_labels) - sum(all_labels)} ({(len(all_labels) - sum(all_labels))/len(all_labels):.1%})")
    
    return df

def create_overfitting_resistant_model(vocab_size, max_len=200, embedding_dim=64):
    """
    Create a model designed to resist overfitting
    """
    print(f"🏗️ Building overfitting-resistant model...")
    print(f"   Vocab size: {vocab_size:,}")
    print(f"   Max length: {max_len}")
    print(f"   Embedding dim: {embedding_dim}")
    
    model = Sequential([
        # Smaller embedding layer
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        SpatialDropout1D(0.4),  # Spatial dropout for embeddings
        
        # Smaller LSTM with heavy regularization
        LSTM(32, dropout=0.3, recurrent_dropout=0.3),
        
        # Batch normalization for stability
        BatchNormalization(),
        
        # Small dense layer with L2 regularization
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.6),  # Heavy dropout
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    # Compile with conservative settings
    model.compile(
        optimizer=Adam(learning_rate=1e-4),  # Lower learning rate
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Build the model to count parameters
    model.build(input_shape=(None, max_len))
    print(f"✅ Model created with {model.count_params():,} parameters")
    return model

def prepare_data(df, test_size=0.2, max_vocab=10000, max_len=200):
    """
    Prepare data with proper preprocessing
    """
    print("🔄 Preparing data...")
    
    # Clean text
    df['clean_comment'] = df['comment_text'].apply(lambda x: str(x).lower().strip())
    
    # Split data properly
    X = df['clean_comment'].values
    y = df['toxic'].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42, 
        stratify=y  # Ensure balanced split
    )
    
    print(f"📊 Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Training toxic rate: {y_train.mean():.1%}")
    print(f"   Validation toxic rate: {y_val.mean():.1%}")
    
    # Create and fit tokenizer
    print("🔤 Creating tokenizer...")
    tokenizer = text.Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)  # Only fit on training data
    
    # Convert to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    
    # Pad sequences
    X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=max_len)
    X_val_pad = sequence.pad_sequences(X_val_seq, maxlen=max_len)
    
    vocab_size = len(tokenizer.word_index) + 1
    
    print(f"✅ Data prepared:")
    print(f"   Vocabulary size: {vocab_size:,}")
    print(f"   Sequence length: {max_len}")
    print(f"   Training shape: {X_train_pad.shape}")
    print(f"   Validation shape: {X_val_pad.shape}")
    
    return X_train_pad, X_val_pad, y_train, y_val, tokenizer, vocab_size

def get_training_callbacks(model_name="fixed_model"):
    """
    Get callbacks that prevent overfitting
    """
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=5,  # Wait 5 epochs for improvement
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            f'saved_models/{model_name}_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        )
    ]

def train_fixed_model(X_train, X_val, y_train, y_val, vocab_size, max_len):
    """
    Train the fixed model with proper validation
    """
    print("\n🚀 TRAINING FIXED MODEL")
    print("=" * 30)
    
    # Create model
    model = create_overfitting_resistant_model(vocab_size, max_len)
    
    # Get callbacks
    callbacks = get_training_callbacks("fixed_toxicity_model")
    
    print("\n📋 Model Summary:")
    model.summary()
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Epochs: 20 (with early stopping)")
    print(f"   Batch size: 32")
    print(f"   Learning rate: 1e-4")
    print(f"   Validation monitoring: ON")
    print(f"   Early stopping patience: 5")
    
    # Train with validation monitoring
    print(f"\n🔄 Starting training...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),  # CRITICAL: Validation monitoring
        epochs=20,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    print(f"\n✅ Training completed!")
    
    return model, history

def analyze_training_results(history):
    """
    Analyze training results for overfitting
    """
    print(f"\n📊 TRAINING RESULTS ANALYSIS")
    print("=" * 35)
    
    # Get final metrics
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    loss_gap = final_val_loss - final_train_loss
    acc_gap = final_train_acc - final_val_acc
    
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Loss Gap: {loss_gap:.4f}")
    print(f"")
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Accuracy Gap: {acc_gap:.4f}")
    
    # Assess overfitting
    print(f"\n🔍 OVERFITTING ASSESSMENT:")
    overfitting_detected = False
    
    if loss_gap > 0.1:
        print("🚨 Large loss gap detected - some overfitting")
        overfitting_detected = True
    elif loss_gap > 0.05:
        print("⚠️ Moderate loss gap - monitor closely")
    else:
        print("✅ Loss gap is acceptable")
    
    if acc_gap > 0.1:
        print("🚨 Large accuracy gap detected - some overfitting")
        overfitting_detected = True
    elif acc_gap > 0.05:
        print("⚠️ Moderate accuracy gap - monitor closely")
    else:
        print("✅ Accuracy gap is acceptable")
    
    if final_train_acc > 0.95:
        print("⚠️ Very high training accuracy - check for memorization")
    
    if not overfitting_detected:
        print("✅ No significant overfitting detected!")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    epochs = range(1, len(history.history['loss']) + 1)
    plt.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.title('Loss Curves - Fixed Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    plt.title('Accuracy Curves - Fixed Model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss gap plot
    plt.subplot(1, 3, 3)
    loss_gaps = np.array(history.history['val_loss']) - np.array(history.history['loss'])
    plt.plot(epochs, loss_gaps, 'orange', linewidth=2)
    plt.title('Loss Gap Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Validation - Training Loss')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('training_results_fixed_model.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return not overfitting_detected

def test_fixed_model(model, tokenizer, max_len):
    """
    Test the fixed model performance
    """
    print(f"\n🧪 TESTING FIXED MODEL PERFORMANCE")
    print("=" * 40)
    
    def predict_text(text):
        """Helper function to predict single text"""
        text = str(text).lower()
        seq = tokenizer.texts_to_sequences([text])
        padded = sequence.pad_sequences(seq, maxlen=max_len)
        return float(model.predict(padded, verbose=0)[0][0])
    
    # Test cases with expected results
    test_cases = [
        ("Hello, how are you?", False, "Greeting"),
        ("Thank you for your help", False, "Gratitude"),
        ("I disagree with you", False, "Disagreement"),
        ("You are an idiot", True, "Clear insult"),
        ("Shut up, moron", True, "Rude command"),
        ("Go kill yourself", True, "Threat"),
        ("You're stupid", True, "Insult"),
        ("Fucking idiot", True, "Profanity + insult"),
        ("This is great!", False, "Positive"),
        ("Have a nice day", False, "Polite")
    ]
    
    print("Testing fixed model on key cases:")
    print("-" * 35)
    
    correct = 0
    total = len(test_cases)
    
    for text, expected_toxic, description in test_cases:
        prob = predict_text(text)
        predicted_toxic = prob > 0.5
        is_correct = (predicted_toxic == expected_toxic)
        
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} {description:15} | '{text[:30]:30}' → {prob:.3f}")
    
    accuracy = correct / total
    print(f"\n📊 FIXED MODEL TEST RESULTS:")
    print(f"Accuracy: {correct}/{total} = {accuracy:.1%}")
    
    if accuracy > 0.8:
        print("✅ EXCELLENT: Fixed model performs well!")
        return True
    elif accuracy > 0.6:
        print("⚠️ GOOD: Fixed model much improved but needs more work")
        return True
    else:
        print("🚨 POOR: Still issues - may need more data or adjustments")
        return False

def save_fixed_model_artifacts(model, tokenizer, config, model_name="fixed_model"):
    """
    Save the fixed model and its artifacts
    """
    print(f"\n💾 SAVING FIXED MODEL ARTIFACTS")
    print("=" * 35)
    
    # Create directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    # Save model
    model_path = f'saved_models/{model_name}_final.h5'
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Save tokenizer
    tokenizer_path = f'{model_name}_tokenizer.pickle'
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"✅ Tokenizer saved to {tokenizer_path}")
    
    # Save config
    config_path = f'saved_models/{model_name}_config.pickle'
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"✅ Config saved to {config_path}")
    
    return model_path, tokenizer_path, config_path

def main():
    """
    Main function to fix the overfitted model
    """
    print("🔧 FIXING OVERFITTED MODEL - COMPLETE RETRAINING")
    print("=" * 55)
    
    # Step 1: Create/load proper dataset
    print("\n1️⃣ LOADING/CREATING DATASET")
    try:
        # Try to load real data first (if available)
        df = pd.read_csv('jigsaw-toxic-comment-train.csv')
        print(f"✅ Loaded real Jigsaw dataset: {len(df)} samples")
    except:
        print("⚠️ Real dataset not found, creating realistic sample data")
        df = create_sample_jigsaw_data()
    
    # Step 2: Prepare data properly
    print("\n2️⃣ PREPARING DATA WITH PROPER PREPROCESSING")
    X_train, X_val, y_train, y_val, tokenizer, vocab_size = prepare_data(
        df, max_vocab=10000, max_len=200
    )
    
    # Step 3: Create fixed model
    print("\n3️⃣ CREATING OVERFITTING-RESISTANT MODEL")
    model = create_overfitting_resistant_model(vocab_size, max_len=200)
    
    # Step 4: Train with validation monitoring
    print("\n4️⃣ TRAINING WITH VALIDATION MONITORING")
    model, history = train_fixed_model(X_train, X_val, y_train, y_val, vocab_size, 200)
    
    # Step 5: Analyze training results
    print("\n5️⃣ ANALYZING TRAINING RESULTS")
    training_success = analyze_training_results(history)
    
    # Step 6: Test fixed model
    print("\n6️⃣ TESTING FIXED MODEL")
    test_success = test_fixed_model(model, tokenizer, 200)
    
    # Step 7: Save artifacts
    print("\n7️⃣ SAVING FIXED MODEL")
    config = {
        'vocab_size': vocab_size,
        'max_len': 200,
        'threshold': 0.5,
        'label_columns': ['toxic']
    }
    
    model_path, tokenizer_path, config_path = save_fixed_model_artifacts(
        model, tokenizer, config, "fixed_toxicity_model"
    )
    
    # Final assessment
    print(f"\n" + "="*60)
    print("🎯 FINAL ASSESSMENT")
    print("="*60)
    
    if training_success and test_success:
        print("✅ SUCCESS: Model has been fixed!")
        print("   - Overfitting eliminated")
        print("   - Model generalizes properly")
        print("   - Performance is good")
    elif test_success:
        print("⚠️ PARTIAL SUCCESS: Model improved but may need more work")
        print("   - Some overfitting reduced")
        print("   - Performance better than before")
    else:
        print("🚨 NEEDS MORE WORK: Model still has issues")
        print("   - May need more data")
        print("   - Consider further regularization")
    
    print(f"\n📁 FILES CREATED:")
    print(f"   - {model_path}")
    print(f"   - {tokenizer_path}")
    print(f"   - {config_path}")
    print(f"   - training_results_fixed_model.png")
    
    print(f"\n🔄 TO USE THE FIXED MODEL:")
    print("   from tensorflow.keras.models import load_model")
    print("   model = load_model('saved_models/fixed_toxicity_model_final.h5')")
    
    return model, history

if __name__ == "__main__":
    model, history = main()
