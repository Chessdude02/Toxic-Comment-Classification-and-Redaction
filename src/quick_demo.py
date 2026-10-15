"""
Quick Demo Script for Toxic Comment Classification System

This script creates a simple demonstration of the toxicity detection system
using a lightweight model trained on sample data, so you can test the functionality
immediately without needing the full Kaggle dataset.
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import re

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    print("📊 Creating sample dataset...")
    
    # Sample toxic and non-toxic comments
    sample_data = {
        'comment_text': [
            # Clean comments
            "This is a great article, thanks for sharing!",
            "I appreciate your perspective on this topic.",
            "Could you provide more information about this?",
            "Thanks for the helpful response.",
            "I'm looking forward to reading more.",
            "Great work on this project!",
            "This is very informative and well-written.",
            "I learned something new today, thank you.",
            "Excellent analysis and clear explanation.",
            "Keep up the good work!",
            
            # Toxic comments (mild examples for demo)
            "This is completely stupid and wrong.",
            "You don't know what you're talking about, idiot.",
            "This is the dumbest thing I've ever read.",
            "What a moronic opinion.",
            "This author is clearly an imbecile.",
            "Shut up, you have no idea.",
            "This is garbage and so are you.",
            "Stop being such a fool about this.",
            "You're wrong and stupid for thinking that.",
            "What kind of idiot writes this nonsense?",
            
            # Neutral comments
            "I disagree with this point of view.",
            "This might not be entirely accurate.", 
            "I have a different opinion on this matter.",
            "Could there be alternative explanations?",
            "This seems questionable to me.",
            "I'm not convinced by this argument.",
            "There might be some issues with this approach.",
            "This could be improved in several ways.",
            "I'm not sure this is the best solution.",
            "This may need further consideration."
        ] * 50,  # Repeat to get more samples
        
        'toxic': [0] * 10 * 50 + [1] * 10 * 50 + [0] * 10 * 50,
        'severe_toxic': [0] * 10 * 50 + [0] * 10 * 50 + [0] * 10 * 50,
        'obscene': [0] * 10 * 50 + [0] * 10 * 50 + [0] * 10 * 50,  
        'threat': [0] * 10 * 50 + [0] * 10 * 50 + [0] * 10 * 50,
        'insult': [0] * 10 * 50 + [1] * 10 * 50 + [0] * 10 * 50,
        'identity_hate': [0] * 10 * 50 + [0] * 10 * 50 + [0] * 10 * 50
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add some variety to toxic labels
    np.random.seed(42)
    for i in range(500, 1000):  # Toxic comments range
        if np.random.random() < 0.3:
            df.loc[i, 'severe_toxic'] = 1
        if np.random.random() < 0.4:
            df.loc[i, 'obscene'] = 1
        if np.random.random() < 0.1:
            df.loc[i, 'threat'] = 1
        if np.random.random() < 0.1:
            df.loc[i, 'identity_hate'] = 1
    
    print(f"✅ Created dataset with {len(df)} samples")
    return df

def clean_text(text):
    """Clean text for processing"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_simple_model(vocab_size, max_len=128, num_classes=6):
    """Create a simple LSTM model for demo"""
    model = Sequential([
        Embedding(vocab_size, 100, input_length=max_len),
        SpatialDropout1D(0.3),
        LSTM(64, dropout=0.3, recurrent_dropout=0.3),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_demo_model():
    """Train a simple model for demonstration"""
    print("🚀 Training demo model...")
    
    # Create sample data
    df = create_sample_dataset()
    
    # Clean text
    df['clean_text'] = df['comment_text'].apply(clean_text)
    
    # Prepare features and labels
    X = df['clean_text'].values
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    y = df[label_columns].values
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y[:, 0]
    )
    
    # Tokenize
    print("🔤 Creating tokenizer...")
    tokenizer = text.Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(list(X_train) + list(X_val))
    
    max_len = 128  # Shorter for demo
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    
    X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=max_len)
    X_val_pad = sequence.pad_sequences(X_val_seq, maxlen=max_len)
    
    vocab_size = len(tokenizer.word_index) + 1
    
    # Create and train model
    print("🏗️ Building model...")
    model = create_simple_model(vocab_size, max_len, len(label_columns))
    
    print(f"📋 Model Summary:")
    model.summary()
    
    print("🎯 Training model...")
    history = model.fit(
        X_train_pad, y_train,
        batch_size=32,
        epochs=5,
        validation_data=(X_val_pad, y_val),
        verbose=1
    )
    
    # Create directories
    os.makedirs('saved_models', exist_ok=True)
    
    # Save model
    model.save('saved_models/demo_toxicity_classifier.h5')
    print("✅ Model saved to saved_models/demo_toxicity_classifier.h5")
    
    # Save tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("✅ Tokenizer saved to tokenizer.pickle")
    
    # Save config
    config = {
        'label_columns': label_columns,
        'max_len': max_len,
        'vocab_size': vocab_size,
        'threshold': 0.5,
        'best_model': 'demo'
    }
    
    with open('saved_models/config.pickle', 'wb') as f:
        pickle.dump(config, f)
    print("✅ Configuration saved to saved_models/config.pickle")
    
    # Test the model
    print("\n🧪 Testing model with sample messages...")
    test_messages = [
        "This is a wonderful day!",
        "You're such an idiot!",
        "Thanks for your help.",
        "This is stupid and wrong."
    ]
    
    for msg in test_messages:
        processed = tokenizer.texts_to_sequences([clean_text(msg)])
        padded = sequence.pad_sequences(processed, maxlen=max_len)
        pred = model.predict(padded, verbose=0)[0]
        
        toxic_prob = pred[0]  # Main toxic probability
        print(f"'{msg}' -> Toxic probability: {toxic_prob:.3f}")
    
    print("✅ Demo model training completed!")
    return True

if __name__ == "__main__":
    print("🚀 Setting up Toxic Comment Classification Demo")
    print("=" * 60)
    
    # Check if model already exists
    if os.path.exists('saved_models/demo_toxicity_classifier.h5') and os.path.exists('tokenizer.pickle'):
        print("✅ Demo model already exists!")
        print("Skipping training. Delete the saved_models folder to retrain.")
    else:
        print("Creating demo model with sample data...")
        success = train_demo_model()
        
        if not success:
            print("❌ Failed to create demo model")
            exit(1)
    
    print("\n🎉 Demo setup complete!")
    print("Now you can:")
    print("1. Run the web app: python toxicity_web_app.py")
    print("2. Use the Python module directly")
    print("3. Open the Jupyter notebook for detailed training")
    
    # Quick test
    print("\n🧪 Quick functionality test:")
    try:
        from toxicity_redactor import load_pretrained_model
        
        # Try to load with demo model
        import tensorflow as tf
        model = tf.keras.models.load_model('saved_models/demo_toxicity_classifier.h5')
        
        with open('tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
            
        with open('saved_models/config.pickle', 'rb') as f:
            config = pickle.load(f)
        
        from toxicity_redactor import ToxicityRedactor
        redactor = ToxicityRedactor(
            model=model,
            tokenizer=tokenizer, 
            label_columns=config['label_columns'],
            threshold=config['threshold'],
            max_len=config['max_len']
        )
        
        # Test message
        test_msg = "This is a test message."
        result = redactor.classify_toxicity(test_msg)
        print(f"✅ Test successful! Toxicity probability: {result['max_toxicity_probability']:.3f}")
        
        print("\n🌐 Ready to run the web application!")
        print("Execute: python toxicity_web_app.py")
        
    except Exception as e:
        print(f"⚠️ Test failed: {str(e)}")
        print("You may need to install additional dependencies or retrain the model.")
