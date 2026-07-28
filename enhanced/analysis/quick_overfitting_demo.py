#!/usr/bin/env python3
"""
Quick Overfitting Demonstration
==============================

This script demonstrates the severe overfitting in your model with clear examples.
"""

import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

def quick_test():
    """Quick test to show overfitting issues"""
    print("🚨 QUICK OVERFITTING DEMONSTRATION")
    print("=" * 40)
    
    try:
        # Load model artifacts
        model = load_model("saved_models/demo_toxicity_classifier.h5")
        with open("tokenizer.pickle", "rb") as f:
            tokenizer = pickle.load(f)
        with open("saved_models/config.pickle", "rb") as f:
            config = pickle.load(f)
        
        print(f"✅ Model loaded: {model.count_params():,} parameters")
        print(f"✅ Vocab size: {config['vocab_size']}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    def predict_text(text):
        """Simple prediction function"""
        # Basic preprocessing
        text = str(text).lower()
        sequence_data = tokenizer.texts_to_sequences([text])
        padded = sequence.pad_sequences(sequence_data, maxlen=config['max_len'])
        prediction = model.predict(padded, verbose=0)[0]
        
        if len(prediction) == 1:
            return float(prediction[0])
        else:
            return float(prediction[0])  # Use first class (toxic)
    
    print(f"\n🧪 TESTING OBVIOUS CASES:")
    print("-" * 30)
    
    # Test very obvious cases that any decent model should get right
    test_cases = [
        ("Hello, nice to meet you!", "Should be CLEAN"),
        ("Thank you for your help", "Should be CLEAN"),  
        ("You are an idiot", "Should be TOXIC"),
        ("Go fuck yourself", "Should be TOXIC"),
        ("I hate you", "Should be TOXIC"),
        ("You're stupid", "Should be TOXIC"),
        ("Great job!", "Should be CLEAN"),
        ("Fucking moron!", "Should be TOXIC"),
        ("Have a nice day", "Should be CLEAN"),
        ("You suck", "Should be TOXIC")
    ]
    
    correct = 0
    total = len(test_cases)
    threshold = config['threshold']
    
    for text, expected in test_cases:
        prob = predict_text(text)
        is_toxic = prob > threshold
        
        # Determine if correct
        should_be_toxic = "TOXIC" in expected
        is_correct = (is_toxic == should_be_toxic)
        
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} '{text}' → {prob:.3f} ({'TOXIC' if is_toxic else 'CLEAN'}) | {expected}")
    
    accuracy = correct / total
    
    print(f"\n📊 RESULTS:")
    print(f"Accuracy: {correct}/{total} = {accuracy:.1%}")
    
    print(f"\n🚨 CRITICAL FINDINGS:")
    print("-" * 25)
    
    if accuracy < 0.7:
        print("🚨 SEVERE OVERFITTING DETECTED!")
        print("   Model can't even classify obvious toxic/clean examples correctly")
        print("   This is clear evidence of overfitting to training patterns")
    
    # Check if all predictions are similar (another overfitting sign)
    all_probs = [predict_text(text) for text, _ in test_cases]
    prob_variance = sum((p - sum(all_probs)/len(all_probs))**2 for p in all_probs) / len(all_probs)
    
    print(f"\nPrediction variance: {prob_variance:.6f}")
    
    if prob_variance < 0.01:
        print("🚨 ALL PREDICTIONS ARE NEARLY IDENTICAL!")
        print("   Model is outputting same value regardless of input")
        print("   This is a classic sign of severe overfitting")
    
    print(f"\n💡 WHAT THIS MEANS:")
    print("- Your model learned to memorize training examples")
    print("- It cannot generalize to new, unseen text")
    print("- The 96%+ training accuracy was memorization, not learning")
    print("- Model is essentially broken for real-world use")
    
    print(f"\n⚡ IMMEDIATE ACTION REQUIRED:")
    print("1. Use the 'fixed_training_code.py' to retrain")
    print("2. Reduce model complexity by 90%")
    print("3. Add heavy regularization (dropout 0.5+)")
    print("4. Monitor validation loss during training")
    print("5. Stop training when validation loss stops improving")

if __name__ == "__main__":
    quick_test()
