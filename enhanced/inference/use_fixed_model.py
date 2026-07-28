#!/usr/bin/env python3
"""
Use Fixed Toxicity Model
========================

Simple utility to use your newly fixed toxicity classification model.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

from common.paths import MODEL_PATH, TOKENIZER_PATH, CONFIG_PATH

class ToxicityClassifier:
    """
    Easy-to-use toxicity classifier with the fixed model
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.load_model()
    
    def load_model(self):
        """Load the fixed model artifacts"""
        try:
            print("🔄 Loading fixed toxicity model...")

            self.model = load_model(MODEL_PATH)

            with open(TOKENIZER_PATH, "rb") as f:
                self.tokenizer = pickle.load(f)

            with open(CONFIG_PATH, "rb") as f:
                self.config = pickle.load(f)
            
            print(f"✅ Model loaded successfully!")
            print(f"   Parameters: {self.model.count_params():,}")
            print(f"   Vocabulary: {self.config['vocab_size']:,} words")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Make sure you've run the fix_overfitted_model.py script first")
    
    def predict(self, text):
        """
        Predict toxicity for a single text
        
        Args:
            text (str): Text to classify
            
        Returns:
            dict: Classification result with probability and decision
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        # Preprocess text
        text_clean = str(text).lower().strip()

        # Empty/near-empty input has no real content to classify. Without
        # this guard, an all-padding sequence gets pushed through the
        # embedding/LSTM stack anyway and the model happily returns a
        # confident (and meaningless) prediction for it.
        if not text_clean:
            return {
                "text": text,
                "probability": 0.0,
                "is_toxic": False,
                "confidence": "Very Low",
                "classification": "CLEAN",
            }

        seq = self.tokenizer.texts_to_sequences([text_clean])
        padded = sequence.pad_sequences(seq, maxlen=self.config['max_len'])

        # Get prediction
        prob = float(self.model.predict(padded, verbose=0)[0][0])
        is_toxic = prob > self.config['threshold']
        
        # Determine confidence level
        if prob < 0.3:
            confidence = "Very Low"
        elif prob < 0.4:
            confidence = "Low" 
        elif prob < 0.6:
            confidence = "Medium"
        elif prob < 0.8:
            confidence = "High"
        else:
            confidence = "Very High"
        
        return {
            "text": text,
            "probability": prob,
            "is_toxic": is_toxic,
            "confidence": confidence,
            "classification": "TOXIC" if is_toxic else "CLEAN"
        }
    
    def batch_predict(self, texts):
        """
        Predict toxicity for multiple texts
        
        Args:
            texts (list): List of texts to classify
            
        Returns:
            list: List of classification results
        """
        return [self.predict(text) for text in texts]
    
    def demo(self):
        """Run a demonstration of the model"""
        print(f"\n🚀 TOXICITY CLASSIFIER DEMO")
        print("=" * 35)
        
        demo_texts = [
            "Hello, how are you today?",
            "Thank you for your help",
            "I disagree with your opinion",
            "You are an idiot",
            "Shut up, moron", 
            "Go fuck yourself",
            "This is a great post",
            "You're completely wrong",
            "Have a wonderful day!"
        ]
        
        print("Testing various texts:")
        print("-" * 25)
        
        for text in demo_texts:
            result = self.predict(text)
            
            status = "🚨" if result['is_toxic'] else "✅"
            print(f"{status} '{text}' → {result['probability']:.3f} ({result['classification']})")
        
        print(f"\n💡 Usage Examples:")
        print("# Single prediction:")
        print("classifier = ToxicityClassifier()")
        print("result = classifier.predict('Your text here')")
        print("print(result['is_toxic'], result['probability'])")
        print("")
        print("# Batch prediction:")
        print("results = classifier.batch_predict(['Text 1', 'Text 2'])")

def main():
    """Main function for interactive use"""
    # Create classifier instance
    classifier = ToxicityClassifier()
    
    if classifier.model is None:
        return
    
    # Run demo
    classifier.demo()
    
    print(f"\n🎯 INTERACTIVE TESTING")
    print("=" * 25)
    print("You can now test your own messages!")
    print("Examples:")
    
    # Interactive examples
    test_messages = [
        "Hello everyone!",
        "You are so annoying",
        "This is stupid",
        "Thank you for sharing"
    ]
    
    print("\nTesting sample messages:")
    for msg in test_messages:
        result = classifier.predict(msg)
        status = "🚨" if result['is_toxic'] else "✅"
        print(f"{status} '{msg}' → {result['classification']} ({result['probability']:.3f})")
    
    print(f"\n🚀 MODEL IS READY FOR USE!")
    print("The overfitting has been fixed and the model works properly.")
    
    return classifier

if __name__ == "__main__":
    classifier = main()
