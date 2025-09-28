#!/usr/bin/env python3
"""
Model Performance Testing Script
===============================

This script tests your trained model against various inputs to evaluate:
1. Overall performance quality
2. Signs of overfitting (poor generalization)
3. Edge cases and robustness
4. Bias and fairness issues
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class ModelTester:
    """
    Comprehensive model testing for toxicity classification
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load model, tokenizer, and configuration"""
        try:
            print("🔄 Loading model artifacts...")
            
            # Load model
            self.model = load_model("saved_models/demo_toxicity_classifier.h5")
            print("✅ Model loaded successfully")
            
            # Load tokenizer
            with open("tokenizer.pickle", "rb") as f:
                self.tokenizer = pickle.load(f)
            print("✅ Tokenizer loaded successfully")
            
            # Load config
            with open("saved_models/config.pickle", "rb") as f:
                self.config = pickle.load(f)
            print("✅ Configuration loaded successfully")
            
            print(f"📊 Model info: {self.model.count_params():,} parameters")
            print(f"📊 Vocab size: {self.config['vocab_size']:,}")
            print(f"📊 Max length: {self.config['max_len']}")
            
        except Exception as e:
            print(f"❌ Error loading artifacts: {e}")
            return False
        
        return True
    
    def preprocess_text(self, text):
        """Preprocess text for model input"""
        if pd.isna(text):
            return ""
        
        # Basic cleaning (same as in training)
        text = str(text).lower()
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Tokenize and pad
        sequence_data = self.tokenizer.texts_to_sequences([text])
        padded = sequence.pad_sequences(sequence_data, maxlen=self.config['max_len'])
        
        return padded
    
    def predict_toxicity(self, text):
        """Predict toxicity for a single text"""
        processed = self.preprocess_text(text)
        prediction = self.model.predict(processed, verbose=0)[0]
        
        # Handle both single and multi-class outputs
        if len(prediction) == 1:
            return {"toxic": float(prediction[0])}
        else:
            return {
                label: float(prediction[i]) 
                for i, label in enumerate(self.config['label_columns'])
            }
    
    def test_basic_functionality(self):
        """Test basic model functionality"""
        print("\n🔍 BASIC FUNCTIONALITY TEST")
        print("=" * 40)
        
        test_cases = [
            ("Hello, how are you today?", False, "Simple greeting"),
            ("I disagree with your opinion.", False, "Mild disagreement"),
            ("You are an idiot!", True, "Clear insult"),
            ("This is stupid and wrong.", True, "Mild toxicity"),
            ("Go kill yourself!", True, "Severe threat"),
            ("", False, "Empty text"),
            ("a", False, "Single character"),
            ("The weather is nice today.", False, "Neutral statement")
        ]
        
        results = []
        
        print("Testing basic cases:")
        print("-" * 20)
        
        for text, expected_toxic, description in test_cases:
            try:
                predictions = self.predict_toxicity(text)
                main_prob = predictions.get('toxic', list(predictions.values())[0])
                is_toxic = main_prob > self.config['threshold']
                
                # Check if prediction matches expectation
                correct = (is_toxic == expected_toxic)
                status = "✅" if correct else "❌"
                
                print(f"{status} {description}")
                print(f"   Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                print(f"   Predicted: {main_prob:.3f} ({'Toxic' if is_toxic else 'Clean'})")
                print(f"   Expected: {'Toxic' if expected_toxic else 'Clean'}")
                print()
                
                results.append({
                    'text': text,
                    'description': description,
                    'predicted_prob': main_prob,
                    'predicted_toxic': is_toxic,
                    'expected_toxic': expected_toxic,
                    'correct': correct
                })
                
            except Exception as e:
                print(f"❌ Error testing '{text[:30]}...': {e}")
                results.append({
                    'text': text,
                    'description': description,
                    'predicted_prob': 0.0,
                    'predicted_toxic': False,
                    'expected_toxic': expected_toxic,
                    'correct': False
                })
        
        # Calculate accuracy
        correct_predictions = sum(1 for r in results if r['correct'])
        accuracy = correct_predictions / len(results)
        
        print(f"📊 BASIC TEST RESULTS:")
        print(f"Correct predictions: {correct_predictions}/{len(results)}")
        print(f"Accuracy: {accuracy:.1%}")
        
        if accuracy < 0.7:
            print("🚨 LOW ACCURACY: Model performing poorly on basic cases!")
        elif accuracy < 0.85:
            print("⚠️ MODERATE ACCURACY: Some issues with basic cases")
        else:
            print("✅ GOOD ACCURACY: Basic functionality working well")
        
        return results
    
    def test_edge_cases(self):
        """Test edge cases that reveal overfitting"""
        print("\n🔍 EDGE CASE TESTING (Overfitting Detection)")
        print("=" * 50)
        
        edge_cases = [
            # Subtle cases that overfitted models often get wrong
            ("I think you might be mistaken about this.", False, "Polite disagreement"),
            ("That's not quite right, let me explain.", False, "Gentle correction"),
            ("You're absolutely wrong about everything!", True, "Strong disagreement"),
            ("This idea is completely ridiculous.", True, "Dismissive language"),
            
            # Cases with context sensitivity
            ("You killed it in that presentation!", False, "Positive use of 'killed'"),
            ("I'm going to kill some time before the meeting.", False, "Innocent use of 'kill'"),
            ("I hate waiting in long lines.", False, "Hate in non-toxic context"),
            ("I hate you so much!", True, "Hate in toxic context"),
            
            # Variations and synonyms
            ("You're an absolute moron!", True, "Direct insult"),
            ("You're intellectually challenged.", True, "Polite insult"),
            ("That's not very smart.", False, "Mild criticism"),
            
            # Length variations
            ("Bad.", True, "Very short toxic"),
            ("This is bad bad bad bad bad bad bad.", True, "Repetitive"),
            ("I think this approach might not be the best way to handle this situation, but I respect your perspective.", False, "Long polite text"),
            
            # Misspellings and variations
            ("You are an idoit!", True, "Misspelled insult"),
            ("Ur so dum!", True, "Text speak insult"),
            ("YOURE STUPID!!!", True, "Caps and punctuation"),
        ]
        
        edge_results = []
        
        print("Testing edge cases:")
        print("-" * 20)
        
        for text, expected_toxic, description in edge_cases:
            try:
                predictions = self.predict_toxicity(text)
                main_prob = predictions.get('toxic', list(predictions.values())[0])
                is_toxic = main_prob > self.config['threshold']
                
                correct = (is_toxic == expected_toxic)
                status = "✅" if correct else "❌"
                
                print(f"{status} {description}")
                print(f"   Text: '{text}'")
                print(f"   Predicted: {main_prob:.3f} ({'Toxic' if is_toxic else 'Clean'})")
                
                if not correct:
                    print(f"   ⚠️ Expected: {'Toxic' if expected_toxic else 'Clean'}")
                
                print()
                
                edge_results.append({
                    'text': text,
                    'description': description,
                    'predicted_prob': main_prob,
                    'predicted_toxic': is_toxic,
                    'expected_toxic': expected_toxic,
                    'correct': correct
                })
                
            except Exception as e:
                print(f"❌ Error testing edge case: {e}")
        
        # Analyze edge case performance
        edge_accuracy = sum(1 for r in edge_results if r['correct']) / len(edge_results)
        
        print(f"📊 EDGE CASE RESULTS:")
        print(f"Edge case accuracy: {edge_accuracy:.1%}")
        
        if edge_accuracy < 0.6:
            print("🚨 POOR GENERALIZATION: Model fails on edge cases - likely overfitted!")
        elif edge_accuracy < 0.8:
            print("⚠️ MODERATE GENERALIZATION: Some edge case issues")
        else:
            print("✅ GOOD GENERALIZATION: Handles edge cases well")
        
        return edge_results
    
    def test_robustness(self):
        """Test model robustness to variations"""
        print("\n🔍 ROBUSTNESS TESTING")
        print("=" * 30)
        
        base_text = "You are stupid"
        variations = [
            "You are stupid",
            "You are stupid.",
            "You are stupid!",
            "You are stupid!!!",
            "YOU ARE STUPID",
            "you are stupid",
            "You  are  stupid",  # Extra spaces
            "You are so stupid",
            "You are really stupid",
            "You are absolutely stupid",
        ]
        
        print("Testing robustness to text variations:")
        print(f"Base text: '{base_text}'")
        print("-" * 30)
        
        base_pred = self.predict_toxicity(base_text)
        base_prob = base_pred.get('toxic', list(base_pred.values())[0])
        
        robust_results = []
        
        for variation in variations:
            pred = self.predict_toxicity(variation)
            prob = pred.get('toxic', list(pred.values())[0])
            diff = abs(prob - base_prob)
            
            print(f"'{variation}' → {prob:.3f} (diff: {diff:.3f})")
            
            robust_results.append({
                'text': variation,
                'probability': prob,
                'difference': diff
            })
        
        # Analyze robustness
        max_diff = max(r['difference'] for r in robust_results)
        avg_diff = np.mean([r['difference'] for r in robust_results])
        
        print(f"\n📊 ROBUSTNESS ANALYSIS:")
        print(f"Maximum difference: {max_diff:.3f}")
        print(f"Average difference: {avg_diff:.3f}")
        
        if max_diff > 0.3:
            print("🚨 POOR ROBUSTNESS: Large variations in similar inputs!")
            print("   This suggests overfitting to specific text patterns")
        elif max_diff > 0.1:
            print("⚠️ MODERATE ROBUSTNESS: Some sensitivity to variations")
        else:
            print("✅ GOOD ROBUSTNESS: Consistent predictions")
        
        return robust_results
    
    def test_confidence_calibration(self):
        """Test if model confidence is well-calibrated"""
        print("\n🔍 CONFIDENCE CALIBRATION TEST")
        print("=" * 40)
        
        # Test a range of texts with varying toxicity levels
        confidence_tests = [
            ("Hello there!", 0.0, "Clearly non-toxic"),
            ("I disagree with you.", 0.1, "Mild disagreement"),
            ("That's not right.", 0.2, "Gentle correction"),
            ("You're wrong about this.", 0.4, "Direct disagreement"),
            ("You don't know what you're talking about.", 0.6, "Moderate rudeness"),
            ("You're an idiot.", 0.8, "Clear insult"),
            ("You're a fucking moron!", 0.95, "Severe toxicity")
        ]
        
        calibration_results = []
        
        print("Testing confidence calibration:")
        print("-" * 30)
        
        for text, expected_confidence, description in confidence_tests:
            pred = self.predict_toxicity(text)
            actual_confidence = pred.get('toxic', list(pred.values())[0])
            
            diff = abs(actual_confidence - expected_confidence)
            
            print(f"{description}:")
            print(f"  Text: '{text}'")
            print(f"  Expected: {expected_confidence:.2f}, Actual: {actual_confidence:.3f}")
            print(f"  Difference: {diff:.3f}")
            print()
            
            calibration_results.append({
                'text': text,
                'expected': expected_confidence,
                'actual': actual_confidence,
                'difference': diff,
                'description': description
            })
        
        # Analyze calibration
        avg_error = np.mean([r['difference'] for r in calibration_results])
        max_error = max(r['difference'] for r in calibration_results)
        
        print(f"📊 CALIBRATION ANALYSIS:")
        print(f"Average prediction error: {avg_error:.3f}")
        print(f"Maximum prediction error: {max_error:.3f}")
        
        if avg_error > 0.3:
            print("🚨 POOR CALIBRATION: Model confidence is unreliable!")
            print("   This often indicates overfitting")
        elif avg_error > 0.15:
            print("⚠️ MODERATE CALIBRATION: Some confidence issues")
        else:
            print("✅ GOOD CALIBRATION: Model confidence seems reasonable")
        
        # Visualize calibration
        plt.figure(figsize=(10, 6))
        
        expected_vals = [r['expected'] for r in calibration_results]
        actual_vals = [r['actual'] for r in calibration_results]
        
        plt.scatter(expected_vals, actual_vals, s=100, alpha=0.7)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        plt.xlabel('Expected Toxicity')
        plt.ylabel('Predicted Toxicity')
        plt.title('Model Confidence Calibration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text labels
        for i, result in enumerate(calibration_results):
            plt.annotate(f"{i+1}", (expected_vals[i], actual_vals[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.show()
        
        return calibration_results
    
    def test_adversarial_examples(self):
        """Test against adversarial examples that might fool an overfitted model"""
        print("\n🔍 ADVERSARIAL TESTING")
        print("=" * 30)
        
        adversarial_cases = [
            # Disguised toxicity
            ("You are so... intelligent (NOT!)", True, "Sarcastic insult"),
            ("What a 'genius' you are", True, "Sarcastic with quotes"),
            ("You're the smartest person... in your dreams", True, "Backhanded compliment"),
            
            # Context-dependent
            ("I love how wrong you are", True, "Disguised criticism"),
            ("Thank you for being so helpful... NOT", True, "Sarcastic gratitude"),
            
            # Innocent but flagged words
            ("I hate brussels sprouts", False, "Hate for food"),
            ("This kills the mood", False, "Metaphorical kill"),
            ("That's insane how good this is", False, "Positive use of 'insane'"),
            
            # Borderline cases
            ("You clearly don't understand", False, "Condescending but not toxic"),
            ("That's the dumbest thing I've heard", True, "Direct insult"),
            ("I find your argument unconvincing", False, "Academic disagreement"),
        ]
        
        adversarial_results = []
        
        print("Testing adversarial examples:")
        print("-" * 30)
        
        for text, expected_toxic, description in adversarial_cases:
            try:
                predictions = self.predict_toxicity(text)
                main_prob = predictions.get('toxic', list(predictions.values())[0])
                is_toxic = main_prob > self.config['threshold']
                
                correct = (is_toxic == expected_toxic)
                status = "✅" if correct else "❌"
                
                print(f"{status} {description}")
                print(f"   Text: '{text}'")
                print(f"   Predicted: {main_prob:.3f} ({'Toxic' if is_toxic else 'Clean'})")
                
                if not correct:
                    print(f"   ❌ Expected: {'Toxic' if expected_toxic else 'Clean'}")
                
                print()
                
                adversarial_results.append({
                    'text': text,
                    'description': description,
                    'predicted_prob': main_prob,
                    'predicted_toxic': is_toxic,
                    'expected_toxic': expected_toxic,
                    'correct': correct
                })
                
            except Exception as e:
                print(f"❌ Error testing adversarial case: {e}")
        
        # Analyze adversarial performance
        adv_accuracy = sum(1 for r in adversarial_results if r['correct']) / len(adversarial_results)
        
        print(f"📊 ADVERSARIAL TEST RESULTS:")
        print(f"Adversarial accuracy: {adv_accuracy:.1%}")
        
        if adv_accuracy < 0.5:
            print("🚨 FAILS ADVERSARIAL TESTS: Model likely overfitted to training patterns!")
        elif adv_accuracy < 0.7:
            print("⚠️ MODERATE ADVERSARIAL PERFORMANCE: Some overfitting signs")
        else:
            print("✅ GOOD ADVERSARIAL PERFORMANCE: Model generalizes reasonably well")
        
        return adversarial_results
    
    def test_length_sensitivity(self):
        """Test sensitivity to text length"""
        print("\n🔍 LENGTH SENSITIVITY TEST")
        print("=" * 35)
        
        base_toxic = "You are stupid"
        base_clean = "Thank you"
        
        # Create variations of different lengths
        length_tests = []
        
        # Short to long toxic variations
        toxic_variations = [
            "stupid",
            "You're stupid",
            "You are really stupid",
            "You are so incredibly stupid it's amazing",
            "You are the most stupid person I have ever encountered in my entire life and I can't believe how stupid you are"
        ]
        
        # Short to long clean variations
        clean_variations = [
            "thanks",
            "Thank you",
            "Thank you very much",
            "Thank you so much for your helpful information",
            "I really appreciate you taking the time to provide such detailed and helpful information about this topic"
        ]
        
        print("Testing length sensitivity:")
        print("-" * 25)
        
        results = {"toxic": [], "clean": []}
        
        # Test toxic variations
        print("Toxic text variations:")
        for i, text in enumerate(toxic_variations):
            pred = self.predict_toxicity(text)
            prob = pred.get('toxic', list(pred.values())[0])
            length = len(text.split())
            
            print(f"  {length:2d} words: {prob:.3f} - '{text[:50]}{'...' if len(text) > 50 else ''}'")
            results["toxic"].append((length, prob))
        
        print("\nClean text variations:")
        for i, text in enumerate(clean_variations):
            pred = self.predict_toxicity(text)
            prob = pred.get('toxic', list(pred.values())[0])
            length = len(text.split())
            
            print(f"  {length:2d} words: {prob:.3f} - '{text[:50]}{'...' if len(text) > 50 else ''}'")
            results["clean"].append((length, prob))
        
        # Analyze length sensitivity
        toxic_probs = [p for _, p in results["toxic"]]
        clean_probs = [p for _, p in results["clean"]]
        
        toxic_variance = np.var(toxic_probs)
        clean_variance = np.var(clean_probs)
        
        print(f"\n📊 LENGTH SENSITIVITY ANALYSIS:")
        print(f"Toxic text variance: {toxic_variance:.4f}")
        print(f"Clean text variance: {clean_variance:.4f}")
        
        if toxic_variance > 0.1 or clean_variance > 0.1:
            print("🚨 HIGH LENGTH SENSITIVITY: Model predictions vary too much with length!")
            print("   This suggests overfitting to specific text lengths")
        else:
            print("✅ GOOD LENGTH CONSISTENCY: Predictions stable across lengths")
        
        return results
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "="*60)
        print("🚨 COMPREHENSIVE MODEL PERFORMANCE REPORT")
        print("="*60)
        
        # Run all tests
        basic_results = self.test_basic_functionality()
        edge_results = self.test_edge_cases()
        robust_results = self.test_robustness()
        calib_results = self.test_confidence_calibration()
        length_results = self.test_length_sensitivity()
        
        # Calculate overall metrics
        basic_acc = sum(1 for r in basic_results if r['correct']) / len(basic_results)
        edge_acc = sum(1 for r in edge_results if r['correct']) / len(edge_results)
        
        print(f"\n📊 OVERALL PERFORMANCE SUMMARY:")
        print("-" * 35)
        print(f"Basic functionality: {basic_acc:.1%}")
        print(f"Edge case handling: {edge_acc:.1%}")
        print(f"Model parameters: {self.model.count_params():,}")
        print(f"Vocabulary size: {self.config['vocab_size']:,}")
        
        # Overall assessment
        overall_score = (basic_acc + edge_acc) / 2
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print("-" * 25)
        
        if overall_score < 0.6:
            print("🚨 POOR PERFORMANCE: Model likely severely overfitted!")
            print("   Recommendation: Retrain with regularization")
        elif overall_score < 0.8:
            print("⚠️ MODERATE PERFORMANCE: Some overfitting signs")
            print("   Recommendation: Add more regularization")
        else:
            print("✅ GOOD PERFORMANCE: Model appears to generalize well")
        
        # Specific recommendations based on results
        print(f"\n💡 SPECIFIC RECOMMENDATIONS:")
        print("-" * 30)
        
        if basic_acc < 0.8:
            print("1. 🔄 Model struggles with basic cases - needs more diverse training data")
        
        if edge_acc < 0.7:
            print("2. 🛡️ Poor edge case performance - add regularization and data augmentation")
        
        if self.model.count_params() > 10_000_000:
            print("3. 📉 Model too complex - reduce size by 80-90%")
        
        print("4. ⏰ Always use validation monitoring during training")
        print("5. 📊 Implement k-fold cross-validation for better evaluation")
        
        return {
            'basic_accuracy': basic_acc,
            'edge_accuracy': edge_acc,
            'overall_score': overall_score,
            'model_params': self.model.count_params()
        }

def main():
    """Main testing function"""
    print("🧪 STARTING COMPREHENSIVE MODEL TESTING")
    print("=" * 50)
    
    # Initialize tester
    tester = ModelTester()
    
    if tester.model is None:
        print("❌ Could not load model. Please ensure model files exist.")
        return
    
    # Run comprehensive testing
    results = tester.generate_performance_report()
    
    print(f"\n🎯 FINAL VERDICT:")
    print("=" * 20)
    
    if results['overall_score'] < 0.6:
        print("🚨 MODEL IS OVERFITTED AND PERFORMS POORLY!")
        print("   Immediate retraining required")
    elif results['overall_score'] < 0.8:
        print("⚠️ MODEL SHOWS SIGNS OF OVERFITTING")
        print("   Improvements recommended")
    else:
        print("✅ MODEL PERFORMANCE IS ACCEPTABLE")
    
    print(f"\nNext steps: Use 'fixed_training_code.py' to retrain properly")

if __name__ == "__main__":
    main()
