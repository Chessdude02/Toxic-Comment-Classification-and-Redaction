#!/usr/bin/env python3
"""
Comprehensive Overfitting Analysis Tool
=====================================

This script analyzes your machine learning model for signs of overfitting and provides
actionable recommendations to improve generalization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import roc_auc_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

class OverfittingAnalyzer:
    """
    A comprehensive tool to detect and analyze overfitting in machine learning models
    """
    
    def __init__(self, model_path=None, config_path=None, tokenizer_path=None):
        self.model = None
        self.config = None
        self.tokenizer = None
        self.training_history = None
        
        # Load artifacts if paths provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
            
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.load_tokenizer(tokenizer_path)
    
    def load_model(self, model_path):
        """Load a trained Keras model"""
        try:
            self.model = load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def load_config(self, config_path):
        """Load model configuration"""
        try:
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
            print(f"✅ Configuration loaded from {config_path}")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
    
    def load_tokenizer(self, tokenizer_path):
        """Load tokenizer"""
        try:
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"✅ Tokenizer loaded from {tokenizer_path}")
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
    
    def analyze_model_complexity(self):
        """
        Analyze model complexity which can lead to overfitting
        """
        print("\n🔍 MODEL COMPLEXITY ANALYSIS")
        print("=" * 50)
        
        if self.model is None:
            print("❌ No model loaded for analysis")
            return
        
        # Model summary
        print("\n📊 Model Architecture:")
        self.model.summary()
        
        # Parameter count analysis
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\n📈 Parameter Analysis:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        
        # Complexity warnings
        complexity_warnings = []
        
        if total_params > 1_000_000:
            complexity_warnings.append("🚨 Very high parameter count (>1M) - high overfitting risk")
        elif total_params > 100_000:
            complexity_warnings.append("⚠️ High parameter count (>100K) - moderate overfitting risk")
        
        # Analyze layers for overfitting risk
        overfitting_layers = []
        dropout_layers = []
        
        for i, layer in enumerate(self.model.layers):
            layer_type = type(layer).__name__
            
            if 'Dense' in layer_type:
                if hasattr(layer, 'units') and layer.units > 256:
                    overfitting_layers.append(f"Layer {i} ({layer_type}): {layer.units} units - high capacity")
            
            if 'Dropout' in layer_type:
                dropout_layers.append(f"Layer {i} ({layer_type}): {layer.rate} rate")
        
        print(f"\n🧠 Layer Analysis:")
        if overfitting_layers:
            print("High-capacity layers (overfitting risk):")
            for layer in overfitting_layers:
                print(f"  • {layer}")
        
        if dropout_layers:
            print("Regularization layers found:")
            for layer in dropout_layers:
                print(f"  ✅ {layer}")
        else:
            complexity_warnings.append("⚠️ No dropout layers found - missing regularization")
        
        # Display warnings
        if complexity_warnings:
            print(f"\n🚨 COMPLEXITY WARNINGS:")
            for warning in complexity_warnings:
                print(f"  {warning}")
        else:
            print(f"\n✅ Model complexity appears reasonable")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'complexity_warnings': complexity_warnings,
            'has_regularization': len(dropout_layers) > 0
        }
    
    def analyze_training_history(self, history_dict=None):
        """
        Analyze training history for overfitting patterns
        """
        print("\n🔍 TRAINING HISTORY ANALYSIS")
        print("=" * 50)
        
        if history_dict is None:
            print("❌ No training history provided")
            print("💡 To analyze training history, pass the history.history dictionary")
            return
        
        self.training_history = history_dict
        
        # Check available metrics
        available_metrics = list(history_dict.keys())
        print(f"📊 Available metrics: {available_metrics}")
        
        # Analyze loss curves
        overfitting_signs = []
        
        if 'loss' in history_dict and 'val_loss' in history_dict:
            train_loss = history_dict['loss']
            val_loss = history_dict['val_loss']
            
            print(f"\n📈 Loss Analysis:")
            print(f"Final training loss: {train_loss[-1]:.4f}")
            print(f"Final validation loss: {val_loss[-1]:.4f}")
            print(f"Loss gap: {val_loss[-1] - train_loss[-1]:.4f}")
            
            # Check for overfitting patterns
            if len(val_loss) > 3:
                # Check if validation loss starts increasing while training loss decreases
                val_loss_trend = np.polyfit(range(len(val_loss)), val_loss, 1)[0]
                train_loss_trend = np.polyfit(range(len(train_loss)), train_loss, 1)[0]
                
                if val_loss_trend > 0 and train_loss_trend < -0.001:
                    overfitting_signs.append("🚨 Validation loss increasing while training loss decreasing")
                
                # Check gap between training and validation loss
                final_gap = val_loss[-1] - train_loss[-1]
                if final_gap > 0.1:
                    overfitting_signs.append(f"🚨 Large gap between train/val loss: {final_gap:.4f}")
                elif final_gap > 0.05:
                    overfitting_signs.append(f"⚠️ Moderate gap between train/val loss: {final_gap:.4f}")
        
        # Analyze accuracy curves
        if 'accuracy' in history_dict and 'val_accuracy' in history_dict:
            train_acc = history_dict['accuracy']
            val_acc = history_dict['val_accuracy']
            
            print(f"\n📊 Accuracy Analysis:")
            print(f"Final training accuracy: {train_acc[-1]:.4f}")
            print(f"Final validation accuracy: {val_acc[-1]:.4f}")
            print(f"Accuracy gap: {train_acc[-1] - val_acc[-1]:.4f}")
            
            # Check for overfitting in accuracy
            acc_gap = train_acc[-1] - val_acc[-1]
            if acc_gap > 0.1:
                overfitting_signs.append(f"🚨 Large training/validation accuracy gap: {acc_gap:.4f}")
            elif acc_gap > 0.05:
                overfitting_signs.append(f"⚠️ Moderate training/validation accuracy gap: {acc_gap:.4f}")
            
            # Check if training accuracy is very high (>95%)
            if train_acc[-1] > 0.95:
                overfitting_signs.append("🚨 Very high training accuracy (>95%) - possible memorization")
        
        # Plot training curves
        self._plot_training_curves(history_dict)
        
        # Display overfitting assessment
        print(f"\n🔍 OVERFITTING ASSESSMENT:")
        if overfitting_signs:
            print("🚨 OVERFITTING DETECTED:")
            for sign in overfitting_signs:
                print(f"  {sign}")
        else:
            print("✅ No clear signs of overfitting detected")
        
        return overfitting_signs
    
    def _plot_training_curves(self, history_dict):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History Analysis for Overfitting Detection', fontsize=16)
        
        # Plot loss
        if 'loss' in history_dict and 'val_loss' in history_dict:
            epochs = range(1, len(history_dict['loss']) + 1)
            axes[0, 0].plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
            axes[0, 0].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epochs')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Highlight overfitting region
            if len(history_dict['val_loss']) > 3:
                min_val_loss_epoch = np.argmin(history_dict['val_loss']) + 1
                axes[0, 0].axvline(x=min_val_loss_epoch, color='green', linestyle='--', 
                                 label=f'Best Val Loss (Epoch {min_val_loss_epoch})')
                axes[0, 0].legend()
        
        # Plot accuracy
        if 'accuracy' in history_dict and 'val_accuracy' in history_dict:
            epochs = range(1, len(history_dict['accuracy']) + 1)
            axes[0, 1].plot(epochs, history_dict['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
            axes[0, 1].plot(epochs, history_dict['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
            axes[0, 1].set_title('Accuracy Curves')
            axes[0, 1].set_xlabel('Epochs')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot loss gap over time
        if 'loss' in history_dict and 'val_loss' in history_dict:
            loss_gap = np.array(history_dict['val_loss']) - np.array(history_dict['loss'])
            epochs = range(1, len(loss_gap) + 1)
            axes[1, 0].plot(epochs, loss_gap, 'orange', linewidth=2)
            axes[1, 0].set_title('Training-Validation Loss Gap')
            axes[1, 0].set_xlabel('Epochs')
            axes[1, 0].set_ylabel('Val Loss - Train Loss')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Plot learning rate if available
        if 'lr' in history_dict:
            epochs = range(1, len(history_dict['lr']) + 1)
            axes[1, 1].plot(epochs, history_dict['lr'], 'purple', linewidth=2)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epochs')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        plt.show()
    
    def check_data_leakage(self, train_data=None, val_data=None, test_data=None):
        """
        Check for potential data leakage between train/validation/test sets
        """
        print("\n🔍 DATA LEAKAGE ANALYSIS")
        print("=" * 50)
        
        if train_data is None or val_data is None:
            print("❌ Training and validation data required for leakage analysis")
            return
        
        # Check for duplicate samples
        if isinstance(train_data, (list, np.ndarray)) and isinstance(val_data, (list, np.ndarray)):
            train_set = set(train_data) if isinstance(train_data[0], str) else set(map(str, train_data))
            val_set = set(val_data) if isinstance(val_data[0], str) else set(map(str, val_data))
            
            overlap = train_set.intersection(val_set)
            overlap_pct = len(overlap) / len(train_set) * 100
            
            print(f"📊 Data Overlap Analysis:")
            print(f"Training samples: {len(train_set):,}")
            print(f"Validation samples: {len(val_set):,}")
            print(f"Overlapping samples: {len(overlap):,} ({overlap_pct:.2f}%)")
            
            if overlap_pct > 5:
                print("🚨 SIGNIFICANT DATA LEAKAGE DETECTED! >5% overlap between train/val")
            elif overlap_pct > 1:
                print("⚠️ Minor data leakage detected (1-5% overlap)")
            else:
                print("✅ No significant data leakage detected")
            
            return overlap_pct
        else:
            print("❌ Unable to analyze data leakage - unsupported data format")
            return None
    
    def analyze_model_performance_gaps(self, train_metrics, val_metrics):
        """
        Analyze performance gaps between training and validation
        """
        print("\n🔍 PERFORMANCE GAP ANALYSIS")
        print("=" * 50)
        
        gap_analysis = {}
        overfitting_flags = []
        
        for metric_name in train_metrics:
            if metric_name in val_metrics:
                train_val = train_metrics[metric_name]
                val_val = val_metrics[metric_name]
                gap = train_val - val_val
                gap_pct = (gap / train_val) * 100 if train_val != 0 else 0
                
                gap_analysis[metric_name] = {
                    'train': train_val,
                    'val': val_val,
                    'gap': gap,
                    'gap_percentage': gap_pct
                }
                
                print(f"\n📊 {metric_name.upper()}:")
                print(f"  Training: {train_val:.4f}")
                print(f"  Validation: {val_val:.4f}")
                print(f"  Gap: {gap:.4f} ({gap_pct:.1f}%)")
                
                # Flag potential overfitting
                if metric_name.lower() in ['accuracy', 'precision', 'recall', 'f1']:
                    if gap_pct > 10:
                        overfitting_flags.append(f"🚨 Large {metric_name} gap: {gap_pct:.1f}%")
                    elif gap_pct > 5:
                        overfitting_flags.append(f"⚠️ Moderate {metric_name} gap: {gap_pct:.1f}%")
                elif metric_name.lower() == 'loss':
                    if gap > 0.2:
                        overfitting_flags.append(f"🚨 Large loss gap: {gap:.4f}")
                    elif gap > 0.1:
                        overfitting_flags.append(f"⚠️ Moderate loss gap: {gap:.4f}")
        
        if overfitting_flags:
            print(f"\n🚨 OVERFITTING INDICATORS:")
            for flag in overfitting_flags:
                print(f"  {flag}")
        else:
            print(f"\n✅ Performance gaps are within acceptable ranges")
        
        return gap_analysis, overfitting_flags
    
    def generate_overfitting_report(self):
        """
        Generate a comprehensive overfitting report
        """
        print("\n" + "="*60)
        print("🚨 COMPREHENSIVE OVERFITTING ANALYSIS REPORT")
        print("="*60)
        
        # Model complexity analysis
        complexity_result = self.analyze_model_complexity()
        
        # Configuration analysis
        if self.config:
            print(f"\n🔧 Configuration Analysis:")
            print(f"Max sequence length: {self.config.get('max_len', 'Unknown')}")
            print(f"Vocabulary size: {self.config.get('vocab_size', 'Unknown'):,}")
            print(f"Number of classes: {len(self.config.get('label_columns', []))}")
            print(f"Decision threshold: {self.config.get('threshold', 'Unknown')}")
        
        # Generate recommendations
        self._generate_recommendations(complexity_result)
    
    def _generate_recommendations(self, complexity_result):
        """
        Generate actionable recommendations to reduce overfitting
        """
        print(f"\n💡 RECOMMENDATIONS TO REDUCE OVERFITTING:")
        print("-" * 50)
        
        recommendations = []
        
        # Based on model complexity
        if complexity_result:
            if complexity_result['total_params'] > 100_000:
                recommendations.append("🔄 Reduce model size: Consider fewer layers or smaller hidden units")
                recommendations.append("📊 Use model pruning to remove unnecessary parameters")
            
            if not complexity_result['has_regularization']:
                recommendations.append("🛡️ Add regularization: Implement dropout layers (0.2-0.5 rate)")
                recommendations.append("⚖️ Add L1/L2 regularization to dense layers")
        
        # Data-related recommendations
        recommendations.extend([
            "📈 Increase dataset size: Collect more diverse training examples",
            "🔀 Improve data augmentation: Use text augmentation techniques",
            "🎯 Cross-validation: Use K-fold CV to better estimate performance",
            "⏰ Early stopping: Stop training when validation loss stops improving",
            "📉 Learning rate scheduling: Reduce learning rate during training",
            "🔄 Ensemble methods: Combine multiple models to reduce overfitting"
        ])
        
        # Training recommendations
        recommendations.extend([
            "📊 Monitor validation metrics: Track multiple metrics during training",
            "🎲 Use different random seeds: Ensure results are not seed-dependent",
            "📝 Implement holdout test set: Keep separate test data for final evaluation",
            "🔍 Analyze prediction confidence: Look for overconfident predictions"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec}")
        
        print(f"\n🎯 IMMEDIATE ACTIONS TO TAKE:")
        print("1. 🚨 Add more dropout layers with rates 0.3-0.5")
        print("2. 📉 Reduce model complexity (fewer neurons/layers)")
        print("3. ⏰ Implement early stopping with patience=3-5")
        print("4. 📊 Use validation loss for model selection, not training loss")
        print("5. 🔀 Increase training data if possible")

def create_overfitting_check_script():
    """
    Create a simple script to quickly check for overfitting signs
    """
    script_content = '''
# Quick Overfitting Check Script
# Run this after training your model

import matplotlib.pyplot as plt
import numpy as np

def quick_overfitting_check(history):
    """
    Quick check for overfitting signs in training history
    """
    print("🔍 QUICK OVERFITTING CHECK")
    print("=" * 30)
    
    # Check if we have the required metrics
    if 'loss' not in history or 'val_loss' not in history:
        print("❌ Training history must contain 'loss' and 'val_loss'")
        return
    
    train_loss = history['loss']
    val_loss = history['val_loss']
    
    # Final loss gap
    final_gap = val_loss[-1] - train_loss[-1]
    print(f"Final loss gap: {final_gap:.4f}")
    
    # Check trends in last few epochs
    if len(val_loss) >= 5:
        last_5_val = val_loss[-5:]
        val_trend = np.polyfit(range(5), last_5_val, 1)[0]
        
        print(f"Validation loss trend (last 5 epochs): {val_trend:.6f}")
        
        if val_trend > 0.001:
            print("🚨 OVERFITTING: Validation loss is increasing!")
        elif final_gap > 0.1:
            print("🚨 OVERFITTING: Large train/val loss gap!")
        elif final_gap > 0.05:
            print("⚠️ WARNING: Moderate train/val loss gap")
        else:
            print("✅ No clear overfitting signs")
    
    # Quick plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if 'accuracy' in history and 'val_accuracy' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage:
# quick_overfitting_check(your_model_history.history)
'''
    
    with open('quick_overfitting_check.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created 'quick_overfitting_check.py' for rapid overfitting detection")

def main():
    """
    Main function to run overfitting analysis
    """
    print("🔍 AUTOMATED OVERFITTING ANALYSIS")
    print("=" * 60)
    
    # Try to load existing model artifacts
    model_path = "saved_models/demo_toxicity_classifier.h5"
    config_path = "saved_models/config.pickle"
    tokenizer_path = "tokenizer.pickle"
    
    # Initialize analyzer
    analyzer = OverfittingAnalyzer(model_path, config_path, tokenizer_path)
    
    # Run analysis
    analyzer.generate_overfitting_report()
    
    # Create helper script
    create_overfitting_check_script()
    
    print(f"\n🎯 SUMMARY:")
    print("1. Model complexity analysis completed")
    print("2. Configuration reviewed")
    print("3. Recommendations generated")
    print("4. Quick check script created")
    
    print(f"\n💡 NEXT STEPS:")
    print("1. Run your training again with added regularization")
    print("2. Monitor training/validation curves closely")
    print("3. Use early stopping to prevent overfitting")
    print("4. Consider model ensemble approaches")

if __name__ == "__main__":
    main()
