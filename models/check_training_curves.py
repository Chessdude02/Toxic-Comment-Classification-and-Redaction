#!/usr/bin/env python3
"""
Quick Training Curves Overfitting Check
======================================

Use this script to quickly analyze your model's training history for overfitting.
"""

import matplotlib.pyplot as plt
import numpy as np

def quick_overfitting_check(history):
    """
    Quick check for overfitting signs in training history
    
    Args:
        history: The history.history dictionary from Keras model training
    """
    print("🔍 QUICK OVERFITTING CHECK")
    print("=" * 30)
    
    # Check if we have the required metrics
    if 'loss' not in history or 'val_loss' not in history:
        print("❌ Training history must contain 'loss' and 'val_loss'")
        print("💡 Pass your_model_history.history to this function")
        return False
    
    train_loss = history['loss']
    val_loss = history['val_loss']
    
    print(f"Training epochs: {len(train_loss)}")
    print(f"Final training loss: {train_loss[-1]:.4f}")
    print(f"Final validation loss: {val_loss[-1]:.4f}")
    
    # Final loss gap
    final_gap = val_loss[-1] - train_loss[-1]
    print(f"Final loss gap: {final_gap:.4f}")
    
    # Overfitting indicators
    overfitting_detected = False
    
    # Check 1: Large loss gap
    if final_gap > 0.1:
        print("🚨 OVERFITTING: Large train/val loss gap (>0.1)!")
        overfitting_detected = True
    elif final_gap > 0.05:
        print("⚠️ WARNING: Moderate train/val loss gap (>0.05)")
    
    # Check 2: Validation loss trend in last epochs
    if len(val_loss) >= 5:
        last_5_val = val_loss[-5:]
        val_trend = np.polyfit(range(5), last_5_val, 1)[0]
        
        print(f"Validation loss trend (last 5 epochs): {val_trend:.6f}")
        
        if val_trend > 0.001:
            print("🚨 OVERFITTING: Validation loss is increasing!")
            overfitting_detected = True
        elif val_trend > 0:
            print("⚠️ WARNING: Validation loss slightly increasing")
    
    # Check 3: Training loss still decreasing rapidly
    if len(train_loss) >= 3:
        train_trend = np.polyfit(range(len(train_loss)), train_loss, 1)[0]
        if train_trend < -0.01:
            print("⚠️ Training loss still decreasing rapidly - may need more epochs to see overfitting")
    
    # Check accuracy if available
    if 'accuracy' in history and 'val_accuracy' in history:
        train_acc = history['accuracy']
        val_acc = history['val_accuracy']
        acc_gap = train_acc[-1] - val_acc[-1]
        
        print(f"\nAccuracy analysis:")
        print(f"Final training accuracy: {train_acc[-1]:.4f}")
        print(f"Final validation accuracy: {val_acc[-1]:.4f}")
        print(f"Accuracy gap: {acc_gap:.4f}")
        
        if acc_gap > 0.1:
            print("🚨 OVERFITTING: Large accuracy gap (>10%)!")
            overfitting_detected = True
        elif acc_gap > 0.05:
            print("⚠️ WARNING: Moderate accuracy gap (>5%)")
        
        if train_acc[-1] > 0.98:
            print("🚨 OVERFITTING: Training accuracy too high (>98%) - possible memorization!")
            overfitting_detected = True
    
    # Overall assessment
    print(f"\n{'='*40}")
    if overfitting_detected:
        print("🚨 OVERFITTING DETECTED!")
        print("Your model is likely overfitting to the training data.")
    else:
        print("✅ No clear overfitting signs detected")
        print("However, monitor for longer training to be sure.")
    print("="*40)
    
    # Quick visualization
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Loss Curves - Overfitting Check')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Highlight potential overfitting region
    if len(val_loss) > 1:
        min_val_loss_epoch = np.argmin(val_loss) + 1
        plt.axvline(x=min_val_loss_epoch, color='green', linestyle='--', 
                   alpha=0.7, label=f'Best Epoch {min_val_loss_epoch}')
        plt.legend()
    
    # Accuracy plot if available
    if 'accuracy' in history and 'val_accuracy' in history:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        plt.title('Accuracy Curves - Overfitting Check')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.subplot(1, 2, 2)
        # Plot loss gap instead
        loss_gap = np.array(val_loss) - np.array(train_loss)
        plt.plot(epochs, loss_gap, 'orange', linewidth=2)
        plt.title('Loss Gap (Val - Train)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Gap')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return not overfitting_detected

# Example usage:
print("To use this script:")
print("1. After training your model, run:")
print("   quick_overfitting_check(your_model_history.history)")
print("")
print("2. If you have a history object, you can also test it like:")
print("   # Simulate some training history for demo")
print("   demo_history = {")
print("       'loss': [0.8, 0.6, 0.4, 0.3, 0.2],")
print("       'val_loss': [0.7, 0.6, 0.5, 0.6, 0.7],  # Notice increasing validation loss")
print("       'accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],")
print("       'val_accuracy': [0.65, 0.72, 0.75, 0.74, 0.73]  # Validation plateauing")
print("   }")
print("   quick_overfitting_check(demo_history)")

# Demonstrate with example
def demo_overfitting_detection():
    """Demonstrate overfitting detection with example data"""
    print(f"\n" + "="*50)
    print("DEMO: OVERFITTING DETECTION")
    print("="*50)
    
    # Example of overfitted model history
    overfitted_history = {
        'loss': [0.8, 0.6, 0.4, 0.25, 0.15, 0.08, 0.05],
        'val_loss': [0.75, 0.65, 0.55, 0.58, 0.62, 0.68, 0.75],  # Starts increasing
        'accuracy': [0.6, 0.7, 0.8, 0.87, 0.92, 0.96, 0.98],    # Very high
        'val_accuracy': [0.62, 0.72, 0.78, 0.79, 0.78, 0.76, 0.75]  # Starts decreasing
    }
    
    print("Example: Clearly overfitted model")
    quick_overfitting_check(overfitted_history)
    
    print(f"\n" + "-"*30)
    
    # Example of well-regularized model
    good_history = {
        'loss': [0.8, 0.6, 0.5, 0.45, 0.42, 0.40, 0.39],
        'val_loss': [0.75, 0.62, 0.52, 0.48, 0.46, 0.45, 0.44],  # Decreasing steadily
        'accuracy': [0.6, 0.7, 0.75, 0.78, 0.80, 0.81, 0.82],
        'val_accuracy': [0.62, 0.72, 0.76, 0.77, 0.78, 0.79, 0.80]  # Following train closely
    }
    
    print("Example: Well-regularized model")
    quick_overfitting_check(good_history)

if __name__ == "__main__":
    demo_overfitting_detection()
