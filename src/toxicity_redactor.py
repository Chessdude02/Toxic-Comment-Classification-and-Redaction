"""
Advanced Toxicity Classification and Real-Time Redaction System

This module provides production-ready functionality for:
1. Multi-class toxicity detection
2. Real-time message redaction
3. Batch message moderation
4. Easy integration with existing applications

Usage:
    from toxicity_redactor import ToxicityRedactor, load_pretrained_model
    
    # Load a pre-trained model
    redactor = load_pretrained_model()
    
    # Check if a message is toxic
    result = redactor.classify_toxicity("Your message here")
    
    # Redact a toxic message
    redacted = redactor.redact_message("Your message here", style="warning")
    
Author: AI Assistant
Date: 2025-08-27
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import numpy as np
import pandas as pd
import pickle
import re
import os
from typing import Dict, List, Tuple, Union, Optional
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
# Registers the root Transformer's custom layers (TransformerBlock,
# AttentionPooling, PositionalEmbedding) with Keras's serialization registry
# as an import side effect, so load_model() below can resolve them by name
# without every caller needing to pass custom_objects explicitly.
import transformer_layers  # noqa: F401
import warnings
warnings.filterwarnings('ignore')


class ToxicityRedactor:
    """
    A comprehensive toxicity detection and redaction system.
    
    This class provides real-time toxicity classification and message redaction
    capabilities with support for multiple toxicity types and redaction styles.
    """
    
    def __init__(self, model, tokenizer, label_columns: List[str], threshold: float = 0.5, max_len: int = 512):
        """
        Initialize the ToxicityRedactor.
        
        Args:
            model: Trained Keras model for toxicity classification
            tokenizer: Fitted Keras tokenizer
            label_columns: List of toxicity label names
            threshold: Classification threshold (default: 0.5)
            max_len: Maximum sequence length for padding (default: 512)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.label_columns = label_columns
        self.threshold = threshold
        self.max_len = max_len
        self.num_classes = len(label_columns)
        
        # Toxic word patterns for enhanced redaction
        self.toxic_patterns = [
            r'\b(idiot|stupid|moron|dumb|fool)\b',
            r'\b(hate|despise|loathe)\b',
            r'\b(kill|die|death)\b',
            r'\b(shut\s+up|stfu)\b'
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for model input.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and handle encoding
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove special characters but preserve spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_message(self, message: str) -> np.ndarray:
        """
        Preprocess a single message for model prediction.
        
        Args:
            message: Input message to preprocess
            
        Returns:
            Padded sequence array ready for model prediction
        """
        cleaned_message = self.clean_text(message)
        sequence_data = self.tokenizer.texts_to_sequences([cleaned_message])
        padded = sequence.pad_sequences(sequence_data, maxlen=self.max_len)
        return padded
    
    def classify_toxicity(self, message: str) -> Dict:
        """
        Classify the toxicity of a message across multiple categories.
        
        Args:
            message: Input message to classify
            
        Returns:
            Dictionary containing toxicity classification results
        """
        # Preprocess the message
        processed_message = self.preprocess_message(message)
        
        # Get model predictions
        predictions = self.model.predict(processed_message, verbose=0)[0]
        
        # Ensure predictions is always an array
        if not isinstance(predictions, np.ndarray):
            predictions = np.array([predictions])
        
        # Handle both single and multi-class outputs
        if len(predictions) != len(self.label_columns):
            # If model outputs single value but we have multiple labels, replicate
            if len(predictions) == 1 and len(self.label_columns) > 1:
                predictions = np.repeat(predictions, len(self.label_columns))
        
        # Create detailed results
        results = {}
        toxic_types = []
        
        for i, label in enumerate(self.label_columns):
            prob = float(predictions[i]) if i < len(predictions) else 0.0
            is_toxic_for_label = prob > self.threshold
            
            results[label] = {
                'probability': prob,
                'is_toxic': is_toxic_for_label
            }
            
            if is_toxic_for_label:
                toxic_types.append(label)
        
        # Overall toxicity assessment
        is_toxic = len(toxic_types) > 0
        max_prob = max([results[label]['probability'] for label in self.label_columns])
        
        return {
            'is_toxic': is_toxic,
            'toxic_types': toxic_types,
            'max_toxicity_probability': max_prob,
            'detailed_results': results,
            'original_message': message,
            'confidence_score': max_prob,
            'classification_threshold': self.threshold
        }
    
    def detect_toxic_words(self, message: str) -> List[str]:
        """
        Detect potentially toxic words using pattern matching.
        
        Args:
            message: Input message
            
        Returns:
            List of detected toxic words
        """
        toxic_words = []
        message_lower = message.lower()
        
        for pattern in self.toxic_patterns:
            matches = re.findall(pattern, message_lower)
            toxic_words.extend(matches)
        
        return toxic_words
    
    def redact_message(self, message: str, redaction_style: str = 'warning') -> Dict:
        """
        Redact toxic content from a message based on the specified style.
        
        Args:
            message: Input message to potentially redact
            redaction_style: Style of redaction ('warning', 'partial', 'complete', 'asterisk')
            
        Returns:
            Dictionary containing redaction results
        """
        classification = self.classify_toxicity(message)
        
        if not classification['is_toxic']:
            return {
                'redacted_message': message,
                'was_redacted': False,
                'toxicity_info': classification,
                'redaction_style': redaction_style
            }
        
        # Apply different redaction styles
        if redaction_style == 'complete':
            # Complete message removal
            toxic_types_str = ', '.join(classification['toxic_types'])
            redacted = f"[MESSAGE REDACTED - CONTAINS {toxic_types_str.upper()} CONTENT]"
            
        elif redaction_style == 'partial':
            # Replace potentially toxic words with asterisks
            words = message.split()
            redacted_words = []
            
            for word in words:
                # Check if word itself is toxic
                if len(word) > 3:
                    word_classification = self.classify_toxicity(word)
                    if word_classification['is_toxic'] or word.lower() in [w for pattern in self.toxic_patterns for w in re.findall(pattern, word.lower())]:
                        redacted_words.append('*' * len(word))
                    else:
                        redacted_words.append(word)
                else:
                    redacted_words.append(word)
            
            redacted = ' '.join(redacted_words)
            
        elif redaction_style == 'asterisk':
            # Replace toxic words with asterisks using pattern matching
            redacted = message
            for pattern in self.toxic_patterns:
                redacted = re.sub(pattern, lambda m: '*' * len(m.group()), redacted, flags=re.IGNORECASE)
            
        else:  # 'warning' style (default)
            # Add warning but keep original message
            toxic_types_str = ', '.join(classification['toxic_types'])
            confidence = classification['max_toxicity_probability']
            redacted = f"⚠️ [WARNING: Contains {toxic_types_str} content - {confidence:.1%} confidence] {message}"
        
        return {
            'redacted_message': redacted,
            'was_redacted': True,
            'toxicity_info': classification,
            'redaction_style': redaction_style,
            'redaction_reason': f"Detected {', '.join(classification['toxic_types'])} (confidence: {classification['max_toxicity_probability']:.3f})"
        }
    
    def moderate_conversation(self, messages: List[str], redaction_style: str = 'warning') -> List[Dict]:
        """
        Moderate a list of messages.
        
        Args:
            messages: List of messages to moderate
            redaction_style: Redaction style to apply
            
        Returns:
            List of moderation results for each message
        """
        moderated_messages = []
        
        for i, message in enumerate(messages):
            result = self.redact_message(message, redaction_style)
            result['message_id'] = i
            result['original_index'] = i
            moderated_messages.append(result)
        
        return moderated_messages
    
    def get_toxicity_summary(self, messages: List[str]) -> Dict:
        """
        Get summary statistics for toxicity in a collection of messages.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            Dictionary with toxicity statistics
        """
        results = []
        toxicity_by_type = {label: 0 for label in self.label_columns}
        
        for message in messages:
            classification = self.classify_toxicity(message)
            results.append(classification)
            
            for toxic_type in classification['toxic_types']:
                toxicity_by_type[toxic_type] += 1
        
        total_messages = len(messages)
        toxic_messages = sum(1 for r in results if r['is_toxic'])
        
        return {
            'total_messages': total_messages,
            'toxic_messages': toxic_messages,
            'clean_messages': total_messages - toxic_messages,
            'toxicity_rate': toxic_messages / total_messages if total_messages > 0 else 0,
            'toxicity_by_type': toxicity_by_type,
            'average_toxicity_probability': np.mean([r['max_toxicity_probability'] for r in results]),
            'max_toxicity_probability': max([r['max_toxicity_probability'] for r in results]) if results else 0
        }
    
    def update_threshold(self, new_threshold: float):
        """
        Update the classification threshold.
        
        Args:
            new_threshold: New threshold value (0.0 to 1.0)
        """
        if 0.0 <= new_threshold <= 1.0:
            self.threshold = new_threshold
            print(f"✅ Threshold updated to {new_threshold}")
        else:
            print("❌ Threshold must be between 0.0 and 1.0")


class ChatModerator:
    """
    Real-time chat moderation system with logging and statistics.
    """
    
    def __init__(self, redactor: ToxicityRedactor, auto_moderate: bool = True, log_messages: bool = True):
        """
        Initialize the ChatModerator.
        
        Args:
            redactor: ToxicityRedactor instance
            auto_moderate: Whether to automatically redact toxic messages
            log_messages: Whether to log all processed messages
        """
        self.redactor = redactor
        self.auto_moderate = auto_moderate
        self.log_messages = log_messages
        self.message_history = []
        self.moderation_stats = {
            'total_messages': 0,
            'toxic_messages': 0,
            'redacted_messages': 0,
            'false_positives': 0,  # Can be updated with user feedback
            'false_negatives': 0   # Can be updated with user feedback
        }
    
    def process_message(self, username: str, message: str, timestamp: Optional[str] = None) -> Dict:
        """
        Process a single chat message through the moderation system.
        
        Args:
            username: Username of the message sender
            message: Message content
            timestamp: Optional timestamp string
            
        Returns:
            Dictionary containing processing results
        """
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Classify the message
        classification = self.redactor.classify_toxicity(message)
        
        # Update statistics
        self.moderation_stats['total_messages'] += 1
        if classification['is_toxic']:
            self.moderation_stats['toxic_messages'] += 1
        
        # Determine display message
        if classification['is_toxic'] and self.auto_moderate:
            redaction_result = self.redactor.redact_message(message, 'warning')
            display_message = redaction_result['redacted_message']
            self.moderation_stats['redacted_messages'] += 1
            status = "🚨 MODERATED"
        elif classification['is_toxic']:
            display_message = message
            status = "⚠️ FLAGGED"
        else:
            display_message = message
            status = "✅ CLEAN"
        
        # Create chat entry
        chat_entry = {
            'timestamp': timestamp,
            'username': username,
            'original_message': message,
            'display_message': display_message,
            'status': status,
            'toxicity_info': classification,
            'was_processed': True
        }
        
        # Log if enabled
        if self.log_messages:
            self.message_history.append(chat_entry)
        
        return chat_entry
    
    def get_chat_log(self, last_n: int = 50) -> List[Dict]:
        """
        Get recent chat messages.
        
        Args:
            last_n: Number of recent messages to return
            
        Returns:
            List of recent chat entries
        """
        return self.message_history[-last_n:] if self.message_history else []
    
    def get_moderation_stats(self) -> Dict:
        """
        Get comprehensive moderation statistics.
        
        Returns:
            Dictionary with moderation statistics
        """
        stats = self.moderation_stats.copy()
        
        if stats['total_messages'] > 0:
            stats['toxicity_rate'] = stats['toxic_messages'] / stats['total_messages']
            stats['redaction_rate'] = stats['redacted_messages'] / stats['total_messages']
            stats['accuracy_estimate'] = 1 - (stats['false_positives'] + stats['false_negatives']) / stats['total_messages']
        else:
            stats['toxicity_rate'] = 0
            stats['redaction_rate'] = 0
            stats['accuracy_estimate'] = 0
        
        return stats
    
    def mark_false_positive(self, message_id: int):
        """Mark a message as falsely flagged as toxic."""
        self.moderation_stats['false_positives'] += 1
    
    def mark_false_negative(self, message_id: int):
        """Mark a message as falsely classified as clean.""" 
        self.moderation_stats['false_negatives'] += 1
    
    def export_chat_log(self, filename: str = 'chat_log.csv'):
        """
        Export chat log to CSV file.
        
        Args:
            filename: Output filename
        """
        if not self.message_history:
            print("❌ No chat history to export.")
            return
        
        # Flatten the data for CSV export
        export_data = []
        for entry in self.message_history:
            row = {
                'timestamp': entry['timestamp'],
                'username': entry['username'],
                'original_message': entry['original_message'],
                'display_message': entry['display_message'],
                'status': entry['status'],
                'is_toxic': entry['toxicity_info']['is_toxic'],
                'toxic_types': ', '.join(entry['toxicity_info']['toxic_types']),
                'max_toxicity_probability': entry['toxicity_info']['max_toxicity_probability']
            }
            export_data.append(row)
        
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"✅ Chat log exported to {filename}")


def create_multiclass_model(vocab_size: int, embedding_dim: int = 300, max_len: int = 512, 
                          num_classes: int = 6, model_type: str = 'lstm') -> tf.keras.Model:
    """
    Create a multi-class toxicity classification model.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Embedding dimension
        max_len: Maximum sequence length
        num_classes: Number of toxicity classes
        model_type: Type of model ('lstm', 'gru', 'bidirectional', 'simple_rnn')
        
    Returns:
        Compiled Keras model
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding, LSTM, GRU, SimpleRNN, Bidirectional,
        Dense, Dropout, SpatialDropout1D
    )
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential()
    
    # Embedding layer
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    
    # Recurrent layers based on model type
    if model_type.lower() == 'simple_rnn':
        model.add(SimpleRNN(128, return_sequences=False))
    elif model_type.lower() == 'gru':
        model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2))
    elif model_type.lower() == 'bidirectional':
        model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
        model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
    else:  # Default to LSTM
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    
    # Dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='sigmoid'))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model


def load_pretrained_model(model_path: str = None, config_path: str = None, tokenizer_path: str = None) -> ToxicityRedactor:
    """
    Load a pre-trained toxicity classification model.
    
    Args:
        model_path: Path to saved model file
        config_path: Path to configuration file
        tokenizer_path: Path to tokenizer file
        
    Returns:
        ToxicityRedactor instance or None if loading fails
    """
    try:
        # Set default paths. Native .keras format (not legacy HDF5) because
        # the root Transformer uses custom Layer subclasses - HDF5's
        # round-tripping of those is far less reliable than the native
        # format's, which stores each layer's get_config() output directly.
        if model_path is None:
            model_path = 'saved_models/demo_toxicity_classifier.keras'
        if config_path is None:
            config_path = 'saved_models/config.pickle'
        if tokenizer_path is None:
            tokenizer_path = 'tokenizer.pickle'
        
        # Load configuration
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Create redactor
        redactor = ToxicityRedactor(
            model=model,
            tokenizer=tokenizer,
            label_columns=config['label_columns'],
            threshold=config.get('threshold', 0.5),
            max_len=config.get('max_len', 512)
        )
        
        print(f"✅ Successfully loaded pre-trained model from {model_path}")
        return redactor
        
    except Exception as e:
        print(f"❌ Error loading pre-trained model: {str(e)}")
        return None


def quick_toxicity_check(message: str, model_path: str = None) -> Dict:
    """
    Quick standalone function to check message toxicity.
    
    Args:
        message: Message to check
        model_path: Optional path to model (loads default if None)
        
    Returns:
        Dictionary with toxicity results
    """
    redactor = load_pretrained_model(model_path)
    
    if redactor is None:
        return {'error': 'Could not load model'}
    
    return redactor.classify_toxicity(message)


def batch_moderate_messages(messages: List[str], redaction_style: str = 'warning', 
                          model_path: str = None) -> List[Dict]:
    """
    Moderate a batch of messages.
    
    Args:
        messages: List of messages to moderate
        redaction_style: Style of redaction to apply
        model_path: Optional path to model
        
    Returns:
        List of moderation results
    """
    redactor = load_pretrained_model(model_path)
    
    if redactor is None:
        return [{'error': 'Could not load model'}] * len(messages)
    
    return redactor.moderate_conversation(messages, redaction_style)


# Utility functions for integration
def create_api_response(message: str, redaction_style: str = 'warning') -> Dict:
    """
    Create an API-style response for toxicity checking.
    
    Args:
        message: Input message
        redaction_style: Redaction style
        
    Returns:
        Standardized API response
    """
    try:
        redactor = load_pretrained_model()
        if redactor is None:
            return {
                'success': False,
                'error': 'Model not available',
                'message': message
            }
        
        classification = redactor.classify_toxicity(message)
        redaction_result = redactor.redact_message(message, redaction_style)
        
        return {
            'success': True,
            'original_message': message,
            'is_toxic': classification['is_toxic'],
            'toxic_types': classification['toxic_types'],
            'confidence': classification['max_toxicity_probability'],
            'moderated_message': redaction_result['redacted_message'],
            'was_redacted': redaction_result['was_redacted'],
            'redaction_style': redaction_style,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': message
        }


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Toxicity Redactor Module")
    print("=" * 50)
    
    # Test messages
    test_messages = [
        "Hello everyone! How are you doing today?",
        "I disagree with your opinion but respect it.",
        "You're such an idiot for thinking that way.",
        "This is completely wrong and stupid.",
        "Thanks for the helpful information!",
        "I hate this stupid idea, it's moronic.",
        "Let's have a civil discussion about this topic."
    ]
    
    print("Sample test messages:")
    for i, msg in enumerate(test_messages, 1):
        print(f"{i}. {msg}")
    
    print("\n💡 To use this module:")
    print("1. Train a model using the notebook")
    print("2. Save the model and tokenizer")
    print("3. Import this module: from toxicity_redactor import ToxicityRedactor")
    print("4. Load your model: redactor = load_pretrained_model()")
    print("5. Use: result = redactor.classify_toxicity('message')")
    
    print("\n🚀 Module ready for integration!")
