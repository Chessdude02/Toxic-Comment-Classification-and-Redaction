#!/usr/bin/env python3
"""
Intelligent Message Redaction System
===================================

A comprehensive redaction system that can:
1. Detect toxic content using the fixed model
2. Identify specific toxic words/phrases
3. Apply intelligent redaction strategies
4. Provide real-time message moderation
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from datetime import datetime
import string

class IntelligentRedactor:
    """
    Advanced redaction system with multiple strategies and word-level detection
    """
    
    def __init__(self, model_path=None, tokenizer_path=None, config_path=None):
        """Initialize the redaction system"""
        self.model = None
        self.tokenizer = None
        self.config = None
        self.toxic_word_patterns = self._load_toxic_patterns()
        self.load_model_artifacts(model_path, tokenizer_path, config_path)
    
    def load_model_artifacts(self, model_path=None, tokenizer_path=None, config_path=None):
        """Load the trained model and artifacts"""
        try:
            print("🔄 Loading toxicity detection model...")
            
            # Use default paths if not provided
            model_path = model_path or "saved_models/fixed_toxicity_model_final.h5"
            tokenizer_path = tokenizer_path or "fixed_toxicity_model_tokenizer.pickle"
            config_path = config_path or "saved_models/fixed_toxicity_model_config.pickle"
            
            # Load model
            self.model = load_model(model_path)
            
            # Load tokenizer
            with open(tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)
            
            # Load config
            with open(config_path, "rb") as f:
                self.config = pickle.load(f)
            
            print(f"✅ Redaction system loaded successfully!")
            print(f"   Model parameters: {self.model.count_params():,}")
            print(f"   Vocabulary size: {self.config['vocab_size']:,}")
            
        except Exception as e:
            print(f"❌ Error loading model artifacts: {e}")
            print("Make sure the fixed model has been trained first")
    
    def _load_toxic_patterns(self):
        """Load patterns for detecting specific toxic words"""
        return {
            'insults': [
                r'\bidiot\b', r'\bmoron\b', r'\bstupid\b', r'\bdumb\b', r'\bimbecile\b',
                r'\basshole\b', r'\bdumbass\b', r'\bloser\b', r'\bpathetic\b'
            ],
            'profanity': [
                r'\bfuck\b', r'\bshit\b', r'\bbitch\b', r'\bdamn\b', r'\bhell\b',
                r'\bfucking\b', r'\bfucker\b', r'\bbullshit\b'
            ],
            'threats': [
                r'\bkill\s+yourself\b', r'\bgo\s+die\b', r'\bdie\s+in\b', 
                r'\bkill\s+you\b', r'\bmurder\b'
            ],
            'hate': [
                r'\bhate\s+you\b', r'\bi\s+hate\b', r'\bhate\s+this\b'
            ]
        }
    
    def predict_toxicity(self, text):
        """
        Predict toxicity probability for text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            return {"error": "Model not loaded", "probability": 0.0, "is_toxic": False}
        
        try:
            # Preprocess text
            text_clean = str(text).lower().strip()
            seq = self.tokenizer.texts_to_sequences([text_clean])
            padded = sequence.pad_sequences(seq, maxlen=self.config['max_len'])
            
            # Get prediction
            prob = float(self.model.predict(padded, verbose=0)[0][0])
            is_toxic = prob > self.config['threshold']
            
            return {
                "text": text,
                "probability": prob,
                "is_toxic": is_toxic,
                "confidence": self._get_confidence_level(prob)
            }
            
        except Exception as e:
            return {"error": str(e), "probability": 0.0, "is_toxic": False}
    
    def _get_confidence_level(self, prob):
        """Get human-readable confidence level"""
        if prob < 0.2:
            return "Very Low"
        elif prob < 0.4:
            return "Low"
        elif prob < 0.6:
            return "Medium"
        elif prob < 0.8:
            return "High"
        else:
            return "Very High"
    
    def identify_toxic_words(self, text):
        """
        Identify specific words/phrases that might be causing toxicity
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Identified toxic elements
        """
        text_lower = text.lower()
        identified = {
            'insults': [],
            'profanity': [],
            'threats': [],
            'hate': [],
            'other_toxic': []
        }
        
        # Check against known patterns
        for category, patterns in self.toxic_word_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    word = match.group()
                    start, end = match.span()
                    identified[category].append({
                        'word': word,
                        'start': start,
                        'end': end,
                        'original': text[start:end]
                    })
        
        # Use word-level toxicity detection
        words = text.split()
        for i, word in enumerate(words):
            # Clean word for testing
            clean_word = re.sub(r'[^a-zA-Z]', '', word.lower())
            if len(clean_word) > 2:
                word_result = self.predict_toxicity(clean_word)
                if word_result['is_toxic'] and word_result['probability'] > 0.6:
                    # Check if not already identified
                    already_found = any(
                        word.lower() in [item['word'] for items in identified.values() for item in items]
                    )
                    if not already_found:
                        identified['other_toxic'].append({
                            'word': clean_word,
                            'original': word,
                            'position': i,
                            'probability': word_result['probability']
                        })
        
        return identified
    
    def redact_message(self, text, style='smart', custom_replacement=None):
        """
        Redact toxic content from message
        
        Args:
            text (str): Input text
            style (str): Redaction style - 'smart', 'partial', 'complete', 'warning', 'custom'
            custom_replacement (str): Custom replacement text for 'custom' style
            
        Returns:
            dict: Redaction result
        """
        # First, check if message is toxic
        toxicity_result = self.predict_toxicity(text)
        
        if not toxicity_result['is_toxic']:
            return {
                'original_text': text,
                'redacted_text': text,
                'was_redacted': False,
                'redaction_style': style,
                'toxicity_info': toxicity_result,
                'redacted_elements': []
            }
        
        # Identify toxic elements
        toxic_elements = self.identify_toxic_words(text)
        
        # Apply redaction based on style
        redacted_text = self._apply_redaction_style(text, toxic_elements, style, custom_replacement)
        
        return {
            'original_text': text,
            'redacted_text': redacted_text,
            'was_redacted': True,
            'redaction_style': style,
            'toxicity_info': toxicity_result,
            'toxic_elements': toxic_elements,
            'redacted_elements': self._get_redacted_elements(toxic_elements)
        }
    
    def _apply_redaction_style(self, text, toxic_elements, style, custom_replacement):
        """Apply the specified redaction style"""
        
        if style == 'complete':
            return "[MESSAGE REDACTED - TOXIC CONTENT DETECTED]"
        
        elif style == 'warning':
            toxic_types = self._get_toxic_types(toxic_elements)
            warning = f"⚠️ [WARNING: {', '.join(toxic_types)} content detected] "
            return warning + text
        
        elif style == 'custom' and custom_replacement:
            return custom_replacement
        
        elif style == 'partial':
            return self._partial_redaction(text, toxic_elements)
        
        elif style == 'smart':
            return self._smart_redaction(text, toxic_elements)
        
        else:
            # Default to partial
            return self._partial_redaction(text, toxic_elements)
    
    def _partial_redaction(self, text, toxic_elements):
        """Replace toxic words with asterisks"""
        result = text
        
        # Sort by position (reverse order to maintain indices)
        all_toxic = []
        for category, items in toxic_elements.items():
            if category != 'other_toxic':
                for item in items:
                    all_toxic.append((item['start'], item['end'], item['original']))
        
        # Add other toxic words
        words = text.split()
        for item in toxic_elements.get('other_toxic', []):
            word_start = text.lower().find(item['word'].lower())
            if word_start != -1:
                all_toxic.append((word_start, word_start + len(item['word']), item['original']))
        
        # Sort by start position (reverse)
        all_toxic.sort(key=lambda x: x[0], reverse=True)
        
        # Replace with asterisks
        for start, end, original in all_toxic:
            replacement = '*' * len(original)
            result = result[:start] + replacement + result[end:]
        
        return result
    
    def _smart_redaction(self, text, toxic_elements):
        """Smart redaction that preserves meaning while removing toxicity"""
        result = text
        
        # Define smart replacements
        smart_replacements = {
            'idiot': '[person]',
            'moron': '[person]',
            'stupid': '[negative adjective]',
            'dumb': '[uninformed]',
            'fuck': '[expletive]',
            'shit': '[expletive]',
            'bitch': '[derogatory term]',
            'asshole': '[rude person]',
            'kill yourself': '[harmful suggestion]',
            'go die': '[harmful suggestion]',
            'hate you': '[strong dislike]'
        }
        
        # Apply smart replacements
        for category, items in toxic_elements.items():
            if category != 'other_toxic':
                for item in items:
                    word = item['word']
                    original = item['original']
                    replacement = smart_replacements.get(word, f'[{category[:-1]}]')
                    result = result.replace(original, replacement)
        
        # Handle other toxic words
        for item in toxic_elements.get('other_toxic', []):
            original = item['original']
            replacement = f'[inappropriate term]'
            result = result.replace(original, replacement)
        
        return result
    
    def _get_toxic_types(self, toxic_elements):
        """Get list of detected toxic types"""
        types = []
        for category, items in toxic_elements.items():
            if items and category != 'other_toxic':
                types.append(category.replace('_', ' '))
        if toxic_elements.get('other_toxic'):
            types.append('inappropriate language')
        return types
    
    def _get_redacted_elements(self, toxic_elements):
        """Get summary of redacted elements"""
        redacted = []
        for category, items in toxic_elements.items():
            for item in items:
                if category == 'other_toxic':
                    redacted.append(f"{item['original']} ({category})")
                else:
                    redacted.append(f"{item['original']} ({category})")
        return redacted

class MessageModerator:
    """
    Real-time message moderation system
    """
    
    def __init__(self, redactor, auto_moderate=True, policy='moderate'):
        """
        Initialize message moderator
        
        Args:
            redactor: IntelligentRedactor instance
            auto_moderate (bool): Whether to automatically redact toxic messages
            policy (str): Moderation policy - 'strict', 'moderate', 'lenient'
        """
        self.redactor = redactor
        self.auto_moderate = auto_moderate
        self.policy = policy
        self.message_history = []
        self.moderation_stats = {
            'total_messages': 0,
            'toxic_detected': 0,
            'messages_redacted': 0,
            'warnings_issued': 0
        }
        
        # Set thresholds based on policy
        self.policy_thresholds = {
            'strict': 0.3,
            'moderate': 0.5,
            'lenient': 0.7
        }
        
        self.current_threshold = self.policy_thresholds.get(policy, 0.5)
        
        print(f"✅ Message moderator initialized")
        print(f"   Policy: {policy} (threshold: {self.current_threshold})")
        print(f"   Auto-moderate: {auto_moderate}")
    
    def moderate_message(self, message, username=None, timestamp=None, redaction_style='smart'):
        """
        Moderate a single message
        
        Args:
            message (str): Message to moderate
            username (str): Username of sender
            timestamp (str): Message timestamp
            redaction_style (str): Style of redaction to apply
            
        Returns:
            dict: Moderation result
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Analyze toxicity
        toxicity_result = self.redactor.predict_toxicity(message)
        
        # Update stats
        self.moderation_stats['total_messages'] += 1
        
        # Determine action based on policy and toxicity level
        action_taken = 'none'
        final_message = message
        
        if toxicity_result['probability'] > self.current_threshold:
            self.moderation_stats['toxic_detected'] += 1
            
            if self.auto_moderate:
                # Apply redaction
                redaction_result = self.redactor.redact_message(message, redaction_style)
                final_message = redaction_result['redacted_text']
                action_taken = 'redacted'
                self.moderation_stats['messages_redacted'] += 1
            else:
                # Just flag for manual review
                action_taken = 'flagged'
                self.moderation_stats['warnings_issued'] += 1
        
        # Create moderation record
        moderation_record = {
            'timestamp': timestamp,
            'username': username or 'Anonymous',
            'original_message': message,
            'final_message': final_message,
            'toxicity_probability': toxicity_result['probability'],
            'is_toxic': toxicity_result['is_toxic'],
            'action_taken': action_taken,
            'policy_used': self.policy,
            'redaction_style': redaction_style if action_taken == 'redacted' else None
        }
        
        self.message_history.append(moderation_record)
        
        return moderation_record
    
    def moderate_conversation(self, messages, redaction_style='smart'):
        """
        Moderate a list of messages (like a chat conversation)
        
        Args:
            messages (list): List of message dictionaries or strings
            redaction_style (str): Redaction style to apply
            
        Returns:
            list: List of moderation results
        """
        results = []
        
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                # Message with metadata
                username = msg.get('username', f'User{i+1}')
                text = msg.get('text', msg.get('message', ''))
                timestamp = msg.get('timestamp')
            else:
                # Simple text message
                username = f'User{i+1}'
                text = str(msg)
                timestamp = None
            
            result = self.moderate_message(text, username, timestamp, redaction_style)
            results.append(result)
        
        return results
    
    def get_moderation_stats(self):
        """Get moderation statistics"""
        stats = self.moderation_stats.copy()
        
        if stats['total_messages'] > 0:
            stats['toxicity_rate'] = stats['toxic_detected'] / stats['total_messages']
            stats['redaction_rate'] = stats['messages_redacted'] / stats['total_messages']
            stats['warning_rate'] = stats['warnings_issued'] / stats['total_messages']
        else:
            stats['toxicity_rate'] = 0
            stats['redaction_rate'] = 0
            stats['warning_rate'] = 0
        
        return stats
    
    def display_conversation(self, last_n=10):
        """Display recent moderated conversation"""
        print(f"\n💬 MODERATED CONVERSATION (Last {last_n} messages)")
        print("=" * 60)
        
        recent = self.message_history[-last_n:] if self.message_history else []
        
        for record in recent:
            # Status icon
            if record['action_taken'] == 'redacted':
                status = "🚨"
            elif record['action_taken'] == 'flagged':
                status = "⚠️"
            else:
                status = "✅"
            
            print(f"[{record['timestamp']}] {status} {record['username']}:")
            print(f"  Original: {record['original_message']}")
            
            if record['action_taken'] == 'redacted':
                print(f"  Redacted: {record['final_message']}")
                print(f"  Toxicity: {record['toxicity_probability']:.3f}")
            elif record['action_taken'] == 'flagged':
                print(f"  ⚠️ FLAGGED for review (toxicity: {record['toxicity_probability']:.3f})")
            
            print()

class RedactionDemo:
    """
    Demonstration class for the redaction system
    """
    
    def __init__(self, redactor):
        self.redactor = redactor
    
    def demo_redaction_styles(self):
        """Demonstrate different redaction styles"""
        print(f"\n🎭 REDACTION STYLES DEMONSTRATION")
        print("=" * 40)
        
        test_messages = [
            "You are such an idiot, go kill yourself!",
            "This is fucking stupid, you moron",
            "I hate you so much, you piece of shit",
            "Shut up, dumbass, nobody cares"
        ]
        
        styles = ['smart', 'partial', 'warning', 'complete']
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n--- Test Message {i} ---")
            print(f"Original: '{message}'")
            print()
            
            for style in styles:
                result = self.redactor.redact_message(message, style)
                print(f"{style.upper():8}: {result['redacted_text']}")
            
            # Show detected elements
            toxic_elements = self.redactor.identify_toxic_words(message)
            detected = []
            for category, items in toxic_elements.items():
                if items:
                    words = [item.get('original', item.get('word', '')) for item in items]
                    detected.extend(words)
            
            if detected:
                print(f"Detected: {', '.join(detected)}")
            
            print("-" * 50)
    
    def demo_real_time_moderation(self):
        """Demonstrate real-time chat moderation"""
        print(f"\n💬 REAL-TIME CHAT MODERATION DEMO")
        print("=" * 40)
        
        # Create moderator with different policies
        moderator = MessageModerator(self.redactor, auto_moderate=True, policy='moderate')
        
        # Simulate a chat conversation
        chat_messages = [
            {"username": "Alice", "text": "Hello everyone! How's your day going?"},
            {"username": "Bob", "text": "Pretty good, thanks for asking Alice!"},
            {"username": "Charlie", "text": "This weather is really annoying today"},
            {"username": "Dave", "text": "You're all idiots if you think that's bad weather"},
            {"username": "Alice", "text": "Let's keep things friendly, please"},
            {"username": "Eve", "text": "I agree with Alice, we should be respectful"},
            {"username": "Dave", "text": "Whatever, you people are so fucking sensitive"},
            {"username": "Bob", "text": "Thanks for sharing your thoughts, everyone"},
            {"username": "Frank", "text": "Dave, shut up you moron, nobody wants to hear it"},
            {"username": "Alice", "text": "Let's focus on having a positive discussion"}
        ]
        
        print("Simulating chat with moderation:")
        print("-" * 35)
        
        # Moderate the conversation
        results = moderator.moderate_conversation(chat_messages, redaction_style='smart')
        
        # Display moderated conversation
        moderator.display_conversation()
        
        # Show statistics
        stats = moderator.get_moderation_stats()
        print(f"📊 MODERATION STATISTICS:")
        print(f"Total messages: {stats['total_messages']}")
        print(f"Toxic detected: {stats['toxic_detected']} ({stats['toxicity_rate']:.1%})")
        print(f"Messages redacted: {stats['messages_redacted']} ({stats['redaction_rate']:.1%})")
        
        return moderator
    
    def interactive_testing(self):
        """Interactive testing interface"""
        print(f"\n🧪 INTERACTIVE REDACTION TESTING")
        print("=" * 40)
        
        test_messages = [
            "Type your own message here to test",
            "You can modify these examples:",
            "This is a great discussion!",
            "You're being really stupid about this",
            "I hate when people do that",
            "Go kill yourself, idiot!"
        ]
        
        print("Try these example messages (you can modify them):")
        print("-" * 45)
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n{i}. Testing: '{message}'")
            
            # Show all redaction styles
            styles = ['smart', 'partial', 'warning']
            for style in styles:
                result = self.redactor.redact_message(message, style)
                status = "🚨" if result['was_redacted'] else "✅"
                
                print(f"   {style:8}: {status} {result['redacted_text']}")
                if result['was_redacted']:
                    print(f"            (Toxicity: {result['toxicity_info']['probability']:.3f})")

def main():
    """Main function to demonstrate the redaction system"""
    print("🛡️ INTELLIGENT MESSAGE REDACTION SYSTEM")
    print("=" * 50)
    
    # Initialize redactor
    redactor = IntelligentRedactor()
    
    if redactor.model is None:
        print("❌ Could not load model. Please run fix_overfitted_model.py first")
        return
    
    # Create demo system
    demo = RedactionDemo(redactor)
    
    # Run demonstrations
    print("\n🎯 Running comprehensive redaction demos...")
    
    # Demo 1: Different redaction styles
    demo.demo_redaction_styles()
    
    # Demo 2: Real-time moderation
    moderator = demo.demo_real_time_moderation()
    
    # Demo 3: Interactive testing
    demo.interactive_testing()
    
    print(f"\n" + "="*60)
    print("🎉 REDACTION SYSTEM READY!")
    print("="*60)
    
    print("✅ FEATURES IMPLEMENTED:")
    print("- Intelligent toxic word detection")
    print("- Multiple redaction styles (smart, partial, warning, complete)")
    print("- Real-time chat moderation")
    print("- Configurable moderation policies")
    print("- Conversation history and statistics")
    print("- Interactive testing interface")
    
    print(f"\n🚀 HOW TO USE:")
    print("# Initialize redactor")
    print("redactor = IntelligentRedactor()")
    print("")
    print("# Redact a message")
    print("result = redactor.redact_message('Your message here', style='smart')")
    print("print(result['redacted_text'])")
    print("")
    print("# Set up chat moderation")
    print("moderator = MessageModerator(redactor, policy='moderate')")
    print("moderated = moderator.moderate_message('Chat message', 'Username')")
    
    return redactor, moderator

if __name__ == "__main__":
    result = main()
    if result is not None:
        redactor, moderator = result
