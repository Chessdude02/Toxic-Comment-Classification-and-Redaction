#!/usr/bin/env python3
"""
Smart Redaction System Demo

This script demonstrates the intelligent redaction capabilities
of the enhanced toxicity detection system with real toxic examples.
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from smart_redaction_system import SmartRedactionSystem, RedactionLevel, RedactionStyle
    redaction_available = True
except ImportError:
    print("Warning: Redaction system not available. Using fallback demo.")
    redaction_available = False

def demo_fallback_redaction():
    """Simple fallback redaction demo."""
    
    print("🔒 Fallback Redaction Demo")
    print("=" * 50)
    
    # Simple pattern-based redaction
    toxic_patterns = {
        'fuck': 'f***', 'fucking': 'f***ing', 'shit': 's***', 'damn': 'd***',
        'hell': 'h***', 'idiot': 'i****', 'stupid': 's*****', 'moron': 'm****',
        'kill': 'k***', 'die': 'd**', 'hate': 'h***', 'pathetic': 'p*******'
    }
    
    test_examples = [
        "You're such a fucking idiot, kill yourself!",
        "SHUT UP YOU STUPID MORON!!!",
        "All you people are disgusting and pathetic",
        "Go to hell you damn fool",
        "This is a great discussion, thank you!"
    ]
    
    for i, text in enumerate(test_examples, 1):
        redacted = text
        redacted_words = []
        
        for word, replacement in toxic_patterns.items():
            if word in text.lower():
                redacted = redacted.replace(word, replacement)
                redacted = redacted.replace(word.upper(), replacement.upper())
                redacted = redacted.replace(word.capitalize(), replacement.capitalize())
                if word in text.lower():
                    redacted_words.append(word)
        
        print(f"\n{i}. Original: \"{text}\"")
        print(f"   Redacted: \"{redacted}\"")
        if redacted_words:
            print(f"   Blocked: {', '.join(redacted_words)}")

def demo_smart_redaction():
    """Demo using the smart redaction system."""
    
    print("🔒 Smart Redaction System Demo")
    print("=" * 50)
    
    # Initialize redaction system
    redactor = SmartRedactionSystem()
    
    # Test examples with various toxicity levels
    test_examples = [
        {
            "text": "You're such a fucking idiot, kill yourself!",
            "description": "Extreme toxicity - profanity + death threat",
            "expected_redaction": True
        },
        {
            "text": "SHUT UP YOU STUPID MORON!!!",
            "description": "Aggressive caps with insults",
            "expected_redaction": True
        },
        {
            "text": "All you people are disgusting and pathetic",
            "description": "Group targeting with derogatory terms",
            "expected_redaction": True
        },
        {
            "text": "Go to hell you damn bastard",
            "description": "Mild profanity with insult",
            "expected_redaction": True
        },
        {
            "text": "This is absolutely terrible, what a joke!",
            "description": "Critical but not personal",
            "expected_redaction": False
        },
        {
            "text": "Sure, that's a brilliant idea... obviously",
            "description": "Sarcastic criticism",
            "expected_redaction": False
        },
        {
            "text": "I disagree with your opinion respectfully",
            "description": "Polite disagreement",
            "expected_redaction": False
        },
        {
            "text": "This is a great discussion, thank you!",
            "description": "Positive comment",
            "expected_redaction": False
        }
    ]
    
    print(f"\n🧪 Testing {len(test_examples)} examples:")
    print("=" * 50)
    
    redaction_stats = {"total": 0, "redacted": 0, "correct": 0}
    
    for i, example in enumerate(test_examples, 1):
        text = example["text"]
        description = example["description"]
        expected = example["expected_redaction"]
        
        print(f"\n{i:2d}. {description}")
        print(f"    Text: \"{text}\"")
        
        # Test different redaction levels
        levels = [RedactionLevel.MINIMAL, RedactionLevel.MODERATE, RedactionLevel.AGGRESSIVE]
        
        for level in levels:
            result = redactor.redact_content(text, redaction_level=level)
            
            print(f"\n    {level.value.upper()} Redaction:")
            print(f"    → \"{result.redacted_text}\"")
            print(f"    → Toxicity: {result.toxicity_score:.3f} | Confidence: {result.confidence:.3f}")
            
            if result.redacted_words:
                print(f"    → Redacted: {', '.join(result.redacted_words)}")
            
            if result.suggestions:
                print(f"    → Suggestion: {result.suggestions[0]}")
            
            # Track accuracy for moderate level
            if level == RedactionLevel.MODERATE:
                redaction_stats["total"] += 1
                was_redacted = len(result.redacted_words) > 0
                if was_redacted == expected:
                    redaction_stats["correct"] += 1
                if was_redacted:
                    redaction_stats["redacted"] += 1
        
        print("    " + "-" * 60)
    
    # Test different styles
    print(f"\n🎨 Testing Different Redaction Styles:")
    print("=" * 50)
    
    toxic_example = "You're such a damn idiot, shut the hell up!"
    styles = [RedactionStyle.ASTERISKS, RedactionStyle.BRACKETS, RedactionStyle.EUPHEMISMS, RedactionStyle.PARTIAL]
    
    print(f"Original: \"{toxic_example}\"")
    
    for style in styles:
        result = redactor.redact_content(toxic_example, 
                                       redaction_level=RedactionLevel.MODERATE,
                                       redaction_style=style)
        print(f"{style.value.capitalize():12s}: \"{result.redacted_text}\"")
    
    # Performance summary
    print(f"\n📊 Performance Summary:")
    print("=" * 50)
    accuracy = redaction_stats["correct"] / redaction_stats["total"] * 100 if redaction_stats["total"] > 0 else 0
    redaction_rate = redaction_stats["redacted"] / redaction_stats["total"] * 100 if redaction_stats["total"] > 0 else 0
    
    print(f"Total Examples: {redaction_stats['total']}")
    print(f"Redaction Accuracy: {accuracy:.1f}% ({redaction_stats['correct']}/{redaction_stats['total']})")
    print(f"Redaction Rate: {redaction_rate:.1f}% ({redaction_stats['redacted']}/{redaction_stats['total']})")
    
    # Generate detailed report for one example
    print(f"\n📋 Detailed Report Example:")
    print("=" * 50)
    
    sample_result = redactor.redact_content("You're a fucking moron!", RedactionLevel.MODERATE)
    report = redactor.create_redaction_report(sample_result)
    print(report)

def main():
    """Main demo function."""
    
    print("🔥 Enhanced Toxicity Redaction Demo")
    print("=" * 60)
    print("Demonstrating intelligent content redaction capabilities")
    print("=" * 60)
    
    if redaction_available:
        demo_smart_redaction()
    else:
        demo_fallback_redaction()
    
    print(f"\n✨ Redaction Demo Complete!")
    
    if redaction_available:
        print("🔒 Smart redaction system successfully demonstrated")
        print("Features: Context-aware, multiple levels, style options, suggestions")
    else:
        print("⚠️  Fallback redaction demonstrated (limited functionality)")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
