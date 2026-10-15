"""
Test Script for Redaction System Validation

This script tests the redaction system with diverse examples to check:
1. Correct redaction of toxic content
2. Preservation of non-toxic content
3. Detection of false positives (overfitting)
4. Edge cases and context sensitivity
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import sys
import os
from typing import List, Dict, Any

# Add upgrade2 directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
upgrade_dir = os.path.join(current_dir, "upgrade2")
sys.path.insert(0, upgrade_dir)

try:
    from redaction.redaction_manager import RedactionManager, RedactionStyle
    from integration.enhanced_toxicity_detector import EnhancedToxicityDetector
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def test_toxic_content_redaction():
    """Test that clearly toxic content is properly redacted."""
    
    print("🔍 Testing Toxic Content Redaction")
    print("-" * 50)
    
    redaction_manager = RedactionManager()
    
    # Clear cases of toxic content that should be redacted
    toxic_samples = [
        "You're such an idiot!",
        "This is complete shit",
        "I hate you, you moron",
        "What the hell is wrong with you?",
        "You're pathetic and worthless",
        "Damn, you're stupid",
        "That's fucking ridiculous",
        "You people are disgusting",
        "Kill yourself, loser",
        "This is garbage, you ass"
    ]
    
    redacted_count = 0
    total_count = len(toxic_samples)
    
    for i, text in enumerate(toxic_samples, 1):
        result = redaction_manager.generate_redaction(text)
        
        is_redacted = len(result.redacted_words) > 0
        redacted_count += is_redacted
        
        status = "✅ REDACTED" if is_redacted else "❌ NOT REDACTED"
        print(f"{i:2d}. {status} | Score: {result.toxicity_score:.3f} | \"{text}\"")
        
        if is_redacted:
            print(f"     → \"{result.redacted_text}\"")
    
    print(f"\nToxic Content Results: {redacted_count}/{total_count} ({redacted_count/total_count*100:.1f}%) redacted")
    
    return redacted_count, total_count


def test_non_toxic_content_preservation():
    """Test that non-toxic content is preserved (no false positives)."""
    
    print("\n🛡️  Testing Non-Toxic Content Preservation")
    print("-" * 50)
    
    redaction_manager = RedactionManager()
    
    # Clear cases of non-toxic content that should NOT be redacted
    clean_samples = [
        "This is a great idea!",
        "I really appreciate your help",
        "Thank you for your hard work",
        "The project looks excellent",
        "I disagree, but I respect your opinion",
        "That's an interesting perspective",
        "Could you help me understand?",
        "I'm not sure about this approach",
        "Let's discuss this further",
        "The results are impressive"
    ]
    
    preserved_count = 0
    total_count = len(clean_samples)
    
    for i, text in enumerate(clean_samples, 1):
        result = redaction_manager.generate_redaction(text)
        
        is_preserved = len(result.redacted_words) == 0
        preserved_count += is_preserved
        
        status = "✅ PRESERVED" if is_preserved else "❌ REDACTED"
        print(f"{i:2d}. {status} | Score: {result.toxicity_score:.3f} | \"{text}\"")
        
        if not is_preserved:
            print(f"     → \"{result.redacted_text}\" (redacted: {result.redacted_words})")
    
    print(f"\nNon-Toxic Content Results: {preserved_count}/{total_count} ({preserved_count/total_count*100:.1f}%) preserved")
    
    return preserved_count, total_count


def test_edge_cases_and_context():
    """Test edge cases and context sensitivity."""
    
    print("\n🎯 Testing Edge Cases and Context Sensitivity")
    print("-" * 50)
    
    redaction_manager = RedactionManager()
    
    # Edge cases that test context sensitivity and potential false positives
    edge_cases = [
        # Context matters
        ("I'm damn proud of this achievement", "Positive context with mild profanity"),
        ("Hell yes, great job everyone!", "Enthusiastic expression"),
        ("That's a hell of a good idea", "Positive emphasis"),
        
        # Technical/neutral usage
        ("The system kill command stopped the process", "Technical terminology"),
        ("We need to destroy the old database", "Technical action"),
        ("This is a critical error", "Technical severity"),
        
        # Sarcasm/irony (tricky cases)
        ("Oh sure, that's totally brilliant", "Sarcastic but not directly insulting"),
        ("Yeah right, like that'll work", "Dismissive sarcasm"),
        
        # Mild frustration vs. toxicity
        ("This is frustrating", "Mild frustration"),
        ("I'm getting annoyed with this", "Personal feeling expression"),
        ("That's pretty stupid", "Mild negative opinion"),
        
        # Borderline cases
        ("You're wrong about this", "Disagreement"),
        ("That doesn't make sense", "Critical assessment"),
        ("I think you're mistaken", "Polite disagreement")
    ]
    
    context_aware_count = 0
    total_count = len(edge_cases)
    
    for i, (text, description) in enumerate(edge_cases, 1):
        result = redaction_manager.generate_redaction(text)
        
        is_redacted = len(result.redacted_words) > 0
        
        # For edge cases, we want to see nuanced handling
        status_icon = "🤔" if is_redacted else "✅"
        status_text = "REDACTED" if is_redacted else "PRESERVED"
        
        print(f"{i:2d}. {status_icon} {status_text} | Score: {result.toxicity_score:.3f}")
        print(f"     Text: \"{text}\"")
        print(f"     Context: {description}")
        
        if is_redacted:
            print(f"     Redacted: \"{result.redacted_text}\"")
            print(f"     Reason: {result.redaction_reason}")
        
        print()
    
    return edge_cases


def test_redaction_styles():
    """Test different redaction styles."""
    
    print("\n🎨 Testing Redaction Styles")
    print("-" * 50)
    
    redaction_manager = RedactionManager()
    
    test_text = "You're an idiot and this shit is terrible"
    
    print(f"Original: \"{test_text}\"")
    print()
    
    for style in RedactionStyle:
        result = redaction_manager.generate_redaction(test_text, style)
        print(f"{style.value.upper():12}: \"{result.redacted_text}\"")


def test_threshold_sensitivity():
    """Test how different thresholds affect redaction decisions."""
    
    print("\n⚖️  Testing Threshold Sensitivity")
    print("-" * 50)
    
    redaction_manager = RedactionManager()
    
    # Test messages with varying toxicity levels
    test_cases = [
        "You're kind of stupid",      # Low-moderate toxicity
        "That's pretty dumb",         # Low toxicity
        "This is damn annoying",      # Mild profanity
        "You're an idiot",           # Direct insult
        "I hate this crap",          # Moderate toxicity
        "What the hell?",            # Mild expression
    ]
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    
    for text in test_cases:
        print(f"\nText: \"{text}\"")
        
        # Get base toxicity score
        base_result = redaction_manager.generate_redaction(text)
        print(f"  Toxicity Score: {base_result.toxicity_score:.3f}")
        
        for threshold in thresholds:
            redaction_manager.set_redaction_threshold(threshold)
            result = redaction_manager.generate_redaction(text)
            
            is_redacted = len(result.redacted_words) > 0
            status = "REDACTED" if is_redacted else "preserved"
            print(f"    Threshold {threshold}: {status}")


def test_batch_processing():
    """Test batch processing capabilities."""
    
    print("\n📦 Testing Batch Processing")
    print("-" * 50)
    
    redaction_manager = RedactionManager()
    
    batch_texts = [
        "Great work everyone!",
        "This is stupid",
        "I appreciate your effort",
        "You're an idiot",
        "Thank you for helping",
        "This is shit",
        "Well done on the project",
        "I hate this garbage",
        "That's a brilliant idea",
        "You're pathetic"
    ]
    
    # Process batch
    results = redaction_manager.batch_redact(batch_texts)
    
    # Get statistics
    stats = redaction_manager.get_redaction_stats(batch_texts)
    
    print("Batch Results:")
    for i, (text, result) in enumerate(zip(batch_texts, results), 1):
        status = "REDACTED" if result.redacted_words else "preserved"
        print(f"{i:2d}. {status:9} | {result.toxicity_score:.3f} | \"{text}\"")
    
    print(f"\nBatch Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")


def run_comprehensive_test():
    """Run all tests and provide summary."""
    
    print("🧪 COMPREHENSIVE REDACTION SYSTEM TEST")
    print("=" * 60)
    
    # Run individual tests
    toxic_redacted, toxic_total = test_toxic_content_redaction()
    clean_preserved, clean_total = test_non_toxic_content_preservation()
    edge_cases = test_edge_cases_and_context()
    
    test_redaction_styles()
    test_threshold_sensitivity()
    test_batch_processing()
    
    # Summary
    print("\n📊 SUMMARY")
    print("=" * 60)
    
    toxic_accuracy = toxic_redacted / toxic_total * 100
    clean_accuracy = clean_preserved / clean_total * 100
    overall_accuracy = (toxic_redacted + clean_preserved) / (toxic_total + clean_total) * 100
    
    print(f"Toxic Content Detection:     {toxic_redacted}/{toxic_total} ({toxic_accuracy:.1f}%)")
    print(f"Clean Content Preservation:  {clean_preserved}/{clean_total} ({clean_accuracy:.1f}%)")
    print(f"Overall Accuracy:           {overall_accuracy:.1f}%")
    
    # Evaluation
    print(f"\n🎯 EVALUATION:")
    
    if toxic_accuracy >= 80:
        print("✅ Good toxic content detection")
    elif toxic_accuracy >= 60:
        print("⚠️  Moderate toxic content detection")
    else:
        print("❌ Poor toxic content detection")
    
    if clean_accuracy >= 80:
        print("✅ Good false positive control")
    elif clean_accuracy >= 60:
        print("⚠️  Some false positives detected")
    else:
        print("❌ High false positive rate (overfitting suspected)")
    
    if overall_accuracy >= 80:
        print("✅ System performs well overall")
    else:
        print("⚠️  System needs improvement")
    
    print(f"\n🔬 RECOMMENDATIONS:")
    if toxic_accuracy < 80:
        print("• Consider lowering redaction threshold")
        print("• Add more toxic patterns to vocabulary")
    
    if clean_accuracy < 80:
        print("• Review vocabulary for overly broad patterns")
        print("• Improve context awareness")
        print("• Consider raising redaction threshold")
    
    print(f"\n✅ TEST COMPLETE")
    print(f"   The redaction system has been thoroughly tested.")
    print(f"   Check the results above for performance details.")


if __name__ == "__main__":
    try:
        run_comprehensive_test()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
