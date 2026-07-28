#!/usr/bin/env python3
"""
Simple Semantic Analyzer Test for Toxic Comments

This script focuses specifically on testing the semantic toxicity analyzer
with clear, toxic examples to demonstrate its capabilities.
"""

import sys
import os
import time

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("🧠 Semantic Toxicity Analyzer - Focused Test")
print("=" * 60)

try:
    from semantic.semantic_toxicity_analyzer import SemanticToxicityAnalyzer
    analyzer = SemanticToxicityAnalyzer()
    print("✅ Semantic Analyzer loaded successfully")
except Exception as e:
    print(f"❌ Failed to load analyzer: {e}")
    exit(1)

# Clear toxic examples for testing
test_examples = [
    {
        "text": "You're such a fucking idiot, kill yourself!",
        "description": "Extreme toxicity: profanity + death threat",
        "expected_features": ["direct_attack", "anger/contempt", "high_toxicity"]
    },
    {
        "text": "WHAT THE HELL IS WRONG WITH YOU MORONS???",
        "description": "Aggressive caps with rhetorical question",
        "expected_features": ["excessive_caps", "anger", "aggressive_questions"]
    },
    {
        "text": "All you people are disgusting and pathetic",
        "description": "Group targeting with derogatory labels",
        "expected_features": ["derogatory_labeling", "disgust", "group_criticism"]
    },
    {
        "text": "Go kill yourself, nobody wants you here",
        "description": "Direct death threat with rejection",
        "expected_features": ["threat", "direct_attack", "high_severity"]
    },
    {
        "text": "You're absolutely terrible at this, what a joke!",
        "description": "Personal criticism with dismissive tone",
        "expected_features": ["direct_attack", "contempt", "dismissive"]
    },
    {
        "text": "Sure, that's a brilliant idea... obviously",
        "description": "Sarcastic criticism",
        "expected_features": ["sarcasm", "contempt", "context_dependent"]
    },
    {
        "text": "I'm getting frustrated with this situation",
        "description": "Personal frustration (not targeted)",
        "expected_features": ["frustration", "personal_story", "low_toxicity"]
    },
    {
        "text": "This is a great discussion, thank you!",
        "description": "Positive comment (control)",
        "expected_features": ["positive", "neutral", "minimal_toxicity"]
    }
]

print(f"\n🧪 Testing {len(test_examples)} examples with semantic analysis")
print("=" * 60)

def analyze_and_display(text, description, expected_features):
    """Analyze text and display comprehensive results."""
    
    print(f"\n📝 Text: \"{text}\"")
    print(f"📋 Description: {description}")
    print(f"🎯 Expected: {', '.join(expected_features)}")
    print("-" * 50)
    
    start_time = time.time()
    try:
        result = analyzer.analyze_semantic_toxicity(text)
        processing_time = time.time() - start_time
        
        # Main scores
        print(f"🔢 Semantic Toxicity Score: {result['semantic_toxicity_score']:.3f}")
        print(f"🎯 Analysis Confidence: {result['confidence']:.3f}")
        
        # Intent analysis
        intent_analysis = result['intent_analysis']
        print(f"\n🎭 Intent Analysis:")
        print(f"   Primary Intent: {intent_analysis['primary_intent']}")
        print(f"   Intent Score: {intent_analysis['primary_intent_score']:.3f}")
        print(f"   Intent Confidence: {intent_analysis['intent_confidence']:.3f}")
        
        if intent_analysis['all_detected_intents']:
            print(f"   All Intents: {list(intent_analysis['all_detected_intents'].keys())}")
        
        # Emotional analysis
        emotional_analysis = result['emotional_analysis']
        print(f"\n😤 Emotional Analysis:")
        print(f"   Dominant Emotion: {emotional_analysis['dominant_emotion']}")
        print(f"   Emotional Intensity: {emotional_analysis['emotional_intensity']:.3f}")
        print(f"   Emotional Complexity: {emotional_analysis['emotional_complexity']} emotions")
        print(f"   Caps Usage: {emotional_analysis['caps_usage']:.3f}")
        print(f"   Exclamation Usage: {emotional_analysis['exclamation_usage']:.3f}")
        
        if emotional_analysis['all_detected_emotions']:
            print(f"   All Emotions: {list(emotional_analysis['all_detected_emotions'].keys())}")
        
        # Context analysis
        context_analysis = result['context_analysis']
        print(f"\n🏷️ Context Analysis:")
        print(f"   Primary Context: {context_analysis['primary_context']}")
        print(f"   Context Modifier: {context_analysis['context_modifier']:.3f}")
        print(f"   Context Confidence: {context_analysis['context_confidence']:.3f}")
        
        if context_analysis['all_detected_contexts']:
            print(f"   All Contexts: {list(context_analysis['all_detected_contexts'].keys())}")
        
        # Pattern analysis
        pattern_analysis = result['pattern_analysis']
        print(f"\n🔍 Pattern Analysis:")
        print(f"   Complexity Score: {pattern_analysis['complexity_score']:.3f}")
        
        if pattern_analysis['pattern_matches']:
            print(f"   Detected Patterns:")
            for pattern_type, data in pattern_analysis['pattern_matches'].items():
                print(f"     {pattern_type}: {data['count']} matches ({data['density']:.3f} density)")
        
        sentence_info = pattern_analysis['sentence_structure']
        print(f"   Text Stats: {sentence_info['word_count']} words, {sentence_info['sentence_count']} sentences")
        
        question_info = pattern_analysis['question_analysis']
        if question_info['question_count'] > 0:
            print(f"   Questions: {question_info['question_count']} total, {question_info['rhetorical_indicators']} rhetorical")
        
        # Contextual insights
        insights = result['contextual_insights']
        print(f"\n💡 Contextual Insights:")
        print(f"   Risk Assessment: {insights['risk_assessment']}")
        
        if insights['toxicity_drivers']:
            print(f"   Toxicity Drivers: {', '.join(insights['toxicity_drivers'])}")
        
        if insights['mitigation_factors']:
            print(f"   Mitigation Factors: {', '.join(insights['mitigation_factors'])}")
        
        if insights['recommendations']:
            print(f"   Recommendations:")
            for rec in insights['recommendations']:
                print(f"     • {rec}")
        
        # Generate human-readable explanation
        explanation = analyzer.generate_explanation(text)
        print(f"\n📖 Explanation:")
        print(f"   {explanation}")
        
        print(f"\n⏱️ Processing Time: {processing_time:.4f}s")
        
        # Feature vector info
        features = result['semantic_features']
        non_zero_features = (features != 0).sum()
        print(f"🔧 Feature Vector: {len(features)} dimensions, {non_zero_features} non-zero")
        
        return result
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

# Run tests
results = []
total_time = 0

for i, example in enumerate(test_examples, 1):
    print(f"\n{'='*15} TEST {i}/{len(test_examples)} {'='*15}")
    
    start = time.time()
    result = analyze_and_display(
        example['text'],
        example['description'],
        example['expected_features']
    )
    test_time = time.time() - start
    total_time += test_time
    
    if result:
        results.append({
            'text': example['text'],
            'toxicity_score': result['semantic_toxicity_score'],
            'confidence': result['confidence'],
            'intent': result['intent_analysis']['primary_intent'],
            'emotion': result['emotional_analysis']['dominant_emotion'],
            'context': result['context_analysis']['primary_context'],
            'processing_time': test_time
        })

# Summary statistics
print("\n" + "=" * 60)
print("📊 SEMANTIC ANALYSIS SUMMARY")
print("=" * 60)

if results:
    scores = [r['toxicity_score'] for r in results]
    confidences = [r['confidence'] for r in results]
    times = [r['processing_time'] for r in results]
    
    print(f"📈 Total Tests: {len(results)}")
    print(f"⏱️ Total Time: {total_time:.4f}s")
    print(f"⚡ Avg Time: {total_time/len(results):.4f}s per analysis")
    
    print(f"\n🔢 Toxicity Scores:")
    print(f"   Range: {min(scores):.3f} - {max(scores):.3f}")
    print(f"   Average: {sum(scores)/len(scores):.3f}")
    print(f"   High toxicity (>0.5): {sum(1 for s in scores if s > 0.5)}/{len(scores)}")
    print(f"   Medium toxicity (0.2-0.5): {sum(1 for s in scores if 0.2 <= s <= 0.5)}/{len(scores)}")
    print(f"   Low toxicity (<0.2): {sum(1 for s in scores if s < 0.2)}/{len(scores)}")
    
    print(f"\n🎯 Confidence Scores:")
    print(f"   Range: {min(confidences):.3f} - {max(confidences):.3f}")
    print(f"   Average: {sum(confidences)/len(confidences):.3f}")
    
    # Intent distribution
    intents = {}
    emotions = {}
    contexts = {}
    
    for r in results:
        intents[r['intent']] = intents.get(r['intent'], 0) + 1
        emotions[r['emotion']] = emotions.get(r['emotion'], 0) + 1
        contexts[r['context']] = contexts.get(r['context'], 0) + 1
    
    print(f"\n🎭 Intent Distribution:")
    for intent, count in sorted(intents.items(), key=lambda x: x[1], reverse=True):
        print(f"   {intent}: {count}")
    
    print(f"\n😤 Emotion Distribution:")
    for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
        print(f"   {emotion}: {count}")
    
    print(f"\n🏷️ Context Distribution:")
    for context, count in sorted(contexts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {context}: {count}")

print(f"\n✨ Semantic Toxicity Analysis Complete!")
print(f"   The system successfully analyzed {len(results)} diverse toxic comment examples")
print(f"   Demonstrating advanced intent, emotion, and context understanding")
print("=" * 60)
