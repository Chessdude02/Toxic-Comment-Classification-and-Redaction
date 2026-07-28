"""
Semantic Toxicity Analyzer for Context-Aware Detection

This module provides advanced semantic analysis for toxic comment detection,
going beyond simple keyword matching to understand context, intent, and nuance.
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import re
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import datetime


@dataclass
class SemanticContext:
    """Represents semantic context of a text."""
    text: str
    intent: str
    emotional_tone: str
    context_type: str
    confidence: float
    semantic_features: np.ndarray


@dataclass
class ToxicityIntent:
    """Represents toxicity intent analysis."""
    intent_type: str
    confidence: float
    indicators: List[str]
    severity: float
    mitigation_suggestions: List[str]


class SemanticToxicityAnalyzer:
    """
    Advanced semantic analyzer for toxic content detection.
    Focuses on understanding context, intent, and nuanced toxicity patterns.
    """
    
    def __init__(self):
        # Intent categories for toxicity
        self.toxicity_intents = {
            'direct_attack': {
                'indicators': ['you are', 'you\'re', 'your', 'yourself'],
                'patterns': [r'\byou\s+are\s+\w+', r'\byou\'re\s+\w+', r'\byour\s+\w+'],
                'severity_multiplier': 1.2
            },
            'threat': {
                'indicators': ['will', 'going to', 'gonna', 'should', 'deserve'],
                'patterns': [r'\bwill\s+\w+', r'\bgoing\s+to\s+\w+', r'\bshould\s+\w+'],
                'severity_multiplier': 1.5
            },
            'derogatory_labeling': {
                'indicators': ['all', 'every', 'these', 'those', 'people like'],
                'patterns': [r'\ball\s+\w+\s+are', r'\bevery\s+\w+', r'\bthese\s+\w+'],
                'severity_multiplier': 1.3
            },
            'dismissive': {
                'indicators': ['whatever', 'doesn\'t matter', 'who cares', 'nobody'],
                'patterns': [r'\bwhatever\b', r'\bwho\s+cares\b', r'\bdoesn\'t\s+matter\b'],
                'severity_multiplier': 0.8
            },
            'comparative_insult': {
                'indicators': ['better than', 'worse than', 'compared to', 'unlike'],
                'patterns': [r'\bbetter\s+than\b', r'\bworse\s+than\b', r'\bunlike\b'],
                'severity_multiplier': 1.1
            }
        }
        
        # Emotional tone indicators
        self.emotional_tones = {
            'anger': {
                'indicators': ['angry', 'mad', 'furious', 'rage', 'pissed', 'livid'],
                'punctuation_patterns': ['!!!', '!!!!', 'CAPS_HEAVY'],
                'intensity_markers': ['so', 'very', 'extremely', 'incredibly']
            },
            'disgust': {
                'indicators': ['disgusting', 'gross', 'sick', 'revolting', 'nasty'],
                'punctuation_patterns': ['ugh', 'eww', 'yuck'],
                'intensity_markers': ['absolutely', 'completely', 'totally']
            },
            'contempt': {
                'indicators': ['pathetic', 'worthless', 'useless', 'waste', 'joke'],
                'punctuation_patterns': ['...', 'lol', 'lmao'],
                'intensity_markers': ['such a', 'what a', 'how']
            },
            'frustration': {
                'indicators': ['frustrated', 'annoyed', 'irritated', 'fed up'],
                'punctuation_patterns': ['seriously?', 'really?', 'come on'],
                'intensity_markers': ['getting', 'becoming', 'making me']
            }
        }
        
        # Context types that modify toxicity interpretation
        self.context_types = {
            'debate': {
                'indicators': ['however', 'but', 'although', 'while', 'whereas'],
                'modifiers': ['argue', 'discuss', 'debate', 'point', 'opinion'],
                'toxicity_modifier': 0.8
            },
            'sarcasm': {
                'indicators': ['sure', 'right', 'obviously', 'clearly', '/s'],
                'modifiers': ['yeah right', 'of course', 'totally'],
                'toxicity_modifier': 0.7
            },
            'frustration_vent': {
                'indicators': ['can\'t believe', 'tired of', 'sick of', 'done with'],
                'modifiers': ['always', 'never', 'constantly', 'every time'],
                'toxicity_modifier': 0.9
            },
            'group_criticism': {
                'indicators': ['these people', 'they all', 'this group', 'everyone'],
                'modifiers': ['always', 'never', 'typical', 'classic'],
                'toxicity_modifier': 1.3
            },
            'personal_story': {
                'indicators': ['my', 'me', 'I', 'personally', 'experience'],
                'modifiers': ['happened to', 'told me', 'did to me'],
                'toxicity_modifier': 1.1
            }
        }
        
        # Semantic feature patterns
        self.semantic_patterns = {
            'escalation_markers': [
                'and another thing', 'furthermore', 'also', 'plus', 'on top of that'
            ],
            'contradiction_markers': [
                'but', 'however', 'although', 'despite', 'nevertheless', 'yet'
            ],
            'emphasis_markers': [
                'especially', 'particularly', 'specifically', 'exactly', 'precisely'
            ],
            'generalization_markers': [
                'always', 'never', 'everyone', 'nobody', 'all', 'none'
            ]
        }
    
    def analyze_semantic_toxicity(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive semantic analysis of text for toxicity detection.
        """
        
        # Basic preprocessing
        text = text.strip()
        if not text:
            return self._empty_analysis()
        
        # Core semantic analysis
        intent_analysis = self._analyze_intent(text)
        emotional_analysis = self._analyze_emotional_tone(text)
        context_analysis = self._analyze_context_type(text)
        pattern_analysis = self._analyze_semantic_patterns(text)
        
        # Generate semantic features
        semantic_features = self._generate_semantic_features(
            text, intent_analysis, emotional_analysis, context_analysis, pattern_analysis
        )
        
        # Calculate overall semantic toxicity
        semantic_toxicity = self._calculate_semantic_toxicity(
            intent_analysis, emotional_analysis, context_analysis
        )
        
        # Generate contextual insights
        insights = self._generate_contextual_insights(
            text, intent_analysis, emotional_analysis, context_analysis
        )
        
        return {
            'text': text,
            'semantic_toxicity_score': semantic_toxicity,
            'intent_analysis': intent_analysis,
            'emotional_analysis': emotional_analysis,
            'context_analysis': context_analysis,
            'pattern_analysis': pattern_analysis,
            'semantic_features': semantic_features,
            'contextual_insights': insights,
            'confidence': self._calculate_analysis_confidence(
                intent_analysis, emotional_analysis, context_analysis
            ),
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_intent(self, text: str) -> Dict[str, Any]:
        """Analyze the intent behind the text."""
        
        text_lower = text.lower()
        detected_intents = {}
        
        for intent_type, intent_data in self.toxicity_intents.items():
            score = 0.0
            matched_indicators = []
            matched_patterns = []
            
            # Check indicators
            for indicator in intent_data['indicators']:
                if indicator in text_lower:
                    score += 0.2
                    matched_indicators.append(indicator)
            
            # Check patterns
            for pattern in intent_data['patterns']:
                matches = re.findall(pattern, text_lower)
                if matches:
                    score += 0.3 * len(matches)
                    matched_patterns.extend(matches)
            
            if score > 0:
                detected_intents[intent_type] = {
                    'score': min(score, 1.0),
                    'severity_multiplier': intent_data['severity_multiplier'],
                    'matched_indicators': matched_indicators,
                    'matched_patterns': matched_patterns
                }
        
        # Determine primary intent
        primary_intent = 'neutral'
        max_score = 0
        if detected_intents:
            primary_intent = max(detected_intents.keys(), 
                               key=lambda x: detected_intents[x]['score'])
            max_score = detected_intents[primary_intent]['score']
        
        return {
            'primary_intent': primary_intent,
            'primary_intent_score': max_score,
            'all_detected_intents': detected_intents,
            'intent_confidence': min(max_score * 2, 1.0)
        }
    
    def _analyze_emotional_tone(self, text: str) -> Dict[str, Any]:
        """Analyze emotional tone of the text."""
        
        text_lower = text.lower()
        detected_emotions = {}
        
        for emotion, emotion_data in self.emotional_tones.items():
            score = 0.0
            matched_indicators = []
            punctuation_matches = []
            intensity_matches = []
            
            # Check emotional indicators
            for indicator in emotion_data['indicators']:
                if indicator in text_lower:
                    score += 0.3
                    matched_indicators.append(indicator)
            
            # Check punctuation patterns
            for pattern in emotion_data['punctuation_patterns']:
                if pattern == 'CAPS_HEAVY':
                    # Check for heavy capitalization
                    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
                    if caps_ratio > 0.3:
                        score += 0.2
                        punctuation_matches.append('heavy_caps')
                else:
                    if pattern in text:
                        score += 0.2
                        punctuation_matches.append(pattern)
            
            # Check intensity markers
            for marker in emotion_data['intensity_markers']:
                if marker in text_lower:
                    score += 0.1
                    intensity_matches.append(marker)
            
            if score > 0:
                detected_emotions[emotion] = {
                    'score': min(score, 1.0),
                    'indicators': matched_indicators,
                    'punctuation': punctuation_matches,
                    'intensity': intensity_matches
                }
        
        # Determine dominant emotion
        dominant_emotion = 'neutral'
        max_emotional_score = 0
        if detected_emotions:
            dominant_emotion = max(detected_emotions.keys(),
                                 key=lambda x: detected_emotions[x]['score'])
            max_emotional_score = detected_emotions[dominant_emotion]['score']
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotional_intensity': max_emotional_score,
            'all_detected_emotions': detected_emotions,
            'emotional_complexity': len(detected_emotions),
            'caps_usage': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'exclamation_usage': text.count('!') / len(text) if text else 0
        }
    
    def _analyze_context_type(self, text: str) -> Dict[str, Any]:
        """Analyze the contextual type of the text."""
        
        text_lower = text.lower()
        detected_contexts = {}
        
        for context_type, context_data in self.context_types.items():
            score = 0.0
            matched_indicators = []
            matched_modifiers = []
            
            # Check context indicators
            for indicator in context_data['indicators']:
                if indicator in text_lower:
                    score += 0.3
                    matched_indicators.append(indicator)
            
            # Check context modifiers
            for modifier in context_data['modifiers']:
                if modifier in text_lower:
                    score += 0.2
                    matched_modifiers.append(modifier)
            
            if score > 0:
                detected_contexts[context_type] = {
                    'score': min(score, 1.0),
                    'toxicity_modifier': context_data['toxicity_modifier'],
                    'indicators': matched_indicators,
                    'modifiers': matched_modifiers
                }
        
        # Determine primary context
        primary_context = 'general'
        context_modifier = 1.0
        if detected_contexts:
            primary_context = max(detected_contexts.keys(),
                                key=lambda x: detected_contexts[x]['score'])
            context_modifier = detected_contexts[primary_context]['toxicity_modifier']
        
        return {
            'primary_context': primary_context,
            'context_modifier': context_modifier,
            'all_detected_contexts': detected_contexts,
            'context_confidence': max([ctx['score'] for ctx in detected_contexts.values()]) if detected_contexts else 0.0
        }
    
    def _analyze_semantic_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze semantic patterns in the text."""
        
        text_lower = text.lower()
        pattern_matches = {}
        
        for pattern_type, patterns in self.semantic_patterns.items():
            matches = []
            for pattern in patterns:
                if pattern in text_lower:
                    matches.append(pattern)
            
            if matches:
                pattern_matches[pattern_type] = {
                    'count': len(matches),
                    'matches': matches,
                    'density': len(matches) / len(text.split()) if text.split() else 0
                }
        
        # Analyze sentence structure
        sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
        word_count = len(text.split())
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Analyze question patterns
        question_count = text.count('?')
        rhetorical_indicators = sum(1 for phrase in ['really?', 'seriously?', 'are you kidding?'] 
                                  if phrase in text_lower)
        
        return {
            'pattern_matches': pattern_matches,
            'sentence_structure': {
                'sentence_count': sentence_count,
                'avg_sentence_length': avg_sentence_length,
                'word_count': word_count
            },
            'question_analysis': {
                'question_count': question_count,
                'rhetorical_indicators': rhetorical_indicators,
                'question_density': question_count / sentence_count if sentence_count > 0 else 0
            },
            'complexity_score': self._calculate_text_complexity(text)
        }
    
    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate semantic complexity of the text."""
        
        words = text.split()
        if not words:
            return 0.0
        
        # Lexical diversity
        unique_words = len(set(words))
        lexical_diversity = unique_words / len(words)
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in words])
        
        # Sentence complexity
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        avg_sentence_complexity = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        
        # Punctuation complexity
        punctuation_density = sum(1 for c in text if c in '.,!?;:') / len(text) if text else 0
        
        # Combine measures
        complexity = (
            lexical_diversity * 0.3 +
            min(avg_word_length / 10, 1.0) * 0.2 +
            min(avg_sentence_complexity / 20, 1.0) * 0.3 +
            punctuation_density * 0.2
        )
        
        return min(complexity, 1.0)
    
    def _generate_semantic_features(self, text: str, intent_analysis: Dict,
                                  emotional_analysis: Dict, context_analysis: Dict,
                                  pattern_analysis: Dict) -> np.ndarray:
        """Generate semantic feature vector for the text."""
        
        features = []
        
        # Intent features (5 dimensions)
        intent_vector = [0.0] * len(self.toxicity_intents)
        for i, intent_type in enumerate(self.toxicity_intents.keys()):
            if intent_type in intent_analysis['all_detected_intents']:
                intent_vector[i] = intent_analysis['all_detected_intents'][intent_type]['score']
        features.extend(intent_vector)
        
        # Emotional features (4 dimensions)
        emotion_vector = [0.0] * len(self.emotional_tones)
        for i, emotion_type in enumerate(self.emotional_tones.keys()):
            if emotion_type in emotional_analysis['all_detected_emotions']:
                emotion_vector[i] = emotional_analysis['all_detected_emotions'][emotion_type]['score']
        features.extend(emotion_vector)
        
        # Context features (5 dimensions)
        context_vector = [0.0] * len(self.context_types)
        for i, context_type in enumerate(self.context_types.keys()):
            if context_type in context_analysis['all_detected_contexts']:
                context_vector[i] = context_analysis['all_detected_contexts'][context_type]['score']
        features.extend(context_vector)
        
        # Pattern features (4 dimensions)
        pattern_counts = []
        for pattern_type in self.semantic_patterns.keys():
            count = 0
            if pattern_type in pattern_analysis['pattern_matches']:
                count = pattern_analysis['pattern_matches'][pattern_type]['density']
            pattern_counts.append(count)
        features.extend(pattern_counts)
        
        # Additional semantic features (6 dimensions)
        additional_features = [
            intent_analysis['intent_confidence'],
            emotional_analysis['emotional_intensity'],
            context_analysis['context_confidence'],
            pattern_analysis['complexity_score'],
            emotional_analysis['caps_usage'],
            emotional_analysis['exclamation_usage']
        ]
        features.extend(additional_features)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_semantic_toxicity(self, intent_analysis: Dict, emotional_analysis: Dict,
                                   context_analysis: Dict) -> float:
        """Calculate overall semantic toxicity score."""
        
        base_score = 0.0
        
        # Intent contribution
        if intent_analysis['primary_intent'] != 'neutral':
            intent_data = intent_analysis['all_detected_intents'][intent_analysis['primary_intent']]
            base_score += intent_data['score'] * intent_data['severity_multiplier'] * 0.4
        
        # Emotional contribution
        if emotional_analysis['dominant_emotion'] != 'neutral':
            emotional_intensity = emotional_analysis['emotional_intensity']
            if emotional_analysis['dominant_emotion'] in ['anger', 'disgust', 'contempt']:
                base_score += emotional_intensity * 0.3
            else:
                base_score += emotional_intensity * 0.2
        
        # Context modification
        context_modifier = context_analysis['context_modifier']
        base_score *= context_modifier
        
        # Apply caps and exclamation penalty
        caps_penalty = emotional_analysis['caps_usage'] * 0.1
        exclamation_penalty = min(emotional_analysis['exclamation_usage'] * 5, 0.2)
        
        base_score += caps_penalty + exclamation_penalty
        
        return min(base_score, 1.0)
    
    def _generate_contextual_insights(self, text: str, intent_analysis: Dict,
                                    emotional_analysis: Dict, context_analysis: Dict) -> Dict[str, Any]:
        """Generate contextual insights about the toxicity."""
        
        insights = {
            'toxicity_drivers': [],
            'mitigation_factors': [],
            'risk_assessment': 'low',
            'recommendations': []
        }
        
        # Identify toxicity drivers
        if intent_analysis['primary_intent'] != 'neutral':
            insights['toxicity_drivers'].append(f"Intent: {intent_analysis['primary_intent']}")
        
        if emotional_analysis['dominant_emotion'] in ['anger', 'contempt', 'disgust']:
            insights['toxicity_drivers'].append(f"Emotion: {emotional_analysis['dominant_emotion']}")
        
        # Identify mitigation factors
        if context_analysis['primary_context'] in ['debate', 'sarcasm']:
            insights['mitigation_factors'].append(f"Context: {context_analysis['primary_context']}")
        
        if emotional_analysis['caps_usage'] < 0.1 and emotional_analysis['exclamation_usage'] < 0.1:
            insights['mitigation_factors'].append("Controlled tone")
        
        # Risk assessment
        if len(insights['toxicity_drivers']) >= 2:
            insights['risk_assessment'] = 'high'
        elif len(insights['toxicity_drivers']) == 1:
            insights['risk_assessment'] = 'medium'
        
        # Generate recommendations
        if intent_analysis['primary_intent'] == 'direct_attack':
            insights['recommendations'].append("Consider redaction due to direct personal attack")
        
        if emotional_analysis['emotional_intensity'] > 0.7:
            insights['recommendations'].append("High emotional intensity detected - monitor for escalation")
        
        if context_analysis['primary_context'] == 'sarcasm':
            insights['recommendations'].append("Sarcastic tone detected - consider context-aware handling")
        
        return insights
    
    def _calculate_analysis_confidence(self, intent_analysis: Dict, emotional_analysis: Dict,
                                     context_analysis: Dict) -> float:
        """Calculate overall confidence in the semantic analysis."""
        
        intent_confidence = intent_analysis['intent_confidence']
        emotional_confidence = emotional_analysis['emotional_intensity']
        context_confidence = context_analysis['context_confidence']
        
        # Weighted average with bias toward clearer signals
        confidence = (
            intent_confidence * 0.4 +
            emotional_confidence * 0.3 +
            context_confidence * 0.3
        )
        
        return min(confidence, 1.0)
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis for invalid input."""
        return {
            'text': '',
            'semantic_toxicity_score': 0.0,
            'intent_analysis': {'primary_intent': 'neutral', 'intent_confidence': 0.0},
            'emotional_analysis': {'dominant_emotion': 'neutral', 'emotional_intensity': 0.0},
            'context_analysis': {'primary_context': 'general', 'context_modifier': 1.0},
            'pattern_analysis': {},
            'semantic_features': np.zeros(24, dtype=np.float32),
            'contextual_insights': {'toxicity_drivers': [], 'mitigation_factors': [], 'risk_assessment': 'none'},
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat()
        }
    
    def compare_semantic_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """Compare semantic similarity between two texts."""
        
        analysis1 = self.analyze_semantic_toxicity(text1)
        analysis2 = self.analyze_semantic_toxicity(text2)
        
        features1 = analysis1['semantic_features']
        features2 = analysis2['semantic_features']
        
        # Cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        cosine_sim = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        
        # Intent similarity
        intent_sim = 1.0 if analysis1['intent_analysis']['primary_intent'] == analysis2['intent_analysis']['primary_intent'] else 0.0
        
        # Emotion similarity
        emotion_sim = 1.0 if analysis1['emotional_analysis']['dominant_emotion'] == analysis2['emotional_analysis']['dominant_emotion'] else 0.0
        
        # Context similarity
        context_sim = 1.0 if analysis1['context_analysis']['primary_context'] == analysis2['context_analysis']['primary_context'] else 0.0
        
        return {
            'overall_similarity': cosine_sim,
            'intent_similarity': intent_sim,
            'emotion_similarity': emotion_sim,
            'context_similarity': context_sim,
            'semantic_distance': 1.0 - cosine_sim
        }
    
    def generate_explanation(self, text: str) -> str:
        """Generate human-readable explanation of semantic analysis."""
        
        analysis = self.analyze_semantic_toxicity(text)
        
        explanation_parts = []
        
        # Overall assessment
        toxicity_score = analysis['semantic_toxicity_score']
        if toxicity_score > 0.7:
            explanation_parts.append("This text shows high semantic toxicity.")
        elif toxicity_score > 0.4:
            explanation_parts.append("This text shows moderate semantic toxicity.")
        else:
            explanation_parts.append("This text shows low semantic toxicity.")
        
        # Intent explanation
        intent = analysis['intent_analysis']['primary_intent']
        if intent != 'neutral':
            explanation_parts.append(f"The primary intent appears to be {intent.replace('_', ' ')}.")
        
        # Emotional explanation
        emotion = analysis['emotional_analysis']['dominant_emotion']
        if emotion != 'neutral':
            intensity = analysis['emotional_analysis']['emotional_intensity']
            explanation_parts.append(f"The dominant emotional tone is {emotion} with intensity {intensity:.1f}.")
        
        # Context explanation
        context = analysis['context_analysis']['primary_context']
        if context != 'general':
            modifier = analysis['context_analysis']['context_modifier']
            explanation_parts.append(f"The context appears to be {context.replace('_', ' ')}, which {'amplifies' if modifier > 1 else 'reduces'} the toxicity.")
        
        # Insights
        insights = analysis['contextual_insights']
        if insights['toxicity_drivers']:
            explanation_parts.append(f"Main toxicity drivers: {', '.join(insights['toxicity_drivers'])}.")
        
        if insights['mitigation_factors']:
            explanation_parts.append(f"Mitigating factors: {', '.join(insights['mitigation_factors'])}.")
        
        return ' '.join(explanation_parts)


if __name__ == "__main__":
    # Test the semantic analyzer
    print("🧠 Testing Semantic Toxicity Analyzer")
    print("=" * 60)
    
    analyzer = SemanticToxicityAnalyzer()
    
    # Test various semantic patterns
    test_messages = [
        "You're such an idiot!",  # Direct attack
        "I can't believe how stupid this is",  # Frustration
        "All politicians are corrupt liars",  # Generalization
        "Sure, that's a great idea... obviously",  # Sarcasm
        "I disagree with your opinion",  # Debate
        "This is really well written",  # Positive
        "You people are always causing problems",  # Group criticism
        "WHAT THE HELL IS WRONG WITH YOU???",  # High emotion + caps
        "I'm getting frustrated with this situation",  # Personal story
        "Everyone knows that's completely wrong",  # Generalization + certainty
    ]
    
    print("\n🧪 Testing Semantic Analysis:")
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. \"{message}\"")
        
        analysis = analyzer.analyze_semantic_toxicity(message)
        
        print(f"   Semantic Toxicity: {analysis['semantic_toxicity_score']:.3f}")
        print(f"   Intent: {analysis['intent_analysis']['primary_intent']} (conf: {analysis['intent_analysis']['intent_confidence']:.2f})")
        print(f"   Emotion: {analysis['emotional_analysis']['dominant_emotion']} (intensity: {analysis['emotional_analysis']['emotional_intensity']:.2f})")
        print(f"   Context: {analysis['context_analysis']['primary_context']} (modifier: {analysis['context_analysis']['context_modifier']:.2f})")
        
        insights = analysis['contextual_insights']
        if insights['toxicity_drivers']:
            print(f"   Drivers: {', '.join(insights['toxicity_drivers'])}")
        if insights['mitigation_factors']:
            print(f"   Mitigating: {', '.join(insights['mitigation_factors'])}")
        
        print(f"   Risk: {insights['risk_assessment']} | Confidence: {analysis['confidence']:.2f}")
        
        # Generate explanation
        explanation = analyzer.generate_explanation(message)
        print(f"   📝 {explanation}")
    
    print(f"\n📊 Feature Analysis:")
    sample_analysis = analyzer.analyze_semantic_toxicity("You're absolutely terrible at this!")
    features = sample_analysis['semantic_features']
    print(f"   Feature vector shape: {features.shape}")
    print(f"   Non-zero features: {np.count_nonzero(features)}")
    print(f"   Max feature value: {features.max():.3f}")
    
    print(f"\n🔍 Similarity Testing:")
    text1 = "You're an idiot!"
    text2 = "You're so stupid!"
    text3 = "Great job everyone!"
    
    sim1 = analyzer.compare_semantic_similarity(text1, text2)
    sim2 = analyzer.compare_semantic_similarity(text1, text3)
    
    print(f"   '{text1}' vs '{text2}': {sim1['overall_similarity']:.3f}")
    print(f"   '{text1}' vs '{text3}': {sim2['overall_similarity']:.3f}")
    
    print(f"\n✅ Semantic Analysis System Ready!")
    print(f"   Features: Intent analysis, emotional tone detection, context awareness")
    print(f"   Advanced: Pattern recognition, risk assessment, similarity comparison")
