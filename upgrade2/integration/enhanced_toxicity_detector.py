"""
Enhanced Toxicity Detector - Integration Module

This module integrates the enhanced vocabulary manager and semantic toxicity analyzer
to provide comprehensive, context-aware toxic comment detection with improved accuracy
and nuanced understanding of toxicity patterns.
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import logging

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
upgrade_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(upgrade_dir)
sys.path.extend([upgrade_dir, project_dir])

try:
    from upgrade2.vocabulary.enhanced_vocabulary_manager import EnhancedToxicVocabularyManager
    from upgrade2.semantic.semantic_toxicity_analyzer import SemanticToxicityAnalyzer
except ImportError:
    # Fallback for direct execution
    print("Warning: Could not import modules. Running in standalone mode.")
    

@dataclass
class ToxicityDetectionResult:
    """Comprehensive toxicity detection result."""
    text: str
    overall_toxicity_score: float
    confidence: float
    detection_components: Dict[str, Any]
    recommendations: List[str]
    redacted_version: str
    severity_level: str
    risk_factors: List[str]
    mitigation_factors: List[str]
    timestamp: str


@dataclass
class DetectionMetrics:
    """Metrics for detection performance tracking."""
    vocabulary_coverage: float
    semantic_confidence: float
    pattern_diversity: int
    processing_time: float
    feature_count: int


class EnhancedToxicityDetector:
    """
    Comprehensive toxicity detection system that integrates multiple approaches:
    - Enhanced vocabulary-based detection
    - Semantic context analysis
    - Pattern recognition
    - Risk assessment and mitigation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced detector with optional configuration."""
        
        self.config = config or self._default_config()
        
        # Initialize components
        try:
            self.vocab_manager = EnhancedToxicVocabularyManager()
            self.semantic_analyzer = SemanticToxicityAnalyzer()
            
            self.vocab_available = True
            self.semantic_available = True
            
        except Exception as e:
            logging.warning(f"Could not initialize all components: {e}")
            self.vocab_available = False
            self.semantic_available = False
            
            # Initialize fallback components
            self._init_fallback_components()
        
        # Detection thresholds
        self.thresholds = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.85
        }
        
        # Weighting for different detection methods
        self.method_weights = {
            'vocabulary': self.config.get('vocabulary_weight', 0.4),
            'semantic': self.config.get('semantic_weight', 0.4),
            'pattern': self.config.get('pattern_weight', 0.2)
        }
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'high_confidence_detections': 0,
            'redaction_recommendations': 0,
            'false_positive_flags': 0
        }
        
        logging.info("Enhanced Toxicity Detector initialized successfully")
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'vocabulary_weight': 0.4,
            'semantic_weight': 0.4,
            'pattern_weight': 0.2,
            'confidence_threshold': 0.6,
            'redaction_threshold': 0.7,
            'enable_explanations': True,
            'enable_redaction': True,
            'enable_similarity_check': True,
            'max_processing_time': 5.0  # seconds
        }
    
    def _init_fallback_components(self):
        """Initialize fallback components when main modules aren't available."""
        
        # Fallback vocabulary patterns
        self.fallback_toxic_patterns = {
            'direct_insults': ['idiot', 'stupid', 'moron', 'dumb', 'pathetic'],
            'profanity': ['damn', 'hell', 'crap'],
            'hate_speech': ['hate', 'despise', 'loathe'],
            'threats': ['kill', 'die', 'destroy'],
            'discriminatory': ['all', 'every', 'these people']
        }
        
        # Fallback semantic indicators
        self.fallback_semantic_indicators = {
            'high_caps': 0.3,  # Threshold for caps ratio
            'exclamation_heavy': 3,  # Number of exclamations
            'question_aggressive': ['what the', 'why the', 'how dare']
        }
    
    def detect_toxicity(self, text: str, context: Optional[Dict[str, Any]] = None) -> ToxicityDetectionResult:
        """
        Perform comprehensive toxicity detection on input text.
        
        Args:
            text: Input text to analyze
            context: Optional context information
            
        Returns:
            Comprehensive detection result
        """
        
        start_time = datetime.now()
        
        # Input validation
        if not text or not isinstance(text, str):
            return self._empty_result(text or "")
        
        text = text.strip()
        if not text:
            return self._empty_result(text)
        
        detection_components = {}
        
        # Vocabulary-based detection
        if self.vocab_available:
            vocab_result = self._vocabulary_detection(text, context)
            detection_components['vocabulary'] = vocab_result
        else:
            detection_components['vocabulary'] = self._fallback_vocabulary_detection(text)
        
        # Semantic analysis
        if self.semantic_available:
            semantic_result = self._semantic_detection(text, context)
            detection_components['semantic'] = semantic_result
        else:
            detection_components['semantic'] = self._fallback_semantic_detection(text)
        
        # Pattern analysis
        pattern_result = self._pattern_detection(text, context)
        detection_components['pattern'] = pattern_result
        
        # Aggregate results
        overall_score, confidence = self._aggregate_scores(detection_components)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(text, detection_components, overall_score)
        
        # Create redacted version if needed
        redacted_text = self._generate_redacted_version(text, detection_components) if self.config.get('enable_redaction', True) else text
        
        # Assess severity and risk factors
        severity_level = self._assess_severity(overall_score)
        risk_factors, mitigation_factors = self._assess_risk_factors(detection_components)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update statistics
        self._update_stats(overall_score, confidence)
        
        return ToxicityDetectionResult(
            text=text,
            overall_toxicity_score=overall_score,
            confidence=confidence,
            detection_components=detection_components,
            recommendations=recommendations,
            redacted_version=redacted_text,
            severity_level=severity_level,
            risk_factors=risk_factors,
            mitigation_factors=mitigation_factors,
            timestamp=datetime.now().isoformat()
        )
    
    def _vocabulary_detection(self, text: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Perform vocabulary-based toxicity detection."""
        
        try:
            # Get enhanced analysis
            vocab_analysis = self.vocab_manager.analyze_text_toxicity(text)
            
            # Extract toxic words for redaction candidates
            redaction_candidates = list(vocab_analysis.get('toxic_words', {}).keys())
            
            return {
                'method': 'enhanced_vocabulary',
                'toxicity_score': vocab_analysis['overall_toxicity'],
                'confidence': 0.8,  # Default confidence for vocab analysis
                'matched_categories': vocab_analysis['category_scores'],
                'context_analysis': vocab_analysis['context_analysis'],
                'slang_detected': vocab_analysis.get('slang_detected', []),
                'escalation_detected': vocab_analysis.get('escalation_detected', {}),
                'redaction_candidates': redaction_candidates
            }
            
        except Exception as e:
            logging.warning(f"Vocabulary detection error: {e}")
            return self._fallback_vocabulary_detection(text)
    
    def _semantic_detection(self, text: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Perform semantic toxicity detection."""
        
        try:
            # Get semantic analysis
            semantic_analysis = self.semantic_analyzer.analyze_semantic_toxicity(text)
            
            return {
                'method': 'semantic_analysis',
                'toxicity_score': semantic_analysis['semantic_toxicity_score'],
                'confidence': semantic_analysis['confidence'],
                'intent_analysis': semantic_analysis['intent_analysis'],
                'emotional_analysis': semantic_analysis['emotional_analysis'],
                'context_analysis': semantic_analysis['context_analysis'],
                'contextual_insights': semantic_analysis['contextual_insights'],
                'semantic_features': semantic_analysis['semantic_features'].tolist()
            }
            
        except Exception as e:
            logging.warning(f"Semantic detection error: {e}")
            return self._fallback_semantic_detection(text)
    
    def _pattern_detection(self, text: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Perform pattern-based toxicity detection."""
        
        text_lower = text.lower()
        pattern_score = 0.0
        detected_patterns = []
        
        # Length-based patterns
        if len(text) < 10:
            pattern_score += 0.1  # Very short aggressive messages
            detected_patterns.append('short_aggressive')
        
        # Repetition patterns
        words = text.split()
        if len(set(words)) / len(words) < 0.5 and len(words) > 3:
            pattern_score += 0.15  # High repetition
            detected_patterns.append('high_repetition')
        
        # Punctuation patterns
        exclamation_ratio = text.count('!') / len(text) if text else 0
        if exclamation_ratio > 0.05:
            pattern_score += exclamation_ratio * 2
            detected_patterns.append('excessive_exclamation')
        
        # Caps patterns
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.3:
            pattern_score += caps_ratio * 0.5
            detected_patterns.append('excessive_caps')
        
        # Question patterns (potentially rhetorical/aggressive)
        aggressive_questions = sum(1 for phrase in ['what the', 'why the', 'how can', 'are you kidding'] 
                                 if phrase in text_lower)
        if aggressive_questions > 0:
            pattern_score += aggressive_questions * 0.2
            detected_patterns.append('aggressive_questions')
        
        # Special character patterns
        special_ratio = sum(1 for c in text if c in '@#$%^&*') / len(text) if text else 0
        if special_ratio > 0.05:
            pattern_score += special_ratio
            detected_patterns.append('special_characters')
        
        return {
            'method': 'pattern_analysis',
            'toxicity_score': min(pattern_score, 1.0),
            'confidence': 0.7,  # Pattern matching has moderate confidence
            'detected_patterns': detected_patterns,
            'text_statistics': {
                'length': len(text),
                'word_count': len(words),
                'unique_word_ratio': len(set(words)) / len(words) if words else 0,
                'caps_ratio': caps_ratio,
                'exclamation_ratio': exclamation_ratio,
                'special_char_ratio': special_ratio
            }
        }
    
    def _fallback_vocabulary_detection(self, text: str) -> Dict[str, Any]:
        """Fallback vocabulary detection when main module unavailable."""
        
        text_lower = text.lower()
        vocab_score = 0.0
        matched_categories = {}
        
        for category, patterns in self.fallback_toxic_patterns.items():
            matches = [pattern for pattern in patterns if pattern in text_lower]
            if matches:
                category_score = len(matches) * 0.2
                vocab_score += category_score
                matched_categories[category] = {
                    'score': min(category_score, 1.0),
                    'matches': matches
                }
        
        return {
            'method': 'fallback_vocabulary',
            'toxicity_score': min(vocab_score, 1.0),
            'confidence': 0.5,  # Lower confidence for fallback
            'matched_categories': matched_categories,
            'note': 'Using fallback vocabulary detection'
        }
    
    def _fallback_semantic_detection(self, text: str) -> Dict[str, Any]:
        """Fallback semantic detection when main module unavailable."""
        
        semantic_score = 0.0
        indicators = []
        
        # Check caps ratio
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > self.fallback_semantic_indicators['high_caps']:
            semantic_score += caps_ratio * 0.3
            indicators.append('high_caps')
        
        # Check exclamation usage
        exclamation_count = text.count('!')
        if exclamation_count >= self.fallback_semantic_indicators['exclamation_heavy']:
            semantic_score += 0.2
            indicators.append('exclamation_heavy')
        
        # Check aggressive questions
        text_lower = text.lower()
        aggressive_q = [phrase for phrase in self.fallback_semantic_indicators['question_aggressive']
                       if phrase in text_lower]
        if aggressive_q:
            semantic_score += len(aggressive_q) * 0.2
            indicators.extend(aggressive_q)
        
        return {
            'method': 'fallback_semantic',
            'toxicity_score': min(semantic_score, 1.0),
            'confidence': 0.4,  # Lower confidence for fallback
            'indicators': indicators,
            'note': 'Using fallback semantic detection'
        }
    
    def _aggregate_scores(self, components: Dict[str, Any]) -> Tuple[float, float]:
        """Aggregate scores from different detection methods."""
        
        weighted_score = 0.0
        total_weight = 0.0
        confidence_scores = []
        
        for method_name, weight in self.method_weights.items():
            if method_name in components:
                component = components[method_name]
                score = component.get('toxicity_score', 0.0)
                confidence = component.get('confidence', 0.0)
                
                weighted_score += score * weight
                total_weight += weight
                confidence_scores.append(confidence * weight)
        
        # Normalize by actual total weight (in case some methods failed)
        if total_weight > 0:
            overall_score = weighted_score / total_weight
            overall_confidence = sum(confidence_scores) / total_weight
        else:
            overall_score = 0.0
            overall_confidence = 0.0
        
        # Apply consensus bonus (when multiple methods agree)
        method_scores = [comp.get('toxicity_score', 0.0) for comp in components.values()]
        score_variance = np.var(method_scores) if method_scores else 0
        
        # Lower variance means better consensus
        if score_variance < 0.1 and overall_score > 0.3:
            overall_confidence = min(overall_confidence * 1.2, 1.0)
        
        return overall_score, overall_confidence
    
    def _generate_recommendations(self, text: str, components: Dict[str, Any], overall_score: float) -> List[str]:
        """Generate actionable recommendations based on detection results."""
        
        recommendations = []
        
        # Score-based recommendations
        if overall_score >= self.thresholds['critical']:
            recommendations.append("Immediate content review recommended - critical toxicity detected")
            recommendations.append("Consider automatic content blocking")
        elif overall_score >= self.thresholds['high']:
            recommendations.append("Human moderation review recommended")
            recommendations.append("Consider content warning or filtering")
        elif overall_score >= self.thresholds['medium']:
            recommendations.append("Monitor for pattern escalation")
            recommendations.append("Consider user education or warning")
        elif overall_score >= self.thresholds['low']:
            recommendations.append("Log for trend analysis")
        
        # Component-specific recommendations
        vocab_comp = components.get('vocabulary', {})
        if vocab_comp.get('toxicity_score', 0) > 0.6:
            recommendations.append("Strong vocabulary-based toxicity detected")
            if 'matched_categories' in vocab_comp:
                # Handle both dict and float values in matched_categories
                categories = vocab_comp['matched_categories']
                if categories:
                    def get_score(cat):
                        value = categories[cat]
                        if isinstance(value, dict):
                            return value.get('score', 0)
                        elif isinstance(value, (int, float)):
                            return value
                        else:
                            return 0
                    
                    top_categories = sorted(categories.keys(), key=get_score, reverse=True)[:2]
                    recommendations.append(f"Primary categories: {', '.join(top_categories)}")
        
        semantic_comp = components.get('semantic', {})
        if semantic_comp.get('toxicity_score', 0) > 0.6:
            recommendations.append("Semantic context indicates toxicity")
            
            # Intent-based recommendations
            if 'intent_analysis' in semantic_comp:
                intent = semantic_comp['intent_analysis'].get('primary_intent')
                if intent == 'direct_attack':
                    recommendations.append("Direct personal attack detected - high priority")
                elif intent == 'threat':
                    recommendations.append("Potential threat detected - requires immediate review")
        
        pattern_comp = components.get('pattern', {})
        if pattern_comp.get('toxicity_score', 0) > 0.5:
            patterns = pattern_comp.get('detected_patterns', [])
            if 'excessive_caps' in patterns:
                recommendations.append("Consider caps filtering or warning")
            if 'aggressive_questions' in patterns:
                recommendations.append("Aggressive questioning pattern detected")
        
        # Redaction recommendations
        if overall_score >= self.config.get('redaction_threshold', 0.7):
            recommendations.append("Content redaction recommended")
        
        return recommendations
    
    def _generate_redacted_version(self, text: str, components: Dict[str, Any]) -> str:
        """Generate redacted version of the text."""
        
        redacted_text = text
        
        # Get redaction candidates from vocabulary analysis
        vocab_comp = components.get('vocabulary', {})
        redaction_candidates = vocab_comp.get('redaction_candidates', [])
        
        # Apply redactions
        for candidate in redaction_candidates:
            if candidate in redacted_text:
                replacement = '*' * len(candidate)
                redacted_text = redacted_text.replace(candidate, replacement)
        
        # Additional pattern-based redaction
        pattern_comp = components.get('pattern', {})
        if 'excessive_caps' in pattern_comp.get('detected_patterns', []):
            # Reduce excessive caps
            redacted_text = ''.join(c.lower() if c.isupper() and i > 0 and text[i-1].isupper() 
                                   else c for i, c in enumerate(redacted_text))
        
        return redacted_text
    
    def _assess_severity(self, score: float) -> str:
        """Assess severity level based on overall score."""
        
        if score >= self.thresholds['critical']:
            return 'critical'
        elif score >= self.thresholds['high']:
            return 'high'
        elif score >= self.thresholds['medium']:
            return 'medium'
        elif score >= self.thresholds['low']:
            return 'low'
        else:
            return 'minimal'
    
    def _assess_risk_factors(self, components: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Assess risk factors and mitigation factors."""
        
        risk_factors = []
        mitigation_factors = []
        
        # Vocabulary risk factors
        vocab_comp = components.get('vocabulary', {})
        if vocab_comp.get('toxicity_score', 0) > 0.5:
            categories = vocab_comp.get('matched_categories', {})
            high_risk_categories = ['threats', 'hate_speech', 'discriminatory']
            for category in high_risk_categories:
                if category in categories:
                    risk_factors.append(f"High-risk vocabulary: {category}")
        
        # Semantic risk factors
        semantic_comp = components.get('semantic', {})
        if semantic_comp.get('method') == 'semantic_analysis':
            intent_analysis = semantic_comp.get('intent_analysis', {})
            if intent_analysis.get('primary_intent') in ['direct_attack', 'threat']:
                risk_factors.append(f"Toxic intent: {intent_analysis['primary_intent']}")
            
            emotional_analysis = semantic_comp.get('emotional_analysis', {})
            if emotional_analysis.get('dominant_emotion') in ['anger', 'contempt', 'disgust']:
                risk_factors.append(f"Negative emotion: {emotional_analysis['dominant_emotion']}")
            
            # Check for mitigation factors
            context_analysis = semantic_comp.get('context_analysis', {})
            if context_analysis.get('primary_context') in ['debate', 'sarcasm']:
                mitigation_factors.append(f"Context: {context_analysis['primary_context']}")
        
        # Pattern risk factors
        pattern_comp = components.get('pattern', {})
        risky_patterns = ['excessive_caps', 'aggressive_questions', 'special_characters']
        for pattern in pattern_comp.get('detected_patterns', []):
            if pattern in risky_patterns:
                risk_factors.append(f"Pattern: {pattern}")
        
        # General mitigation factors
        if len([comp for comp in components.values() if comp.get('confidence', 0) > 0.8]) < 2:
            mitigation_factors.append("Low multi-method consensus")
        
        return risk_factors, mitigation_factors
    
    def _update_stats(self, score: float, confidence: float):
        """Update detection statistics."""
        
        self.detection_stats['total_detections'] += 1
        
        if confidence > 0.8:
            self.detection_stats['high_confidence_detections'] += 1
        
        if score >= self.config.get('redaction_threshold', 0.7):
            self.detection_stats['redaction_recommendations'] += 1
    
    def _empty_result(self, text: str) -> ToxicityDetectionResult:
        """Return empty result for invalid input."""
        
        return ToxicityDetectionResult(
            text=text,
            overall_toxicity_score=0.0,
            confidence=0.0,
            detection_components={},
            recommendations=["No analysis performed - invalid input"],
            redacted_version=text,
            severity_level='minimal',
            risk_factors=[],
            mitigation_factors=[],
            timestamp=datetime.now().isoformat()
        )
    
    def batch_detect(self, texts: List[str], context: Optional[Dict[str, Any]] = None) -> List[ToxicityDetectionResult]:
        """Perform batch toxicity detection on multiple texts."""
        
        results = []
        for text in texts:
            result = self.detect_toxicity(text, context)
            results.append(result)
        
        return results
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get current detection statistics."""
        
        stats = self.detection_stats.copy()
        
        if stats['total_detections'] > 0:
            stats['high_confidence_rate'] = stats['high_confidence_detections'] / stats['total_detections']
            stats['redaction_rate'] = stats['redaction_recommendations'] / stats['total_detections']
        else:
            stats['high_confidence_rate'] = 0.0
            stats['redaction_rate'] = 0.0
        
        return stats
    
    def explain_detection(self, text: str) -> str:
        """Generate detailed explanation of toxicity detection."""
        
        result = self.detect_toxicity(text)
        
        explanation_parts = []
        
        # Overall assessment
        explanation_parts.append(f"Overall toxicity score: {result.overall_toxicity_score:.3f} ({result.severity_level} severity)")
        explanation_parts.append(f"Detection confidence: {result.confidence:.3f}")
        
        # Component breakdown
        explanation_parts.append("\nDetection method breakdown:")
        for method, component in result.detection_components.items():
            score = component.get('toxicity_score', 0)
            conf = component.get('confidence', 0)
            explanation_parts.append(f"  {method.capitalize()}: {score:.3f} (confidence: {conf:.3f})")
        
        # Risk factors
        if result.risk_factors:
            explanation_parts.append(f"\nRisk factors: {', '.join(result.risk_factors)}")
        
        if result.mitigation_factors:
            explanation_parts.append(f"Mitigation factors: {', '.join(result.mitigation_factors)}")
        
        # Recommendations
        if result.recommendations:
            explanation_parts.append(f"\nRecommendations:")
            for rec in result.recommendations[:3]:  # Top 3 recommendations
                explanation_parts.append(f"  - {rec}")
        
        return '\n'.join(explanation_parts)
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compare toxicity between two texts."""
        
        result1 = self.detect_toxicity(text1)
        result2 = self.detect_toxicity(text2)
        
        comparison = {
            'text1_score': result1.overall_toxicity_score,
            'text2_score': result2.overall_toxicity_score,
            'score_difference': abs(result1.overall_toxicity_score - result2.overall_toxicity_score),
            'text1_severity': result1.severity_level,
            'text2_severity': result2.severity_level,
            'more_toxic': 'text1' if result1.overall_toxicity_score > result2.overall_toxicity_score else 'text2',
            'confidence_difference': abs(result1.confidence - result2.confidence)
        }
        
        # Similarity in risk factors
        common_risks = set(result1.risk_factors) & set(result2.risk_factors)
        comparison['common_risk_factors'] = list(common_risks)
        comparison['risk_similarity'] = len(common_risks) / max(len(result1.risk_factors) + len(result2.risk_factors), 1)
        
        return comparison


if __name__ == "__main__":
    # Test the enhanced detector
    print("🔍 Testing Enhanced Toxicity Detector")
    print("=" * 60)
    
    detector = EnhancedToxicityDetector()
    
    # Test various types of content
    test_messages = [
        "You're a complete idiot and should just shut up!",  # High toxicity
        "I disagree with your political opinion",  # Low toxicity
        "WHAT THE HELL IS WRONG WITH YOU PEOPLE???",  # High emotion + patterns
        "All politicians are corrupt lying scumbags",  # Generalization + profanity
        "Great job on that project, well done!",  # Positive content
        "This is frustrating but I understand your point",  # Mild frustration
        "You people always do this stupid stuff",  # Implicit bias
        "I'm getting really annoyed with this situation",  # Personal expression
        "Sure, that's totally reasonable... NOT!",  # Sarcasm
        "Everyone knows you're just making things up",  # Dismissive + generalization
    ]
    
    print("\n🧪 Testing Comprehensive Detection:")
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. \"{message}\"")
        
        result = detector.detect_toxicity(message)
        
        print(f"   Overall Score: {result.overall_toxicity_score:.3f} | Confidence: {result.confidence:.3f}")
        print(f"   Severity: {result.severity_level.upper()}")
        
        # Show component scores
        for method, component in result.detection_components.items():
            score = component.get('toxicity_score', 0)
            print(f"     {method.capitalize()}: {score:.3f}")
        
        if result.risk_factors:
            print(f"   Risk Factors: {', '.join(result.risk_factors)}")
        
        if result.mitigation_factors:
            print(f"   Mitigating: {', '.join(result.mitigation_factors)}")
        
        print(f"   Recommendations: {len(result.recommendations)} generated")
        
        # Show redacted version if different
        if result.redacted_version != result.text:
            print(f"   Redacted: \"{result.redacted_version}\"")
    
    print(f"\n📊 Detection Statistics:")
    stats = detector.get_detection_stats()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🔍 Detailed Analysis Example:")
    sample_text = "You're absolutely terrible at this, what a joke!"
    explanation = detector.explain_detection(sample_text)
    print(f"Text: \"{sample_text}\"")
    print(explanation)
    
    print(f"\n⚖️  Comparison Example:")
    text1 = "You're an idiot!"
    text2 = "I think you're mistaken."
    comparison = detector.compare_texts(text1, text2)
    print(f"'{text1}' vs '{text2}':")
    print(f"   Scores: {comparison['text1_score']:.3f} vs {comparison['text2_score']:.3f}")
    print(f"   More toxic: {comparison['more_toxic']}")
    print(f"   Difference: {comparison['score_difference']:.3f}")
    
    print(f"\n✅ Enhanced Toxicity Detection System Ready!")
    print(f"   Features: Multi-method detection, risk assessment, redaction, explanations")
    print(f"   Components: Vocabulary analysis, semantic analysis, pattern detection")
    print(f"   Capabilities: Batch processing, comparison, statistics tracking")
