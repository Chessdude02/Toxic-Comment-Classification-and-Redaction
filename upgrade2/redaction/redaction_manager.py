"""
Redaction Manager for Toxic Comment Classification

This module provides intelligent content redaction using the enhanced toxicity detection system.
It generates context-aware redactions with configurable masking styles and provides alternative
suggestions for toxic content.
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import sys
import os
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from integration.enhanced_toxicity_detector import EnhancedToxicityDetector
except ImportError as e:
    print(f"Warning: Could not import enhanced toxicity detector: {e}")


class RedactionStyle(Enum):
    """Different redaction visual styles."""
    ASTERISKS = "asterisks"         # f***
    HASHES = "hashes"               # f###
    DASHES = "dashes"               # f---
    UNDERSCORES = "underscores"     # f___
    BRACKETS = "brackets"           # [REDACTED]
    PARTIAL = "partial"             # f**k, st*pid
    COMPLETE = "complete"           # [REMOVED]


@dataclass
class RedactionResult:
    """Result of content redaction."""
    original_text: str
    redacted_text: str
    toxicity_score: float
    confidence: float
    redacted_words: List[str]
    redaction_positions: List[Tuple[int, int]]
    redaction_reason: str
    alternative_suggestions: List[str]
    style_used: RedactionStyle


class RedactionManager:
    """
    Intelligent redaction system that uses enhanced toxicity detection
    to redact toxic content while preserving meaning and readability.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the redaction manager."""
        
        self.config = config or self._default_config()
        
        # Initialize toxicity detector
        try:
            self.detector = EnhancedToxicityDetector()
            self.enhanced_available = True
        except Exception as e:
            print(f"Warning: Enhanced detection unavailable: {e}")
            self.enhanced_available = False
        
        # Always initialize fallback redaction (as backup)
        self._init_fallback_redaction()
        
        # Redaction thresholds
        self.redaction_threshold = self.config.get('redaction_threshold', 0.5)
        
        # Initialize replacement mappings
        self._init_replacement_mappings()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'default_style': RedactionStyle.ASTERISKS,
            'redaction_threshold': 0.3,  # More sensitive default
            'preserve_length': True,
            'provide_alternatives': True,
            'min_word_length_for_partial': 4,
            'context_aware': True
        }
    
    def _init_fallback_redaction(self):
        """Initialize fallback redaction when enhanced detection isn't available."""
        
        self.fallback_patterns = {
            'profanity': ['fuck', 'shit', 'damn', 'hell', 'ass', 'bitch', 'crap'],
            'insults': ['idiot', 'stupid', 'moron', 'dumb', 'loser', 'pathetic'],
            'threats': ['kill', 'die', 'murder', 'destroy'],
            'slurs': ['hate', 'disgusting', 'worthless']
        }
    
    def _init_replacement_mappings(self):
        """Initialize replacement word mappings."""
        
        # Alternative suggestions for toxic words
        self.suggestion_map = {
            'idiot': ['unwise person', 'someone who made a mistake'],
            'stupid': ['unthinking', 'misguided', 'ill-considered'],
            'moron': ['person who erred', 'someone who misjudged'],
            'pathetic': ['disappointing', 'unfortunate'],
            'worthless': ['of little value', 'not helpful'],
            'disgusting': ['unpleasant', 'disagreeable'],
            'hate': ['dislike', 'disagree with', 'find problematic'],
            'damn': ['darn', 'blast'],
            'hell': ['heck', 'trouble'],
            'crap': ['nonsense', 'junk'],
            'shit': ['nonsense', 'rubbish'],
            'fuck': ['frick', 'darn'],
            'bitch': ['person', 'individual'],
            'ass': ['person', 'individual']
        }
    
    def generate_redaction(self, text: str, style: Optional[RedactionStyle] = None) -> RedactionResult:
        """
        Generate redacted version of text based on toxicity detection.
        
        Args:
            text: Input text to redact
            style: Redaction style to use (default from config)
            
        Returns:
            RedactionResult with original and redacted text
        """
        
        if not text or not isinstance(text, str):
            return RedactionResult(
                original_text=text or "",
                redacted_text=text or "",
                toxicity_score=0.0,
                confidence=0.0,
                redacted_words=[],
                redaction_positions=[],
                redaction_reason="Invalid input",
                alternative_suggestions=[],
                style_used=style or self.config['default_style']
            )
        
        # Use enhanced detection if available
        if self.enhanced_available:
            result = self._enhanced_redaction(text, style)
        else:
            result = self._fallback_redaction(text, style)
        
        return result
    
    def _enhanced_redaction(self, text: str, style: Optional[RedactionStyle]) -> RedactionResult:
        """Perform redaction using enhanced toxicity detection."""
        
        # Get toxicity analysis
        detection_result = self.detector.detect_toxicity(text)
        
        # Check if redaction is needed
        if detection_result.overall_toxicity_score < self.redaction_threshold:
            return RedactionResult(
                original_text=text,
                redacted_text=text,
                toxicity_score=detection_result.overall_toxicity_score,
                confidence=detection_result.confidence,
                redacted_words=[],
                redaction_positions=[],
                redaction_reason="Below redaction threshold",
                alternative_suggestions=[],
                style_used=style or self.config['default_style']
            )
        
        # Extract redaction candidates from vocabulary analysis
        vocab_component = detection_result.detection_components.get('vocabulary', {})
        redaction_candidates = vocab_component.get('redaction_candidates', [])
        
        # If no candidates from enhanced detection, use fallback patterns
        if not redaction_candidates:
            # Use fallback patterns to find toxic words
            for category, patterns in self.fallback_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in text.lower():
                        redaction_candidates.append(pattern)
        
        # Apply redaction
        redacted_text = text
        redacted_words = []
        redaction_positions = []
        
        redaction_style = style or self.config['default_style']
        
        for candidate in redaction_candidates:
            # Find all occurrences of the word (case insensitive)
            pattern = re.compile(r'\b' + re.escape(candidate) + r'\b', re.IGNORECASE)
            matches = list(pattern.finditer(text))
            
            for match in reversed(matches):  # Reverse to maintain positions
                start, end = match.span()
                original_word = text[start:end]
                
                # Generate replacement based on style
                replacement = self._generate_replacement(original_word, redaction_style)
                
                # Apply replacement
                redacted_text = redacted_text[:start] + replacement + redacted_text[end:]
                
                if original_word not in redacted_words:  # Avoid duplicates
                    redacted_words.append(original_word)
                    redaction_positions.append((start, end))
        
        # Generate alternative suggestions
        alternatives = self._generate_alternatives(redacted_words)
        
        # Generate redaction reason
        reason = self._generate_redaction_reason(detection_result) if redacted_words else "Above threshold but no toxic words found"
        
        return RedactionResult(
            original_text=text,
            redacted_text=redacted_text,
            toxicity_score=detection_result.overall_toxicity_score,
            confidence=detection_result.confidence,
            redacted_words=redacted_words,
            redaction_positions=redaction_positions,
            redaction_reason=reason,
            alternative_suggestions=alternatives,
            style_used=redaction_style
        )
    
    def _fallback_redaction(self, text: str, style: Optional[RedactionStyle]) -> RedactionResult:
        """Fallback redaction when enhanced detection isn't available."""
        
        redacted_text = text.lower()
        redacted_words = []
        redaction_positions = []
        toxicity_score = 0.0
        
        redaction_style = style or self.config['default_style']
        
        # Check for fallback patterns
        for category, patterns in self.fallback_patterns.items():
            for pattern in patterns:
                if pattern in redacted_text:
                    # Find all occurrences
                    start = 0
                    while True:
                        pos = redacted_text.find(pattern, start)
                        if pos == -1:
                            break
                        
                        # Generate replacement
                        replacement = self._generate_replacement(pattern, redaction_style)
                        
                        # Apply replacement (case-preserving)
                        original_case = text[pos:pos+len(pattern)]
                        redacted_text = redacted_text[:pos] + replacement.lower() + redacted_text[pos+len(pattern):]
                        
                        redacted_words.append(original_case)
                        redaction_positions.append((pos, pos + len(pattern)))
                        toxicity_score += 0.2  # Simple scoring
                        
                        start = pos + len(replacement)
        
        # Restore original casing for non-redacted parts
        final_redacted = ""
        last_pos = 0
        
        for start, end in redaction_positions:
            final_redacted += text[last_pos:start]
            replacement = self._generate_replacement(text[start:end], redaction_style)
            final_redacted += replacement
            last_pos = end
        
        final_redacted += text[last_pos:]
        
        # Generate alternatives
        alternatives = self._generate_alternatives(redacted_words)
        
        return RedactionResult(
            original_text=text,
            redacted_text=final_redacted if redacted_words else text,
            toxicity_score=min(toxicity_score, 1.0),
            confidence=0.6,  # Lower confidence for fallback
            redacted_words=redacted_words,
            redaction_positions=redaction_positions,
            redaction_reason="Fallback pattern matching" if redacted_words else "No toxic content detected",
            alternative_suggestions=alternatives,
            style_used=redaction_style
        )
    
    def _generate_replacement(self, word: str, style: RedactionStyle) -> str:
        """Generate replacement text based on style."""
        
        if style == RedactionStyle.ASTERISKS:
            if self.config.get('preserve_length', True):
                if len(word) >= self.config.get('min_word_length_for_partial', 4):
                    return word[0] + '*' * (len(word) - 1)
                else:
                    return '*' * len(word)
            else:
                return '***'
        
        elif style == RedactionStyle.HASHES:
            if self.config.get('preserve_length', True):
                if len(word) >= self.config.get('min_word_length_for_partial', 4):
                    return word[0] + '#' * (len(word) - 1)
                else:
                    return '#' * len(word)
            else:
                return '###'
        
        elif style == RedactionStyle.DASHES:
            if self.config.get('preserve_length', True):
                return '-' * len(word)
            else:
                return '---'
        
        elif style == RedactionStyle.UNDERSCORES:
            if self.config.get('preserve_length', True):
                return '_' * len(word)
            else:
                return '___'
        
        elif style == RedactionStyle.BRACKETS:
            return '[REDACTED]'
        
        elif style == RedactionStyle.PARTIAL:
            if len(word) >= 4:
                mid_length = len(word) - 2
                return word[0] + '*' * mid_length + word[-1]
            else:
                return '*' * len(word)
        
        elif style == RedactionStyle.COMPLETE:
            return '[REMOVED]'
        
        else:
            # Default to asterisks
            return '*' * len(word)
    
    def _generate_alternatives(self, redacted_words: List[str]) -> List[str]:
        """Generate alternative suggestions for redacted words."""
        
        if not self.config.get('provide_alternatives', True):
            return []
        
        alternatives = []
        
        for word in redacted_words:
            word_lower = word.lower()
            if word_lower in self.suggestion_map:
                alternatives.extend(self.suggestion_map[word_lower])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_alternatives = []
        for alt in alternatives:
            if alt not in seen:
                seen.add(alt)
                unique_alternatives.append(alt)
        
        return unique_alternatives[:5]  # Return top 5 alternatives
    
    def _generate_redaction_reason(self, detection_result) -> str:
        """Generate human-readable reason for redaction."""
        
        reasons = []
        
        # Check toxicity level
        score = detection_result.overall_toxicity_score
        if score >= 0.8:
            reasons.append("High toxicity detected")
        elif score >= 0.6:
            reasons.append("Moderate toxicity detected")
        else:
            reasons.append("Low-level toxicity detected")
        
        # Check specific categories
        vocab_comp = detection_result.detection_components.get('vocabulary', {})
        categories = vocab_comp.get('matched_categories', {})
        
        if categories:
            top_category = max(categories.keys(), key=lambda x: categories[x] if isinstance(categories[x], (int, float)) else categories[x].get('score', 0))
            reasons.append(f"Primary category: {top_category}")
        
        # Check for specific risks
        if detection_result.risk_factors:
            high_risk = any('threat' in factor.lower() or 'hate' in factor.lower() 
                           for factor in detection_result.risk_factors)
            if high_risk:
                reasons.append("Contains high-risk content")
        
        return "; ".join(reasons)
    
    def batch_redact(self, texts: List[str], style: Optional[RedactionStyle] = None) -> List[RedactionResult]:
        """Perform batch redaction on multiple texts."""
        
        results = []
        for text in texts:
            result = self.generate_redaction(text, style)
            results.append(result)
        
        return results
    
    def get_redaction_stats(self, texts: List[str]) -> Dict[str, Any]:
        """Get redaction statistics for a set of texts."""
        
        results = self.batch_redact(texts)
        
        total_texts = len(results)
        redacted_texts = sum(1 for r in results if r.redacted_words)
        total_words_redacted = sum(len(r.redacted_words) for r in results)
        avg_toxicity = sum(r.toxicity_score for r in results) / total_texts if total_texts > 0 else 0
        avg_confidence = sum(r.confidence for r in results) / total_texts if total_texts > 0 else 0
        
        return {
            'total_texts': total_texts,
            'redacted_texts': redacted_texts,
            'redaction_rate': redacted_texts / total_texts if total_texts > 0 else 0,
            'total_words_redacted': total_words_redacted,
            'avg_words_per_redacted_text': total_words_redacted / redacted_texts if redacted_texts > 0 else 0,
            'avg_toxicity_score': avg_toxicity,
            'avg_confidence': avg_confidence,
            'texts_above_threshold': sum(1 for r in results if r.toxicity_score >= self.redaction_threshold)
        }
    
    def compare_redaction_styles(self, text: str) -> Dict[RedactionStyle, str]:
        """Compare different redaction styles on the same text."""
        
        comparisons = {}
        
        for style in RedactionStyle:
            result = self.generate_redaction(text, style)
            comparisons[style] = result.redacted_text
        
        return comparisons
    
    def set_redaction_threshold(self, threshold: float):
        """Update the redaction threshold."""
        
        if 0.0 <= threshold <= 1.0:
            self.redaction_threshold = threshold
            self.config['redaction_threshold'] = threshold
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")
    
    def enable_context_awareness(self, enabled: bool):
        """Enable or disable context-aware redaction."""
        
        self.config['context_aware'] = enabled
    
    def process_batch(self, texts: List[str], style: Optional[RedactionStyle] = None) -> List[RedactionResult]:
        """Process a batch of texts for redaction (alias for batch_redact)."""
        return self.batch_redact(texts, style)


def demo_redaction_system():
    """Demonstrate the redaction system capabilities."""
    
    print("🔒 Redaction Manager Demo")
    print("=" * 60)
    
    # Initialize redaction manager
    redaction_manager = RedactionManager()
    
    # Test messages with varying toxicity levels
    test_messages = [
        "You're such an idiot for thinking that!",
        "This is a great idea, well done!",
        "What the hell were you thinking?",
        "I hate stupid people like you",
        "That's a damn good point",
        "You're absolutely pathetic and worthless",
        "I disagree with your opinion respectfully",
        "Shut up you moron, nobody cares",
        "This project is shit, complete garbage",
        "I'm frustrated but I understand your position"
    ]
    
    print("\n🧪 Testing Different Messages:")
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. Original: \"{message}\"")
        
        result = redaction_manager.generate_redaction(message)
        
        print(f"   Toxicity: {result.toxicity_score:.3f} | Confidence: {result.confidence:.3f}")
        
        if result.redacted_words:
            print(f"   Redacted: \"{result.redacted_text}\"")
            print(f"   Words Redacted: {', '.join(result.redacted_words)}")
            print(f"   Reason: {result.redaction_reason}")
            
            if result.alternative_suggestions:
                print(f"   Alternatives: {', '.join(result.alternative_suggestions)}")
        else:
            print("   No redaction needed")
    
    # Test different redaction styles
    print(f"\n🎨 Testing Redaction Styles:")
    toxic_message = "You're an idiot and this is shit!"
    
    style_comparisons = redaction_manager.compare_redaction_styles(toxic_message)
    
    print(f"Original: \"{toxic_message}\"")
    for style, redacted in style_comparisons.items():
        print(f"  {style.value.title():12}: \"{redacted}\"")
    
    # Batch processing demo
    print(f"\n📊 Batch Processing Stats:")
    batch_messages = [
        "Great work everyone!",
        "This is stupid nonsense",
        "I hate this garbage project",
        "You're all idiots",
        "Excellent presentation, thank you",
        "What the hell is going on?",
        "This is really well done",
        "Damn, that's impressive",
        "You people are pathetic",
        "I appreciate your hard work"
    ]
    
    stats = redaction_manager.get_redaction_stats(batch_messages)
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Threshold testing
    print(f"\n⚖️  Threshold Impact:")
    test_message = "That's pretty stupid"
    
    for threshold in [0.3, 0.5, 0.7]:
        redaction_manager.set_redaction_threshold(threshold)
        result = redaction_manager.generate_redaction(test_message)
        
        redaction_status = "REDACTED" if result.redacted_words else "NOT REDACTED"
        print(f"  Threshold {threshold}: {redaction_status} (score: {result.toxicity_score:.3f})")
    
    print(f"\n✅ Redaction Manager Demo Complete!")
    print(f"   Features demonstrated: Multiple styles, batch processing, threshold control")
    print(f"   Integration: Enhanced toxicity detection with fallback support")


if __name__ == "__main__":
    demo_redaction_system()
