"""
Smart Redaction System for Toxic Comments

This module provides intelligent content redaction using the enhanced toxicity detection system.
Features include context-aware redaction, partial masking, alternative suggestions, and
configurable redaction policies.
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import sys
import os
import re
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from integration.enhanced_toxicity_detector import EnhancedToxicityDetector
    from semantic.semantic_toxicity_analyzer import SemanticToxicityAnalyzer
except ImportError as e:
    print(f"Warning: Could not import all components: {e}")


class RedactionLevel(Enum):
    """Redaction intensity levels."""
    NONE = "none"
    MINIMAL = "minimal"      # Light censoring, preserve readability
    MODERATE = "moderate"    # Standard redaction
    AGGRESSIVE = "aggressive" # Heavy redaction
    COMPLETE = "complete"    # Full removal/replacement


class RedactionStyle(Enum):
    """Different redaction visual styles."""
    ASTERISKS = "asterisks"         # f***
    BLOCKS = "blocks"               # ████
    DASHES = "dashes"               # ----
    UNDERSCORES = "underscores"     # ____
    BRACKETS = "brackets"           # [REDACTED]
    EUPHEMISMS = "euphemisms"       # freaking, darn
    PARTIAL = "partial"             # f**k, st*pid


@dataclass
class RedactionRule:
    """Rule for content redaction."""
    pattern: str
    replacement: str
    condition: Optional[str] = None
    severity_threshold: float = 0.5
    preserve_length: bool = True
    case_sensitive: bool = False


@dataclass
class RedactionResult:
    """Result of content redaction."""
    original_text: str
    redacted_text: str
    redaction_level: RedactionLevel
    redacted_words: List[str]
    redaction_positions: List[Tuple[int, int]]
    toxicity_score: float
    confidence: float
    redaction_reason: str
    suggestions: List[str]
    timestamp: str


class SmartRedactionSystem:
    """
    Advanced redaction system that uses enhanced toxicity detection
    to intelligently redact toxic content while preserving meaning and readability.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the smart redaction system."""
        
        self.config = config or self._default_config()
        
        # Initialize toxicity detector
        try:
            self.detector = EnhancedToxicityDetector()
            self.semantic_analyzer = SemanticToxicityAnalyzer()
            self.enhanced_available = True
        except Exception as e:
            print(f"Warning: Enhanced detection unavailable: {e}")
            self.enhanced_available = False
            self._init_fallback_redaction()
        
        # Redaction thresholds
        self.redaction_thresholds = {
            RedactionLevel.MINIMAL: 0.3,
            RedactionLevel.MODERATE: 0.5,
            RedactionLevel.AGGRESSIVE: 0.7,
            RedactionLevel.COMPLETE: 0.9
        }
        
        # Initialize redaction rules
        self._init_redaction_rules()
        self._init_replacement_mappings()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'default_redaction_level': RedactionLevel.MODERATE,
            'default_redaction_style': RedactionStyle.ASTERISKS,
            'preserve_context': True,
            'provide_alternatives': True,
            'min_word_length_for_partial': 4,
            'context_window': 5,  # Words around redacted content
            'enable_smart_suggestions': True,
            'custom_patterns': []
        }
    
    def _init_fallback_redaction(self):
        """Initialize fallback redaction when enhanced detection isn't available."""
        
        self.fallback_patterns = {
            'profanity': ['fuck', 'shit', 'damn', 'hell', 'ass', 'bitch'],
            'insults': ['idiot', 'stupid', 'moron', 'dumb', 'loser'],
            'threats': ['kill', 'die', 'murder', 'destroy'],
            'slurs': ['hate', 'disgusting', 'pathetic', 'worthless']
        }
    
    def _init_redaction_rules(self):
        """Initialize comprehensive redaction rules."""
        
        self.redaction_rules = []
        
        # Profanity rules
        profanity_patterns = [
            r'\bf\*+ck\b', r'\bsh\*+t\b', r'\bd\*+mn\b', r'\bh\*+ll\b',
            r'\bf+u+c+k+\b', r'\bs+h+i+t+\b', r'\bd+a+m+n+\b'
        ]
        
        for pattern in profanity_patterns:
            self.redaction_rules.append(RedactionRule(
                pattern=pattern,
                replacement="[PROFANITY]",
                severity_threshold=0.6
            ))
        
        # Threat rules
        threat_patterns = [
            r'\bkill\s+you\b', r'\bdie\b', r'\bmurder\b', r'\bdestroy\s+you\b',
            r'\bkill\s+yourself\b', r'\bgo\s+die\b'
        ]
        
        for pattern in threat_patterns:
            self.redaction_rules.append(RedactionRule(
                pattern=pattern,
                replacement="[THREAT]",
                severity_threshold=0.8
            ))
        
        # Insult rules
        insult_patterns = [
            r'\bidiot\b', r'\bmoron\b', r'\bstupid\b', r'\bdumb\b',
            r'\bpathetic\b', r'\bworthless\b', r'\bloser\b'
        ]
        
        for pattern in insult_patterns:
            self.redaction_rules.append(RedactionRule(
                pattern=pattern,
                replacement="[INSULT]",
                severity_threshold=0.4
            ))
    
    def _init_replacement_mappings(self):
        """Initialize replacement word mappings."""
        
        # Euphemism replacements
        self.euphemism_map = {
            'fuck': 'frick', 'fucking': 'freaking', 'shit': 'crud',
            'damn': 'darn', 'hell': 'heck', 'ass': 'butt',
            'bitch': 'jerk', 'bastard': 'jerk', 'crap': 'crud'
        }
        
        # Alternative suggestions
        self.suggestion_map = {
            'idiot': ['unwise person', 'someone who made a mistake'],
            'stupid': ['unthinking', 'misguided', 'ill-considered'],
            'moron': ['person who erred', 'someone who misjudged'],
            'pathetic': ['disappointing', 'unfortunate'],
            'worthless': ['of little value', 'not helpful'],
            'disgusting': ['unpleasant', 'disagreeable'],
            'hate': ['dislike', 'disagree with', 'find problematic']
        }
    
    def redact_content(self, text: str, 
                      redaction_level: Optional[RedactionLevel] = None,
                      redaction_style: Optional[RedactionStyle] = None) -> RedactionResult:
        """
        Main redaction function that intelligently redacts toxic content.
        
        Args:
            text: Input text to redact
            redaction_level: Intensity of redaction
            redaction_style: Visual style of redaction
            
        Returns:
            RedactionResult with original, redacted text and metadata
        """
        
        if not text or not text.strip():
            return self._empty_result(text)
        
        # Use defaults if not specified
        redaction_level = redaction_level or self.config['default_redaction_level']
        redaction_style = redaction_style or self.config['default_redaction_style']
        
        # Analyze toxicity
        if self.enhanced_available:
            toxicity_analysis = self.detector.detect_toxicity(text)
            toxicity_score = toxicity_analysis.overall_toxicity_score
            confidence = toxicity_analysis.confidence
            redaction_reason = f"Toxicity score: {toxicity_score:.3f}"
            
            # Get semantic context
            semantic_analysis = self.semantic_analyzer.analyze_semantic_toxicity(text)
            
        else:
            # Fallback analysis
            toxicity_score, confidence, redaction_reason = self._fallback_analysis(text)
            semantic_analysis = None
        
        # Determine if redaction is needed
        threshold = self.redaction_thresholds.get(redaction_level, 0.5)
        
        if toxicity_score < threshold:
            return RedactionResult(
                original_text=text,
                redacted_text=text,
                redaction_level=RedactionLevel.NONE,
                redacted_words=[],
                redaction_positions=[],
                toxicity_score=toxicity_score,
                confidence=confidence,
                redaction_reason="Below redaction threshold",
                suggestions=[],
                timestamp=datetime.now().isoformat()
            )
        
        # Perform redaction
        redacted_text, redacted_words, positions = self._apply_redaction(
            text, redaction_level, redaction_style, toxicity_analysis if self.enhanced_available else None
        )
        
        # Generate suggestions
        suggestions = self._generate_suggestions(redacted_words) if self.config.get('provide_alternatives', True) else []
        
        return RedactionResult(
            original_text=text,
            redacted_text=redacted_text,
            redaction_level=redaction_level,
            redacted_words=redacted_words,
            redaction_positions=positions,
            toxicity_score=toxicity_score,
            confidence=confidence,
            redaction_reason=redaction_reason,
            suggestions=suggestions,
            timestamp=datetime.now().isoformat()
        )
    
    def _apply_redaction(self, text: str, level: RedactionLevel, style: RedactionStyle,
                        analysis: Optional[Any] = None) -> Tuple[str, List[str], List[Tuple[int, int]]]:
        """Apply redaction to text based on level and style."""
        
        redacted_text = text
        redacted_words = []
        positions = []
        
        # Get words to redact based on analysis
        words_to_redact = self._identify_redaction_targets(text, level, analysis)
        
        # Apply redaction for each word
        for word_info in words_to_redact:
            word = word_info['word']
            severity = word_info.get('severity', 0.5)
            start_pos = word_info.get('position', 0)
            
            # Determine redaction method based on style and severity
            replacement = self._create_replacement(word, style, severity, level)
            
            # Apply replacement
            pattern = r'\b' + re.escape(word) + r'\b'
            match = re.search(pattern, redacted_text, re.IGNORECASE)
            
            if match:
                start, end = match.span()
                redacted_text = redacted_text[:start] + replacement + redacted_text[end:]
                redacted_words.append(word)
                positions.append((start, start + len(replacement)))
        
        return redacted_text, redacted_words, positions
    
    def _identify_redaction_targets(self, text: str, level: RedactionLevel, 
                                  analysis: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Identify words that should be redacted."""
        
        targets = []
        
        if self.enhanced_available and analysis:
            # Use enhanced analysis to identify targets
            
            # Get redaction candidates from vocabulary analysis
            vocab_components = analysis.detection_components.get('vocabulary', {})
            if 'redaction_candidates' in vocab_components:
                for candidate in vocab_components['redaction_candidates']:
                    targets.append({
                        'word': candidate,
                        'severity': 0.7,
                        'reason': 'vocabulary_match'
                    })
            
            # Use semantic analysis to find additional targets
            semantic_components = analysis.detection_components.get('semantic', {})
            if semantic_components.get('intent_analysis', {}).get('primary_intent') == 'direct_attack':
                # Look for attack-related words
                attack_words = self._find_attack_words(text)
                for word in attack_words:
                    targets.append({
                        'word': word,
                        'severity': 0.6,
                        'reason': 'direct_attack_intent'
                    })
        
        else:
            # Fallback pattern matching
            for category, patterns in self.fallback_patterns.items():
                for pattern in patterns:
                    if re.search(r'\b' + re.escape(pattern) + r'\b', text, re.IGNORECASE):
                        targets.append({
                            'word': pattern,
                            'severity': 0.5,
                            'reason': f'fallback_{category}'
                        })
        
        # Apply custom redaction rules
        for rule in self.redaction_rules:
            matches = re.finditer(rule.pattern, text, re.IGNORECASE if not rule.case_sensitive else 0)
            for match in matches:
                word = match.group()
                targets.append({
                    'word': word,
                    'severity': rule.severity_threshold,
                    'reason': 'custom_rule',
                    'position': match.start()
                })
        
        return targets
    
    def _find_attack_words(self, text: str) -> List[str]:
        """Find words commonly used in personal attacks."""
        
        attack_indicators = [
            'idiot', 'stupid', 'moron', 'dumb', 'pathetic', 'worthless',
            'loser', 'freak', 'weirdo', 'creep', 'scum'
        ]
        
        found_words = []
        for word in attack_indicators:
            if re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE):
                found_words.append(word)
        
        return found_words
    
    def _create_replacement(self, word: str, style: RedactionStyle, severity: float, level: RedactionLevel) -> str:
        """Create replacement text based on redaction style."""
        
        if level == RedactionLevel.COMPLETE:
            return "[REMOVED]"
        
        if style == RedactionStyle.ASTERISKS:
            if level == RedactionLevel.MINIMAL and len(word) >= 4:
                # Partial redaction: preserve first and last letter
                return word[0] + '*' * (len(word) - 2) + word[-1]
            else:
                return '*' * len(word)
        
        elif style == RedactionStyle.BLOCKS:
            return '█' * len(word)
        
        elif style == RedactionStyle.DASHES:
            return '-' * len(word)
        
        elif style == RedactionStyle.UNDERSCORES:
            return '_' * len(word)
        
        elif style == RedactionStyle.BRACKETS:
            if severity > 0.8:
                return "[THREAT]"
            elif severity > 0.6:
                return "[PROFANITY]"
            elif severity > 0.4:
                return "[INSULT]"
            else:
                return "[REDACTED]"
        
        elif style == RedactionStyle.EUPHEMISMS:
            # Use euphemism replacement if available
            return self.euphemism_map.get(word.lower(), word)
        
        elif style == RedactionStyle.PARTIAL:
            if len(word) >= self.config.get('min_word_length_for_partial', 4):
                mid_point = len(word) // 2
                return word[:1] + '*' * (mid_point - 1) + word[mid_point:]
            else:
                return '*' * len(word)
        
        return '*' * len(word)  # Default fallback
    
    def _generate_suggestions(self, redacted_words: List[str]) -> List[str]:
        """Generate alternative word suggestions for redacted content."""
        
        suggestions = []
        
        for word in redacted_words[:3]:  # Limit to top 3 words
            word_lower = word.lower()
            if word_lower in self.suggestion_map:
                alternatives = self.suggestion_map[word_lower]
                suggestions.extend(alternatives[:2])  # Max 2 per word
        
        # Add general suggestions
        if redacted_words:
            suggestions.extend([
                "Consider rephrasing more constructively",
                "Express disagreement respectfully",
                "Focus on specific issues rather than personal attacks"
            ])
        
        return suggestions[:5]  # Limit total suggestions
    
    def _fallback_analysis(self, text: str) -> Tuple[float, float, str]:
        """Fallback toxicity analysis when enhanced detection unavailable."""
        
        score = 0.0
        matches = []
        
        for category, patterns in self.fallback_patterns.items():
            for pattern in patterns:
                if re.search(r'\b' + re.escape(pattern) + r'\b', text, re.IGNORECASE):
                    if category == 'threats':
                        score += 0.3
                    elif category == 'profanity':
                        score += 0.25
                    elif category == 'slurs':
                        score += 0.2
                    else:
                        score += 0.15
                    matches.append(f"{category}:{pattern}")
        
        # Check for caps and exclamation patterns
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.3:
            score += 0.1
            matches.append("excessive_caps")
        
        if text.count('!') > 2:
            score += 0.05
            matches.append("excessive_exclamation")
        
        confidence = 0.6 if matches else 0.3
        reason = f"Fallback analysis: {', '.join(matches)}" if matches else "No toxic patterns detected"
        
        return min(score, 1.0), confidence, reason
    
    def _empty_result(self, text: str) -> RedactionResult:
        """Return empty result for invalid input."""
        
        return RedactionResult(
            original_text=text,
            redacted_text=text,
            redaction_level=RedactionLevel.NONE,
            redacted_words=[],
            redaction_positions=[],
            toxicity_score=0.0,
            confidence=0.0,
            redaction_reason="Empty or invalid input",
            suggestions=[],
            timestamp=datetime.now().isoformat()
        )
    
    def batch_redact(self, texts: List[str], 
                    redaction_level: Optional[RedactionLevel] = None,
                    redaction_style: Optional[RedactionStyle] = None) -> List[RedactionResult]:
        """Perform batch redaction on multiple texts."""
        
        results = []
        for text in texts:
            result = self.redact_content(text, redaction_level, redaction_style)
            results.append(result)
        
        return results
    
    def create_redaction_report(self, result: RedactionResult) -> str:
        """Create detailed redaction report."""
        
        report = []
        report.append("=== REDACTION REPORT ===")
        report.append(f"Timestamp: {result.timestamp}")
        report.append(f"Original Length: {len(result.original_text)} chars")
        report.append(f"Redacted Length: {len(result.redacted_text)} chars")
        report.append("")
        
        report.append("ANALYSIS:")
        report.append(f"  Toxicity Score: {result.toxicity_score:.3f}")
        report.append(f"  Confidence: {result.confidence:.3f}")
        report.append(f"  Redaction Level: {result.redaction_level.value}")
        report.append(f"  Reason: {result.redaction_reason}")
        report.append("")
        
        if result.redacted_words:
            report.append("REDACTED CONTENT:")
            report.append(f"  Words Redacted: {len(result.redacted_words)}")
            report.append(f"  Redacted Words: {', '.join(result.redacted_words)}")
            report.append("")
        
        report.append("ORIGINAL TEXT:")
        report.append(f'  "{result.original_text}"')
        report.append("")
        
        report.append("REDACTED TEXT:")
        report.append(f'  "{result.redacted_text}"')
        report.append("")
        
        if result.suggestions:
            report.append("SUGGESTIONS:")
            for i, suggestion in enumerate(result.suggestions, 1):
                report.append(f"  {i}. {suggestion}")
            report.append("")
        
        report.append("=== END REPORT ===")
        
        return '\n'.join(report)
    
    def get_redaction_stats(self, results: List[RedactionResult]) -> Dict[str, Any]:
        """Generate statistics from multiple redaction results."""
        
        if not results:
            return {}
        
        stats = {
            'total_texts': len(results),
            'redacted_texts': sum(1 for r in results if r.redacted_words),
            'redaction_rate': 0,
            'avg_toxicity_score': 0,
            'avg_confidence': 0,
            'total_words_redacted': 0,
            'redaction_levels': {},
            'common_redacted_words': {},
            'suggestions_generated': sum(len(r.suggestions) for r in results)
        }
        
        # Calculate averages
        scores = [r.toxicity_score for r in results]
        confidences = [r.confidence for r in results]
        
        if scores:
            stats['avg_toxicity_score'] = sum(scores) / len(scores)
        if confidences:
            stats['avg_confidence'] = sum(confidences) / len(confidences)
        
        stats['redaction_rate'] = stats['redacted_texts'] / stats['total_texts'] if stats['total_texts'] > 0 else 0
        
        # Count redaction levels
        for result in results:
            level = result.redaction_level.value
            stats['redaction_levels'][level] = stats['redaction_levels'].get(level, 0) + 1
            stats['total_words_redacted'] += len(result.redacted_words)
            
            # Count common redacted words
            for word in result.redacted_words:
                stats['common_redacted_words'][word] = stats['common_redacted_words'].get(word, 0) + 1
        
        return stats


if __name__ == "__main__":
    # Test the smart redaction system
    print("🔒 Smart Redaction System - Test")
    print("=" * 50)
    
    # Initialize redaction system
    redactor = SmartRedactionSystem()
    
    # Test cases with various toxicity levels
    test_cases = [
        ("You're such a fucking idiot, kill yourself!", "Extreme toxicity with threat"),
        ("SHUT UP YOU STUPID MORON!!!", "Caps with insults"),
        ("All you people are disgusting and pathetic", "Group targeting"),
        ("This is absolutely terrible, what a joke!", "Mild criticism"),
        ("Sure, that's a brilliant idea... obviously", "Sarcasm"),
        ("I disagree with your opinion respectfully", "Respectful disagreement"),
        ("Great discussion, thank you for sharing!", "Positive comment")
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} examples with different redaction levels")
    print("=" * 50)
    
    # Test different redaction levels and styles
    levels_to_test = [RedactionLevel.MINIMAL, RedactionLevel.MODERATE, RedactionLevel.AGGRESSIVE]
    styles_to_test = [RedactionStyle.ASTERISKS, RedactionStyle.BRACKETS, RedactionStyle.EUPHEMISMS]
    
    all_results = []
    
    for i, (text, description) in enumerate(test_cases, 1):
        print(f"\n{'─'*10} TEST {i}: {description} {'─'*10}")
        print(f"Original: \"{text}\"")
        
        # Test with different redaction levels
        for level in levels_to_test:
            result = redactor.redact_content(text, redaction_level=level)
            all_results.append(result)
            
            print(f"\n{level.value.upper()} Redaction:")
            print(f"  Redacted: \"{result.redacted_text}\"")
            print(f"  Score: {result.toxicity_score:.3f} | Confidence: {result.confidence:.3f}")
            
            if result.redacted_words:
                print(f"  Words Redacted: {', '.join(result.redacted_words)}")
            
            if result.suggestions:
                print(f"  Suggestions: {result.suggestions[0]}")
    
    # Test different styles on a toxic example
    print(f"\n{'─'*20} STYLE COMPARISON {'─'*20}")
    toxic_example = "You're such a damn idiot, shut the hell up!"
    print(f"Original: \"{toxic_example}\"")
    
    for style in styles_to_test:
        result = redactor.redact_content(toxic_example, 
                                       redaction_level=RedactionLevel.MODERATE,
                                       redaction_style=style)
        print(f"\n{style.value.upper()}: \"{result.redacted_text}\"")
    
    # Generate comprehensive report
    print(f"\n{'─'*20} DETAILED REPORT {'─'*20}")
    sample_result = redactor.redact_content("You're a fucking moron!", RedactionLevel.MODERATE)
    report = redactor.create_redaction_report(sample_result)
    print(report)
    
    # Batch processing test
    print(f"\n{'─'*20} BATCH PROCESSING {'─'*20}")
    batch_texts = [case[0] for case in test_cases]
    batch_results = redactor.batch_redact(batch_texts, RedactionLevel.MODERATE)
    
    stats = redactor.get_redaction_stats(batch_results)
    print(f"Batch Statistics:")
    print(f"  Total Texts: {stats['total_texts']}")
    print(f"  Redaction Rate: {stats['redaction_rate']:.1%}")
    print(f"  Average Toxicity: {stats['avg_toxicity_score']:.3f}")
    print(f"  Total Words Redacted: {stats['total_words_redacted']}")
    print(f"  Suggestions Generated: {stats['suggestions_generated']}")
    
    if stats['common_redacted_words']:
        print(f"  Most Redacted Words: {list(stats['common_redacted_words'].keys())[:3]}")
    
    print(f"\n✅ Smart Redaction System Test Complete!")
    print(f"   Features: Context-aware, multiple styles, batch processing, detailed reporting")
    print(f"   Capabilities: {len(levels_to_test)} redaction levels, {len(styles_to_test)} visual styles")
    print("=" * 50)
