"""
Enhanced Vocabulary Manager for Toxic Comment Classification

This module provides advanced vocabulary management specifically designed for toxicity detection.
Features include expanded dictionaries, slang detection, context-aware matching, and dynamic learning.
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import re
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass


@dataclass
class ToxicToken:
    """Represents a toxic token with enhanced metadata."""
    word: str
    category: str  # toxic, severe_toxic, obscene, threat, insult, identity_hate
    severity: float  # 0.0 to 1.0
    context_patterns: List[str]
    variants: List[str]  # spelling variations, abbreviations
    confidence: float


class EnhancedToxicVocabularyManager:
    """
    Advanced vocabulary manager specifically for toxic content detection.
    Includes comprehensive toxic language patterns, slang, and context awareness.
    """
    
    def __init__(self, vocab_dir: str = None):
        if vocab_dir is None:
            self.vocab_dir = Path(__file__).parent / "data"
        else:
            self.vocab_dir = Path(vocab_dir)
        
        self.vocab_dir.mkdir(exist_ok=True)
        
        # Core toxicity categories
        self.toxicity_categories = [
            'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
        ]
        
        # Enhanced vocabularies
        self.toxic_patterns = {}
        self.slang_dictionary = {}
        self.context_patterns = {}
        self.severity_weights = {}
        self.word_variants = defaultdict(set)
        
        # Semantic mappings
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_embeddings = {}
        
        # Advanced features
        self.leetspeak_mappings = {}
        self.euphemism_patterns = {}
        self.escalation_sequences = {}
        
        self._initialize_vocabularies()
        self._build_advanced_patterns()
    
    def _initialize_vocabularies(self):
        """Initialize comprehensive toxic vocabularies."""
        
        # Base toxic vocabulary with severity scores
        self.toxic_patterns = {
            'toxic': {
                'stupid': {'severity': 0.6, 'variants': ['st*pid', 'stoopid', 'stewpid']},
                'idiot': {'severity': 0.7, 'variants': ['id10t', '1diot', 'idiotic']},
                'moron': {'severity': 0.7, 'variants': ['m0ron', 'moronic']},
                'dumb': {'severity': 0.5, 'variants': ['d*mb', 'dumm']},
                'fool': {'severity': 0.4, 'variants': ['f00l', 'foolish']},
                'pathetic': {'severity': 0.6, 'variants': ['pathethic']},
                'loser': {'severity': 0.6, 'variants': ['l0ser', 'looser']},
                'worthless': {'severity': 0.8, 'variants': ['worthles']},
                'garbage': {'severity': 0.5, 'variants': ['garb4ge', 'trash']},
                'terrible': {'severity': 0.4, 'variants': ['terrib1e']},
            },
            
            'severe_toxic': {
                'kill': {'severity': 0.9, 'variants': ['k1ll', 'kil', 'ki11']},
                'die': {'severity': 0.9, 'variants': ['d1e', 'dy3']},
                'murder': {'severity': 0.95, 'variants': ['murd3r', 'murdr']},
                'destroy': {'severity': 0.8, 'variants': ['destr0y', 'destory']},
                'eliminate': {'severity': 0.85, 'variants': ['elimin8']},
                'annihilate': {'severity': 0.9, 'variants': ['ann1hilate']},
            },
            
            'obscene': {
                'damn': {'severity': 0.4, 'variants': ['d*mn', 'd4mn', 'damm']},
                'hell': {'severity': 0.3, 'variants': ['h*ll', 'h3ll']},
                'crap': {'severity': 0.5, 'variants': ['cr*p', 'c4rap']},
                'shit': {'severity': 0.7, 'variants': ['sh*t', 'sh1t', 's**t']},
                'fuck': {'severity': 0.8, 'variants': ['f*ck', 'f**k', 'fck', 'f*k']},
                'bitch': {'severity': 0.8, 'variants': ['b*tch', 'b1tch', 'bi7ch']},
                'ass': {'severity': 0.6, 'variants': ['a*s', 'a$$', '@ss']},
            },
            
            'threat': {
                'threat': {'severity': 0.8, 'variants': ['thr3at', 'threaten']},
                'harm': {'severity': 0.7, 'variants': ['h*rm']},
                'hurt': {'severity': 0.6, 'variants': ['h*rt']},
                'attack': {'severity': 0.8, 'variants': ['att*ck', 'atta*k']},
                'violence': {'severity': 0.9, 'variants': ['viol3nce']},
                'revenge': {'severity': 0.7, 'variants': ['rev3nge']},
                'payback': {'severity': 0.6, 'variants': ['pay_back']},
            },
            
            'insult': {
                'ugly': {'severity': 0.6, 'variants': ['ug1y', 'uglie']},
                'fat': {'severity': 0.5, 'variants': ['f*t', 'phat']},
                'freak': {'severity': 0.7, 'variants': ['fr3ak', 'freek']},
                'weirdo': {'severity': 0.6, 'variants': ['w3irdo', 'weird0']},
                'creep': {'severity': 0.7, 'variants': ['cr33p', 'creap']},
                'pervert': {'severity': 0.8, 'variants': ['perv3rt', 'perv']},
                'scum': {'severity': 0.8, 'variants': ['sc*m']},
                'slut': {'severity': 0.8, 'variants': ['sl*t', 's1ut']},
            },
            
            'identity_hate': {
                'racist': {'severity': 0.9, 'variants': ['rac1st', 'racisst']},
                'nazi': {'severity': 0.95, 'variants': ['n*zi', 'naz1']},
                'fascist': {'severity': 0.9, 'variants': ['fasc1st']},
                'bigot': {'severity': 0.8, 'variants': ['b1got']},
                'homophobic': {'severity': 0.9, 'variants': ['homoph0bic']},
                'sexist': {'severity': 0.8, 'variants': ['sex1st']},
            }
        }
        
        # Modern slang and internet toxicity
        self.slang_dictionary = {
            'kys': {'meaning': 'kill yourself', 'category': 'severe_toxic', 'severity': 0.95},
            'stfu': {'meaning': 'shut the f*** up', 'category': 'obscene', 'severity': 0.8},
            'gtfo': {'meaning': 'get the f*** out', 'category': 'obscene', 'severity': 0.7},
            'pos': {'meaning': 'piece of shit', 'category': 'obscene', 'severity': 0.8},
            'af': {'meaning': 'as f***', 'category': 'obscene', 'severity': 0.5},
            'smh': {'meaning': 'shake my head', 'category': 'mild', 'severity': 0.2},
            'fml': {'meaning': 'f*** my life', 'category': 'obscene', 'severity': 0.6},
            'wtf': {'meaning': 'what the f***', 'category': 'obscene', 'severity': 0.6},
            'omfg': {'meaning': 'oh my f***ing god', 'category': 'obscene', 'severity': 0.5},
            'lmfao': {'meaning': 'laughing my f***ing ass off', 'category': 'obscene', 'severity': 0.4},
            'thot': {'meaning': 'that ho over there', 'category': 'insult', 'severity': 0.7},
            'simp': {'meaning': 'simpleton/submissive', 'category': 'insult', 'severity': 0.5},
            'karen': {'meaning': 'entitled woman', 'category': 'insult', 'severity': 0.6},
            'incel': {'meaning': 'involuntary celibate', 'category': 'identity_hate', 'severity': 0.7},
        }
        
        # Context patterns that modify toxicity
        self.context_patterns = {
            'amplifiers': ['very', 'extremely', 'totally', 'completely', 'absolutely', 'really', 'so', 'such'],
            'diminishers': ['kinda', 'somewhat', 'a bit', 'slightly', 'maybe', 'possibly'],
            'negations': ['not', "don't", "won't", "can't", "shouldn't", 'never', 'nothing', 'nobody'],
            'questions': ['?', 'what', 'how', 'why', 'when', 'where', 'who'],
            'sarcasm_indicators': ['yeah right', 'sure', 'obviously', '/s', 'totally', 'great job'],
        }
        
        self._build_word_mappings()
    
    def _build_advanced_patterns(self):
        """Build advanced pattern matching systems."""
        
        # Leetspeak mappings
        self.leetspeak_mappings = {
            'a': ['@', '4'], 'e': ['3'], 'i': ['1', '!'], 'o': ['0'], 
            's': ['$', '5'], 't': ['7'], 'l': ['1', '|'], 'g': ['9'],
            'b': ['6'], 'z': ['2'], 'f': ['ph'], 'c': ['k'], 'u': ['v']
        }
        
        # Euphemism patterns (mild substitutions for stronger language)
        self.euphemism_patterns = {
            'fudge': 'fuck', 'frick': 'fuck', 'darn': 'damn', 'heck': 'hell',
            'shoot': 'shit', 'crud': 'crap', 'dang': 'damn', 'freaking': 'fucking',
            'frigging': 'fucking', 'effing': 'fucking', 'bs': 'bullshit',
            'pos': 'piece of shit', 'sob': 'son of a bitch'
        }
        
        # Escalation sequences (progressively more toxic)
        self.escalation_sequences = {
            'annoyance': ['annoying', 'irritating', 'stupid', 'idiot'],
            'anger': ['mad', 'angry', 'pissed', 'furious', 'rage'],
            'hostility': ['hate', 'despise', 'kill', 'destroy', 'die'],
        }
    
    def _build_word_mappings(self):
        """Build comprehensive word-to-ID mappings."""
        all_words = set()
        
        # Add all toxic words and variants
        for category, words in self.toxic_patterns.items():
            for word, data in words.items():
                all_words.add(word)
                all_words.update(data['variants'])
        
        # Add slang terms
        all_words.update(self.slang_dictionary.keys())
        
        # Add context words
        for context_list in self.context_patterns.values():
            all_words.update(context_list)
        
        # Add euphemisms
        all_words.update(self.euphemism_patterns.keys())
        all_words.update(self.euphemism_patterns.values())
        
        # Add special tokens
        all_words.update(['<UNK>', '<PAD>', '<TOXIC>', '<CLEAN>', '<CONTEXT>'])
        
        # Create mappings
        self.word_to_id = {word: idx for idx, word in enumerate(sorted(all_words))}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        
        # Initialize basic embeddings
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize word embeddings with toxicity-aware representations."""
        embedding_dim = 128
        np.random.seed(42)
        
        for word in self.word_to_id.keys():
            # Base random embedding
            embedding = np.random.randn(embedding_dim) * 0.1
            
            # Enhance with toxicity information
            toxicity_score = self.get_word_toxicity_score(word)
            if toxicity_score > 0:
                # Toxic words get distinct embedding patterns
                embedding[:10] += toxicity_score * 0.5
            
            self.word_embeddings[word] = embedding
    
    def get_word_toxicity_score(self, word: str) -> float:
        """Get toxicity score for a word."""
        word_lower = word.lower()
        
        # Check direct matches
        for category, words in self.toxic_patterns.items():
            if word_lower in words:
                return words[word_lower]['severity']
            
            # Check variants
            for base_word, data in words.items():
                if word_lower in data['variants']:
                    return data['severity']
        
        # Check slang
        if word_lower in self.slang_dictionary:
            return self.slang_dictionary[word_lower]['severity']
        
        # Check euphemisms
        if word_lower in self.euphemism_patterns:
            mapped_word = self.euphemism_patterns[word_lower]
            return self.get_word_toxicity_score(mapped_word) * 0.7  # Reduced severity
        
        return 0.0
    
    def detect_leetspeak(self, text: str) -> str:
        """Convert leetspeak back to normal text."""
        converted = text.lower()
        
        for char, replacements in self.leetspeak_mappings.items():
            for replacement in replacements:
                converted = converted.replace(replacement, char)
        
        return converted
    
    def expand_text_variants(self, text: str) -> List[str]:
        """Generate possible text variants including leetspeak conversions."""
        variants = [text, text.lower(), self.detect_leetspeak(text)]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)
        
        return unique_variants
    
    def analyze_text_toxicity(self, text: str) -> Dict[str, any]:
        """Comprehensive toxicity analysis of text."""
        
        # Basic preprocessing
        text_variants = self.expand_text_variants(text)
        words = []
        for variant in text_variants:
            words.extend(re.findall(r'\b\w+\b', variant))
        
        # Remove duplicates
        unique_words = list(set(words))
        
        # Analyze each word
        word_analysis = {}
        category_scores = {cat: 0.0 for cat in self.toxicity_categories}
        
        for word in unique_words:
            toxicity_score = self.get_word_toxicity_score(word)
            if toxicity_score > 0:
                category = self._get_word_category(word)
                word_analysis[word] = {
                    'toxicity_score': toxicity_score,
                    'category': category,
                    'variants_detected': self._get_detected_variants(word)
                }
                category_scores[category] += toxicity_score
        
        # Context analysis
        context_modifiers = self._analyze_context(text)
        
        # Apply context modifiers
        for category in category_scores:
            category_scores[category] *= context_modifiers['multiplier']
        
        # Overall toxicity
        max_category_score = max(category_scores.values())
        overall_toxicity = min(max_category_score, 1.0)
        
        return {
            'overall_toxicity': overall_toxicity,
            'category_scores': category_scores,
            'toxic_words': word_analysis,
            'context_analysis': context_modifiers,
            'text_variants_checked': text_variants,
            'slang_detected': self._detect_slang(text),
            'escalation_detected': self._detect_escalation_pattern(unique_words)
        }
    
    def _get_word_category(self, word: str) -> str:
        """Determine the primary toxicity category of a word."""
        word_lower = word.lower()
        
        for category, words in self.toxic_patterns.items():
            if word_lower in words:
                return category
            for base_word, data in words.items():
                if word_lower in data['variants']:
                    return category
        
        if word_lower in self.slang_dictionary:
            return self.slang_dictionary[word_lower]['category']
        
        return 'toxic'  # default category
    
    def _get_detected_variants(self, word: str) -> List[str]:
        """Get list of variants that matched for this word."""
        word_lower = word.lower()
        detected = []
        
        for category, words in self.toxic_patterns.items():
            for base_word, data in words.items():
                if word_lower == base_word or word_lower in data['variants']:
                    detected.extend([base_word] + data['variants'])
                    break
        
        return list(set(detected))
    
    def _analyze_context(self, text: str) -> Dict[str, any]:
        """Analyze contextual factors that modify toxicity."""
        text_lower = text.lower()
        
        # Check for amplifiers/diminishers
        amplifier_count = sum(1 for amp in self.context_patterns['amplifiers'] if amp in text_lower)
        diminisher_count = sum(1 for dim in self.context_patterns['diminishers'] if dim in text_lower)
        negation_count = sum(1 for neg in self.context_patterns['negations'] if neg in text_lower)
        question_indicators = sum(1 for q in self.context_patterns['questions'] if q in text_lower)
        sarcasm_indicators = sum(1 for s in self.context_patterns['sarcasm_indicators'] if s in text_lower)
        
        # Calculate multiplier
        multiplier = 1.0
        multiplier += amplifier_count * 0.3  # Amplifiers increase toxicity
        multiplier -= diminisher_count * 0.2  # Diminishers decrease toxicity
        multiplier *= (0.5 if negation_count > 0 else 1.0)  # Negations reduce toxicity
        multiplier *= (0.8 if question_indicators > 0 else 1.0)  # Questions are less toxic
        multiplier *= (0.7 if sarcasm_indicators > 0 else 1.0)  # Sarcasm is complex
        
        multiplier = max(0.1, min(2.0, multiplier))  # Bound between 0.1 and 2.0
        
        return {
            'multiplier': multiplier,
            'amplifiers': amplifier_count,
            'diminishers': diminisher_count,
            'negations': negation_count,
            'questions': question_indicators,
            'sarcasm': sarcasm_indicators,
            'context_confidence': min(1.0, (amplifier_count + diminisher_count + negation_count) * 0.2)
        }
    
    def _detect_slang(self, text: str) -> List[Dict[str, any]]:
        """Detect slang terms in text."""
        detected_slang = []
        text_lower = text.lower()
        
        for slang_term, data in self.slang_dictionary.items():
            if slang_term in text_lower:
                detected_slang.append({
                    'term': slang_term,
                    'meaning': data['meaning'],
                    'category': data['category'],
                    'severity': data['severity']
                })
        
        return detected_slang
    
    def _detect_escalation_pattern(self, words: List[str]) -> Dict[str, any]:
        """Detect escalation patterns in word sequence."""
        escalation_detected = {}
        
        for escalation_type, sequence in self.escalation_sequences.items():
            matches = [word for word in words if word.lower() in sequence]
            if matches:
                # Calculate escalation level based on position in sequence
                max_level = 0
                for word in matches:
                    try:
                        level = sequence.index(word.lower()) + 1
                        max_level = max(max_level, level)
                    except ValueError:
                        continue
                
                escalation_detected[escalation_type] = {
                    'level': max_level,
                    'max_level': len(sequence),
                    'matched_words': matches,
                    'escalation_ratio': max_level / len(sequence)
                }
        
        return escalation_detected
    
    def get_redaction_suggestions(self, text: str, style: str = 'smart') -> Dict[str, any]:
        """Get intelligent redaction suggestions based on enhanced analysis."""
        
        analysis = self.analyze_text_toxicity(text)
        
        if analysis['overall_toxicity'] < 0.3:
            return {
                'needs_redaction': False,
                'confidence': analysis['overall_toxicity'],
                'suggested_action': 'none'
            }
        
        suggestions = {
            'needs_redaction': True,
            'confidence': analysis['overall_toxicity'],
            'toxic_spans': [],
            'redaction_strategies': []
        }
        
        # Identify specific toxic spans
        for word, word_data in analysis['toxic_words'].items():
            # Find word positions in text
            word_positions = []
            start = 0
            while True:
                pos = text.lower().find(word.lower(), start)
                if pos == -1:
                    break
                word_positions.append((pos, pos + len(word)))
                start = pos + 1
            
            for start_pos, end_pos in word_positions:
                suggestions['toxic_spans'].append({
                    'word': word,
                    'start': start_pos,
                    'end': end_pos,
                    'severity': word_data['toxicity_score'],
                    'category': word_data['category'],
                    'replacement_suggestions': self._get_replacement_suggestions(word, word_data['category'])
                })
        
        # Redaction strategies based on style
        if style == 'smart':
            if analysis['overall_toxicity'] > 0.8:
                suggestions['redaction_strategies'].append('complete_removal')
            elif analysis['overall_toxicity'] > 0.6:
                suggestions['redaction_strategies'].append('word_replacement')
            else:
                suggestions['redaction_strategies'].append('warning_label')
        
        # Context-aware adjustments
        if analysis['context_analysis']['sarcasm'] > 0:
            suggestions['redaction_strategies'].append('context_warning')
        
        if analysis['escalation_detected']:
            suggestions['redaction_strategies'].append('escalation_detected')
        
        return suggestions
    
    def _get_replacement_suggestions(self, word: str, category: str) -> List[str]:
        """Get replacement suggestions for toxic words."""
        
        replacements = {
            'toxic': ['[REMOVED]', '***', 'inappropriate'],
            'severe_toxic': ['[SEVERE CONTENT REMOVED]', '***'],
            'obscene': ['[CENSORED]', '***', 'inappropriate language'],
            'threat': ['[THREATENING CONTENT REMOVED]', '***'],
            'insult': ['[INSULT REMOVED]', '***', 'unkind words'],
            'identity_hate': ['[HATE SPEECH REMOVED]', '***']
        }
        
        return replacements.get(category, ['***', '[INAPPROPRIATE]'])
    
    def update_vocabulary(self, new_words: Dict[str, Dict[str, any]]):
        """Dynamically update vocabulary with new toxic patterns."""
        
        for word, data in new_words.items():
            category = data.get('category', 'toxic')
            severity = data.get('severity', 0.5)
            variants = data.get('variants', [])
            
            if category not in self.toxic_patterns:
                self.toxic_patterns[category] = {}
            
            self.toxic_patterns[category][word] = {
                'severity': severity,
                'variants': variants
            }
        
        # Rebuild mappings
        self._build_word_mappings()
    
    def save_vocabulary(self, filename: str = 'enhanced_toxic_vocabulary.pkl'):
        """Save the enhanced vocabulary system."""
        
        vocab_data = {
            'toxic_patterns': self.toxic_patterns,
            'slang_dictionary': self.slang_dictionary,
            'context_patterns': self.context_patterns,
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'word_embeddings': self.word_embeddings,
            'leetspeak_mappings': self.leetspeak_mappings,
            'euphemism_patterns': self.euphemism_patterns,
            'escalation_sequences': self.escalation_sequences
        }
        
        filepath = self.vocab_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print(f"💾 Enhanced vocabulary saved to: {filepath}")
    
    def load_vocabulary(self, filename: str = 'enhanced_toxic_vocabulary.pkl'):
        """Load enhanced vocabulary system."""
        
        filepath = self.vocab_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                vocab_data = pickle.load(f)
            
            self.toxic_patterns = vocab_data['toxic_patterns']
            self.slang_dictionary = vocab_data['slang_dictionary']
            self.context_patterns = vocab_data['context_patterns']
            self.word_to_id = vocab_data['word_to_id']
            self.id_to_word = vocab_data['id_to_word']
            self.word_embeddings = vocab_data['word_embeddings']
            self.leetspeak_mappings = vocab_data['leetspeak_mappings']
            self.euphemism_patterns = vocab_data['euphemism_patterns']
            self.escalation_sequences = vocab_data['escalation_sequences']
            
            print(f"✅ Enhanced vocabulary loaded from: {filepath}")
        else:
            print(f"⚠️ Vocabulary file not found: {filepath}")
    
    def get_vocabulary_statistics(self) -> Dict[str, any]:
        """Get comprehensive vocabulary statistics."""
        
        total_toxic_words = sum(len(words) for words in self.toxic_patterns.values())
        total_variants = sum(len(data['variants']) for category in self.toxic_patterns.values() 
                           for data in category.values())
        
        return {
            'total_vocabulary_size': len(self.word_to_id),
            'total_toxic_words': total_toxic_words,
            'total_variants': total_variants,
            'slang_terms': len(self.slang_dictionary),
            'euphemisms': len(self.euphemism_patterns),
            'escalation_sequences': len(self.escalation_sequences),
            'categories': {
                category: len(words) for category, words in self.toxic_patterns.items()
            },
            'context_patterns': {
                pattern: len(words) for pattern, words in self.context_patterns.items()
            },
            'leetspeak_mappings': len(self.leetspeak_mappings)
        }


if __name__ == "__main__":
    # Test the enhanced vocabulary system
    print("🚀 Testing Enhanced Toxic Vocabulary Manager")
    print("=" * 60)
    
    manager = EnhancedToxicVocabularyManager()
    
    # Test various toxic patterns
    test_messages = [
        "You're such an idiot!",
        "kys you f***ing moron",
        "st*pid 1diot",
        "This is really great work!",  # Clean message
        "You're kinda dumb, maybe?",  # Context modifiers
        "What the hell is this garbage?",
        "omfg this is so stupid af",  # Slang
        "frick this stupid crap",  # Euphemisms
        "I'm not happy with this decision",  # Negation
    ]
    
    print("\n🧪 Testing Toxicity Analysis:")
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. \"{message}\"")
        analysis = manager.analyze_text_toxicity(message)
        
        print(f"   Overall Toxicity: {analysis['overall_toxicity']:.3f}")
        print(f"   Categories: {', '.join(f'{k}: {v:.2f}' for k, v in analysis['category_scores'].items() if v > 0)}")
        
        if analysis['toxic_words']:
            print(f"   Toxic Words: {', '.join(analysis['toxic_words'].keys())}")
        
        if analysis['slang_detected']:
            slang_terms = [s['term'] for s in analysis['slang_detected']]
            print(f"   Slang Detected: {', '.join(slang_terms)}")
        
        # Test redaction suggestions
        suggestions = manager.get_redaction_suggestions(message)
        if suggestions['needs_redaction']:
            print(f"   Redaction Needed: Yes (confidence: {suggestions['confidence']:.3f})")
            print(f"   Strategies: {', '.join(suggestions['redaction_strategies'])}")
    
    print(f"\n📊 Vocabulary Statistics:")
    stats = manager.get_vocabulary_statistics()
    print(f"   Total Vocabulary: {stats['total_vocabulary_size']:,} words")
    print(f"   Toxic Words: {stats['total_toxic_words']} (+ {stats['total_variants']} variants)")
    print(f"   Slang Terms: {stats['slang_terms']}")
    print(f"   Categories: {', '.join(f'{k}: {v}' for k, v in stats['categories'].items())}")
    
    print(f"\n✅ Enhanced Vocabulary System Ready!")
    print(f"   Advanced features: Leetspeak detection, slang analysis, context awareness")
    print(f"   Escalation detection, euphemism handling, dynamic learning capability")
