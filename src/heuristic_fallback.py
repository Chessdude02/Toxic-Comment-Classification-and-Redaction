"""
Heuristic fallback for the Flask web app.

Wraps upgrade2's EnhancedToxicityDetector (keyword/regex/pattern heuristics -
NOT a trained model, see upgrade2/README.md) behind the same
classify_toxicity()/redact_message()/moderate_conversation() interface that
ToxicityRedactor exposes in toxicity_redactor.py. This lets toxicity_web_app.py
and ChatModerator use either object interchangeably: when no trained model
weights are available (e.g. a fresh cloud deployment with nothing trained
yet), the app falls back to this instead of returning HTTP 503 on every
request, so there's still something real and working to try.

Also re-implements ChatModerator (originally in toxicity_redactor.py) with no
dependency on that module. toxicity_redactor.py does `import tensorflow` at
module level, so importing anything from it - even just ChatModerator, which
doesn't touch TensorFlow itself - pulls in TensorFlow's ~600MB RSS footprint
regardless. Confirmed in practice on a 512MB-RAM deploy: that alone is enough
to OOM before a single request is served. Keeping ChatModerator here means
the common "no trained model, running on heuristics" deployment path never
imports TensorFlow at all - see toxicity_web_app.py's lazy import gate.
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from upgrade2.integration.enhanced_toxicity_detector import EnhancedToxicityDetector

IS_HEURISTIC_FALLBACK = True


class ChatModerator:
    """
    Real-time chat moderation system with logging and statistics.

    Identical behavior to toxicity_redactor.ChatModerator - duplicated here
    (rather than imported) specifically to avoid that module's top-level
    `import tensorflow`. Works with any redactor exposing
    classify_toxicity()/redact_message(), trained or heuristic alike.
    """

    def __init__(self, redactor, auto_moderate: bool = True, log_messages: bool = True):
        self.redactor = redactor
        self.auto_moderate = auto_moderate
        self.log_messages = log_messages
        self.message_history = []
        self.moderation_stats = {
            'total_messages': 0,
            'toxic_messages': 0,
            'redacted_messages': 0,
            'false_positives': 0,
            'false_negatives': 0,
        }

    def process_message(self, username: str, message: str, timestamp: Optional[str] = None) -> Dict:
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S")

        classification = self.redactor.classify_toxicity(message)

        self.moderation_stats['total_messages'] += 1
        if classification['is_toxic']:
            self.moderation_stats['toxic_messages'] += 1

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

        chat_entry = {
            'timestamp': timestamp,
            'username': username,
            'original_message': message,
            'display_message': display_message,
            'status': status,
            'toxicity_info': classification,
            'was_processed': True,
        }

        if self.log_messages:
            self.message_history.append(chat_entry)

        return chat_entry

    def get_chat_log(self, last_n: int = 50) -> List[Dict]:
        return self.message_history[-last_n:] if self.message_history else []

    def get_moderation_stats(self) -> Dict:
        stats = self.moderation_stats.copy()
        if stats['total_messages'] > 0:
            stats['toxicity_rate'] = stats['toxic_messages'] / stats['total_messages']
            stats['redaction_rate'] = stats['redacted_messages'] / stats['total_messages']
            stats['accuracy_estimate'] = 1 - (
                stats['false_positives'] + stats['false_negatives']
            ) / stats['total_messages']
        else:
            stats['toxicity_rate'] = 0
            stats['redaction_rate'] = 0
            stats['accuracy_estimate'] = 0
        return stats


class HeuristicRedactor:
    """Rule-based stand-in for ToxicityRedactor - no trained weights involved."""

    def __init__(self, threshold: float = 0.5):
        self.detector = EnhancedToxicityDetector()
        self.threshold = threshold

    def classify_toxicity(self, message: str) -> dict:
        result = self.detector.detect_toxicity(message)
        is_toxic = result.overall_toxicity_score > self.threshold

        return {
            'is_toxic': is_toxic,
            # The heuristic detector doesn't produce the trained classifier's
            # six-category labels (toxic/severe_toxic/obscene/threat/insult/
            # identity_hate) - its risk_factors are free-text explanations of
            # what fired, which is the closest honest equivalent here.
            'toxic_types': result.risk_factors if is_toxic else [],
            'max_toxicity_probability': result.overall_toxicity_score,
            'detailed_results': {
                'heuristic': {
                    'probability': result.overall_toxicity_score,
                    'is_toxic': is_toxic,
                    'severity_level': result.severity_level,
                }
            },
            'original_message': message,
            'confidence_score': result.confidence,
            'classification_threshold': self.threshold,
        }

    def redact_message(self, message: str, redaction_style: str = 'warning') -> dict:
        classification = self.classify_toxicity(message)

        if not classification['is_toxic']:
            return {
                'redacted_message': message,
                'was_redacted': False,
                'toxicity_info': classification,
                'redaction_style': redaction_style,
            }

        result = self.detector.detect_toxicity(message)

        if redaction_style == 'complete':
            factors = ', '.join(classification['toxic_types']) or 'flagged content'
            redacted = f"[MESSAGE REDACTED - {factors.upper()}]"
        elif redaction_style in ('partial', 'asterisk'):
            redacted = result.redacted_version
        else:  # 'warning' (default) and any unrecognized style
            redacted = (
                f"⚠️ [WARNING: heuristic severity {result.severity_level}, "
                f"score {result.overall_toxicity_score:.2f}] {message}"
            )

        return {
            'redacted_message': redacted,
            'was_redacted': True,
            'toxicity_info': classification,
            'redaction_style': redaction_style,
        }

    def moderate_conversation(self, messages, redaction_style: str = 'warning') -> list:
        moderated = []
        for i, message in enumerate(messages):
            result = self.redact_message(message, redaction_style)
            result['message_id'] = i
            result['original_index'] = i
            moderated.append(result)
        return moderated
