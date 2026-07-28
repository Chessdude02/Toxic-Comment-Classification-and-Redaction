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
"""

import os
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from upgrade2.integration.enhanced_toxicity_detector import EnhancedToxicityDetector

IS_HEURISTIC_FALLBACK = True


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
