#!/usr/bin/env python3
"""
Toxic Comment Redaction
========================

This is the "Redaction" half of "Toxic Comment Classification and Redaction" -
previously the repository contained no redaction code at all, only the
classifier's overfitting-diagnosis scripts.

`ToxicRedactor` masks toxic words/phrases in a comment. It uses a curated
lexicon for span matching, and (when available) the trained classifier to
decide whether a comment needs redaction at all - so contextual, non-toxic
uses of a trigger word ("this kills the mood", "I hate mondays") are left
alone instead of being redacted just because they contain a flagged word.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REDACTION_TOKEN = "[redacted]"

# Curated phrases/words to mask, longest-first so multi-word phrases are
# matched before their component words.
TOXIC_LEXICON = [
    "go kill yourself", "kill yourself", "i hope you die", "you should just disappear",
    "die in a fire", "i will hurt you", "go to hell", "fuck you", "go fuck yourself",
    "fucking idiot", "fucking moron", "piece of shit", "shut the hell up",
    "shut up", "shut your mouth", "get lost", "screw you",
    "idiot", "moron", "stupid", "pathetic", "worthless", "loser", "disgusting",
    "imbecile", "dumbass", "brain-dead", "braindead", "trash", "garbage", "clown",
    "insufferable", "repulsive", "asshole", "bitch", "bastard", "retarded",
    "scum", "fuck", "shit", "damn",
]

_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in sorted(TOXIC_LEXICON, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def mask_toxic_spans(text):
    """Replace toxic words/phrases with a redaction token. Returns (redacted_text, matches)."""
    matches = [m.group(0) for m in _PATTERN.finditer(text)]
    redacted = _PATTERN.sub(REDACTION_TOKEN, text)
    return redacted, matches


class ToxicRedactor:
    """
    Combines the trained classifier (gate: is this comment toxic at all?)
    with lexicon-based span masking (what exactly to redact).

    Falls back to lexicon-only mode if no trained model/tokenizer is found,
    so this module works even before `models/training/fix_overfitted_model.py`
    has been run.
    """

    def __init__(self, use_classifier=True, toxicity_threshold=0.5):
        self.classifier = None
        self.toxicity_threshold = toxicity_threshold
        if use_classifier:
            try:
                from inference.use_fixed_model import ToxicityClassifier
                clf = ToxicityClassifier()
                if clf.model is not None:
                    self.classifier = clf
            except Exception as e:
                print(f"Redactor: no trained classifier available ({e}); using lexicon-only mode")

    def redact(self, text):
        redacted, matches = mask_toxic_spans(text)
        result = {
            "original": text,
            "redacted": redacted if matches else text,
            "redacted_spans": matches,
            "was_redacted": bool(matches),
        }

        if self.classifier is not None:
            prediction = self.classifier.predict(text)
            result["probability"] = prediction["probability"]
            result["is_toxic"] = prediction["is_toxic"]
            # Only apply lexicon redaction when the classifier agrees the
            # comment is actually toxic - keeps innocent trigger-word uses
            # ("you killed it out there") from being redacted.
            if not prediction["is_toxic"]:
                result["redacted"] = text
                result["was_redacted"] = False
        else:
            result["probability"] = None
            result["is_toxic"] = bool(matches)

        return result

    def batch_redact(self, texts):
        return [self.redact(t) for t in texts]


def demo():
    redactor = ToxicRedactor()
    samples = [
        "You are an idiot and should shut up",
        "Go kill yourself, nobody likes you",
        "Thank you for your help today",
        "You killed it in that presentation",
        "I hate waiting in long lines",
        "Fucking moron, get lost",
    ]
    print("TOXIC COMMENT REDACTION DEMO")
    print("=" * 40)
    for text in samples:
        result = redactor.redact(text)
        status = "REDACTED" if result["was_redacted"] else "unchanged"
        print(f"[{status}] '{text}'")
        print(f"       -> '{result['redacted']}'")
    return redactor


if __name__ == "__main__":
    demo()
