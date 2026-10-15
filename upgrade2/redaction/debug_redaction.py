#!/usr/bin/env python3
"""
Debug script to test and fix redaction system
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import sys
import os
sys.path.insert(0, '.')

from redaction_manager import RedactionManager, RedactionStyle

def debug_redaction():
    print('🔍 Diagnosing redaction issues...')
    
    rm = RedactionManager()
    print(f'Enhanced available: {rm.enhanced_available}')
    print(f'Current threshold: {rm.redaction_threshold}')
    
    if rm.enhanced_available:
        print('Using enhanced detection')
    else:
        print('Using fallback patterns')
        print(f'Fallback patterns: {rm.fallback_patterns}')
    
    # Test with various toxic phrases
    test_phrases = [
        'you idiot',
        'you are an idiot', 
        'stupid person',
        'you moron',
        'go kill yourself',
        'this is shit',
        'damn it',
        'what the hell'
    ]
    
    print('\n📝 Testing phrases:')
    print('=' * 60)
    
    for phrase in test_phrases:
        result = rm.generate_redaction(phrase)
        print(f'Input: "{phrase}"')
        print(f'  Output: "{result.redacted_text}"')
        print(f'  Score: {result.toxicity_score:.3f}')
        print(f'  Redacted words: {result.redacted_words}')
        print(f'  Reason: {result.redaction_reason}')
        print()
    
    # Test fallback system specifically
    print('\n🔧 Testing fallback system directly:')
    print('=' * 60)
    
    # Force fallback mode
    rm.enhanced_available = False
    
    for phrase in test_phrases:
        result = rm.generate_redaction(phrase)
        print(f'Fallback - "{phrase}" → "{result.redacted_text}" | Words: {result.redacted_words}')

if __name__ == '__main__':
    debug_redaction()
