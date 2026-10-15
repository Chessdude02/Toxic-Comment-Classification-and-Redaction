#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from toxicity_redactor import load_pretrained_model

print('🧪 Testing toxicity detection module...')

redactor = load_pretrained_model()

if redactor:
    test_messages = [
        'Hello everyone, how are you?',
        'You are such an idiot!', 
        'This is a great discussion.',
        'Stop being so stupid about this.'
    ]
    
    for msg in test_messages:
        result = redactor.classify_toxicity(msg)
        print(f'Message: "{msg}"')
        print(f'  Is toxic: {result["is_toxic"]}')
        print(f'  Confidence: {result["max_toxicity_probability"]:.3f}')
        print(f'  Types: {result["toxic_types"]}')
        
        # Test redaction
        redacted = redactor.redact_message(msg, redaction_style='warning')
        if redacted['was_redacted']:
            print(f'  Redacted: "{redacted["redacted_message"]}"')
        print()
        
    print('✅ Module test completed successfully!')
else:
    print('❌ Could not load model')
