"""
Quick Redaction Demo Script
"""

from redaction.redaction_manager import RedactionManager, RedactionStyle

print('🔒 REDACTION SYSTEM DEMO')
print('='*50)

# Initialize redaction manager
rm = RedactionManager()

# Test messages
tests = [
    'You are such an idiot!',
    'This is complete garbage',
    'What a great job!',
    'That is fucking ridiculous',
    'I hate you moron'
]

print('\n📊 Testing with different thresholds:')
for threshold in [0.3, 0.5, 0.7]:
    print(f'\n--- THRESHOLD: {threshold} ---')
    rm.set_redaction_threshold(threshold)
    
    for text in tests:
        result = rm.generate_redaction(text)
        status = 'REDACTED' if result.redacted_words else 'preserved'
        if result.redacted_words:
            print(f'✅ {status} | {result.toxicity_score:.3f} | "{text}" → "{result.redacted_text}"')
        else:
            print(f'⚪ {status} | {result.toxicity_score:.3f} | "{text}"')

print('\n🎨 Testing different redaction styles:')
test_text = 'You are an idiot and pathetic'
rm.set_redaction_threshold(0.3)  # Lower threshold to catch more

for style in [RedactionStyle.ASTERISKS, RedactionStyle.BRACKETS, RedactionStyle.PARTIAL]:
    result = rm.generate_redaction(test_text, style)
    print(f'{style.value:10}: "{result.redacted_text}"')

print('\n✅ Demo complete!')
