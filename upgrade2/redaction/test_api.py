#!/usr/bin/env python3
"""
API Test Script for Smart Redaction System Flask Server
Tests all available endpoints
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import requests
import json

def test_api():
    print('🧪 Testing Flask API endpoints...')
    
    base_url = 'http://localhost:5000'
    
    # Test health endpoint
    try:
        health = requests.get(f'{base_url}/api/health', timeout=5)
        print('✅ Health check:', health.json())
    except Exception as e:
        print('❌ Health check failed:', e)
        return
    
    # Test redact endpoint
    try:
        test_data = {
            'text': 'You are such an idiot!',
            'redaction_level': 'medium',
            'style': 'ASTERISKS'
        }
        
        response = requests.post(
            f'{base_url}/api/redact',
            json=test_data,
            timeout=5
        )
        
        result = response.json()
        print('\n✅ Redaction test result:')
        print('  Original:', result.get('original_text'))
        print('  Redacted:', result.get('redacted_text'))
        print('  Score:', result.get('toxicity_score'))
        print('  Words redacted:', result.get('redacted_words'))
        print('  Success:', result.get('success'))
        
    except Exception as e:
        print('❌ Redaction test failed:', e)
    
    # Test batch endpoint
    try:
        batch_data = {
            'texts': [
                'You are stupid!',
                'This is nice.',
                'Go kill yourself!'
            ],
            'redaction_level': 'high',
            'style': 'BRACKETS'
        }
        
        batch_response = requests.post(
            f'{base_url}/api/batch',
            json=batch_data,
            timeout=10
        )
        
        batch_result = batch_response.json()
        print('\n✅ Batch test result:')
        print('  Success:', batch_result.get('success'))
        print('  Processed count:', batch_result.get('processed_count'))
        
        if batch_result.get('success'):
            for i, result in enumerate(batch_result.get('results', [])):
                print(f'  Text {i+1}:')
                print(f'    Original: {result.get("original_text")}')
                print(f'    Redacted: {result.get("redacted_text")}')
                print(f'    Score: {result.get("toxicity_score"):.3f}')
        
    except Exception as e:
        print('❌ Batch test failed:', e)
    
    print(f'\n🌐 Web interface available at: {base_url}')

if __name__ == '__main__':
    test_api()
