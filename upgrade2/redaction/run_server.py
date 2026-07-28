#!/usr/bin/env python3
"""
Flask Server Runner for Smart Redaction System
Runs the web interface with comprehensive logging and error handling
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print('🚀 Starting Flask server with detailed logging...')
print('Working directory:', os.getcwd())
print('Python path includes:', sys.path[0])

try:
    from flask import Flask, request, jsonify
    from redaction_manager import RedactionManager, RedactionStyle
    
    print('✅ All imports successful')
    
    app = Flask(__name__)
    app.config['DEBUG'] = True
    
    # Initialize RedactionManager
    print('Initializing RedactionManager...')
    redaction_manager = RedactionManager()
    print('✅ RedactionManager initialized')
    
    @app.route('/')
    def home():
        print('📡 Home route accessed')
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Smart Redaction System</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
        textarea { width: 100%; height: 100px; margin: 10px 0; }
        button { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { background: white; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }
        .redacted { color: red; font-weight: bold; }
        .original { color: gray; }
        .controls { margin: 15px 0; }
        .controls label { margin: 0 10px 0 0; font-weight: bold; }
        .controls select { margin: 0 15px 0 5px; padding: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Smart Redaction System</h1>
        <p>Enter text below to detect and redact toxic content using AI-powered analysis.</p>
        
        <textarea id="input-text" placeholder="Enter text to redact... (e.g., 'You are such an idiot!')"></textarea>
        
        <div class="controls">
            <label for="redaction-level">Redaction Level:</label>
            <select id="redaction-level">
                <option value="low">Low (0.7) - Only very toxic</option>
                <option value="medium" selected>Medium (0.5) - Moderately toxic</option>
                <option value="high">High (0.3) - Mildly toxic</option>
                <option value="strict">Strict (0.1) - Any questionable</option>
            </select>
            
            <label for="redaction-style">Style:</label>
            <select id="redaction-style">
                <option value="ASTERISKS" selected>Asterisks (****)</option>
                <option value="BRACKETS">[REDACTED]</option>
                <option value="DASHES">----</option>
                <option value="DOTS">...</option>
            </select>
        </div>
        
        <button onclick="redactText()">🔒 Redact Text</button>
        <button onclick="clearAll()">🗑️ Clear</button>
        <button onclick="testSamples()">🧪 Test Samples</button>
        
        <div id="result"></div>
    </div>
    
    <script>
        async function redactText() {
            const text = document.getElementById('input-text').value;
            const level = document.getElementById('redaction-level').value;
            const style = document.getElementById('redaction-style').value;
            
            if (!text.trim()) {
                alert('Please enter some text to redact');
                return;
            }
            
            try {
                const response = await fetch('/api/redact', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: text,
                        redaction_level: level,
                        style: style
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('result').innerHTML = `
                        <div class="result">
                            <h3>📊 Redaction Results</h3>
                            <p><strong>Original:</strong> <span class="original">${escapeHtml(data.original_text)}</span></p>
                            <p><strong>Redacted:</strong> <span class="redacted">${escapeHtml(data.redacted_text)}</span></p>
                            <p><strong>Toxicity Score:</strong> ${data.toxicity_score.toFixed(3)}</p>
                            <p><strong>Confidence:</strong> ${data.confidence.toFixed(3)}</p>
                            <p><strong>Words Redacted:</strong> ${data.redacted_words.join(', ') || 'None'}</p>
                            <p><strong>Reason:</strong> ${data.redaction_reason}</p>
                            <p><strong>Level Used:</strong> ${data.redaction_level}</p>
                            ${data.alternative_suggestions.length > 0 ? 
                                '<p><strong>Suggestions:</strong> ' + data.alternative_suggestions.join(', ') + '</p>' : ''
                            }
                        </div>
                    `;
                } else {
                    document.getElementById('result').innerHTML = `
                        <div class="result" style="border-left-color: red;">
                            <h3>❌ Error</h3>
                            <p>${data.error}</p>
                        </div>
                    `;
                }
            } catch (error) {
                document.getElementById('result').innerHTML = `
                    <div class="result" style="border-left-color: red;">
                        <h3>❌ Network Error</h3>
                        <p>Failed to connect to server: ${error.message}</p>
                    </div>
                `;
            }
        }
        
        async function testSamples() {
            const samples = [
                "You are such an idiot!",
                "This is a nice day.",
                "Go kill yourself!",
                "I love programming.",
                "You stupid moron!"
            ];
            
            const level = document.getElementById('redaction-level').value;
            const style = document.getElementById('redaction-style').value;
            
            let resultsHtml = '<div class="result"><h3>🧪 Test Results</h3>';
            
            for (const text of samples) {
                try {
                    const response = await fetch('/api/redact', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: text,
                            redaction_level: level,
                            style: style
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        const wasRedacted = data.redacted_words.length > 0;
                        resultsHtml += `
                            <div style="margin: 10px 0; padding: 10px; background: ${wasRedacted ? '#ffe6e6' : '#e6ffe6'}; border-radius: 5px;">
                                <strong>Input:</strong> ${escapeHtml(text)}<br>
                                <strong>Output:</strong> ${escapeHtml(data.redacted_text)}<br>
                                <strong>Score:</strong> ${data.toxicity_score.toFixed(3)} | 
                                <strong>Redacted:</strong> ${wasRedacted ? data.redacted_words.join(', ') : 'None'}
                            </div>
                        `;
                    }
                } catch (error) {
                    resultsHtml += `<div style="color: red;">Error processing: ${escapeHtml(text)}</div>`;
                }
            }
            
            resultsHtml += '</div>';
            document.getElementById('result').innerHTML = resultsHtml;
        }
        
        function clearAll() {
            document.getElementById('input-text').value = '';
            document.getElementById('result').innerHTML = '';
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
        '''
    
    @app.route('/api/redact', methods=['POST'])
    def api_redact():
        try:
            print('📡 API redact endpoint accessed')
            data = request.get_json()
            
            if not data or 'text' not in data:
                return jsonify({'success': False, 'error': 'No text provided'})
            
            text = data['text']
            redaction_level = data.get('redaction_level', 'medium')
            style = data.get('style', 'ASTERISKS')
            
            print(f'Processing text: "{text[:50]}..."')
            print(f'Level: {redaction_level}, Style: {style}')
            
            # Set threshold based on level
            thresholds = {
                'low': 0.6,      # More sensitive
                'medium': 0.3,    # Default better sensitivity 
                'high': 0.2,     # High sensitivity
                'strict': 0.1    # Maximum sensitivity
            }
            threshold = thresholds.get(redaction_level, 0.5)
            redaction_manager.set_redaction_threshold(threshold)
            
            # Convert style string to enum
            try:
                style_enum = RedactionStyle[style.upper()]
            except KeyError:
                style_enum = RedactionStyle.ASTERISKS
            
            # Generate redaction
            result = redaction_manager.generate_redaction(text, style_enum)
            
            response_data = {
                'success': True,
                'original_text': result.original_text,
                'redacted_text': result.redacted_text,
                'toxicity_score': float(result.toxicity_score),
                'confidence': float(result.confidence),
                'redacted_words': result.redacted_words,
                'redaction_reason': result.redaction_reason,
                'alternative_suggestions': result.alternative_suggestions,
                'redaction_level': redaction_level
            }
            
            print(f'✅ Success: {len(result.redacted_words)} words redacted')
            return jsonify(response_data)
            
        except Exception as e:
            print(f'❌ Error in API redact: {str(e)}')
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        print('📡 Health check accessed')
        return jsonify({
            'status': 'healthy',
            'redaction_levels': ['low', 'medium', 'high', 'strict'],
            'redaction_styles': ['ASTERISKS', 'BRACKETS', 'DASHES', 'DOTS']
        })
    
    @app.route('/api/batch', methods=['POST'])
    def api_batch_redact():
        try:
            print('📡 API batch redact endpoint accessed')
            data = request.get_json()
            
            if not data or 'texts' not in data:
                return jsonify({'success': False, 'error': 'No texts provided'})
            
            texts = data['texts']
            if not isinstance(texts, list):
                return jsonify({'success': False, 'error': 'Texts must be a list'})
            
            redaction_level = data.get('redaction_level', 'medium')
            style = data.get('style', 'ASTERISKS')
            
            print(f'Processing {len(texts)} texts in batch')
            
            # Set threshold based on level
            thresholds = {
                'low': 0.7,
                'medium': 0.5,
                'high': 0.3,
                'strict': 0.1
            }
            threshold = thresholds.get(redaction_level, 0.5)
            redaction_manager.set_redaction_threshold(threshold)
            
            # Convert style string to enum
            try:
                style_enum = RedactionStyle[style.upper()]
            except KeyError:
                style_enum = RedactionStyle.ASTERISKS
            
            # Process batch
            results = redaction_manager.process_batch(texts, style_enum)
            
            # Convert results to JSON-serializable format
            batch_results = []
            for result in results:
                batch_results.append({
                    'original_text': result.original_text,
                    'redacted_text': result.redacted_text,
                    'toxicity_score': float(result.toxicity_score),
                    'confidence': float(result.confidence),
                    'redacted_words': result.redacted_words,
                    'redaction_reason': result.redaction_reason,
                    'alternative_suggestions': result.alternative_suggestions
                })
            
            response_data = {
                'success': True,
                'results': batch_results,
                'redaction_level': redaction_level,
                'processed_count': len(results)
            }
            
            print(f'✅ Batch success: {len(results)} texts processed')
            return jsonify(response_data)
            
        except Exception as e:
            print(f'❌ Error in API batch redact: {str(e)}')
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)})
    
    if __name__ == '__main__':
        print('🚀 Starting Flask server on http://localhost:5000')
        print('📄 Visit http://localhost:5000 in your browser to use the web interface')
        print('🔌 API endpoints:')
        print('  - GET  /api/health')
        print('  - POST /api/redact')
        print('  - POST /api/batch')
        print()
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    
except Exception as e:
    print(f'❌ Critical error: {str(e)}')
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
