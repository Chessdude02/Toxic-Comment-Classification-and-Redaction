"""
Web Interface for Smart Redaction System

A simple Flask web application that provides an easy-to-use interface
for the intelligent content redaction system.
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import sys
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime
import json

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from redaction_manager import RedactionManager, RedactionStyle
    redaction_available = True
except ImportError as e:
    print(f"Warning: Could not import redaction system: {e}")
    redaction_available = False

app = Flask(__name__)
app.secret_key = 'redaction_system_secret_key_2024'

# Initialize redaction system
if redaction_available:
    redactor = RedactionManager()
else:
    redactor = None

@app.route('/')
def index():
    """Main page of the redaction interface."""
    
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔒 Smart Redaction System</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            padding: 30px;
            margin: 20px 0;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: #4a5568;
        }
        
        .header h1 {
            color: #2d3748;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .input-section, .output-section, .controls-section {
            margin: 25px 0;
        }
        
        .controls-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #4a5568;
        }
        
        select, textarea, button {
            width: 100%;
            padding: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 16px;
            font-family: inherit;
            transition: border-color 0.3s;
        }
        
        select:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        textarea {
            resize: vertical;
            height: 120px;
        }
        
        button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .result-container {
            display: none;
            margin-top: 30px;
        }
        
        .result-box {
            background: #f8fafc;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }
        
        .result-text {
            font-size: 18px;
            line-height: 1.6;
            margin: 10px 0;
        }
        
        .original-text {
            background: #fed7d7;
            border-left-color: #e53e3e;
        }
        
        .redacted-text {
            background: #c6f6d5;
            border-left-color: #38a169;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 14px;
            color: #6b7280;
            margin-top: 5px;
        }
        
        .suggestions {
            margin-top: 20px;
        }
        
        .suggestions ul {
            list-style: none;
            padding: 0;
        }
        
        .suggestions li {
            background: #e6fffa;
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 8px;
            border-left: 4px solid #38b2ac;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .examples {
            margin-top: 30px;
        }
        
        .example-button {
            display: inline-block;
            margin: 5px;
            padding: 8px 15px;
            background: #f7fafc;
            border: 2px solid #e2e8f0;
            border-radius: 20px;
            color: #4a5568;
            text-decoration: none;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .example-button:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            color: #6b7280;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔒 Smart Redaction System</h1>
            <p>Intelligent content redaction using enhanced toxicity detection</p>
        </div>
        
        <div class="input-section">
            <label for="input-text">Enter text to analyze and redact:</label>
            <textarea id="input-text" placeholder="Type or paste your text here..."></textarea>
        </div>
        
        <div class="controls-section">
            <div>
                <label for="redaction-level">Redaction Level:</label>
                <select id="redaction-level">
                    <option value="minimal">Minimal - Light censoring</option>
                    <option value="moderate" selected>Moderate - Standard redaction</option>
                    <option value="aggressive">Aggressive - Heavy redaction</option>
                    <option value="complete">Complete - Full removal</option>
                </select>
            </div>
            
            <div>
                <label for="redaction-style">Redaction Style:</label>
                <select id="redaction-style">
                    <option value="asterisks" selected>Asterisks (f***)</option>
                    <option value="blocks">Blocks (████)</option>
                    <option value="dashes">Dashes (----)</option>
                    <option value="brackets">Brackets ([REDACTED])</option>
                    <option value="euphemisms">Euphemisms (frick, darn)</option>
                    <option value="partial">Partial (f**k)</option>
                </select>
            </div>
        </div>
        
        <button onclick="processText()" id="process-btn">
            🔍 Analyze & Redact Text
        </button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing your text...</p>
        </div>
        
        <div class="result-container" id="results">
            <h3>📊 Redaction Results</h3>
            
            <div class="stats-grid" id="stats">
                <!-- Stats will be populated here -->
            </div>
            
            <div class="result-box original-text">
                <h4>📝 Original Text:</h4>
                <div class="result-text" id="original-text"></div>
            </div>
            
            <div class="result-box redacted-text">
                <h4>🔒 Redacted Text:</h4>
                <div class="result-text" id="redacted-text"></div>
            </div>
            
            <div class="suggestions" id="suggestions-section" style="display: none;">
                <h4>💡 Alternative Suggestions:</h4>
                <ul id="suggestions-list"></ul>
            </div>
        </div>
        
        <div class="examples">
            <h3>📋 Try These Examples:</h3>
            <span class="example-button" onclick="setExample('You\\'re such a fucking idiot, kill yourself!')">Extreme Toxicity</span>
            <span class="example-button" onclick="setExample('SHUT UP YOU STUPID MORON!!!')">Caps + Insults</span>
            <span class="example-button" onclick="setExample('All you people are disgusting and pathetic')">Group Targeting</span>
            <span class="example-button" onclick="setExample('This is absolutely terrible, what a joke!')">Mild Criticism</span>
            <span class="example-button" onclick="setExample('Sure, that\\'s a brilliant idea... obviously')">Sarcasm</span>
            <span class="example-button" onclick="setExample('I disagree with your opinion respectfully')">Respectful</span>
        </div>
        
        <div class="footer">
            <p>Enhanced Toxicity Detection System v2.0 | Powered by Advanced NLP</p>
        </div>
    </div>
    
    <script>
        function setExample(text) {
            document.getElementById('input-text').value = text;
        }
        
        async function processText() {
            const inputText = document.getElementById('input-text').value.trim();
            const redactionLevel = document.getElementById('redaction-level').value;
            const redactionStyle = document.getElementById('redaction-style').value;
            
            if (!inputText) {
                alert('Please enter some text to analyze!');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('process-btn').disabled = true;
            
            try {
                const response = await fetch('/api/redact', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: inputText,
                        redaction_level: redactionLevel,
                        redaction_style: redactionStyle
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayResults(result.data);
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Failed to process text. Please try again.');
            } finally {
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                document.getElementById('process-btn').disabled = false;
            }
        }
        
        function displayResults(data) {
            // Show results section
            document.getElementById('results').style.display = 'block';
            
            // Display stats
            const statsContainer = document.getElementById('stats');
            statsContainer.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${(data.toxicity_score * 100).toFixed(1)}%</div>
                    <div class="stat-label">Toxicity Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${(data.confidence * 100).toFixed(1)}%</div>
                    <div class="stat-label">Confidence</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.redaction_level.toUpperCase()}</div>
                    <div class="stat-label">Redaction Level</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.redacted_words.length}</div>
                    <div class="stat-label">Words Redacted</div>
                </div>
            `;
            
            // Display original and redacted text
            document.getElementById('original-text').textContent = data.original_text;
            document.getElementById('redacted-text').textContent = data.redacted_text;
            
            // Display suggestions if available
            const suggestionsSection = document.getElementById('suggestions-section');
            const suggestionsList = document.getElementById('suggestions-list');
            
            if (data.suggestions && data.suggestions.length > 0) {
                suggestionsSection.style.display = 'block';
                suggestionsList.innerHTML = data.suggestions.map(suggestion => 
                    `<li>${suggestion}</li>`
                ).join('');
            } else {
                suggestionsSection.style.display = 'none';
            }
            
            // Scroll to results
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        }
        
        // Allow Enter key to process text
        document.getElementById('input-text').addEventListener('keydown', function(event) {
            if (event.ctrlKey && event.key === 'Enter') {
                processText();
            }
        });
    </script>
</body>
</html>
    """

@app.route('/api/redact', methods=['POST'])
def redact_api():
    """API endpoint for text redaction."""
    
    if not redaction_available or not redactor:
        return jsonify({
            'success': False,
            'error': 'Redaction system not available'
        })
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            })
        
        text = data['text']
        redaction_level = data.get('redaction_level', 'moderate')
        redaction_style = data.get('redaction_style', 'asterisks')
        
        # Convert string to enum
        try:
            style_enum = RedactionStyle(redaction_style)
        except ValueError:
            style_enum = RedactionStyle.ASTERISKS
        
        # Set threshold based on redaction level
        if redaction_level == 'minimal':
            redactor.set_redaction_threshold(0.7)
        elif redaction_level == 'moderate':
            redactor.set_redaction_threshold(0.5)
        elif redaction_level == 'aggressive':
            redactor.set_redaction_threshold(0.3)
        elif redaction_level == 'complete':
            redactor.set_redaction_threshold(0.1)
        
        # Perform redaction
        result = redactor.generate_redaction(text, style_enum)
        
        return jsonify({
            'success': True,
            'data': {
                'original_text': result.original_text,
                'redacted_text': result.redacted_text,
                'redaction_level': redaction_level,
                'redacted_words': result.redacted_words,
                'toxicity_score': result.toxicity_score,
                'confidence': result.confidence,
                'redaction_reason': result.redaction_reason,
                'suggestions': result.suggestions,
                'timestamp': result.timestamp
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/batch', methods=['POST'])
def batch_redact_api():
    """API endpoint for batch redaction."""
    
    if not redaction_available or not redactor:
        return jsonify({
            'success': False,
            'error': 'Redaction system not available'
        })
    
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'success': False,
                'error': 'No texts provided'
            })
        
        texts = data['texts']
        redaction_level = data.get('redaction_level', 'moderate')
        redaction_style = data.get('redaction_style', 'asterisks')
        
        # Convert string to enum
        try:
            style_enum = RedactionStyle(redaction_style)
        except ValueError:
            style_enum = RedactionStyle.ASTERISKS
        
        # Set threshold based on redaction level
        if redaction_level == 'minimal':
            redactor.set_redaction_threshold(0.7)
        elif redaction_level == 'moderate':
            redactor.set_redaction_threshold(0.5)
        elif redaction_level == 'aggressive':
            redactor.set_redaction_threshold(0.3)
        elif redaction_level == 'complete':
            redactor.set_redaction_threshold(0.1)
        
        # Perform batch redaction
        results = redactor.batch_redact(texts, style_enum)
        
        # Convert results to JSON-serializable format
        batch_data = []
        for result in results:
            batch_data.append({
                'original_text': result.original_text,
                'redacted_text': result.redacted_text,
                'redaction_level': redaction_level,
                'redacted_words': result.redacted_words,
                'toxicity_score': result.toxicity_score,
                'confidence': result.confidence,
                'redaction_reason': result.redaction_reason,
                'suggestions': result.suggestions,
                'timestamp': result.timestamp
            })
        
        # Generate batch statistics
        stats = redactor.get_redaction_stats(results)
        
        return jsonify({
            'success': True,
            'data': {
                'results': batch_data,
                'statistics': stats
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health_check():
    """Health check endpoint."""
    
    status = {
        'status': 'healthy',
        'redaction_available': redaction_available,
        'timestamp': datetime.now().isoformat()
    }
    
    if redaction_available and redactor:
        if hasattr(redactor, 'enhanced_available'):
            status['enhanced_detection'] = redactor.enhanced_available
        status['redaction_levels'] = ['minimal', 'moderate', 'aggressive', 'complete']
        status['redaction_styles'] = list(RedactionStyle.__members__.keys())
    
    return jsonify(status)

if __name__ == '__main__':
    print("🌐 Starting Smart Redaction Web Interface...")
    print("=" * 50)
    
    if redaction_available:
        print("✅ Redaction system loaded successfully")
        print("🔒 Enhanced detection available:", getattr(redactor, 'enhanced_available', False))
    else:
        print("⚠️ Redaction system not available - running in demo mode")
    
    print("\n🌍 Web interface will be available at:")
    print("   http://localhost:5000")
    print("\n📋 API Endpoints:")
    print("   POST /api/redact - Single text redaction")
    print("   POST /api/batch - Batch text redaction")
    print("   GET /health - Health check")
    
    print("\n🚀 Starting server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
