#!/usr/bin/env python3
"""
Interactive Redaction Demo Web Interface
=======================================

A simple web interface for testing the redaction system interactively.
Users can type messages and see real-time redaction results.
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


from flask import Flask, render_template_string, request, jsonify
import json
from datetime import datetime
from intelligent_redaction_system import IntelligentRedactor

app = Flask(__name__)

# Global redactor instance
redactor = None

def initialize_redactor():
    """Initialize the redaction system"""
    global redactor
    try:
        print("🔄 Initializing redaction system for demo...")
        redactor = IntelligentRedactor()
        if redactor.model:
            print("✅ Redaction system ready for demo!")
            return True
        else:
            print("❌ Model not loaded")
            return False
    except Exception as e:
        print(f"❌ Failed to initialize redactor: {e}")
        return False

@app.route('/')
def home():
    """Main demo page"""
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ Interactive Message Redaction Demo</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(45deg, #1e3c72, #2a5298);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .main-content {
            padding: 40px;
        }
        
        .input-section {
            margin-bottom: 30px;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        .message-input {
            width: 100%;
            padding: 15px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s ease;
            resize: vertical;
            min-height: 80px;
        }
        
        .message-input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .style-selector {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        
        .style-option {
            flex: 1;
            min-width: 120px;
        }
        
        .style-option input[type="radio"] {
            display: none;
        }
        
        .style-option label {
            display: block;
            padding: 12px 16px;
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        
        .style-option input[type="radio"]:checked + label {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        
        .test-button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        .test-button:hover {
            transform: translateY(-2px);
        }
        
        .test-button:active {
            transform: translateY(0);
        }
        
        .test-button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .results-section {
            margin-top: 30px;
            display: none;
        }
        
        .result-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 5px solid #667eea;
        }
        
        .result-card.toxic {
            border-left-color: #dc3545;
            background: #fff5f5;
        }
        
        .result-card.clean {
            border-left-color: #28a745;
            background: #f8fff8;
        }
        
        .result-header {
            display: flex;
            justify-content: between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .result-status {
            font-size: 1.2em;
            font-weight: 600;
        }
        
        .result-status.toxic {
            color: #dc3545;
        }
        
        .result-status.clean {
            color: #28a745;
        }
        
        .toxicity-score {
            background: #e9ecef;
            padding: 5px 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 0.9em;
        }
        
        .message-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 15px;
        }
        
        .message-box {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        
        .message-box h4 {
            margin-bottom: 10px;
            color: #495057;
        }
        
        .message-text {
            font-size: 16px;
            line-height: 1.4;
            word-wrap: break-word;
        }
        
        .original-text {
            color: #495057;
        }
        
        .redacted-text {
            color: #007bff;
            font-weight: 500;
        }
        
        .details {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #dee2e6;
        }
        
        .detail-item {
            margin-bottom: 8px;
            font-size: 14px;
        }
        
        .detail-label {
            font-weight: 600;
            color: #495057;
        }
        
        .detected-elements {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 5px;
        }
        
        .element-tag {
            background: #dc3545;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
        }
        
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            padding: 20px;
        }
        
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .examples {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .example-button {
            display: inline-block;
            padding: 8px 12px;
            margin: 5px;
            background: #e9ecef;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s ease;
        }
        
        .example-button:hover {
            background: #dee2e6;
        }
        
        .stats {
            margin-top: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #dee2e6;
        }
        
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 5px;
        }
        
        @media (max-width: 768px) {
            .message-comparison {
                grid-template-columns: 1fr;
            }
            
            .style-selector {
                flex-direction: column;
            }
            
            .style-option {
                min-width: auto;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Interactive Message Redaction Demo</h1>
            <p>Test the intelligent redaction system with your own messages</p>
        </div>
        
        <div class="main-content">
            <div class="input-section">
                <div class="input-group">
                    <label for="messageInput">Enter your message:</label>
                    <textarea 
                        id="messageInput" 
                        class="message-input" 
                        placeholder="Type any message here to test the redaction system... Try something like 'You're such an idiot!' or 'This is a great discussion!'"
                        rows="4"
                    ></textarea>
                </div>
                
                <div class="input-group">
                    <label>Redaction Style:</label>
                    <div class="style-selector">
                        <div class="style-option">
                            <input type="radio" id="smart" name="style" value="smart" checked>
                            <label for="smart">Smart</label>
                        </div>
                        <div class="style-option">
                            <input type="radio" id="partial" name="style" value="partial">
                            <label for="partial">Partial</label>
                        </div>
                        <div class="style-option">
                            <input type="radio" id="warning" name="style" value="warning">
                            <label for="warning">Warning</label>
                        </div>
                        <div class="style-option">
                            <input type="radio" id="complete" name="style" value="complete">
                            <label for="complete">Complete</label>
                        </div>
                    </div>
                </div>
                
                <button id="testButton" class="test-button" onclick="testMessage()">
                    🔍 Test Message
                </button>
            </div>
            
            <div id="resultsSection" class="results-section">
                <!-- Results will be populated here -->
            </div>
            
            <div class="examples">
                <h3>📝 Try These Examples:</h3>
                <button class="example-button" onclick="setExample('Hello everyone! How are you doing today?')">Friendly Message</button>
                <button class="example-button" onclick="setExample('You\\'re such an idiot, nobody cares!')">Insults</button>
                <button class="example-button" onclick="setExample('This is fucking stupid')">Profanity</button>
                <button class="example-button" onclick="setExample('Go kill yourself, loser!')">Threats</button>
                <button class="example-button" onclick="setExample('I hate you so much')">Hate Speech</button>
                <button class="example-button" onclick="setExample('Thanks for sharing your thoughts with us!')">Positive Message</button>
                <button class="example-button" onclick="setExample('This discussion is getting really frustrating')">Mild Negativity</button>
            </div>
        </div>
    </div>

    <script>
        let sessionStats = {
            totalTested: 0,
            toxicDetected: 0,
            messagesRedacted: 0
        };

        function setExample(text) {
            document.getElementById('messageInput').value = text;
            document.getElementById('messageInput').focus();
        }

        function getSelectedStyle() {
            const styleInputs = document.querySelectorAll('input[name="style"]');
            for (const input of styleInputs) {
                if (input.checked) {
                    return input.value;
                }
            }
            return 'smart';
        }

        async function testMessage() {
            const messageInput = document.getElementById('messageInput');
            const testButton = document.getElementById('testButton');
            const resultsSection = document.getElementById('resultsSection');
            
            const message = messageInput.value.trim();
            
            if (!message) {
                alert('Please enter a message to test!');
                return;
            }
            
            // Show loading state
            testButton.disabled = true;
            testButton.innerHTML = '<div class="spinner"></div> Testing...';
            resultsSection.style.display = 'block';
            resultsSection.innerHTML = '<div class="loading"><div class="spinner"></div> Analyzing message...</div>';
            
            try {
                const style = getSelectedStyle();
                
                // Call the redaction API
                const response = await fetch('/test', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: message,
                        style: style
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const result = await response.json();
                
                // Update session stats
                sessionStats.totalTested++;
                if (result.was_redacted) {
                    sessionStats.toxicDetected++;
                    sessionStats.messagesRedacted++;
                }
                
                // Display results
                displayResults(result);
                
            } catch (error) {
                resultsSection.innerHTML = `
                    <div class="result-card" style="border-left-color: #dc3545;">
                        <h3 style="color: #dc3545;">❌ Error</h3>
                        <p>Failed to test message: ${error.message}</p>
                        <p style="font-size: 0.9em; color: #6c757d; margin-top: 10px;">
                            Make sure the redaction system is properly initialized.
                        </p>
                    </div>
                `;
            } finally {
                // Reset button
                testButton.disabled = false;
                testButton.innerHTML = '🔍 Test Message';
            }
        }

        function displayResults(result) {
            const resultsSection = document.getElementById('resultsSection');
            
            const statusClass = result.was_redacted ? 'toxic' : 'clean';
            const statusIcon = result.was_redacted ? '🚨' : '✅';
            const statusText = result.was_redacted ? 'TOXIC CONTENT DETECTED' : 'CLEAN MESSAGE';
            
            const detectedElementsHtml = result.redacted_elements && result.redacted_elements.length > 0 
                ? `<div class="detected-elements">
                     ${result.redacted_elements.map(elem => `<span class="element-tag">${elem}</span>`).join('')}
                   </div>`
                : '';
            
            const html = `
                <div class="result-card ${statusClass}">
                    <div class="result-header">
                        <div class="result-status ${statusClass}">
                            ${statusIcon} ${statusText}
                        </div>
                        <div class="toxicity-score">
                            Toxicity: ${(result.toxicity_probability * 100).toFixed(1)}%
                        </div>
                    </div>
                    
                    <div class="message-comparison">
                        <div class="message-box">
                            <h4>📝 Original Message</h4>
                            <div class="message-text original-text">${escapeHtml(result.original_text)}</div>
                        </div>
                        <div class="message-box">
                            <h4>🛡️ ${result.was_redacted ? 'Redacted' : 'Approved'} Message</h4>
                            <div class="message-text redacted-text">${escapeHtml(result.redacted_text)}</div>
                        </div>
                    </div>
                    
                    <div class="details">
                        <div class="detail-item">
                            <span class="detail-label">Redaction Style:</span> ${result.redaction_style}
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Confidence Level:</span> ${result.confidence}
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Processing Time:</span> ${result.processing_time_ms.toFixed(2)}ms
                        </div>
                        ${result.was_redacted ? `
                        <div class="detail-item">
                            <span class="detail-label">Detected Issues:</span>
                            ${detectedElementsHtml}
                        </div>
                        ` : ''}
                        <div class="detail-item">
                            <span class="detail-label">Timestamp:</span> ${new Date(result.timestamp).toLocaleString()}
                        </div>
                    </div>
                </div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value">${sessionStats.totalTested}</div>
                        <div class="stat-label">Messages Tested</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${sessionStats.toxicDetected}</div>
                        <div class="stat-label">Toxic Detected</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${sessionStats.messagesRedacted}</div>
                        <div class="stat-label">Messages Redacted</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${sessionStats.totalTested > 0 ? ((sessionStats.toxicDetected / sessionStats.totalTested) * 100).toFixed(1) : 0}%</div>
                        <div class="stat-label">Toxicity Rate</div>
                    </div>
                </div>
            `;
            
            resultsSection.innerHTML = html;
            resultsSection.style.display = 'block';
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Allow Enter key to submit
        document.addEventListener('DOMContentLoaded', function() {
            const messageInput = document.getElementById('messageInput');
            messageInput.addEventListener('keydown', function(event) {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    testMessage();
                }
            });
            
            // Focus on input
            messageInput.focus();
        });
    </script>
</body>
</html>
    """
    
    return render_template_string(html_template)

@app.route('/test', methods=['POST'])
def test_message():
    """Test endpoint for message redaction"""
    if not redactor:
        return jsonify({"error": "Redaction system not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data['text']
        style = data.get('style', 'smart')
        
        # Record start time for performance measurement
        start_time = datetime.now()
        
        # Test the message
        result = redactor.redact_message(text, style)
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Format response
        response = {
            "original_text": result['original_text'],
            "redacted_text": result['redacted_text'],
            "was_redacted": result['was_redacted'],
            "redaction_style": result['redaction_style'],
            "toxicity_probability": result['toxicity_info']['probability'],
            "confidence": result['toxicity_info']['confidence'],
            "redacted_elements": result.get('redacted_elements', []),
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def get_status():
    """Get system status"""
    return jsonify({
        "status": "ready" if redactor and redactor.model else "not_ready",
        "model_loaded": redactor is not None and redactor.model is not None,
        "timestamp": datetime.now().isoformat()
    })

def run_interactive_demo():
    """Run the interactive demo server"""
    print("🚀 STARTING INTERACTIVE REDACTION DEMO")
    print("=" * 50)
    
    # Initialize the redaction system
    if not initialize_redactor():
        print("❌ Cannot start demo without redaction system")
        return
    
    print(f"✅ Demo server ready!")
    print(f"🌐 Open your browser and go to: http://localhost:8080")
    print(f"💡 Type messages and press Enter to test redaction")
    print(f"🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        app.run(host='localhost', port=8080, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    run_interactive_demo()
