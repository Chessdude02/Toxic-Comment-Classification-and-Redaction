"""
Flask Web Application for Real-Time Toxicity Detection and Redaction

This web application demonstrates the toxicity classification and redaction system
with a simple web interface for testing messages in real-time.

Features:
- Real-time message toxicity checking
- Multiple redaction styles
- Chat simulation interface
- API endpoints for integration
- Message history and statistics

To run:
    python toxicity_web_app.py

Requirements:
    pip install flask flask-cors
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import logging
from datetime import datetime
from typing import Dict, List
import os
import sys

# heuristic_fallback also provides ChatModerator (a copy of the one in
# toxicity_redactor.py, deliberately NOT imported from there - see the
# module docstring in heuristic_fallback.py: importing anything from
# toxicity_redactor.py drags in `import tensorflow` at ~600MB RSS, which
# alone is enough to OOM a 512MB deploy target before serving a request).
# We only pay that cost if a trained model actually looks present on disk -
# see initialize_models() below.
try:
    from heuristic_fallback import HeuristicRedactor, ChatModerator
except ImportError as _e:
    print(f"❌ Could not import heuristic_fallback module ({_e}). Make sure it's in the same directory.")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
redactor = None
chat_moderator = None
# One of: 'trained' (real model loaded), 'heuristic' (rule-based fallback,
# no trained model), 'unavailable' (neither could be initialized).
redactor_mode = 'unavailable'

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Real-time toxic comment detection and redaction — test messages, simulate chat moderation, and inspect live statistics.">
    <meta name="color-scheme" content="light dark">
    <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%9B%A1%EF%B8%8F%3C/text%3E%3C/svg%3E">
    <title>Toxic Comment Detection & Redaction System</title>
    <style>
        :root {
            --bg-gradient-start: #667eea;
            --bg-gradient-end: #764ba2;
            --surface: #ffffff;
            --surface-alt: #f7f8fc;
            --text: #24262b;
            --text-muted: #666a73;
            --border: #e1e5e9;
            --accent: #667eea;
            --accent-2: #764ba2;
            --toxic-bg: #ffe6e6;
            --toxic-border: #ff6b6b;
            --toxic-text: #c92a2a;
            --clean-bg: #e6ffe6;
            --clean-border: #51cf66;
            --clean-text: #2b8a3e;
            --warn-bg: #fff3cd;
            --warn-border: #ffc107;
            --warn-text: #856404;
            --shadow: 0 15px 35px rgba(0,0,0,0.12);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --surface: #1b1d22;
                --surface-alt: #24262c;
                --text: #eef0f4;
                --text-muted: #a3a7b2;
                --border: #34363d;
                --toxic-bg: #3a1f22;
                --toxic-border: #ff6b6b;
                --toxic-text: #ff9b9b;
                --clean-bg: #1c3324;
                --clean-border: #51cf66;
                --clean-text: #7ee2a0;
                --warn-bg: #3a3319;
                --warn-border: #ffc107;
                --warn-text: #ffd873;
                --shadow: 0 15px 35px rgba(0,0,0,0.4);
            }
        }

        * { box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
            min-height: 100vh;
            color: var(--text);
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: var(--surface);
            border-radius: 15px;
            box-shadow: var(--shadow);
            padding: 30px;
        }

        h1 {
            text-align: center;
            color: var(--text);
            margin-bottom: 12px;
            font-size: clamp(1.6em, 4vw, 2.5em);
        }

        .subtitle {
            text-align: center;
            color: var(--text-muted);
            margin-bottom: 24px;
            font-size: 1.1em;
        }

        .status-banner {
            display: flex;
            align-items: center;
            gap: 10px;
            justify-content: center;
            margin: 0 auto 30px;
            padding: 10px 16px;
            border-radius: 8px;
            font-size: 0.9em;
            max-width: fit-content;
            background: var(--warn-bg);
            border: 1px solid var(--warn-border);
            color: var(--warn-text);
        }

        .status-banner.ok {
            background: var(--clean-bg);
            border-color: var(--clean-border);
            color: var(--clean-text);
        }

        .section {
            margin-bottom: 30px;
            padding: 20px;
            border-radius: 10px;
            background: var(--surface-alt);
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }

        .test-section {
            border-left: 4px solid var(--accent);
        }

        .chat-section {
            border-left: 4px solid #ff6b6b;
        }

        .stats-section {
            border-left: 4px solid #51cf66;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: var(--text);
        }

        textarea, input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            background: var(--surface);
            color: var(--text);
            transition: border-color 0.2s ease;
            font-family: inherit;
        }

        textarea:focus, input:focus, select:focus {
            outline: 3px solid transparent;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.25);
        }

        button {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(102, 126, 234, 0.35);
        }

        button:focus-visible {
            outline: 3px solid var(--accent-2);
            outline-offset: 2px;
        }

        button:disabled {
            cursor: progress;
            opacity: 0.75;
        }

        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            animation: fade-in 0.2s ease;
        }

        @keyframes fade-in {
            from { opacity: 0; transform: translateY(-4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .toxic {
            background: var(--toxic-bg);
            border-left: 4px solid var(--toxic-border);
            color: var(--toxic-text);
        }

        .clean {
            background: var(--clean-bg);
            border-left: 4px solid var(--clean-border);
            color: var(--clean-text);
        }

        .warning {
            background: var(--warn-bg);
            border-left: 4px solid var(--warn-border);
            color: var(--warn-text);
        }

        .chat-log {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 15px;
            background: var(--surface);
        }

        .message {
            margin-bottom: 10px;
            padding: 8px 12px;
            border-radius: 6px;
        }

        .message.clean {
            background: var(--clean-bg);
        }

        .message.toxic {
            background: var(--toxic-bg);
        }

        .message.moderated {
            background: var(--warn-bg);
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .stat-box {
            background: var(--surface);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 2px solid var(--border);
        }

        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: var(--accent);
        }

        .stat-label {
            color: var(--text-muted);
            font-size: 0.9em;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }

        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
            body {
                padding: 10px;
            }
            .container {
                padding: 18px;
                border-radius: 10px;
            }
        }

        .emoji {
            font-size: 1.2em;
        }

        .loading {
            opacity: 0.7;
            pointer-events: none;
        }

        .hidden {
            display: none;
        }

        footer {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85em;
            margin-top: 10px;
        }

        footer a {
            color: var(--accent);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Toxic Comment Detection & Redaction System</h1>
        <p class="subtitle">Real-time AI-powered toxicity detection with multi-class classification and intelligent redaction</p>
        <div id="modelStatusBanner" class="status-banner hidden" role="status" aria-live="polite"></div>

        <div class="grid">
            <!-- Message Testing Section -->
            <div class="section test-section">
                <h2>🧪 Test Message Toxicity</h2>
                <div class="form-group">
                    <label for="testMessage">Enter a message to test:</label>
                    <textarea id="testMessage" rows="4" placeholder="Type your message here to check for toxicity..."></textarea>
                </div>
                
                <div class="form-group">
                    <label for="redactionStyle">Redaction Style:</label>
                    <select id="redactionStyle">
                        <option value="warning">Warning (show original + warning)</option>
                        <option value="partial">Partial (replace toxic words)</option>
                        <option value="complete">Complete (remove entire message)</option>
                        <option value="asterisk">Asterisk (replace with ***)</option>
                    </select>
                </div>
                
                <button onclick="testMessage()" id="testBtn">🔍 Analyze Message</button>
                
                <div id="testResult"></div>
            </div>
            
            <!-- Chat Simulation Section -->
            <div class="section chat-section">
                <h2>💬 Chat Simulation</h2>
                <div class="form-group">
                    <label for="username">Username:</label>
                    <input type="text" id="username" value="TestUser" placeholder="Enter your username">
                </div>
                
                <div class="form-group">
                    <label for="chatMessage">Chat Message:</label>
                    <textarea id="chatMessage" rows="3" placeholder="Type a chat message..."></textarea>
                </div>
                
                <button onclick="sendChatMessage()" id="chatBtn">📤 Send Message</button>
                <button onclick="clearChat()" style="background: #ff6b6b; margin-left: 10px;">🗑️ Clear Chat</button>
                
                <h3>Chat Log:</h3>
                <div id="chatLog" class="chat-log">
                    <p style="text-align: center; color: #999;">No messages yet. Send a message to start chatting!</p>
                </div>
            </div>
        </div>
        
        <!-- Statistics Section -->
        <div class="section stats-section">
            <h2>📊 Moderation Statistics</h2>
            <div class="stats" id="statsContainer">
                <div class="stat-box">
                    <div class="stat-number" id="totalMessages">0</div>
                    <div class="stat-label">Total Messages</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" id="toxicMessages">0</div>
                    <div class="stat-label">Toxic Messages</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" id="cleanMessages">0</div>
                    <div class="stat-label">Clean Messages</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" id="toxicityRate">0%</div>
                    <div class="stat-label">Toxicity Rate</div>
                </div>
            </div>
            
            <button onclick="refreshStats()" style="margin-top: 15px;">🔄 Refresh Statistics</button>
        </div>
        
        <!-- API Information -->
        <div class="section">
            <h2>🔗 API Endpoints</h2>
            <p><strong>POST /api/check-toxicity</strong> - Check if a message is toxic</p>
            <p><strong>POST /api/moderate-message</strong> - Moderate a single message</p>
            <p><strong>POST /api/moderate-batch</strong> - Moderate multiple messages</p>
            <p><strong>GET /api/stats</strong> - Get moderation statistics</p>
            <p><strong>GET /api/chat-log</strong> - Get recent chat messages</p>
        </div>

        <footer>
            Built during the BISAG-N internship project · see the repository's root README.md
            for architecture, training, and evaluation details.
        </footer>
    </div>

    <script>
        // Test message toxicity
        async function testMessage() {
            const message = document.getElementById('testMessage').value;
            const style = document.getElementById('redactionStyle').value;
            const btn = document.getElementById('testBtn');
            
            if (!message.trim()) {
                alert('Please enter a message to test!');
                return;
            }
            
            btn.disabled = true;
            btn.textContent = '🔄 Analyzing...';
            
            try {
                const response = await fetch('/api/check-toxicity', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: message,
                        redaction_style: style
                    })
                });
                
                const result = await response.json();
                displayTestResult(result);
            } catch (error) {
                document.getElementById('testResult').innerHTML = 
                    '<div class="result warning">❌ Error: ' + error.message + '</div>';
            }
            
            btn.disabled = false;
            btn.textContent = '🔍 Analyze Message';
        }
        
        function displayTestResult(result) {
            const resultDiv = document.getElementById('testResult');
            
            if (!result.success) {
                resultDiv.innerHTML = '<div class="result warning">❌ Error: ' + result.error + '</div>';
                return;
            }
            
            const cssClass = result.is_toxic ? 'toxic' : 'clean';
            const icon = result.is_toxic ? '🚨' : '✅';
            const status = result.is_toxic ? 'TOXIC CONTENT DETECTED' : 'CLEAN CONTENT';
            
            let html = `
                <div class="result ${cssClass}">
                    <h3>${icon} ${status}</h3>
                    <p><strong>Original:</strong> "${result.original_message}"</p>
                    <p><strong>Moderated:</strong> "${result.moderated_message}"</p>
                    <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
            `;
            
            if (result.toxic_types && result.toxic_types.length > 0) {
                html += `<p><strong>Toxic Types:</strong> ${result.toxic_types.join(', ')}</p>`;
            }
            
            if (result.was_redacted) {
                html += `<p><strong>Redaction Applied:</strong> ${result.redaction_style}</p>`;
            }

            if (result.mode === 'heuristic') {
                html += `<p><em>Scored by the rule-based heuristic fallback, not a trained model.</em></p>`;
            }

            html += '</div>';
            resultDiv.innerHTML = html;
        }
        
        // Send chat message
        async function sendChatMessage() {
            const username = document.getElementById('username').value;
            const message = document.getElementById('chatMessage').value;
            const btn = document.getElementById('chatBtn');
            
            if (!username.trim() || !message.trim()) {
                alert('Please enter both username and message!');
                return;
            }
            
            btn.disabled = true;
            btn.textContent = '📤 Sending...';
            
            try {
                const response = await fetch('/api/chat/send', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        username: username,
                        message: message
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('chatMessage').value = '';
                    updateChatLog();
                    updateStats();
                }
            } catch (error) {
                console.error('Error sending message:', error);
            }
            
            btn.disabled = false;
            btn.textContent = '📤 Send Message';
        }
        
        // Update chat log
        async function updateChatLog() {
            try {
                const response = await fetch('/api/chat-log');
                const data = await response.json();
                
                const chatLog = document.getElementById('chatLog');
                
                if (data.messages && data.messages.length > 0) {
                    let html = '';
                    data.messages.forEach(msg => {
                        const statusClass = msg.status.includes('CLEAN') ? 'clean' : 
                                          msg.status.includes('TOXIC') ? 'toxic' : 'moderated';
                        const statusIcon = msg.status.split(' ')[0];
                        
                        html += `
                            <div class="message ${statusClass}">
                                <strong>[${msg.timestamp}] ${statusIcon} ${msg.username}:</strong><br>
                                ${msg.display_message}
                            </div>
                        `;
                    });
                    chatLog.innerHTML = html;
                } else {
                    chatLog.innerHTML = '<p style="text-align: center; color: #999;">No messages yet.</p>';
                }
                
                // Scroll to bottom
                chatLog.scrollTop = chatLog.scrollHeight;
            } catch (error) {
                console.error('Error updating chat log:', error);
            }
        }
        
        // Clear chat
        async function clearChat() {
            if (confirm('Are you sure you want to clear the chat log?')) {
                try {
                    await fetch('/api/chat/clear', {method: 'POST'});
                    updateChatLog();
                    updateStats();
                } catch (error) {
                    console.error('Error clearing chat:', error);
                }
            }
        }
        
        // Update statistics
        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();

                document.getElementById('totalMessages').textContent = stats.total_messages || 0;
                document.getElementById('toxicMessages').textContent = stats.toxic_messages || 0;
                document.getElementById('cleanMessages').textContent = stats.clean_messages || 0;
                document.getElementById('toxicityRate').textContent =
                    ((stats.toxicity_rate || 0) * 100).toFixed(1) + '%';

                updateModelStatusBanner(stats.mode);
            } catch (error) {
                console.error('Error updating stats:', error);
            }
        }

        // Reflect which of three states is actually backing predictions,
        // since the API returns 200s in both the trained and heuristic
        // cases - only the banner (and each result's "mode" field) tells
        // you which one you're looking at.
        function updateModelStatusBanner(mode) {
            const banner = document.getElementById('modelStatusBanner');
            banner.classList.remove('hidden', 'ok');
            if (mode === 'trained') {
                banner.classList.add('ok');
                banner.innerHTML = '✅ Trained model loaded — live predictions are active.';
            } else if (mode === 'heuristic') {
                banner.innerHTML = '⚠️ No trained model loaded — running on the rule-based heuristic ' +
                    'fallback (upgrade2), not a trained model. Predictions are real but limited to ' +
                    'keyword/pattern matching.';
            } else {
                banner.innerHTML = '❌ No trained model or heuristic fallback available — ' +
                    'toxicity-check endpoints will return HTTP 503.';
            }
        }
        
        // Refresh statistics
        async function refreshStats() {
            await updateStats();
        }
        
        // Enter key handlers
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('testMessage').addEventListener('keydown', function(event) {
                if (event.ctrlKey && event.key === 'Enter') {
                    testMessage();
                }
            });
            
            document.getElementById('chatMessage').addEventListener('keydown', function(event) {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    sendChatMessage();
                }
            });
            
            // Initial load
            updateStats();
        });
    </script>
</body>
</html>
"""


def _trained_model_files_present() -> bool:
    """Cheap on-disk check, done BEFORE importing toxicity_redactor.

    That module does `import tensorflow` at module level, which costs
    ~600MB of RSS on its own - more than the entire memory budget on a
    typical free-tier deploy (Render/Railway's 512MB, confirmed by an actual
    OOM kill in practice). There is no point paying that cost if nobody has
    actually provided a trained model to load.
    """
    return os.path.exists(os.path.join('saved_models', 'config.pickle'))


def initialize_models():
    """Initialize the toxicity detection models.

    Tries the trained model first, but only imports toxicity_redactor (and
    therefore TensorFlow) if trained model files actually appear to be
    present - see _trained_model_files_present(). Otherwise falls straight to
    the rule-based heuristic detector, which needs neither TensorFlow nor a
    trained model, instead of leaving the app unable to respond to anything.
    Both paths implement the same classify_toxicity()/redact_message()
    interface, so ChatModerator and every API route below work unchanged
    either way.
    """
    global redactor, chat_moderator, redactor_mode

    if _trained_model_files_present():
        try:
            from toxicity_redactor import load_pretrained_model
            redactor = load_pretrained_model()

            if redactor is not None:
                chat_moderator = ChatModerator(redactor, auto_moderate=True)
                redactor_mode = 'trained'
                logger.info("✅ Trained model loaded successfully!")
                return True

            logger.warning("Trained model files were present but failed to load.")

        except Exception as e:
            logger.error(f"❌ Error loading trained model: {str(e)}")
    else:
        logger.info("No trained model files found - skipping the TensorFlow-heavy import entirely.")

    try:
        redactor = HeuristicRedactor()
        chat_moderator = ChatModerator(redactor, auto_moderate=True)
        redactor_mode = 'heuristic'
        logger.warning(
            "⚠️  No trained model available - falling back to the rule-based "
            "heuristic detector (upgrade2). This is NOT a trained model; see "
            "upgrade2/README.md."
        )
        return True
    except Exception as e:
        logger.error(f"❌ Error initializing heuristic fallback: {str(e)}")

    redactor = None
    chat_moderator = None
    redactor_mode = 'unavailable'
    logger.error("❌ Neither a trained model nor the heuristic fallback could be initialized.")
    return False


# Run at import time (not just under `if __name__ == '__main__'`) so a
# production WSGI server (gunicorn toxicity_web_app:app) initializes the
# model the same way `python toxicity_web_app.py` does - the __main__ guard
# alone would never run under gunicorn, since it imports this module rather
# than executing it as a script.
_models_ready = initialize_models()


@app.route('/')
def index():
    """Main page with the web interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/check-toxicity', methods=['POST'])
def check_toxicity_api():
    """
    API endpoint to check message toxicity.
    
    Expected JSON: {
        "message": "text to check",
        "redaction_style": "warning"  // optional
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'success': False, 'error': 'Message is required'}), 400
        
        message = data['message']
        redaction_style = data.get('redaction_style', 'warning')
        
        if redactor is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded',
                'message': message
            }), 503
        
        # Get classification and redaction
        classification = redactor.classify_toxicity(message)
        redaction_result = redactor.redact_message(message, redaction_style)
        
        return jsonify({
            'success': True,
            'original_message': message,
            'is_toxic': classification['is_toxic'],
            'toxic_types': classification['toxic_types'],
            'confidence': classification['max_toxicity_probability'],
            'moderated_message': redaction_result['redacted_message'],
            'was_redacted': redaction_result['was_redacted'],
            'redaction_style': redaction_style,
            'detailed_results': classification['detailed_results'],
            'mode': redactor_mode,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in check_toxicity_api: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/moderate-message', methods=['POST'])
def moderate_message_api():
    """
    API endpoint to moderate a single message.
    
    Expected JSON: {
        "message": "text to moderate",
        "style": "warning"  // optional
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'success': False, 'error': 'Message is required'}), 400
        
        message = data['message']
        style = data.get('style', 'warning')
        
        if redactor is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 503
        
        result = redactor.redact_message(message, style)
        
        return jsonify({
            'success': True,
            'original_message': message,
            'moderated_message': result['redacted_message'],
            'was_redacted': result['was_redacted'],
            'toxicity_info': result['toxicity_info'],
            'redaction_reason': result.get('redaction_reason', '')
        })
        
    except Exception as e:
        logger.error(f"Error in moderate_message_api: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/moderate-batch', methods=['POST'])
def moderate_batch_api():
    """
    API endpoint to moderate multiple messages.
    
    Expected JSON: {
        "messages": ["message1", "message2"],
        "style": "warning"  // optional
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'messages' not in data:
            return jsonify({'success': False, 'error': 'Messages list is required'}), 400
        
        messages = data['messages']
        style = data.get('style', 'warning')
        
        if redactor is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 503
        
        results = redactor.moderate_conversation(messages, style)
        
        return jsonify({
            'success': True,
            'results': results,
            'total_messages': len(messages),
            'toxic_count': sum(1 for r in results if r['was_redacted'])
        })
        
    except Exception as e:
        logger.error(f"Error in moderate_batch_api: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat/send', methods=['POST'])
def send_chat_message_api():
    """
    API endpoint to send a chat message through the moderation system.
    
    Expected JSON: {
        "username": "user123",
        "message": "chat message"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'username' not in data or 'message' not in data:
            return jsonify({'success': False, 'error': 'Username and message are required'}), 400
        
        username = data['username']
        message = data['message']
        
        if chat_moderator is None:
            return jsonify({'success': False, 'error': 'Chat moderator not initialized'}), 503
        
        # Process the message
        result = chat_moderator.process_message(username, message)
        
        return jsonify({
            'success': True,
            'chat_entry': result
        })
        
    except Exception as e:
        logger.error(f"Error in send_chat_message_api: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat-log', methods=['GET'])
def get_chat_log_api():
    """Get recent chat messages."""
    try:
        if chat_moderator is None:
            return jsonify({'success': False, 'error': 'Chat moderator not initialized'}), 503
        
        last_n = request.args.get('last_n', 50, type=int)
        messages = chat_moderator.get_chat_log(last_n)
        
        return jsonify({
            'success': True,
            'messages': messages,
            'count': len(messages)
        })
        
    except Exception as e:
        logger.error(f"Error in get_chat_log_api: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat/clear', methods=['POST'])
def clear_chat_api():
    """Clear chat log."""
    try:
        if chat_moderator is None:
            return jsonify({'success': False, 'error': 'Chat moderator not initialized'}), 503
        
        chat_moderator.message_history = []
        chat_moderator.moderation_stats = {
            'total_messages': 0,
            'toxic_messages': 0,
            'redacted_messages': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        return jsonify({'success': True, 'message': 'Chat log cleared'})
        
    except Exception as e:
        logger.error(f"Error in clear_chat_api: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats_api():
    """Get moderation statistics."""
    try:
        if chat_moderator is None:
            return jsonify({
                'total_messages': 0,
                'toxic_messages': 0,
                'clean_messages': 0,
                'toxicity_rate': 0,
                'redacted_messages': 0,
                'mode': redactor_mode,
                'model_loaded': redactor_mode == 'trained'
            })

        stats = chat_moderator.get_moderation_stats()
        stats['clean_messages'] = stats['total_messages'] - stats['toxic_messages']
        stats['mode'] = redactor_mode
        stats['model_loaded'] = redactor_mode == 'trained'

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error in get_stats_api: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'mode': redactor_mode,
        'model_loaded': redactor_mode == 'trained',
        'chat_moderator_loaded': chat_moderator is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


def create_sample_data():
    """Create sample chat data for demonstration."""
    if chat_moderator is None:
        return
    
    sample_messages = [
        ("Alice", "Hello everyone! Welcome to our chat!"),
        ("Bob", "Thanks Alice! Great to be here."),
        ("Charlie", "This platform looks amazing."),
        ("Dave", "I think this is stupid and pointless."),
        ("Alice", "Let's try to keep things positive!"),
        ("Eve", "I agree with Alice, constructive feedback is better."),
        ("Frank", "You people are all idiots if you think this works."),
        ("Bob", "Thanks for sharing your thoughts, everyone.")
    ]
    
    logger.info("Creating sample chat data...")
    for username, message in sample_messages:
        chat_moderator.process_message(username, message)
    
    logger.info(f"✅ Created {len(sample_messages)} sample messages")


if __name__ == '__main__':
    print("🚀 Starting Toxicity Detection Web Application")
    print("=" * 60)

    # Models were already initialized at import time (see _models_ready
    # above) so gunicorn and `python toxicity_web_app.py` behave the same.
    if redactor_mode == 'trained':
        create_sample_data()
    elif redactor_mode == 'heuristic':
        print("⚠️ No trained model found - running on the rule-based heuristic fallback")
        print("Train and save a model using the notebook for the trained classifier instead")
    else:
        print("❌ Neither a trained model nor the heuristic fallback could be initialized")

    print("\n🌐 Starting Flask server...")
    print(f"📱 Open your browser and go to: http://localhost:{os.environ.get('PORT', 5000)}")
    print("\n🔗 Available API endpoints:")
    print("  POST /api/check-toxicity - Check message toxicity")
    print("  POST /api/moderate-message - Moderate a message")
    print("  POST /api/moderate-batch - Moderate multiple messages")
    print("  POST /api/chat/send - Send chat message")
    print("  GET  /api/chat-log - Get chat history")
    print("  GET  /api/stats - Get statistics")
    print("  GET  /api/health - Health check")
    
    print("\n💡 Usage tips:")
    print("  • Test individual messages in the left panel")
    print("  • Simulate chat conversations in the right panel")
    print("  • Monitor statistics in the bottom section")
    print("  • Use the API endpoints for integration")
    
    print("\n🛠️ Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Run the Flask app. Cloud platforms (Render, Railway, Fly.io, etc.)
    # inject the port to bind via $PORT rather than a fixed value, and
    # debug=True would expose Werkzeug's interactive debugger (arbitrary
    # code execution) to the internet - never enable it outside local dev.
    # For real deployments, prefer running via gunicorn (see Dockerfile)
    # rather than this development server.
    try:
        app.run(
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            debug=os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true'),
            use_reloader=False  # Disable reloader to prevent model reloading
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {str(e)}")
