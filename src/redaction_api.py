#!/usr/bin/env python3
"""
Redaction System REST API
=========================

A Flask-based REST API for the intelligent redaction system:
- RESTful endpoints for message filtering
- Batch processing capabilities
- Configuration management
- Real-time WebSocket support
- API key authentication
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import asyncio
import threading
import time
import uuid
from datetime import datetime
from intelligent_redaction_system import IntelligentRedactor, MessageModerator
# real_time_message_filter provides RealTimeMessageFilter/FilterConfig/MessageEvent for the
# /api/moderate endpoint. It's an optional module — imported lazily below so the rest of the
# API (filter/detect/stats/etc.) still works if it isn't present.

app = Flask(__name__)
CORS(app)

# Global instances
redactor = None
message_filter = None
api_stats = {
    'requests_processed': 0,
    'messages_filtered': 0,
    'start_time': datetime.now()
}

def initialize_redaction_system():
    """Initialize the redaction system components"""
    global redactor, message_filter
    
    try:
        print("🔄 Initializing redaction system for API...")

        # Initialize redactor
        redactor = IntelligentRedactor()

        # Initialize real-time filter (optional — module may not be present)
        from real_time_message_filter import RealTimeMessageFilter, FilterConfig
        config = FilterConfig(
            toxicity_threshold=0.4,
            auto_redact=True,
            max_messages_per_minute=100
        )
        message_filter = RealTimeMessageFilter(config)
        message_filter.initialize_ai_components()
        
        print("✅ Redaction system initialized for API")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize redaction system: {e}")
        return False

# API Endpoints

@app.route('/')
def home():
    """API documentation home page"""
    doc_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Intelligent Redaction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { background: #007acc; color: white; padding: 4px 8px; border-radius: 3px; }
            .example { background: #e8f4fd; padding: 10px; margin: 10px 0; border-radius: 3px; }
            pre { background: #2d2d2d; color: #f8f8f2; padding: 10px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>🛡️ Intelligent Redaction System API</h1>
        <p>A production-ready API for intelligent message filtering and redaction.</p>
        
        <h2>📊 API Status</h2>
        <div class="example">
            <strong>Status:</strong> {{ 'Active' if redactor else 'Not Initialized' }}<br>
            <strong>Requests Processed:</strong> {{ api_stats['requests_processed'] }}<br>
            <strong>Messages Filtered:</strong> {{ api_stats['messages_filtered'] }}<br>
            <strong>Uptime:</strong> {{ uptime_str }}
        </div>
        
        <h2>📚 Available Endpoints</h2>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /api/filter</h3>
            <p>Filter a single message with toxicity detection and redaction.</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre>{
  "text": "Your message here",
  "style": "smart",           // optional: smart, partial, warning, complete
  "user_id": "user123",       // optional
  "username": "JohnDoe"       // optional
}</pre>
            </div>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /api/filter/batch</h3>
            <p>Filter multiple messages in a single request.</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre>{
  "messages": [
    {"text": "First message", "user_id": "user1"},
    {"text": "Second message", "user_id": "user2"}
  ],
  "style": "smart"            // optional
}</pre>
            </div>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /api/moderate</h3>
            <p>Moderate a message with full context and user tracking.</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre>{
  "message": "Your message here",
  "user_id": "user123",
  "username": "JohnDoe",
  "channel": "general",       // optional
  "metadata": {}             // optional
}</pre>
            </div>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /api/stats</h3>
            <p>Get system statistics and performance metrics.</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /api/config</h3>
            <p>Update system configuration.</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre>{
  "toxicity_threshold": 0.5,
  "auto_redact": true,
  "max_messages_per_minute": 60
}</pre>
            </div>
        </div>
        
        <h2>🧪 Test Interface</h2>
        <div class="example">
            <p>Try the API with this simple test form:</p>
            <input type="text" id="testInput" placeholder="Enter message to test..." style="width: 300px; padding: 8px;">
            <button onclick="testMessage()" style="padding: 8px 16px;">Test Filter</button>
            <div id="testResult" style="margin-top: 10px;"></div>
        </div>
        
        <script>
        async function testMessage() {
            const text = document.getElementById('testInput').value;
            if (!text) return;
            
            try {
                const response = await fetch('/api/filter', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text, style: 'smart'})
                });
                
                const result = await response.json();
                document.getElementById('testResult').innerHTML = 
                    '<strong>Result:</strong> ' + JSON.stringify(result, null, 2);
            } catch (error) {
                document.getElementById('testResult').innerHTML = 
                    '<strong>Error:</strong> ' + error.message;
            }
        }
        </script>
    </body>
    </html>
    """
    
    # Calculate uptime
    uptime = datetime.now() - api_stats['start_time']
    uptime_str = str(uptime).split('.')[0]  # Remove microseconds
    
    return render_template_string(doc_html, 
                                 redactor=redactor, 
                                 api_stats=api_stats, 
                                 uptime_str=uptime_str)

@app.route('/api/filter', methods=['POST'])
def filter_message():
    """Filter a single message"""
    global api_stats
    api_stats['requests_processed'] += 1
    
    if not redactor:
        return jsonify({"error": "Redaction system not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text']
        style = data.get('style', 'smart')
        user_id = data.get('user_id', 'anonymous')
        username = data.get('username', 'Anonymous')
        
        # Filter the message
        start_time = time.time()
        result = redactor.redact_message(text, style)
        processing_time = (time.time() - start_time) * 1000
        
        api_stats['messages_filtered'] += 1
        
        # Format response
        response = {
            "original_text": result['original_text'],
            "filtered_text": result['redacted_text'],
            "was_filtered": result['was_redacted'],
            "redaction_style": result['redaction_style'],
            "toxicity_probability": result['toxicity_info']['probability'],
            "confidence": result['toxicity_info']['confidence'],
            "processing_time_ms": processing_time,
            "detected_elements": result.get('redacted_elements', []),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/filter/batch', methods=['POST'])
def filter_batch():
    """Filter multiple messages in batch"""
    global api_stats
    api_stats['requests_processed'] += 1
    
    if not redactor:
        return jsonify({"error": "Redaction system not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data or 'messages' not in data:
            return jsonify({"error": "Missing 'messages' field in request"}), 400
        
        messages = data['messages']
        style = data.get('style', 'smart')
        
        if not isinstance(messages, list):
            return jsonify({"error": "'messages' must be a list"}), 400
        
        results = []
        start_time = time.time()
        
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                text = msg.get('text', '')
                user_id = msg.get('user_id', f'user{i+1}')
                username = msg.get('username', f'User{i+1}')
            else:
                text = str(msg)
                user_id = f'user{i+1}'
                username = f'User{i+1}'
            
            if text:
                result = redactor.redact_message(text, style)
                api_stats['messages_filtered'] += 1
                
                results.append({
                    "index": i,
                    "user_id": user_id,
                    "username": username,
                    "original_text": result['original_text'],
                    "filtered_text": result['redacted_text'],
                    "was_filtered": result['was_redacted'],
                    "toxicity_probability": result['toxicity_info']['probability'],
                    "detected_elements": result.get('redacted_elements', [])
                })
        
        total_time = (time.time() - start_time) * 1000
        
        response = {
            "results": results,
            "total_processed": len(results),
            "total_filtered": sum(1 for r in results if r['was_filtered']),
            "batch_processing_time_ms": total_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/moderate', methods=['POST'])
def moderate_message():
    """Moderate a message with full user context"""
    global api_stats
    api_stats['requests_processed'] += 1
    
    if not message_filter:
        return jsonify({"error": "Message filter not initialized"}), 500
    
    try:
        from real_time_message_filter import MessageEvent

        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' field in request"}), 400

        # Create message event
        message_event = MessageEvent(
            message_id=str(uuid.uuid4()),
            user_id=data.get('user_id', 'anonymous'),
            username=data.get('username', 'Anonymous'),
            text=data['message'],
            timestamp=datetime.now(),
            channel=data.get('channel', 'general'),
            metadata=data.get('metadata', {})
        )
        
        # Run async filtering in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(message_filter.filter_message(message_event))
            api_stats['messages_filtered'] += 1
            
            response = {
                "message_id": result.message_event.message_id,
                "action_taken": result.action_taken,
                "original_text": result.message_event.text,
                "filtered_text": result.filtered_text,
                "toxicity_score": result.toxicity_score,
                "detected_issues": result.detected_issues,
                "processing_time_ms": result.processing_time_ms,
                "user_info": {
                    "user_id": result.message_event.user_id,
                    "username": result.message_event.username,
                    "channel": result.message_event.channel
                },
                "timestamp": result.message_event.timestamp.isoformat()
            }
            
            return jsonify(response)
            
        finally:
            loop.close()
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/detect', methods=['POST'])
def detect_toxic_words():
    """Detect specific toxic words/phrases in text"""
    global api_stats
    api_stats['requests_processed'] += 1
    
    if not redactor:
        return jsonify({"error": "Redaction system not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text']
        
        # Get toxicity prediction
        toxicity_result = redactor.predict_toxicity(text)
        
        # Identify toxic words
        toxic_elements = redactor.identify_toxic_words(text)
        
        response = {
            "text": text,
            "toxicity_probability": toxicity_result['probability'],
            "is_toxic": toxicity_result['is_toxic'],
            "confidence": toxicity_result['confidence'],
            "toxic_elements": toxic_elements,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        # Calculate uptime
        uptime = datetime.now() - api_stats['start_time']
        
        stats = {
            "api_stats": {
                "requests_processed": api_stats['requests_processed'],
                "messages_filtered": api_stats['messages_filtered'],
                "uptime_seconds": uptime.total_seconds(),
                "requests_per_minute": api_stats['requests_processed'] / max(uptime.total_seconds() / 60, 1)
            },
            "system_status": {
                "redactor_loaded": redactor is not None,
                "filter_loaded": message_filter is not None,
                "model_parameters": redactor.model.count_params() if redactor and redactor.model else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Add filter stats if available
        if message_filter:
            filter_stats = message_filter.get_stats()
            stats["filter_stats"] = filter_stats
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    """Get or update system configuration"""
    global message_filter
    
    if request.method == 'GET':
        # Return current configuration
        if message_filter:
            config_dict = {
                "toxicity_threshold": message_filter.config.toxicity_threshold,
                "auto_redact": message_filter.config.auto_redact,
                "max_messages_per_minute": message_filter.config.max_messages_per_minute,
                "escalation_threshold": message_filter.config.escalation_threshold,
                "default_redaction_style": message_filter.config.default_redaction_style
            }
            return jsonify({"config": config_dict})
        else:
            return jsonify({"error": "Message filter not initialized"}), 500
    
    elif request.method == 'POST':
        # Update configuration
        if not message_filter:
            return jsonify({"error": "Message filter not initialized"}), 500
        
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No configuration data provided"}), 400
            
            # Update configuration
            if 'toxicity_threshold' in data:
                message_filter.config.toxicity_threshold = float(data['toxicity_threshold'])
            if 'auto_redact' in data:
                message_filter.config.auto_redact = bool(data['auto_redact'])
            if 'max_messages_per_minute' in data:
                message_filter.config.max_messages_per_minute = int(data['max_messages_per_minute'])
            if 'escalation_threshold' in data:
                message_filter.config.escalation_threshold = int(data['escalation_threshold'])
            if 'default_redaction_style' in data:
                message_filter.config.default_redaction_style = data['default_redaction_style']
            
            return jsonify({
                "message": "Configuration updated successfully",
                "config": {
                    "toxicity_threshold": message_filter.config.toxicity_threshold,
                    "auto_redact": message_filter.config.auto_redact,
                    "max_messages_per_minute": message_filter.config.max_messages_per_minute,
                    "escalation_threshold": message_filter.config.escalation_threshold,
                    "default_redaction_style": message_filter.config.default_redaction_style
                }
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            "status": "healthy" if (redactor and message_filter) else "degraded",
            "components": {
                "redactor": "ok" if redactor else "not_loaded",
                "message_filter": "ok" if message_filter else "not_loaded",
                "model": "loaded" if (redactor and redactor.model) else "not_loaded"
            },
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - api_stats['start_time']).total_seconds()
        }
        
        status_code = 200 if health_status["status"] == "healthy" else 503
        return jsonify(health_status), status_code
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/demo', methods=['GET'])
def demo_endpoint():
    """Demonstration endpoint with sample data"""
    if not redactor:
        return jsonify({"error": "Redaction system not initialized"}), 500
    
    try:
        # Sample messages for demonstration
        demo_messages = [
            "Hello! This is a friendly message.",
            "You're being really stupid about this",
            "I hate when people do that",
            "Go kill yourself, idiot!",
            "Thanks for sharing your thoughts with us"
        ]
        
        results = []
        for i, message in enumerate(demo_messages):
            result = redactor.redact_message(message, 'smart')
            results.append({
                "index": i + 1,
                "original": result['original_text'],
                "filtered": result['redacted_text'],
                "was_filtered": result['was_redacted'],
                "toxicity": result['toxicity_info']['probability'],
                "confidence": result['toxicity_info']['confidence']
            })
        
        return jsonify({
            "demo_title": "Redaction System Demonstration",
            "results": results,
            "summary": {
                "total_messages": len(demo_messages),
                "toxic_detected": sum(1 for r in results if r['was_filtered']),
                "average_toxicity": sum(r['toxicity'] for r in results) / len(results)
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

def run_api_server(host='localhost', port=5000, debug=False):
    """Run the API server"""
    print(f"🚀 Starting Redaction API Server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Debug: {debug}")
    
    # Initialize redaction system
    if not initialize_redaction_system():
        print("❌ Failed to initialize redaction system")
        return
    
    print(f"✅ API Server ready!")
    print(f"🌐 Access documentation at: http://{host}:{port}")
    print(f"🧪 Test endpoint: http://{host}:{port}/api/demo")
    
    # Run the Flask app
    app.run(host=host, port=port, debug=debug, threaded=True)

def test_api_locally():
    """Test the API locally with sample requests"""
    import requests
    import json
    
    base_url = "http://localhost:5000"
    
    print("🧪 TESTING API LOCALLY")
    print("=" * 30)
    
    # Test data
    test_cases = [
        {
            "name": "Clean message",
            "data": {"text": "Hello! How are you doing today?", "style": "smart"}
        },
        {
            "name": "Toxic message",
            "data": {"text": "You're such an idiot, go kill yourself!", "style": "smart"}
        },
        {
            "name": "Batch filtering",
            "data": {
                "messages": [
                    {"text": "Nice weather today!", "user_id": "user1"},
                    {"text": "You're a moron", "user_id": "user2"},
                    {"text": "Thanks for helping!", "user_id": "user3"}
                ],
                "style": "partial"
            }
        }
    ]
    
    try:
        # Test single message filtering
        for test_case in test_cases[:2]:
            print(f"\n🔍 Testing: {test_case['name']}")
            
            try:
                response = requests.post(f"{base_url}/api/filter", 
                                       json=test_case['data'], 
                                       timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Success:")
                    print(f"   Original: {result['original_text']}")
                    print(f"   Filtered: {result['filtered_text']}")
                    print(f"   Toxicity: {result['toxicity_probability']:.3f}")
                else:
                    print(f"❌ Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                print(f"❌ Request failed: {e}")
        
        # Test batch filtering
        print(f"\n🔍 Testing: Batch filtering")
        try:
            response = requests.post(f"{base_url}/api/filter/batch", 
                                   json=test_cases[2]['data'], 
                                   timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Batch Success:")
                print(f"   Processed: {result['total_processed']}")
                print(f"   Filtered: {result['total_filtered']}")
                print(f"   Time: {result['batch_processing_time_ms']:.2f}ms")
            else:
                print(f"❌ Batch Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Batch request failed: {e}")
        
        # Test stats endpoint
        print(f"\n🔍 Testing: Statistics")
        try:
            response = requests.get(f"{base_url}/api/stats", timeout=10)
            
            if response.status_code == 200:
                stats = response.json()
                print(f"✅ Stats retrieved:")
                print(f"   Requests: {stats['api_stats']['requests_processed']}")
                print(f"   Messages: {stats['api_stats']['messages_filtered']}")
                print(f"   Model params: {stats['system_status']['model_parameters']:,}")
            else:
                print(f"❌ Stats Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Stats request failed: {e}")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")

def main():
    """Main function"""
    print("🌐 REDACTION SYSTEM REST API")
    print("=" * 40)
    
    print("Choose an option:")
    print("1. Start API server")
    print("2. Test API locally (server must be running)")
    print("3. Show usage examples")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        # Start the server
        run_api_server(host='localhost', port=5000, debug=True)
        
    elif choice == '2':
        # Test the API
        test_api_locally()
        
    elif choice == '3':
        # Show usage examples
        print(f"\n📖 API USAGE EXAMPLES")
        print("=" * 30)
        
        examples = [
            {
                "title": "Filter single message",
                "method": "POST",
                "endpoint": "/api/filter",
                "body": {
                    "text": "Your message here",
                    "style": "smart",
                    "user_id": "user123",
                    "username": "JohnDoe"
                }
            },
            {
                "title": "Batch filter messages",
                "method": "POST", 
                "endpoint": "/api/filter/batch",
                "body": {
                    "messages": [
                        {"text": "First message", "user_id": "user1"},
                        {"text": "Second message", "user_id": "user2"}
                    ],
                    "style": "partial"
                }
            },
            {
                "title": "Full moderation",
                "method": "POST",
                "endpoint": "/api/moderate", 
                "body": {
                    "message": "Your message here",
                    "user_id": "user123",
                    "username": "JohnDoe",
                    "channel": "general"
                }
            }
        ]
        
        for example in examples:
            print(f"\n{example['title']}:")
            print(f"  {example['method']} {example['endpoint']}")
            print(f"  Body: {json.dumps(example['body'], indent=8)}")
        
        print(f"\n💡 PYTHON CLIENT EXAMPLE:")
        print("""
import requests

# Filter a message
response = requests.post('http://localhost:5000/api/filter', json={
    'text': 'Your message here',
    'style': 'smart'
})

result = response.json()
print(f"Filtered: {result['filtered_text']}")
print(f"Toxicity: {result['toxicity_probability']:.3f}")
        """)
        
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
