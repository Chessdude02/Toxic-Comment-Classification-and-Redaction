# 🛡️ Advanced Toxic Comment Classification & Real-Time Redaction System

An AI-powered system for detecting and redacting toxic content in real-time with support for multiclass toxicity detection, multiple redaction styles, and production-ready deployment options.

## ✨ Key Features

- **🎯 Multiclass Toxicity Detection**: Identifies 6 types of toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate)
- **⚡ Real-Time Message Redaction**: Automatically detects and redacts toxic content as messages are sent
- **🎨 Multiple Redaction Styles**: Choose from warning, partial, complete, or asterisk redaction methods
- **🤖 Multiple Model Architectures**: Compare SimpleRNN, LSTM, GRU, and Bidirectional LSTM models
- **🌐 Web Interface**: Beautiful, responsive web app for testing and demonstration
- **🔌 REST API**: Production-ready API endpoints for easy integration
- **📊 Real-Time Analytics**: Track moderation statistics and toxicity patterns
- **💬 Chat Simulation**: Test the system with realistic chat scenarios
- **📱 Mobile-Friendly**: Responsive design works on all devices

## 📁 Project Structure

```
├── improved_toxic_comment_classifier.ipynb  # Main training notebook
├── toxicity_redactor.py                    # Core module (production-ready)
├── toxicity_web_app.py                     # Flask web application
├── toxic-comment-classification-65cbb2.ipynb # Original notebook
├── README.md                               # This documentation
├── requirements.txt                        # Python dependencies
├── saved_models/                          # Trained model files (created after training)
└── tokenizer.pickle                       # Saved tokenizer (created after training)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn plotly flask flask-cors tqdm
```

### 2. Train Models

Open and run the **`improved_toxic_comment_classifier.ipynb`** notebook:

1. **Option A - Kaggle Environment**: 
   - Upload the notebook to Kaggle
   - Use the Jigsaw Toxic Comment dataset
   - Run all cells to train multiple models

2. **Option B - Local Environment**:
   - Download the Jigsaw Toxic Comment dataset from Kaggle
   - Update the data paths in the notebook
   - Run the notebook to train models

### 3. Test the System

#### Option A: Interactive Notebook
Run cells in the notebook to test individual messages and see detailed analysis.

#### Option B: Python Module
```python
from toxicity_redactor import load_pretrained_model

# Load the trained model
redactor = load_pretrained_model()

# Test a message
result = redactor.classify_toxicity("This is a test message")
print(result)

# Redact a toxic message
redacted = redactor.redact_message("You're an idiot!", style="warning")
print(redacted['redacted_message'])
```

#### Option C: Web Application
```bash
python toxicity_web_app.py
```
Then open http://localhost:5000 in your browser.

## 📊 Model Performance

The system trains and compares multiple architectures:

| Model | Description | Use Case |
|-------|-------------|----------|
| **SimpleRNN** | Basic RNN architecture | Lightweight, fast inference |
| **LSTM** | Long Short-Term Memory | Better long-term dependencies |
| **GRU** | Gated Recurrent Unit | Good balance of speed/performance |
| **Bidirectional LSTM** | Processes text in both directions | Best overall performance |

## 🎨 Redaction Styles

Choose from multiple redaction approaches:

### 1. Warning Style (Default)
```
Original: "You're such an idiot!"
Redacted: "⚠️ [WARNING: Contains insult content - 89.2% confidence] You're such an idiot!"
```

### 2. Partial Redaction
```
Original: "You're such an idiot!"
Redacted: "You're such an ******!"
```

### 3. Complete Redaction
```
Original: "You're such an idiot!"
Redacted: "[MESSAGE REDACTED - CONTAINS INSULT CONTENT]"
```

### 4. Asterisk Pattern
```
Original: "You're such an idiot!"
Redacted: "You're such an *****!"
```

## 🌐 Web Interface Features

The web application provides:

### 🧪 Message Testing Panel
- Test individual messages for toxicity
- Choose redaction style
- See detailed classification results
- Real-time confidence scores

### 💬 Chat Simulation
- Simulate real chat conversations
- Automatic moderation in real-time
- Message history with status indicators
- Clear visual feedback for toxic content

### 📊 Statistics Dashboard
- Total messages processed
- Toxicity detection rates
- Real-time analytics
- Performance metrics

## 🔌 API Endpoints

The system provides RESTful API endpoints for integration:

### POST /api/check-toxicity
Check if a message is toxic:
```json
{
  "message": "Your message here",
  "redaction_style": "warning"
}
```

### POST /api/moderate-message
Moderate a single message:
```json
{
  "message": "Message to moderate",
  "style": "partial"
}
```

### POST /api/moderate-batch
Moderate multiple messages:
```json
{
  "messages": ["msg1", "msg2", "msg3"],
  "style": "warning"
}
```

### GET /api/stats
Get moderation statistics

### GET /api/chat-log
Get recent chat messages

## 🔧 Production Deployment

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "toxicity_web_app.py"]
```

Build and run:
```bash
docker build -t toxicity-detector .
docker run -p 5000:5000 toxicity-detector
```

### Cloud Deployment

Deploy to major cloud platforms:

- **AWS**: Use Elastic Beanstalk or ECS
- **Google Cloud**: Deploy to Cloud Run or App Engine
- **Azure**: Use App Service or Container Instances
- **Heroku**: Deploy directly with `Procfile`

## 🧪 Usage Examples

### Basic Usage

```python
from toxicity_redactor import ToxicityRedactor, load_pretrained_model

# Load model
redactor = load_pretrained_model()

# Check toxicity
result = redactor.classify_toxicity("This is great!")
print(f"Is toxic: {result['is_toxic']}")
print(f"Confidence: {result['max_toxicity_probability']:.3f}")

# Get detailed results
for label, info in result['detailed_results'].items():
    print(f"{label}: {info['probability']:.3f}")
```

### Chat Moderation

```python
from toxicity_redactor import ChatModerator, load_pretrained_model

# Initialize
redactor = load_pretrained_model()
moderator = ChatModerator(redactor, auto_moderate=True)

# Process messages
messages = [
    ("Alice", "Hello everyone!"),
    ("Bob", "This is stupid!"),
    ("Charlie", "Let's be respectful.")
]

for username, message in messages:
    result = moderator.process_message(username, message)
    print(f"{result['status']} {username}: {result['display_message']}")

# Get statistics
stats = moderator.get_moderation_stats()
print(f"Toxicity rate: {stats['toxicity_rate']:.1%}")
```

### Batch Processing

```python
# Process multiple messages
messages = [
    "Great job on the project!",
    "This is terrible work.",
    "Thanks for your help.",
    "You're all idiots!"
]

results = redactor.moderate_conversation(messages, style='warning')
for result in results:
    print(f"Original: {result['toxicity_info']['original_message']}")
    print(f"Moderated: {result['redacted_message']}")
    print(f"Was redacted: {result['was_redacted']}")
    print("---")
```

## 📈 Model Training Tips

### Data Preparation
- Clean text data (remove URLs, special characters)
- Handle class imbalance with appropriate sampling
- Use stratified splits to maintain class distribution
- Consider text length distribution for sequence padding

### Model Optimization
- Experiment with different architectures
- Tune hyperparameters (learning rate, dropout, etc.)
- Use early stopping to prevent overfitting
- Monitor validation metrics during training

### Performance Evaluation
- Use ROC-AUC for binary classification metrics
- Analyze per-class performance for multiclass problems
- Create confusion matrices to understand error patterns
- Test with real-world examples

## 🛠️ Customization

### Adding Custom Toxicity Categories

1. **Modify the training data** to include your custom labels
2. **Update the model architecture** to output the correct number of classes
3. **Retrain the models** with the new data
4. **Update the redactor** to handle new categories

### Custom Redaction Logic

```python
def custom_redaction(message, classification):
    if classification['is_toxic']:
        if 'severe_toxic' in classification['toxic_types']:
            return "[SEVERELY TOXIC CONTENT REMOVED]"
        elif 'threat' in classification['toxic_types']:
            return "[THREATENING CONTENT FLAGGED FOR REVIEW]"
        else:
            return f"[TOXIC: {', '.join(classification['toxic_types'])}] {message}"
    return message
```

### Integration with Existing Systems

```python
# Example: Discord Bot Integration
import discord
from toxicity_redactor import load_pretrained_model

redactor = load_pretrained_model()

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    result = redactor.classify_toxicity(message.content)
    
    if result['is_toxic']:
        await message.delete()
        warning = f"⚠️ {message.author.mention}, your message was removed for containing {', '.join(result['toxic_types'])} content."
        await message.channel.send(warning)
```

## 📋 System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- 2GB free disk space
- CPU with 4+ cores

### Recommended for Production
- Python 3.9+
- 16GB RAM
- GPU with CUDA support (for training)
- SSD storage
- Load balancer for high traffic

## 🐛 Troubleshooting

### Common Issues

**Model not loading**:
- Ensure models are trained and saved properly
- Check file paths in the configuration
- Verify all dependencies are installed

**Memory errors during training**:
- Reduce batch size
- Use gradient accumulation
- Consider model parallelism

**Poor model performance**:
- Check data quality and preprocessing
- Increase training epochs
- Try different architectures
- Tune hyperparameters

**Web app not starting**:
- Install Flask dependencies: `pip install flask flask-cors`
- Check port availability (default: 5000)
- Ensure models are trained and saved

## 📚 Additional Resources

### Documentation
- [TensorFlow Documentation](https://tensorflow.org/api_docs)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Flask Documentation](https://flask.palletsprojects.com/)

### Research Papers
- "Attention Is All You Need" (Transformer Architecture)
- "BERT: Bidirectional Encoder Representations from Transformers"
- "Toxic Comment Classification Challenge" (Kaggle)

### Datasets
- [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [Hate Speech Detection](https://hasocfire.github.io/hasoc/2021/)
- [Offensive Language Identification](https://sites.google.com/site/offensevalsharedtask/)

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report bugs** by creating detailed issue reports
2. **Suggest features** for new functionality
3. **Submit pull requests** with improvements
4. **Improve documentation** to help other users
5. **Share datasets** to improve model performance

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd toxic-comment-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Start development server
python toxicity_web_app.py
```

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- **Jigsaw/Conversation AI** for the toxic comment dataset
- **TensorFlow team** for the deep learning framework
- **Kaggle community** for inspiration and datasets
- **Open source contributors** who made this possible

## 📞 Support

Need help? Here are your options:

- **Documentation**: Check this README and inline code comments
- **Issues**: Create a GitHub issue for bugs or feature requests
- **Discussions**: Join the community discussions
- **Email**: Contact the maintainers directly

---

**🎉 Ready to build safer online communities with AI-powered content moderation!**

*This system helps create positive online spaces by automatically detecting and handling toxic content while maintaining transparency and user control over moderation policies.*
