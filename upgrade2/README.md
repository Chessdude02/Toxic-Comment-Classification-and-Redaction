# Toxic Comment Classification - Enhanced Features (Upgrade2)

This directory extends the base classifier with two genuinely different kinds of components — read this before using either:

- **Rule-based heuristic layer** (`vocabulary/`, `semantic/`, `integration/`): keyword lists, regex pattern matching, and caps/punctuation heuristics. Despite the "Enhanced"/"Semantic" naming, **this is not a trained model** — no learning happens here. Its value is as a fast, dependency-free pre-filter (sub-millisecond scoring, no model weights to load), not a replacement for the Transformer classifier in the main notebook.
- **Enhanced BiLSTM model** (`models/enhanced_bilstm_model.py`): a real, trainable PyTorch `nn.Module` architecture that fuses text embeddings with features from the heuristic layer. It is untrained — no weights are shipped — and uses **PyTorch**, a different framework from the TensorFlow stack used by the main classifier notebook. See Installation below.

## 🚀 Overview

The Upgrade2 system provides:
- **Vocabulary Manager**: Keyword/regex pattern matching with configurable pattern lists
- **Semantic Toxicity Analyzer**: Rule-based intent, emotion, and context heuristics (sarcasm/debate detection via pattern matching, not learned representations)
- **Integrated Detection System**: Combines the above heuristics with confidence scoring
- **Enhanced BiLSTM Model**: A separate, untrained PyTorch neural architecture for feature fusion (see note above)

## 📁 Directory Structure

```
upgrade2/
├── vocabulary/
│   └── enhanced_vocabulary_manager.py    # Advanced vocabulary pattern detection
├── semantic/
│   └── semantic_toxicity_analyzer.py     # Context-aware semantic analysis
├── integration/
│   └── enhanced_toxicity_detector.py     # Multi-method integration system
├── models/
│   └── enhanced_bilstm_model.py          # Neural model with feature fusion
└── README.md                             # This file
```

## 🔧 Installation

1. Ensure you have the required dependencies:
```bash
pip install numpy torch scikit-learn pandas matplotlib seaborn
```

2. The modules are designed to work independently or as part of the integrated system.

## 💡 Key Features

All features below are implemented via keyword/regex pattern lists and hand-written heuristics — not a trained model. Framed here in terms of what they mechanically do, not what they imply:

### Vocabulary Manager
- **Configurable Pattern Lists**: Toxicity keyword/phrase lists that can be extended at runtime
- **Regional Term Variants**: Pattern lists include some regional/generational slang variants
- **Severity Scoring**: Combines multiple pattern matches into a severity score
- **Redaction**: Replaces matched spans in the text

### Semantic Toxicity Analyzer
- **Intent Heuristics**: Flags toxic intent (direct attack, threat, dismissive, etc.) via keyword/structure rules
- **Emotional Tone Heuristics**: Flags anger/disgust/contempt/frustration via keyword and punctuation cues
- **Context Heuristics**: Flags likely debate/sarcasm/group-criticism context via surface patterns (e.g. "obviously...", excessive caps)
- **Pattern Analysis**: Sentence-structure and lexical-diversity statistics

### Integration System
- **Multi-Heuristic Combination**: Combines vocabulary, semantic, and pattern scores with configurable weights
- **Risk Assessment**: Surfaces which heuristics fired as risk/mitigation factors
- **Confidence Scoring**: Heuristic agreement used as a proxy for confidence
- **Batch Processing**: Runs the above over multiple texts

### Enhanced BiLSTM Model
- **Feature Fusion**: Integrates text embeddings with vocabulary and semantic features
- **Attention Mechanism**: Multi-head attention for improved focus
- **Multi-Task Learning**: Joint training for toxicity, severity, and category prediction
- **Advanced Architecture**: BiLSTM + Attention + Feature Processing

## 🎯 Quick Start

### Basic Usage

```python
from upgrade2.integration.enhanced_toxicity_detector import EnhancedToxicityDetector

# Initialize detector
detector = EnhancedToxicityDetector()

# Analyze text
result = detector.detect_toxicity("Your text here")
print(f"Toxicity Score: {result.overall_toxicity_score:.3f}")
print(f"Severity: {result.severity_level}")
print(f"Recommendations: {result.recommendations}")
```

### Using Individual Components

```python
# Vocabulary Analysis
from upgrade2.vocabulary.enhanced_vocabulary_manager import EnhancedToxicVocabularyManager
vocab_manager = EnhancedToxicVocabularyManager()
vocab_analysis = vocab_manager.analyze_toxicity_patterns("text")

# Semantic Analysis
from upgrade2.semantic.semantic_toxicity_analyzer import SemanticToxicityAnalyzer
semantic_analyzer = SemanticToxicityAnalyzer()
semantic_analysis = semantic_analyzer.analyze_semantic_toxicity("text")
```

### Neural Model Training

```python
from upgrade2.models.enhanced_bilstm_model import EnhancedBiLSTM, ToxicityDataset
import torch

# Configuration
config = {
    'embedding_dim': 128,
    'hidden_dim': 128,
    'vocab_size': 10000,
    'num_classes': 2
}

# Create model and dataset
model = EnhancedBiLSTM(config)
dataset = ToxicityDataset(texts, labels, vocab_manager, semantic_analyzer)

# Training would follow standard PyTorch patterns
```

## 📊 Performance Features

### Detection Metrics
- **Multi-Method Consensus**: Agreement between different detection approaches
- **Confidence Scoring**: Reliability metrics for each prediction
- **Risk Assessment**: Categorized risk factors and mitigation factors
- **Processing Statistics**: Performance tracking and analytics

### Advanced Analytics
- **Pattern Recognition**: Dynamic detection of emerging toxicity patterns
- **Cultural Sensitivity**: Region and generation-aware analysis
- **Context Understanding**: Differentiates between toxicity types and contexts
- **Explanation Generation**: Human-readable analysis explanations

## 🔍 Testing and Validation

Each component includes comprehensive testing:

```bash
# Test individual components
cd upgrade2/vocabulary/
python enhanced_vocabulary_manager.py

cd upgrade2/semantic/
python semantic_toxicity_analyzer.py

cd upgrade2/integration/
python enhanced_toxicity_detector.py

cd upgrade2/models/
python enhanced_bilstm_model.py
```

## 📈 Advanced Usage

### Batch Processing
```python
texts = ["text1", "text2", "text3"]
results = detector.batch_detect(texts)
for result in results:
    print(f"Text: {result.text[:50]}... | Score: {result.overall_toxicity_score:.3f}")
```

### Text Comparison
```python
comparison = detector.compare_texts("text1", "text2")
print(f"More toxic: {comparison['more_toxic']}")
print(f"Score difference: {comparison['score_difference']:.3f}")
```

### Detailed Analysis
```python
explanation = detector.explain_detection("your text here")
print(explanation)
```

## 🛠️ Configuration

The system supports extensive configuration:

```python
config = {
    'vocabulary_weight': 0.4,      # Weight for vocabulary-based detection
    'semantic_weight': 0.4,        # Weight for semantic analysis
    'pattern_weight': 0.2,         # Weight for pattern analysis
    'confidence_threshold': 0.6,   # Minimum confidence for reliable detection
    'redaction_threshold': 0.7,    # Threshold for redaction recommendations
    'enable_explanations': True,   # Generate human-readable explanations
    'enable_redaction': True,      # Generate redacted versions
}

detector = EnhancedToxicityDetector(config)
```

## 📝 Output Format

### Detection Result Structure
```python
ToxicityDetectionResult:
    text: str                           # Original text
    overall_toxicity_score: float       # Combined toxicity score (0-1)
    confidence: float                   # Detection confidence (0-1)
    detection_components: Dict          # Individual method results
    recommendations: List[str]          # Actionable recommendations
    redacted_version: str              # Filtered version of text
    severity_level: str                # minimal/low/medium/high/critical
    risk_factors: List[str]            # Identified risk factors
    mitigation_factors: List[str]      # Factors that reduce risk
    timestamp: str                     # Analysis timestamp
```

## 🔧 Integration with Existing System

This heuristic layer can be used alongside the trained classifier in `src/toxicity_redactor.py` — e.g. as a fast pre-filter before the model runs, or as a fallback when model weights aren't loaded:

```python
import sys
sys.path.insert(0, '../src')
from toxicity_redactor import load_pretrained_model
from upgrade2.integration.enhanced_toxicity_detector import EnhancedToxicityDetector

# Combine the trained classifier and the heuristic layer
trained_redactor = load_pretrained_model()
heuristic_detector = EnhancedToxicityDetector()

# Compare results
trained_result = trained_redactor.classify_toxicity(text)
heuristic_result = heuristic_detector.detect_toxicity(text)
```

## 📚 API Reference

### EnhancedToxicityDetector Methods
- `detect_toxicity(text, context=None)`: Main detection method
- `batch_detect(texts, context=None)`: Batch processing
- `compare_texts(text1, text2)`: Compare two texts
- `explain_detection(text)`: Generate detailed explanation
- `get_detection_stats()`: Get performance statistics

### EnhancedToxicVocabularyManager Methods
- `analyze_toxicity_patterns(text)`: Comprehensive vocabulary analysis
- `add_dynamic_pattern(pattern, category)`: Add new pattern
- `update_cultural_context(context)`: Update cultural awareness
- `generate_redacted_version(text)`: Create filtered version

### SemanticToxicityAnalyzer Methods
- `analyze_semantic_toxicity(text)`: Full semantic analysis
- `compare_semantic_similarity(text1, text2)`: Compare semantic similarity
- `generate_explanation(text)`: Human-readable explanation

## 🚨 Important Notes

1. **Privacy**: The system processes text locally and doesn't transmit data externally
2. **Performance**: The heuristic layer adds negligible latency (sub-millisecond); the BiLSTM model's cost depends on whether it's trained/run
3. **Compatibility**: The heuristic layer needs only the stdlib + regex; `enhanced_bilstm_model.py` additionally requires PyTorch (not in the root `requirements.txt` — install separately, see Installation above)
4. **Extensibility**: All components can be extended with custom patterns and rules

## 🤝 Contributing

To extend the system:
1. Add new patterns to vocabulary manager
2. Implement new semantic analysis features
3. Create additional neural model architectures
4. Improve integration and fusion methods

## 📄 License

This enhancement package follows the same license as the main project.

## 🔗 Related Files

- Main project: `../src/`
- Web application: `../src/toxicity_web_app.py`
- Documentation: `../docs/`
- Tests: `../tests/`

---

**Upgrade2 System**: Advanced toxic comment classification with enhanced vocabulary, semantic understanding, and integrated detection capabilities.
