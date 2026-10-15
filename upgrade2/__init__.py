"""
Upgrade2: Enhanced Toxic Comment Classification System

This package provides advanced toxicity detection capabilities including:
- Enhanced vocabulary management with cultural context
- Semantic analysis for intent, emotion, and context understanding  
- Multi-method integration for comprehensive detection
- Enhanced neural models with feature fusion

Author: Enhanced Toxicity Detection System
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Enhanced Toxicity Detection System"

# Import main classes for easy access
try:
    from .vocabulary.enhanced_vocabulary_manager import EnhancedToxicVocabularyManager
    from .semantic.semantic_toxicity_analyzer import SemanticToxicityAnalyzer
    from .integration.enhanced_toxicity_detector import EnhancedToxicityDetector, ToxicityDetectionResult
    from .models.enhanced_bilstm_model import EnhancedBiLSTM, ToxicityDataset, EnhancedBiLSTMTrainer

    # Make main components easily accessible
    __all__ = [
        'EnhancedToxicVocabularyManager',
        'SemanticToxicityAnalyzer', 
        'EnhancedToxicityDetector',
        'ToxicityDetectionResult',
        'EnhancedBiLSTM',
        'ToxicityDataset',
        'EnhancedBiLSTMTrainer'
    ]
    
except ImportError as e:
    print(f"Warning: Some upgrade2 components could not be imported: {e}")
    print("You may need to install missing dependencies: numpy, torch, scikit-learn")
    
    # Define minimal interface
    __all__ = []


def get_version():
    """Return the current version of the upgrade2 package."""
    return __version__


def quick_start_example():
    """
    Print a quick start example for using the enhanced detection system.
    """
    example = """
# Quick Start Example - Enhanced Toxicity Detection

from upgrade2 import EnhancedToxicityDetector

# Initialize the enhanced detector
detector = EnhancedToxicityDetector()

# Analyze a text for toxicity
text = "Your text here"
result = detector.detect_toxicity(text)

# Access results
print(f"Toxicity Score: {result.overall_toxicity_score:.3f}")
print(f"Severity Level: {result.severity_level}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Risk Factors: {result.risk_factors}")
print(f"Recommendations: {result.recommendations}")

# Get detailed explanation
explanation = detector.explain_detection(text)
print(f"Analysis: {explanation}")

# Batch processing
texts = ["text1", "text2", "text3"]
results = detector.batch_detect(texts)

# Compare two texts
comparison = detector.compare_texts("text1", "text2")
print(f"More toxic: {comparison['more_toxic']}")
"""
    print(example)


def system_info():
    """
    Print information about the upgrade2 system.
    """
    info = f"""
🔍 Enhanced Toxic Comment Classification System (Upgrade2)
Version: {__version__}

📦 Components:
• Enhanced Vocabulary Manager - Advanced pattern matching with cultural context
• Semantic Toxicity Analyzer - Intent, emotion, and context analysis  
• Enhanced Toxicity Detector - Multi-method integration system
• Enhanced BiLSTM Model - Neural model with vocabulary/semantic features

💡 Key Features:
• Multi-method toxicity detection
• Context-aware analysis
• Risk assessment and mitigation factors
• Confidence scoring
• Batch processing
• Text comparison
• Detailed explanations
• Redaction capabilities

📊 Performance Benefits:
• Improved accuracy through multi-method consensus
• Better handling of context and nuance
• Reduced false positives through semantic understanding
• Cultural and generational awareness
• Real-time processing capabilities

🚀 Usage:
from upgrade2 import EnhancedToxicityDetector
detector = EnhancedToxicityDetector()
result = detector.detect_toxicity("your text")
"""
    print(info)
