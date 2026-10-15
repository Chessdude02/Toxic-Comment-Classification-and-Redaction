"""
Enhanced BiLSTM Model with Vocabulary and Semantic Features

This module provides an advanced BiLSTM architecture that integrates:
- Traditional text embeddings
- Enhanced vocabulary features
- Semantic toxicity features
- Attention mechanisms
- Multi-task learning capabilities
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
upgrade_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(upgrade_dir)
sys.path.extend([upgrade_dir, project_dir])

try:
    from upgrade2.vocabulary.enhanced_vocabulary_manager import EnhancedToxicVocabularyManager
    from upgrade2.semantic.semantic_toxicity_analyzer import SemanticToxicityAnalyzer
    from upgrade2.integration.enhanced_toxicity_detector import EnhancedToxicityDetector
except ImportError:
    logging.warning("Could not import some modules. Some features may be limited.")


class ToxicityDataset(Dataset):
    """Dataset class for toxicity classification with enhanced features."""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None,
                 vocab_manager: Optional[EnhancedToxicVocabularyManager] = None,
                 semantic_analyzer: Optional[SemanticToxicityAnalyzer] = None,
                 max_length: int = 512):
        
        self.texts = texts
        self.labels = labels if labels is not None else [0] * len(texts)
        self.vocab_manager = vocab_manager
        self.semantic_analyzer = semantic_analyzer
        self.max_length = max_length
        
        # Pre-compute enhanced features if analyzers are available
        self.vocab_features = []
        self.semantic_features = []
        
        if self.vocab_manager is not None:
            logging.info("Pre-computing vocabulary features...")
            for text in texts:
                try:
                    analysis = self.vocab_manager.analyze_toxicity_patterns(text)
                    features = self._extract_vocab_features(analysis)
                    self.vocab_features.append(features)
                except Exception as e:
                    logging.warning(f"Error computing vocab features for text: {e}")
                    self.vocab_features.append(np.zeros(50))  # Default fallback
        
        if self.semantic_analyzer is not None:
            logging.info("Pre-computing semantic features...")
            for text in texts:
                try:
                    analysis = self.semantic_analyzer.analyze_semantic_toxicity(text)
                    features = analysis['semantic_features']
                    self.semantic_features.append(features)
                except Exception as e:
                    logging.warning(f"Error computing semantic features for text: {e}")
                    self.semantic_features.append(np.zeros(24))  # Default fallback
        
        logging.info(f"Dataset initialized with {len(self.texts)} samples")
    
    def _extract_vocab_features(self, analysis: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from vocabulary analysis."""
        
        features = []
        
        # Overall toxicity and confidence
        features.extend([
            analysis.get('overall_toxicity_score', 0.0),
            analysis.get('confidence', 0.0)
        ])
        
        # Category scores (assume up to 10 categories)
        categories = analysis.get('matched_categories', {})
        category_scores = []
        for i in range(10):
            if i < len(categories):
                category_name = list(categories.keys())[i]
                category_scores.append(categories[category_name].get('score', 0.0))
            else:
                category_scores.append(0.0)
        features.extend(category_scores)
        
        # Severity factors
        severity = analysis.get('severity_analysis', {})
        features.extend([
            severity.get('overall_severity', 0.0),
            len(severity.get('factors', [])) / 10.0,  # Normalized factor count
            severity.get('base_score', 0.0),
            severity.get('multiplier', 1.0)
        ])
        
        # Statistical features
        stats = analysis.get('text_statistics', {})
        features.extend([
            stats.get('word_count', 0) / 100.0,  # Normalized
            stats.get('unique_ratio', 0.0),
            stats.get('caps_ratio', 0.0),
            stats.get('punctuation_ratio', 0.0),
            len(analysis.get('redaction_candidates', [])) / 5.0  # Normalized
        ])
        
        # Pattern features
        patterns = analysis.get('dynamic_patterns', [])
        pattern_features = []
        for i in range(15):  # Up to 15 pattern features
            if i < len(patterns):
                pattern_features.append(1.0)  # Pattern present
            else:
                pattern_features.append(0.0)
        features.extend(pattern_features)
        
        # Linguistic complexity
        features.extend([
            analysis.get('linguistic_complexity', 0.0),
            analysis.get('toxicity_distribution', {}).get('variance', 0.0),
            len(analysis.get('context_modifiers', [])) / 5.0  # Normalized
        ])
        
        # Cultural and contextual factors
        cultural = analysis.get('cultural_analysis', {})
        features.extend([
            cultural.get('regional_score', 0.0),
            cultural.get('generational_score', 0.0),
            len(cultural.get('cultural_indicators', [])) / 3.0  # Normalized
        ])
        
        return np.array(features[:50], dtype=np.float32)  # Ensure fixed size
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Basic text features (to be tokenized later)
        item = {
            'text': text,
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        # Add vocabulary features if available
        if self.vocab_features:
            vocab_feat = self.vocab_features[idx] if idx < len(self.vocab_features) else np.zeros(50)
            item['vocab_features'] = torch.tensor(vocab_feat, dtype=torch.float32)
        
        # Add semantic features if available
        if self.semantic_features:
            semantic_feat = self.semantic_features[idx] if idx < len(self.semantic_features) else np.zeros(24)
            item['semantic_features'] = torch.tensor(semantic_feat, dtype=torch.float32)
        
        return item


class EnhancedBiLSTM(nn.Module):
    """
    Enhanced BiLSTM model with vocabulary and semantic feature integration.
    
    Architecture:
    1. Embedding layer for text
    2. BiLSTM for sequential processing
    3. Attention mechanism
    4. Feature fusion (text + vocabulary + semantic)
    5. Classification layers
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(EnhancedBiLSTM, self).__init__()
        
        self.config = config
        
        # Text processing layers
        self.embedding_dim = config.get('embedding_dim', 128)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.vocab_size = config.get('vocab_size', 10000)
        self.num_classes = config.get('num_classes', 2)
        
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=config.get('num_lstm_layers', 2),
            batch_first=True,
            bidirectional=True,
            dropout=config.get('lstm_dropout', 0.2)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,  # BiLSTM output dimension
            num_heads=config.get('attention_heads', 8),
            dropout=config.get('attention_dropout', 0.1),
            batch_first=True
        )
        
        # Feature dimensions
        self.vocab_feature_dim = config.get('vocab_feature_dim', 50)
        self.semantic_feature_dim = config.get('semantic_feature_dim', 24)
        
        # Feature processing layers
        self.vocab_processor = nn.Sequential(
            nn.Linear(self.vocab_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        self.semantic_processor = nn.Sequential(
            nn.Linear(self.semantic_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # Feature fusion
        fusion_input_dim = (self.hidden_dim * 2) + 32 + 32  # BiLSTM + vocab + semantic
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, config.get('fusion_dim', 256)),
            nn.ReLU(),
            nn.BatchNorm1d(config.get('fusion_dim', 256)),
            nn.Dropout(config.get('fusion_dropout', 0.3))
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(config.get('fusion_dim', 256), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.num_classes)
        )
        
        # Auxiliary task heads (multi-task learning)
        self.severity_head = nn.Linear(config.get('fusion_dim', 256), 5)  # 5 severity levels
        self.category_head = nn.Linear(config.get('fusion_dim', 256), 10)  # Multiple toxicity categories
        
        # Initialize weights
        self._initialize_weights()
        
        logging.info(f"Enhanced BiLSTM model initialized with {self._count_parameters()} parameters")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def _count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids: torch.Tensor, 
                vocab_features: Optional[torch.Tensor] = None,
                semantic_features: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized text input [batch_size, seq_len]
            vocab_features: Vocabulary-based features [batch_size, vocab_feat_dim]
            semantic_features: Semantic features [batch_size, semantic_feat_dim]
            attention_mask: Attention mask for padded tokens
            
        Returns:
            Dictionary with logits and auxiliary outputs
        """
        
        batch_size = input_ids.size(0)
        
        # Text processing
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # BiLSTM processing
        lstm_output, (hidden, cell) = self.bilstm(embedded)  # [batch_size, seq_len, hidden_dim*2]
        
        # Apply attention
        if attention_mask is not None:
            # Create attention mask for padding
            attn_mask = attention_mask == 0  # True for padded positions
            attn_output, attn_weights = self.attention(lstm_output, lstm_output, lstm_output,
                                                      key_padding_mask=attn_mask)
        else:
            attn_output, attn_weights = self.attention(lstm_output, lstm_output, lstm_output)
        
        # Global average pooling over sequence length
        text_representation = attn_output.mean(dim=1)  # [batch_size, hidden_dim*2]
        
        # Process additional features
        feature_representations = []
        
        # Vocabulary features
        if vocab_features is not None:
            vocab_repr = self.vocab_processor(vocab_features)  # [batch_size, 32]
            feature_representations.append(vocab_repr)
        else:
            # Use zero features if not provided
            vocab_repr = torch.zeros(batch_size, 32, device=input_ids.device)
            feature_representations.append(vocab_repr)
        
        # Semantic features
        if semantic_features is not None:
            semantic_repr = self.semantic_processor(semantic_features)  # [batch_size, 32]
            feature_representations.append(semantic_repr)
        else:
            # Use zero features if not provided
            semantic_repr = torch.zeros(batch_size, 32, device=input_ids.device)
            feature_representations.append(semantic_repr)
        
        # Feature fusion
        all_features = torch.cat([text_representation] + feature_representations, dim=1)
        fused_features = self.fusion_layer(all_features)  # [batch_size, fusion_dim]
        
        # Main classification
        main_logits = self.classifier(fused_features)  # [batch_size, num_classes]
        
        # Auxiliary tasks
        severity_logits = self.severity_head(fused_features)  # [batch_size, 5]
        category_logits = self.category_head(fused_features)  # [batch_size, 10]
        
        return {
            'logits': main_logits,
            'severity_logits': severity_logits,
            'category_logits': category_logits,
            'attention_weights': attn_weights,
            'fused_features': fused_features
        }


class EnhancedBiLSTMTrainer:
    """Trainer class for the Enhanced BiLSTM model."""
    
    def __init__(self, model: EnhancedBiLSTM, tokenizer: Any, device: str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Loss functions
        self.main_criterion = nn.CrossEntropyLoss()
        self.severity_criterion = nn.CrossEntropyLoss()
        self.category_criterion = nn.BCEWithLogitsLoss()  # Multi-label classification
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=2e-4,
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        logging.info("Enhanced BiLSTM trainer initialized")
    
    def tokenize_batch(self, texts: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts."""
        
        # Simple tokenization (can be replaced with more sophisticated tokenizer)
        tokenized_texts = []
        attention_masks = []
        
        for text in texts:
            # Basic word-level tokenization
            tokens = text.lower().split()[:max_length]
            
            # Convert to IDs (simplified)
            token_ids = [hash(token) % self.model.vocab_size for token in tokens]
            
            # Pad or truncate
            if len(token_ids) < max_length:
                padding_length = max_length - len(token_ids)
                token_ids.extend([0] * padding_length)  # 0 is padding token
                attention_mask = [1] * len(tokens) + [0] * padding_length
            else:
                token_ids = token_ids[:max_length]
                attention_mask = [1] * max_length
            
            tokenized_texts.append(token_ids)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': torch.tensor(tokenized_texts, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch in train_loader:
            # Move batch to device
            texts = batch['text']
            labels = batch['label'].to(self.device)
            
            vocab_features = batch.get('vocab_features')
            if vocab_features is not None:
                vocab_features = vocab_features.to(self.device)
            
            semantic_features = batch.get('semantic_features')
            if semantic_features is not None:
                semantic_features = semantic_features.to(self.device)
            
            # Tokenize texts
            tokenized = self.tokenize_batch(texts)
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, vocab_features, semantic_features, attention_mask)
            
            # Calculate losses
            main_loss = self.main_criterion(outputs['logits'], labels)
            
            # Auxiliary losses (if targets available)
            total_loss_batch = main_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += total_loss_batch.item()
            predictions = torch.argmax(outputs['logits'], dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model."""
        
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                texts = batch['text']
                labels = batch['label'].to(self.device)
                
                vocab_features = batch.get('vocab_features')
                if vocab_features is not None:
                    vocab_features = vocab_features.to(self.device)
                
                semantic_features = batch.get('semantic_features')
                if semantic_features is not None:
                    semantic_features = semantic_features.to(self.device)
                
                # Tokenize texts
                tokenized = self.tokenize_batch(texts)
                input_ids = tokenized['input_ids'].to(self.device)
                attention_mask = tokenized['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, vocab_features, semantic_features, attention_mask)
                
                # Calculate loss
                loss = self.main_criterion(outputs['logits'], labels)
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(outputs['logits'], dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 10, save_path: str = None):
        """Train the model."""
        
        best_val_loss = float('inf')
        
        logging.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            start_time = datetime.now()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.model.config
                }, save_path)
                logging.info(f"Best model saved with val_loss: {val_loss:.4f}")
            
            epoch_time = (datetime.now() - start_time).total_seconds()
            
            logging.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
        
        logging.info("Training completed!")
    
    def predict(self, texts: List[str], 
                vocab_manager: Optional[EnhancedToxicVocabularyManager] = None,
                semantic_analyzer: Optional[SemanticToxicityAnalyzer] = None) -> np.ndarray:
        """Make predictions on new texts."""
        
        self.model.eval()
        
        # Create dataset
        dataset = ToxicityDataset(texts, labels=None, 
                                vocab_manager=vocab_manager,
                                semantic_analyzer=semantic_analyzer)
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                texts_batch = batch['text']
                
                vocab_features = batch.get('vocab_features')
                if vocab_features is not None:
                    vocab_features = vocab_features.to(self.device)
                
                semantic_features = batch.get('semantic_features')
                if semantic_features is not None:
                    semantic_features = semantic_features.to(self.device)
                
                # Tokenize texts
                tokenized = self.tokenize_batch(texts_batch)
                input_ids = tokenized['input_ids'].to(self.device)
                attention_mask = tokenized['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, vocab_features, semantic_features, attention_mask)
                
                # Get predictions
                probs = torch.softmax(outputs['logits'], dim=1)
                batch_predictions = probs.cpu().numpy()
                predictions.append(batch_predictions)
        
        return np.vstack(predictions)


if __name__ == "__main__":
    # Test the enhanced BiLSTM model
    print("🧠 Testing Enhanced BiLSTM Model")
    print("=" * 60)
    
    # Model configuration
    config = {
        'embedding_dim': 128,
        'hidden_dim': 128,
        'vocab_size': 10000,
        'num_classes': 2,
        'num_lstm_layers': 2,
        'attention_heads': 8,
        'vocab_feature_dim': 50,
        'semantic_feature_dim': 24,
        'fusion_dim': 256
    }
    
    # Create model
    model = EnhancedBiLSTM(config)
    
    print(f"✅ Model created with {model._count_parameters():,} parameters")
    
    # Test dataset
    sample_texts = [
        "You're such an idiot, shut up!",
        "I disagree with your opinion",
        "This is really well written",
        "What the hell is wrong with you?",
        "Great job on that project!"
    ]
    
    sample_labels = [1, 0, 0, 1, 0]  # 1 = toxic, 0 = non-toxic
    
    # Create dataset (without feature extraction for testing)
    dataset = ToxicityDataset(sample_texts, sample_labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # Test forward pass
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            texts = batch['text']
            labels = batch['label']
            
            # Simple tokenization for testing
            trainer = EnhancedBiLSTMTrainer(model, None)
            tokenized = trainer.tokenize_batch(texts)
            
            vocab_features = batch.get('vocab_features')
            semantic_features = batch.get('semantic_features')
            
            outputs = model(
                tokenized['input_ids'],
                vocab_features,
                semantic_features,
                tokenized['attention_mask']
            )
            
            print(f"\n📊 Batch Results:")
            print(f"   Texts: {texts}")
            print(f"   Labels: {labels}")
            print(f"   Main logits shape: {outputs['logits'].shape}")
            print(f"   Severity logits shape: {outputs['severity_logits'].shape}")
            print(f"   Category logits shape: {outputs['category_logits'].shape}")
            print(f"   Attention weights shape: {outputs['attention_weights'].shape}")
            
            # Show predictions
            predictions = torch.softmax(outputs['logits'], dim=1)
            print(f"   Predictions: {predictions}")
            
            break  # Test only first batch
    
    print(f"\n✅ Enhanced BiLSTM Model Ready!")
    print(f"   Features: BiLSTM, attention, vocabulary integration, semantic integration")
    print(f"   Architecture: Multi-task learning with auxiliary heads")
    print(f"   Capabilities: Text + enhanced features fusion, attention visualization")
