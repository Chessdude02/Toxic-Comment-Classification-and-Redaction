"""
Dataset Manager for Toxic Comment Classification

This module handles dataset loading, preprocessing, and management for the toxic comment classification system.
Supports multiple data sources and formats.
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import pandas as pd
import numpy as np
import os
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import zipfile
import json
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


class ToxicCommentDatasetManager:
    """
    Manages toxic comment datasets with support for multiple data sources.
    """
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            self.data_dir = Path(__file__).parent
        else:
            self.data_dir = Path(data_dir)
        
        self.data_dir.mkdir(exist_ok=True)
        
        # Standard toxicity labels
        self.toxicity_labels = [
            'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
        ]
        
        # Dataset metadata
        self.dataset_info = {}
        
    def download_jigsaw_dataset(self) -> bool:
        """
        Download the Jigsaw Toxic Comment Classification dataset.
        Note: This requires Kaggle API setup or manual download.
        """
        
        kaggle_dataset_url = "jigsaw-toxic-comment-classification-challenge"
        
        print("📥 To download the Jigsaw dataset, you have two options:")
        print("\n1. KAGGLE API (Recommended):")
        print("   - Install kaggle: pip install kaggle")
        print("   - Setup API key: https://www.kaggle.com/docs/api")
        print("   - Run: kaggle competitions download -c jigsaw-toxic-comment-classification-challenge")
        print("\n2. MANUAL DOWNLOAD:")
        print("   - Visit: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data")
        print("   - Download train.csv and test.csv")
        print(f"   - Place files in: {self.data_dir}")
        
        return False
    
    def load_sample_dataset(self) -> pd.DataFrame:
        """
        Create a sample dataset for testing when the full dataset isn't available.
        """
        
        sample_data = {
            'id': range(1, 21),
            'comment_text': [
                "This is a great article, thanks for sharing!",
                "I completely disagree with this viewpoint.",
                "You're absolutely stupid if you believe this nonsense.",
                "Thanks for the detailed explanation.",
                "This is helpful information, much appreciated.",
                "What a load of garbage, this writer is an idiot.",
                "I found this very informative and well-written.",
                "Complete waste of time reading this trash.",
                "Excellent points made throughout the article.",
                "The author is clearly biased and ignorant.",
                "Great job on researching this topic thoroughly.",
                "This is the dumbest thing I've ever read.",
                "I appreciate the different perspective offered here.",
                "Whoever wrote this should be ashamed.",
                "Well-structured argument with solid evidence.",
                "This makes me so angry, what terrible content.",
                "Thank you for taking the time to explain this.",
                "I hate this kind of misleading information.",
                "Very educational and easy to understand.",
                "The comments section is full of morons."
            ],
            'toxic': [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'severe_toxic': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'obscene': [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'threat': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'insult': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            'identity_hate': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Save sample dataset
        sample_path = self.data_dir / "sample_dataset.csv"
        df.to_csv(sample_path, index=False)
        
        print(f"✅ Created sample dataset with {len(df)} comments at: {sample_path}")
        return df
    
    def load_dataset(self, dataset_name: str = "jigsaw") -> Optional[pd.DataFrame]:
        """
        Load the specified dataset.
        """
        
        if dataset_name == "jigsaw":
            # Try to load the full Jigsaw dataset
            train_path = self.data_dir / "train.csv"
            if train_path.exists():
                print(f"📁 Loading Jigsaw dataset from: {train_path}")
                df = pd.read_csv(train_path)
                print(f"✅ Loaded {len(df)} comments from Jigsaw dataset")
                return df
            else:
                print("⚠️ Jigsaw dataset not found. Creating sample dataset...")
                return self.load_sample_dataset()
        
        elif dataset_name == "sample":
            return self.load_sample_dataset()
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def prepare_training_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training with proper splits.
        """
        
        # Ensure required columns exist
        required_columns = ['comment_text'] + self.toxicity_labels
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean the data
        df = df.dropna(subset=['comment_text'])
        df = df[df['comment_text'].str.strip() != '']
        
        # Features and labels
        X = df['comment_text']
        y = df[self.toxicity_labels]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y.iloc[:, 0]
        )
        
        print(f"📊 Dataset split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing: {len(X_test)} samples")
        print(f"   Features: {len(self.toxicity_labels)} toxicity labels")
        
        # Calculate class distribution
        self._print_class_distribution(y_train, "Training")
        self._print_class_distribution(y_test, "Testing")
        
        return X_train, X_test, y_train, y_test
    
    def _print_class_distribution(self, y: pd.DataFrame, split_name: str):
        """Print class distribution statistics."""
        print(f"\n📈 {split_name} set class distribution:")
        for label in self.toxicity_labels:
            positive_count = y[label].sum()
            positive_rate = positive_count / len(y)
            print(f"   {label}: {positive_count} ({positive_rate:.1%})")
    
    def get_class_weights(self, y_train: pd.DataFrame) -> Dict[str, Dict[int, float]]:
        """
        Calculate class weights for imbalanced data.
        """
        
        class_weights = {}

        for label in self.toxicity_labels:
            # Calculate weights for binary classification. A label may have
            # only one class present in a given split (common with small/
            # skewed datasets), which compute_class_weight can't balance —
            # fall back to neutral weights in that case.
            present_classes = np.unique(y_train[label])
            if len(present_classes) < 2:
                class_weights[label] = {0: 1.0, 1: 1.0}
                continue

            weights = compute_class_weight(
                'balanced',
                classes=np.array([0, 1]),
                y=y_train[label]
            )
            class_weights[label] = {0: weights[0], 1: weights[1]}

        return class_weights
    
    def create_dataset_summary(self, df: pd.DataFrame) -> Dict:
        """
        Create a comprehensive dataset summary.
        """
        
        summary = {
            'total_samples': len(df),
            'columns': list(df.columns),
            'toxicity_distribution': {},
            'text_statistics': {},
            'missing_data': {},
            'dataset_quality': {}
        }
        
        # Toxicity distribution
        for label in self.toxicity_labels:
            if label in df.columns:
                positive_count = df[label].sum()
                summary['toxicity_distribution'][label] = {
                    'positive_samples': int(positive_count),
                    'negative_samples': int(len(df) - positive_count),
                    'positive_rate': float(positive_count / len(df))
                }
        
        # Text statistics
        if 'comment_text' in df.columns:
            text_lengths = df['comment_text'].str.len()
            word_counts = df['comment_text'].str.split().str.len()
            
            summary['text_statistics'] = {
                'avg_character_length': float(text_lengths.mean()),
                'max_character_length': int(text_lengths.max()),
                'min_character_length': int(text_lengths.min()),
                'avg_word_count': float(word_counts.mean()),
                'max_word_count': int(word_counts.max()),
                'min_word_count': int(word_counts.min())
            }
        
        # Missing data analysis
        summary['missing_data'] = df.isnull().sum().to_dict()
        
        # Dataset quality metrics
        if 'comment_text' in df.columns:
            empty_comments = (df['comment_text'].str.strip() == '').sum()
            duplicate_comments = df['comment_text'].duplicated().sum()
            
            summary['dataset_quality'] = {
                'empty_comments': int(empty_comments),
                'duplicate_comments': int(duplicate_comments),
                'completeness_rate': float(1 - empty_comments / len(df)),
                'uniqueness_rate': float(1 - duplicate_comments / len(df))
            }
        
        return summary
    
    def save_dataset_info(self, df: pd.DataFrame, filename: str = "dataset_info.json"):
        """
        Save dataset information to file.
        """
        
        summary = self.create_dataset_summary(df)
        info_path = self.data_dir / filename
        
        with open(info_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Dataset information saved to: {info_path}")
    
    def load_or_create_dataset(self, prefer_sample: bool = False) -> pd.DataFrame:
        """
        Load the best available dataset or create a sample one.
        """
        
        if prefer_sample:
            return self.load_sample_dataset()
        
        # Try to load the full dataset first
        try:
            df = self.load_dataset("jigsaw")
            self.save_dataset_info(df)
            return df
        except Exception as e:
            print(f"⚠️ Could not load full dataset: {e}")
            print("📝 Creating sample dataset for demonstration...")
            return self.load_sample_dataset()
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate dataset integrity and format.
        """
        
        validation_results = {}
        
        # Check required columns
        validation_results['has_comment_text'] = 'comment_text' in df.columns
        validation_results['has_toxicity_labels'] = all(label in df.columns for label in self.toxicity_labels)
        
        # Check data types
        validation_results['comment_text_is_string'] = df['comment_text'].dtype == 'object'
        validation_results['labels_are_numeric'] = all(
            pd.api.types.is_numeric_dtype(df[label]) for label in self.toxicity_labels if label in df.columns
        )
        
        # Check for empty data
        validation_results['no_empty_comments'] = not df['comment_text'].isnull().any()
        validation_results['has_sufficient_data'] = len(df) >= 10  # Minimum for training
        
        # Check label range
        for label in self.toxicity_labels:
            if label in df.columns:
                validation_results[f'{label}_valid_range'] = df[label].isin([0, 1]).all()
        
        return validation_results
    
    def print_dataset_summary(self, df: pd.DataFrame):
        """
        Print a comprehensive dataset summary.
        """
        
        print("=" * 60)
        print("📊 DATASET SUMMARY")
        print("=" * 60)
        
        summary = self.create_dataset_summary(df)
        
        print(f"📁 Total Samples: {summary['total_samples']:,}")
        print(f"📋 Columns: {len(summary['columns'])}")
        
        print("\n🎯 TOXICITY DISTRIBUTION:")
        for label, stats in summary['toxicity_distribution'].items():
            print(f"   {label:15}: {stats['positive_samples']:6,} toxic ({stats['positive_rate']:6.1%}) | {stats['negative_samples']:6,} clean")
        
        if 'text_statistics' in summary:
            print("\n📝 TEXT STATISTICS:")
            text_stats = summary['text_statistics']
            print(f"   Avg Length: {text_stats['avg_character_length']:.0f} chars | {text_stats['avg_word_count']:.0f} words")
            print(f"   Max Length: {text_stats['max_character_length']:,} chars | {text_stats['max_word_count']:,} words")
            print(f"   Min Length: {text_stats['min_character_length']} chars | {text_stats['min_word_count']} words")
        
        if 'dataset_quality' in summary:
            print("\n✅ QUALITY METRICS:")
            quality = summary['dataset_quality']
            print(f"   Completeness: {quality['completeness_rate']:.1%}")
            print(f"   Uniqueness: {quality['uniqueness_rate']:.1%}")
            print(f"   Empty Comments: {quality['empty_comments']}")
            print(f"   Duplicate Comments: {quality['duplicate_comments']}")
        
        # Validation
        validation = self.validate_dataset(df)
        print("\n🔍 VALIDATION RESULTS:")
        for check, result in validation.items():
            status = "✅" if result else "❌"
            print(f"   {status} {check.replace('_', ' ').title()}")
        
        print("=" * 60)


if __name__ == "__main__":
    # Test the dataset manager
    manager = ToxicCommentDatasetManager()
    
    # Load dataset
    df = manager.load_or_create_dataset(prefer_sample=True)

    # Save dataset metadata
    manager.save_dataset_info(df)

    # Print summary
    manager.print_dataset_summary(df)
    
    # Prepare training data
    X_train, X_test, y_train, y_test = manager.prepare_training_data(df)
    
    # Calculate class weights
    class_weights = manager.get_class_weights(y_train)
    print(f"\n⚖️ Class weights calculated for {len(class_weights)} labels")
