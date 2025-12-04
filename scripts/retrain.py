"""
Auto-Retraining Pipeline for Customer Churn Model
MLOps Series - Part 2

This script:
1. Checks for new data
2. Validates data quality
3. Retrains the model
4. Compares with existing model
5. Promotes if better, otherwise keeps old model

Usage:
    python scripts/retrain.py
    python scripts/retrain.py --force  # Force retrain even without new data
"""

import os
import sys
import json
import argparse
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from src.preprocess import ChurnPreprocessor


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_HASH_PATH = os.path.join(BASE_DIR, "data", "processed", "data_hash.txt")


def get_data_hash(filepath: str) -> str:
    """Calculate MD5 hash of data file to detect changes."""
    if not os.path.exists(filepath):
        return ""
    
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def has_new_data() -> bool:
    """Check if data has changed since last training."""
    current_hash = get_data_hash(DATA_PATH)
    
    if not os.path.exists(DATA_HASH_PATH):
        return True
    
    with open(DATA_HASH_PATH, 'r') as f:
        stored_hash = f.read().strip()
    
    return current_hash != stored_hash


def save_data_hash():
    """Save current data hash for future comparison."""
    current_hash = get_data_hash(DATA_PATH)
    os.makedirs(os.path.dirname(DATA_HASH_PATH), exist_ok=True)
    
    with open(DATA_HASH_PATH, 'w') as f:
        f.write(current_hash)


def get_next_version() -> str:
    """Get next model version number."""
    existing_versions = []
    
    if os.path.exists(MODELS_DIR):
        for name in os.listdir(MODELS_DIR):
            if name.startswith('v') and os.path.isdir(os.path.join(MODELS_DIR, name)):
                try:
                    version_num = int(name[1:])
                    existing_versions.append(version_num)
                except ValueError:
                    continue
    
    next_version = max(existing_versions, default=0) + 1
    return f"v{next_version}"


def get_current_version() -> Optional[str]:
    """Get current (latest) model version."""
    existing_versions = []
    
    if os.path.exists(MODELS_DIR):
        for name in os.listdir(MODELS_DIR):
            if name.startswith('v') and os.path.isdir(os.path.join(MODELS_DIR, name)):
                try:
                    version_num = int(name[1:])
                    existing_versions.append(version_num)
                except ValueError:
                    continue
    
    if existing_versions:
        return f"v{max(existing_versions)}"
    return None


def load_current_metrics() -> Optional[Dict[str, Any]]:
    """Load metrics from current model version."""
    current_version = get_current_version()
    if not current_version:
        return None
    
    metrics_path = os.path.join(MODELS_DIR, current_version, "metrics.json")
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def validate_data(df: pd.DataFrame) -> bool:
    """Validate data quality before training."""
    print("\nValidating data quality...")
    
    # Check minimum rows
    if len(df) < 100:
        print(f"  FAIL: Not enough data ({len(df)} rows, need 100+)")
        return False
    print(f"  OK: {len(df)} rows")
    
    # Check required columns
    required_cols = ['Churn', 'tenure', 'MonthlyCharges', 'Contract']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"  FAIL: Missing columns: {missing}")
        return False
    print(f"  OK: All required columns present")
    
    # Check target distribution
    churn_rate = df['Churn'].value_counts(normalize=True).get('Yes', 0)
    if churn_rate < 0.05 or churn_rate > 0.95:
        print(f"  FAIL: Extreme churn rate ({churn_rate:.2%})")
        return False
    print(f"  OK: Churn rate is {churn_rate:.2%}")
    
    # Check for excessive missing values
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if missing_pct > 0.20:
        print(f"  FAIL: Too many missing values ({missing_pct:.2%})")
        return False
    print(f"  OK: Missing values {missing_pct:.2%}")
    
    print("Data validation passed!")
    return True


def train_new_model(df: pd.DataFrame) -> tuple:
    """Train a new model and return model, preprocessor, and metrics."""
    print("\nTraining new model...")
    
    # Preprocess
    preprocessor = ChurnPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4),
        "test_samples": len(y_test),
        "train_samples": len(y_train),
        "churn_rate": round(y.mean(), 4),
        "trained_at": datetime.now().isoformat()
    }
    
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1 Score:  {metrics['f1_score']:.2%}")
    print(f"  CV F1:     {metrics['cv_f1_mean']:.4f} (+/- {metrics['cv_f1_std']*2:.4f})")
    
    return model, preprocessor, metrics


def compare_models(new_metrics: Dict, old_metrics: Optional[Dict]) -> bool:
    """Compare new model with old model. Returns True if new is better."""
    if old_metrics is None:
        print("\nNo existing model to compare. New model will be saved.")
        return True
    
    print("\nComparing models...")
    print(f"  {'Metric':<15} {'Old':>10} {'New':>10} {'Change':>10}")
    print(f"  {'-'*45}")
    
    # Primary metric: F1 Score
    old_f1 = old_metrics.get('f1_score', 0)
    new_f1 = new_metrics['f1_score']
    change = new_f1 - old_f1
    
    print(f"  {'F1 Score':<15} {old_f1:>10.4f} {new_f1:>10.4f} {change:>+10.4f}")
    
    # Secondary metrics
    for metric in ['accuracy', 'precision', 'recall']:
        old_val = old_metrics.get(metric, 0)
        new_val = new_metrics[metric]
        change = new_val - old_val
        print(f"  {metric.title():<15} {old_val:>10.4f} {new_val:>10.4f} {change:>+10.4f}")
    
    # Decision: New model must have better or equal F1 score
    # and not significantly worse on other metrics
    is_better = new_f1 >= old_f1
    
    if is_better:
        print("\n  New model is BETTER or EQUAL. Will be promoted.")
    else:
        print("\n  New model is WORSE. Keeping old model.")
    
    return is_better


def save_model(model, preprocessor, metrics: Dict, version: str):
    """Save model artifacts to versioned directory."""
    version_dir = os.path.join(MODELS_DIR, version)
    os.makedirs(version_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(version_dir, "model.pkl")
    joblib.dump(model, model_path)
    
    # Save preprocessor
    preprocessor_path = os.path.join(version_dir, "preprocessor.pkl")
    preprocessor.save(preprocessor_path)
    
    # Save metrics
    metrics['version'] = version
    metrics_path = os.path.join(version_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nModel saved to {version_dir}")
    print(f"  - model.pkl")
    print(f"  - preprocessor.pkl")
    print(f"  - metrics.json")


def main():
    """Main retraining pipeline."""
    parser = argparse.ArgumentParser(description="Retrain churn prediction model")
    parser.add_argument('--force', action='store_true', help='Force retrain even without new data')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Auto-Retraining Pipeline")
    print("=" * 60)
    
    # Check for new data
    if not args.force and not has_new_data():
        print("\nNo new data detected. Skipping retraining.")
        print("Use --force to retrain anyway.")
        return
    
    if args.force:
        print("\nForce retrain enabled.")
    else:
        print("\nNew data detected!")
    
    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"\nError: Data file not found at {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded {len(df)} rows from dataset")
    
    # Validate data
    if not validate_data(df):
        print("\nData validation failed. Aborting retraining.")
        return
    
    # Train new model
    model, preprocessor, new_metrics = train_new_model(df)
    
    # Load current metrics for comparison
    old_metrics = load_current_metrics()
    
    # Compare models
    if compare_models(new_metrics, old_metrics):
        # Save new model
        version = get_next_version()
        save_model(model, preprocessor, new_metrics, version)
        
        # Update data hash
        save_data_hash()
        
        print("\n" + "=" * 60)
        print(f"Retraining complete! New model: {version}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Retraining complete. No model update needed.")
        print("=" * 60)


if __name__ == "__main__":
    main()
