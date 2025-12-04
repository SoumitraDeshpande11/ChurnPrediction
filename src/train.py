"""
Model Training Script for Customer Churn Prediction
MLOps Series - Part 1

Usage:
    python src/train.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib
import os
import json
from datetime import datetime
from preprocess import ChurnPreprocessor


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "v1")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from CSV."""
    if not os.path.exists(path):
        print(f"Data file not found at {path}")
        print("\nPlease download the Telco Customer Churn dataset:")
        print("   1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("   2. Download 'WA_Fn-UseC_-Telco-Customer-Churn.csv'")
        print(f"   3. Place it in: {os.path.dirname(path)}")
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from dataset")
    return df


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train the churn prediction model.
    
    Using RandomForest for:
    - Good baseline performance
    - Feature importance interpretability
    - Handles imbalanced data reasonably well
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',  # Handle class imbalance
        n_jobs=-1
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    print("Model training complete!")
    
    return model


def evaluate_model(
    model: RandomForestClassifier, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    feature_names: list
) -> dict:
    """
    Evaluate model performance and return metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "test_samples": len(y_test),
        "churn_rate": round(y_test.mean(), 4),
        "trained_at": datetime.now().isoformat()
    }
    
    print("\nModel Evaluation:")
    print(f"   Accuracy:  {metrics['accuracy']:.2%}")
    print(f"   Precision: {metrics['precision']:.2%}")
    print(f"   Recall:    {metrics['recall']:.2%}")
    print(f"   F1 Score:  {metrics['f1_score']:.2%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Top feature importances
    print("\nTop 10 Important Features:")
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importances.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    metrics['top_features'] = importances.head(10).to_dict('records')
    
    return metrics


def save_artifacts(
    model: RandomForestClassifier, 
    preprocessor: ChurnPreprocessor, 
    metrics: dict
):
    """Save model, preprocessor, and metrics."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    
    # Save preprocessor
    preprocessor.save(PREPROCESSOR_PATH)
    
    # Save metrics
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Customer Churn Prediction - Training Pipeline")
    print("=" * 60)
    
    # Load data
    df = load_data(DATA_PATH)
    
    # Initialize preprocessor
    preprocessor = ChurnPreprocessor()
    
    # Preprocess data
    print("\nPreprocessing data...")
    X, y = preprocessor.fit_transform(df)
    print(f"Features shape: {X.shape}")
    print(f"Churn rate: {y.mean():.2%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Maintain class distribution
    )
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Cross-validation
    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"CV F1 Scores: {cv_scores.round(4)}")
    print(f"CV F1 Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, preprocessor.feature_columns)
    metrics['cv_f1_mean'] = round(cv_scores.mean(), 4)
    metrics['cv_f1_std'] = round(cv_scores.std(), 4)
    
    # Save everything
    save_artifacts(model, preprocessor, metrics)
    
    print("\n" + "=" * 60)
    print("Training Pipeline Complete!")
    print("=" * 60)
    print(f"\nArtifacts saved in: {MODEL_DIR}")
    print("   - model.pkl")
    print("   - preprocessor.pkl") 
    print("   - metrics.json")
    print("\nNext: Run the API with 'python api/main.py'")


if __name__ == "__main__":
    main()
