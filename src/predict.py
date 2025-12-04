"""
Prediction Module for Customer Churn
MLOps Series - Part 1
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import ChurnPreprocessor


class ChurnPredictor:
    """
    Handles loading model and making predictions.
    Designed to be used by the FastAPI service.
    """
    
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, "models", "v1")
        
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model and preprocessor from disk."""
        model_path = os.path.join(self.model_dir, "model.pkl")
        preprocessor_path = os.path.join(self.model_dir, "preprocessor.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run 'python src/train.py' first."
            )
        
        self.model = joblib.load(model_path)
        self.preprocessor = ChurnPreprocessor()
        self.preprocessor.load(preprocessor_path)
        
        print(f"Model and preprocessor loaded from {self.model_dir}")
    
    def predict(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a churn prediction for a single customer.
        
        Args:
            customer_data: Dictionary with customer features
            
        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Add dummy customerID if not present
        if 'customerID' not in df.columns:
            df['customerID'] = 'API_REQUEST'
        
        # Preprocess
        X = self.preprocessor.transform(df)
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        return {
            "churn_prediction": bool(prediction),
            "churn_label": "Yes" if prediction == 1 else "No",
            "churn_probability": round(float(probability[1]), 4),
            "retention_probability": round(float(probability[0]), 4),
            "risk_level": self._get_risk_level(probability[1])
        }
    
    def predict_batch(self, customers: list) -> list:
        """
        Make predictions for multiple customers.
        
        Args:
            customers: List of customer dictionaries
            
        Returns:
            List of prediction results
        """
        return [self.predict(customer) for customer in customers]
    
    def _get_risk_level(self, churn_prob: float) -> str:
        """Categorize churn risk level."""
        if churn_prob >= 0.7:
            return "HIGH"
        elif churn_prob >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        metrics_path = os.path.join(self.model_dir, "metrics.json")
        
        info = {
            "model_type": type(self.model).__name__,
            "model_dir": self.model_dir,
            "feature_count": len(self.preprocessor.feature_columns),
            "features": self.preprocessor.feature_columns
        }
        
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r') as f:
                info["metrics"] = json.load(f)
        
        return info
