"""
Data Preprocessing Module for Customer Churn Prediction
MLOps Series - Part 1
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, Any
import joblib
import os


class ChurnPreprocessor:
    """
    Handles all data preprocessing for the Churn prediction model.
    Includes encoding, scaling, and feature engineering.
    """
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessors and transform training data.
        
        Args:
            df: Raw training DataFrame
            
        Returns:
            Tuple of (X features array, y target array)
        """
        df = self._clean_data(df)
        
        # Separate target
        y = df['Churn'].map({'Yes': 1, 'No': 0}).values
        X_df = df.drop(columns=['Churn', 'customerID'])
        
        # Identify column types
        self.categorical_columns = X_df.select_dtypes(include=['object']).columns.tolist()
        self.numerical_columns = X_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Encode categorical columns
        for col in self.categorical_columns:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
            self.label_encoders[col] = le
        
        # Scale numerical columns
        X_df[self.numerical_columns] = self.scaler.fit_transform(X_df[self.numerical_columns])
        
        self.feature_columns = X_df.columns.tolist()
        
        return X_df.values, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessors.
        
        Args:
            df: Raw DataFrame to transform
            
        Returns:
            Transformed features array
        """
        df = self._clean_data(df)
        
        # Remove target and ID if present
        if 'Churn' in df.columns:
            df = df.drop(columns=['Churn'])
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
        
        # Encode categorical columns
        for col in self.categorical_columns:
            if col in df.columns:
                # Handle unseen categories
                le = self.label_encoders[col]
                df[col] = df[col].apply(
                    lambda x: le.transform([str(x)])[0] 
                    if str(x) in le.classes_ 
                    else -1
                )
        
        # Scale numerical columns
        df[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])
        
        # Ensure column order matches training
        df = df[self.feature_columns]
        
        return df.values
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data: handle missing values, convert types.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # TotalCharges has some empty strings, convert to numeric
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        
        # Fill any remaining missing values
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('Unknown')
        
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def save(self, path: str):
        """Save preprocessor to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns
        }, path)
        print(f"✅ Preprocessor saved to {path}")
    
    def load(self, path: str):
        """Load preprocessor from disk."""
        data = joblib.load(path)
        self.label_encoders = data['label_encoders']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.categorical_columns = data['categorical_columns']
        self.numerical_columns = data['numerical_columns']
        print(f"✅ Preprocessor loaded from {path}")
        return self


def get_sample_input() -> Dict[str, Any]:
    """
    Returns a sample input for testing the API.
    """
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.2
    }
