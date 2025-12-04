"""
Comprehensive Tests for Churn Prediction System
MLOps Series - Part 2

Run with:
    pytest tests/ -v
"""

import pytest
import sys
import os
import json
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import ChurnPreprocessor, get_sample_input


# ============== Unit Tests ==============

class TestSampleInput:
    """Tests for sample input validation."""
    
    def test_sample_input_has_all_required_fields(self):
        """Verify sample input contains all required fields."""
        sample = get_sample_input()
        
        required_fields = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]
        
        for field in required_fields:
            assert field in sample, f"Missing required field: {field}"
    
    def test_sample_input_correct_types(self):
        """Verify sample input has correct data types."""
        sample = get_sample_input()
        
        assert isinstance(sample['gender'], str)
        assert isinstance(sample['SeniorCitizen'], int)
        assert sample['SeniorCitizen'] in [0, 1]
        assert isinstance(sample['tenure'], int)
        assert sample['tenure'] >= 0
        assert isinstance(sample['MonthlyCharges'], (int, float))
        assert isinstance(sample['TotalCharges'], (int, float))
    
    def test_sample_input_valid_categories(self):
        """Verify categorical values are valid."""
        sample = get_sample_input()
        
        assert sample['gender'] in ['Male', 'Female']
        assert sample['Partner'] in ['Yes', 'No']
        assert sample['Dependents'] in ['Yes', 'No']
        assert sample['Contract'] in ['Month-to-month', 'One year', 'Two year']


class TestPreprocessor:
    """Tests for ChurnPreprocessor class."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame([
            {
                'customerID': 'TEST001',
                'gender': 'Male',
                'SeniorCitizen': 0,
                'Partner': 'Yes',
                'Dependents': 'No',
                'tenure': 12,
                'PhoneService': 'Yes',
                'MultipleLines': 'No',
                'InternetService': 'DSL',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'Yes',
                'DeviceProtection': 'No',
                'TechSupport': 'Yes',
                'StreamingTV': 'No',
                'StreamingMovies': 'No',
                'Contract': 'One year',
                'PaperlessBilling': 'No',
                'PaymentMethod': 'Bank transfer (automatic)',
                'MonthlyCharges': 55.0,
                'TotalCharges': 660.0,
                'Churn': 'No'
            },
            {
                'customerID': 'TEST002',
                'gender': 'Female',
                'SeniorCitizen': 1,
                'Partner': 'No',
                'Dependents': 'No',
                'tenure': 2,
                'PhoneService': 'Yes',
                'MultipleLines': 'Yes',
                'InternetService': 'Fiber optic',
                'OnlineSecurity': 'No',
                'OnlineBackup': 'No',
                'DeviceProtection': 'No',
                'TechSupport': 'No',
                'StreamingTV': 'Yes',
                'StreamingMovies': 'Yes',
                'Contract': 'Month-to-month',
                'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 95.0,
                'TotalCharges': 190.0,
                'Churn': 'Yes'
            }
        ])
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initializes correctly."""
        preprocessor = ChurnPreprocessor()
        
        assert preprocessor.label_encoders == {}
        assert preprocessor.feature_columns == []
    
    def test_fit_transform_returns_correct_shape(self, sample_dataframe):
        """Test fit_transform returns correct shapes."""
        preprocessor = ChurnPreprocessor()
        X, y = preprocessor.fit_transform(sample_dataframe)
        
        assert X.shape[0] == 2  # 2 samples
        assert y.shape[0] == 2
    
    def test_fit_transform_correct_labels(self, sample_dataframe):
        """Test target encoding is correct."""
        preprocessor = ChurnPreprocessor()
        X, y = preprocessor.fit_transform(sample_dataframe)
        
        assert y[0] == 0  # 'No' -> 0
        assert y[1] == 1  # 'Yes' -> 1


class TestPredictorIntegration:
    """Integration tests - only run if model exists."""
    
    @pytest.fixture
    def predictor(self):
        """Load predictor if model exists."""
        from src.predict import ChurnPredictor
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "v1", "model.pkl")
        
        if not os.path.exists(model_path):
            pytest.skip("Model not trained yet. Run 'python src/train.py' first.")
        
        return ChurnPredictor()
    
    def test_single_prediction(self, predictor):
        """Test single customer prediction."""
        sample = get_sample_input()
        result = predictor.predict(sample)
        
        assert 'churn_prediction' in result
        assert 'churn_probability' in result
        assert 'risk_level' in result
        assert isinstance(result['churn_prediction'], bool)
        assert 0 <= result['churn_probability'] <= 1
        assert result['risk_level'] in ['LOW', 'MEDIUM', 'HIGH']
    
    def test_batch_prediction(self, predictor):
        """Test batch prediction."""
        samples = [get_sample_input(), get_sample_input()]
        results = predictor.predict_batch(samples)
        
        assert len(results) == 2
        for result in results:
            assert 'churn_prediction' in result
    
    def test_model_info(self, predictor):
        """Test model info retrieval."""
        info = predictor.get_model_info()
        
        assert 'model_type' in info
        assert 'feature_count' in info
        assert info['feature_count'] > 0
    
    def test_high_risk_customer(self, predictor):
        """Test high-risk customer gets elevated churn probability."""
        high_risk = {
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 100.0,
            "TotalCharges": 100.0
        }
        
        result = predictor.predict(high_risk)
        assert result['churn_probability'] > 0.3
    
    def test_low_risk_customer(self, predictor):
        """Test low-risk customer gets low churn probability."""
        low_risk = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 72,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 90.0,
            "TotalCharges": 6480.0
        }
        
        result = predictor.predict(low_risk)
        assert result['churn_probability'] < 0.5


class TestModelQuality:
    """Tests for model quality metrics."""
    
    @pytest.fixture
    def metrics(self):
        """Load model metrics."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        metrics_path = os.path.join(base_dir, "models", "v1", "metrics.json")
        
        if not os.path.exists(metrics_path):
            pytest.skip("Metrics file not found. Train model first.")
        
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    def test_accuracy_threshold(self, metrics):
        """Test model accuracy meets minimum threshold."""
        assert metrics['accuracy'] >= 0.70, "Accuracy below 70% threshold"
    
    def test_f1_score_threshold(self, metrics):
        """Test F1 score meets minimum threshold."""
        assert metrics['f1_score'] >= 0.50, "F1 score below 50% threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
