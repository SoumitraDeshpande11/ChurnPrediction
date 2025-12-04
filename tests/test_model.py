"""
Tests for the Churn Prediction Model
MLOps Series - Part 1
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import ChurnPreprocessor, get_sample_input


class TestPreprocessor:
    """Tests for the ChurnPreprocessor class."""
    
    def test_sample_input_has_required_fields(self):
        """Test that sample input has all required fields."""
        sample = get_sample_input()
        
        required_fields = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]
        
        for field in required_fields:
            assert field in sample, f"Missing field: {field}"
    
    def test_sample_input_types(self):
        """Test that sample input has correct types."""
        sample = get_sample_input()
        
        assert isinstance(sample['gender'], str)
        assert isinstance(sample['SeniorCitizen'], int)
        assert isinstance(sample['tenure'], int)
        assert isinstance(sample['MonthlyCharges'], (int, float))
        assert isinstance(sample['TotalCharges'], (int, float))


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
