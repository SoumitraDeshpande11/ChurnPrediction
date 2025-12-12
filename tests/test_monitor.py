"""
Tests for the Monitoring Agent
MLOps Series - Part 3
"""

import pytest
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.monitor import MLMonitorAgent


class TestMLMonitorAgent:
    """Test suite for monitoring agent."""
    
    @pytest.fixture
    def agent(self, tmp_path):
        """Create a monitoring agent with temporary directories."""
        agent = MLMonitorAgent(base_dir=str(tmp_path))
        
        # Create necessary directories
        os.makedirs(os.path.join(tmp_path, "data", "raw"), exist_ok=True)
        os.makedirs(os.path.join(tmp_path, "data", "processed"), exist_ok=True)
        os.makedirs(os.path.join(tmp_path, "models", "v1"), exist_ok=True)
        
        return agent
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.alert_threshold == "medium"
        assert agent.api_url == "https://churnprediction-bgio.onrender.com"
        assert agent.alerts == []
        assert agent.status["healthy"] == True
    
    def test_add_alert(self, agent):
        """Test alert addition."""
        agent.add_alert("TEST_ALERT", "Test message", "high")
        
        assert len(agent.alerts) == 1
        assert agent.alerts[0]["type"] == "TEST_ALERT"
        assert agent.alerts[0]["message"] == "Test message"
        assert agent.alerts[0]["severity"] == "high"
    
    @patch('requests.get')
    def test_api_health_check_success(self, mock_get, agent):
        """Test successful API health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = agent.check_api_health()
        
        assert result["status"] == "healthy"
        assert "latency_ms" in result["details"]
        assert result["details"]["status_code"] == 200
    
    @patch('requests.get')
    def test_api_health_check_failure(self, mock_get, agent):
        """Test failed API health check."""
        mock_get.side_effect = Exception("Connection error")
        
        result = agent.check_api_health()
        
        assert result["status"] == "failed"
        assert len(agent.alerts) > 0
        assert any(a["type"] == "API_DOWN" for a in agent.alerts)
    
    def test_model_performance_check_no_metrics(self, agent):
        """Test performance check when metrics file doesn't exist."""
        result = agent.check_model_performance()
        
        assert result["status"] == "no_data"
    
    def test_model_performance_check_with_metrics(self, agent, tmp_path):
        """Test performance check with metrics file."""
        metrics = {
            "accuracy": 0.75,
            "f1_score": 0.60,
            "precision": 0.58,
            "recall": 0.65
        }
        
        metrics_path = os.path.join(tmp_path, "models", "v1", "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        
        result = agent.check_model_performance()
        
        assert result["status"] == "healthy"
        assert result["details"]["current_metrics"]["accuracy"] == 0.75
    
    def test_model_performance_check_degraded(self, agent, tmp_path):
        """Test performance check with degraded metrics."""
        metrics = {
            "accuracy": 0.65,  # Below threshold
            "f1_score": 0.50,
            "precision": 0.48,
            "recall": 0.55
        }
        
        metrics_path = os.path.join(tmp_path, "models", "v1", "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        
        result = agent.check_model_performance()
        
        assert result["status"] == "degraded"
        assert len(agent.alerts) > 0
        assert any(a["type"] == "PERFORMANCE_DROP" for a in agent.alerts)
    
    @patch('requests.post')
    def test_prediction_patterns_check(self, mock_post, agent):
        """Test prediction pattern analysis."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "churn_probability": 0.72,
            "risk_level": "HIGH"
        }
        mock_post.return_value = mock_response
        
        result = agent.check_prediction_patterns()
        
        assert result["status"] == "healthy"
        assert result["details"]["sample_prediction"]["churn_probability"] == 0.72
    
    def test_system_resources_check(self, agent, tmp_path):
        """Test system resources check."""
        # Create model files
        model_dir = os.path.join(tmp_path, "models", "v1")
        for file in ["model.pkl", "preprocessor.pkl", "metrics.json"]:
            path = os.path.join(model_dir, file)
            with open(path, 'w') as f:
                f.write("dummy content")
        
        result = agent.check_system_resources()
        
        assert result["status"] == "healthy"
        assert "model.pkl" in result["details"]
    
    def test_system_resources_check_missing_files(self, agent):
        """Test system resources check with missing files."""
        result = agent.check_system_resources()
        
        assert result["status"] == "incomplete"
        assert len(agent.alerts) > 0
    
    def test_generate_intelligent_summary_healthy(self, agent):
        """Test summary generation for healthy system."""
        results = {
            "overall_status": "healthy",
            "alert_count": 0,
            "alerts": []
        }
        
        summary = agent.generate_intelligent_summary(results)
        
        assert "All systems operational" in summary
        assert "No issues detected" in summary
    
    def test_generate_intelligent_summary_with_alerts(self, agent):
        """Test summary generation with alerts."""
        agent.add_alert("DATA_DRIFT", "Feature drift detected", "medium")
        agent.add_alert("API_LATENCY", "High latency", "high")
        
        results = {
            "overall_status": "degraded",
            "alert_count": 2,
            "alerts": agent.alerts
        }
        
        summary = agent.generate_intelligent_summary(results)
        
        assert "degraded" in summary.lower()
        assert "2 issue(s)" in summary
        assert "DATA_DRIFT" in summary or "drift" in summary.lower()
    
    def test_save_monitor_log(self, agent, tmp_path):
        """Test monitor log saving."""
        results = {
            "timestamp": "2025-12-12T10:00:00",
            "overall_status": "healthy",
            "alert_count": 0
        }
        
        agent.save_monitor_log(results)
        
        log_path = os.path.join(tmp_path, "data", "processed", "monitor_log.json")
        assert os.path.exists(log_path)
        
        with open(log_path, 'r') as f:
            logs = json.load(f)
        
        assert "monitoring_runs" in logs
        assert len(logs["monitoring_runs"]) == 1
        assert logs["monitoring_runs"][0]["overall_status"] == "healthy"
    
    def test_thresholds_configuration(self):
        """Test different alert threshold configurations."""
        low_agent = MLMonitorAgent(alert_threshold="low")
        medium_agent = MLMonitorAgent(alert_threshold="medium")
        high_agent = MLMonitorAgent(alert_threshold="high")
        
        assert low_agent.thresholds["drift"] == 0.15
        assert medium_agent.thresholds["drift"] == 0.10
        assert high_agent.thresholds["drift"] == 0.05
        
        assert low_agent.thresholds["api_latency"] == 5000
        assert medium_agent.thresholds["api_latency"] == 3000
        assert high_agent.thresholds["api_latency"] == 1000


def test_monitor_script_imports():
    """Test that monitor script imports correctly."""
    from scripts import monitor
    assert hasattr(monitor, 'MLMonitorAgent')
    assert hasattr(monitor, 'main')


def test_monitor_agent_has_required_methods():
    """Test that agent has all required methods."""
    required_methods = [
        'run_full_check',
        'check_api_health',
        'check_model_performance',
        'check_data_drift',
        'check_prediction_patterns',
        'check_system_resources',
        'add_alert',
        'generate_intelligent_summary',
        'save_monitor_log'
    ]
    
    for method in required_methods:
        assert hasattr(MLMonitorAgent, method)
