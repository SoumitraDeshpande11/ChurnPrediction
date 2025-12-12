"""
Intelligent Monitoring Agent for ML Pipeline
MLOps Series - Part 3

This agent monitors:
1. Data drift (statistical changes in features)
2. Model performance degradation
3. API health and availability
4. Feature distribution shifts
5. Prediction patterns

Usage:
    python scripts/monitor.py
    python scripts/monitor.py --alert-threshold high
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from scipy import stats
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MLMonitorAgent:
    """Intelligent agent for monitoring ML pipeline health."""
    
    def __init__(
        self,
        base_dir: str = None,
        api_url: str = None,
        alert_threshold: str = "medium"
    ):
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.api_url = api_url or "https://churnprediction-bgio.onrender.com"
        self.alert_threshold = alert_threshold
        
        # Paths
        self.data_path = os.path.join(self.base_dir, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
        self.model_dir = os.path.join(self.base_dir, "models", "v1")
        self.metrics_path = os.path.join(self.model_dir, "metrics.json")
        self.baseline_path = os.path.join(self.base_dir, "data", "processed", "baseline_stats.json")
        self.monitor_log_path = os.path.join(self.base_dir, "data", "processed", "monitor_log.json")
        
        # Alert thresholds
        self.thresholds = {
            "low": {"drift": 0.15, "performance_drop": 0.10, "api_latency": 5000},
            "medium": {"drift": 0.10, "performance_drop": 0.05, "api_latency": 3000},
            "high": {"drift": 0.05, "performance_drop": 0.03, "api_latency": 1000}
        }[alert_threshold]
        
        self.alerts = []
        self.status = {"healthy": True, "issues": []}
    
    def run_full_check(self) -> Dict[str, Any]:
        """Run all monitoring checks and return comprehensive report."""
        print("=" * 60)
        print("ML PIPELINE MONITORING AGENT")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Alert Threshold: {self.alert_threshold.upper()}")
        print()
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {}
        }
        
        # 1. API Health Check
        print("[1/5] Checking API Health...")
        api_result = self.check_api_health()
        results["checks"]["api_health"] = api_result
        
        # 2. Model Performance Check
        print("[2/5] Checking Model Performance...")
        perf_result = self.check_model_performance()
        results["checks"]["model_performance"] = perf_result
        
        # 3. Data Drift Check
        print("[3/5] Checking Data Drift...")
        drift_result = self.check_data_drift()
        results["checks"]["data_drift"] = drift_result
        
        # 4. Prediction Pattern Analysis
        print("[4/5] Analyzing Prediction Patterns...")
        pattern_result = self.check_prediction_patterns()
        results["checks"]["prediction_patterns"] = pattern_result
        
        # 5. System Resources Check
        print("[5/5] Checking System Resources...")
        resource_result = self.check_system_resources()
        results["checks"]["system_resources"] = resource_result
        
        # Determine overall status
        if self.alerts:
            results["overall_status"] = "degraded" if len(self.alerts) < 3 else "critical"
            self.status["healthy"] = False
        
        results["alerts"] = self.alerts
        results["alert_count"] = len(self.alerts)
        
        # Generate intelligent summary
        print("\n" + "=" * 60)
        print("MONITORING SUMMARY")
        print("=" * 60)
        summary = self.generate_intelligent_summary(results)
        results["summary"] = summary
        print(summary)
        print()
        
        # Save monitoring log
        self.save_monitor_log(results)
        
        return results
    
    def check_api_health(self) -> Dict[str, Any]:
        """Check API availability and response time."""
        result = {"status": "unknown", "details": {}}
        
        try:
            # Test health endpoint
            start_time = datetime.now()
            response = requests.get(f"{self.api_url}/health", timeout=10)
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            result["details"]["latency_ms"] = round(latency, 2)
            result["details"]["status_code"] = response.status_code
            
            if response.status_code == 200:
                result["status"] = "healthy"
                print(f"   ✓ API responding (latency: {latency:.0f}ms)")
                
                # Check if latency is concerning
                if latency > self.thresholds["api_latency"]:
                    self.add_alert(
                        "API_LATENCY",
                        f"API latency is {latency:.0f}ms (threshold: {self.thresholds['api_latency']}ms)",
                        "medium"
                    )
            else:
                result["status"] = "degraded"
                self.add_alert("API_ERROR", f"API returned status code {response.status_code}", "high")
        
        except requests.exceptions.RequestException as e:
            result["status"] = "failed"
            result["details"]["error"] = str(e)
            self.add_alert("API_DOWN", f"API is unreachable: {str(e)}", "critical")
            print(f"   ✗ API check failed: {str(e)}")
        
        return result
    
    def check_model_performance(self) -> Dict[str, Any]:
        """Check if model performance has degraded."""
        result = {"status": "unknown", "details": {}}
        
        try:
            # Load current metrics
            if not os.path.exists(self.metrics_path):
                result["status"] = "no_data"
                print("   - No metrics file found")
                return result
            
            with open(self.metrics_path, 'r') as f:
                current_metrics = json.load(f)
            
            result["details"]["current_metrics"] = {
                "accuracy": current_metrics["accuracy"],
                "f1_score": current_metrics["f1_score"],
                "precision": current_metrics["precision"],
                "recall": current_metrics["recall"]
            }
            
            # Check against minimum thresholds
            min_accuracy = 0.70
            min_f1 = 0.55
            
            if current_metrics["accuracy"] < min_accuracy:
                self.add_alert(
                    "PERFORMANCE_DROP",
                    f"Accuracy dropped to {current_metrics['accuracy']:.2%} (min: {min_accuracy:.0%})",
                    "high"
                )
                result["status"] = "degraded"
            elif current_metrics["f1_score"] < min_f1:
                self.add_alert(
                    "PERFORMANCE_DROP",
                    f"F1 score dropped to {current_metrics['f1_score']:.2%} (min: {min_f1:.0%})",
                    "high"
                )
                result["status"] = "degraded"
            else:
                result["status"] = "healthy"
                print(f"   ✓ Model performing well (F1: {current_metrics['f1_score']:.2%})")
        
        except Exception as e:
            result["status"] = "error"
            result["details"]["error"] = str(e)
            print(f"   ✗ Performance check failed: {str(e)}")
        
        return result
    
    def check_data_drift(self) -> Dict[str, Any]:
        """Detect statistical drift in feature distributions."""
        result = {"status": "unknown", "details": {"drifted_features": []}}
        
        try:
            # Load current data
            if not os.path.exists(self.data_path):
                result["status"] = "no_data"
                print("   - No data file found")
                return result
            
            df = pd.read_csv(self.data_path)
            
            # Load or create baseline
            if os.path.exists(self.baseline_path):
                with open(self.baseline_path, 'r') as f:
                    baseline = json.load(f)
            else:
                # First run - create baseline
                baseline = self.create_baseline_stats(df)
                result["status"] = "baseline_created"
                print("   ✓ Baseline statistics created")
                return result
            
            # Check numerical features for drift
            numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
            drifted_features = []
            
            for feature in numerical_features:
                if feature not in df.columns:
                    continue
                
                current_mean = df[feature].mean()
                current_std = df[feature].std()
                
                baseline_mean = baseline.get(feature, {}).get("mean", current_mean)
                baseline_std = baseline.get(feature, {}).get("std", current_std)
                
                # Calculate drift (relative change in mean)
                if baseline_mean != 0:
                    drift = abs(current_mean - baseline_mean) / baseline_mean
                    
                    if drift > self.thresholds["drift"]:
                        drifted_features.append({
                            "feature": feature,
                            "drift_score": round(drift, 4),
                            "baseline_mean": round(baseline_mean, 2),
                            "current_mean": round(current_mean, 2)
                        })
            
            result["details"]["drifted_features"] = drifted_features
            
            if drifted_features:
                result["status"] = "drift_detected"
                for feat in drifted_features:
                    self.add_alert(
                        "DATA_DRIFT",
                        f"{feat['feature']} drifted {feat['drift_score']:.1%} from baseline",
                        "medium"
                    )
                print(f"   ⚠ Drift detected in {len(drifted_features)} feature(s)")
            else:
                result["status"] = "stable"
                print("   ✓ No significant drift detected")
        
        except Exception as e:
            result["status"] = "error"
            result["details"]["error"] = str(e)
            print(f"   ✗ Drift check failed: {str(e)}")
        
        return result
    
    def check_prediction_patterns(self) -> Dict[str, Any]:
        """Analyze prediction patterns for anomalies."""
        result = {"status": "unknown", "details": {}}
        
        try:
            # Test prediction endpoint with sample data
            sample_data = {
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
            
            response = requests.post(
                f"{self.api_url}/predict",
                json=sample_data,
                timeout=10
            )
            
            if response.status_code == 200:
                pred = response.json()
                result["details"]["sample_prediction"] = {
                    "churn_probability": pred.get("churn_probability"),
                    "risk_level": pred.get("risk_level")
                }
                result["status"] = "healthy"
                print(f"   ✓ Predictions working (sample churn prob: {pred.get('churn_probability', 0):.2%})")
            else:
                result["status"] = "error"
                self.add_alert("PREDICTION_ERROR", f"Prediction failed with status {response.status_code}", "high")
        
        except Exception as e:
            result["status"] = "error"
            result["details"]["error"] = str(e)
            print(f"   ✗ Prediction check failed: {str(e)}")
        
        return result
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check model file sizes and data availability."""
        result = {"status": "healthy", "details": {}}
        
        try:
            # Check model files
            model_files = ["model.pkl", "preprocessor.pkl", "metrics.json"]
            missing_files = []
            
            for file in model_files:
                path = os.path.join(self.model_dir, file)
                if os.path.exists(path):
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    result["details"][file] = f"{size_mb:.2f} MB"
                else:
                    missing_files.append(file)
            
            if missing_files:
                result["status"] = "incomplete"
                self.add_alert("MISSING_FILES", f"Missing files: {', '.join(missing_files)}", "high")
                print(f"   ⚠ Missing files: {', '.join(missing_files)}")
            else:
                print("   ✓ All model files present")
        
        except Exception as e:
            result["status"] = "error"
            result["details"]["error"] = str(e)
            print(f"   ✗ Resource check failed: {str(e)}")
        
        return result
    
    def create_baseline_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create baseline statistics for drift detection."""
        baseline = {}
        
        numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
        
        for feature in numerical_features:
            if feature in df.columns:
                baseline[feature] = {
                    "mean": float(df[feature].mean()),
                    "std": float(df[feature].std()),
                    "min": float(df[feature].min()),
                    "max": float(df[feature].max())
                }
        
        baseline["created_at"] = datetime.now().isoformat()
        baseline["sample_size"] = len(df)
        
        # Save baseline
        os.makedirs(os.path.dirname(self.baseline_path), exist_ok=True)
        with open(self.baseline_path, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        return baseline
    
    def add_alert(self, alert_type: str, message: str, severity: str):
        """Add an alert to the monitoring report."""
        self.alerts.append({
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        })
        self.status["issues"].append(message)
    
    def generate_intelligent_summary(self, results: Dict[str, Any]) -> str:
        """Generate an intelligent summary of monitoring results."""
        status = results["overall_status"]
        alert_count = results["alert_count"]
        
        if status == "healthy":
            return (
                "✓ All systems operational. No issues detected.\n"
                "  The ML pipeline is performing within expected parameters."
            )
        
        summary_lines = [
            f"⚠ System Status: {status.upper()}",
            f"  {alert_count} issue(s) detected:\n"
        ]
        
        # Group alerts by severity
        critical = [a for a in self.alerts if a["severity"] == "critical"]
        high = [a for a in self.alerts if a["severity"] == "high"]
        medium = [a for a in self.alerts if a["severity"] == "medium"]
        
        if critical:
            summary_lines.append(f"  🔴 CRITICAL ({len(critical)}):")
            for alert in critical:
                summary_lines.append(f"     - {alert['message']}")
        
        if high:
            summary_lines.append(f"  🟠 HIGH ({len(high)}):")
            for alert in high:
                summary_lines.append(f"     - {alert['message']}")
        
        if medium:
            summary_lines.append(f"  🟡 MEDIUM ({len(medium)}):")
            for alert in medium:
                summary_lines.append(f"     - {alert['message']}")
        
        # Add recommendations
        summary_lines.append("\n  Recommendations:")
        if any(a["type"] == "DATA_DRIFT" for a in self.alerts):
            summary_lines.append("     → Consider retraining the model with recent data")
        if any(a["type"] == "PERFORMANCE_DROP" for a in self.alerts):
            summary_lines.append("     → Investigate model degradation, check feature quality")
        if any(a["type"].startswith("API") for a in self.alerts):
            summary_lines.append("     → Check API deployment and infrastructure")
        
        return "\n".join(summary_lines)
    
    def save_monitor_log(self, results: Dict[str, Any]):
        """Save monitoring results to log file."""
        try:
            # Load existing logs
            if os.path.exists(self.monitor_log_path):
                with open(self.monitor_log_path, 'r') as f:
                    logs = json.load(f)
            else:
                logs = {"monitoring_runs": []}
            
            # Append new results
            logs["monitoring_runs"].append(results)
            
            # Keep only last 100 runs
            logs["monitoring_runs"] = logs["monitoring_runs"][-100:]
            
            # Save
            os.makedirs(os.path.dirname(self.monitor_log_path), exist_ok=True)
            with open(self.monitor_log_path, 'w') as f:
                json.dump(logs, f, indent=2)
            
            print(f"\n✓ Monitoring log saved to: {self.monitor_log_path}")
        
        except Exception as e:
            print(f"\n✗ Failed to save monitoring log: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor ML pipeline health")
    parser.add_argument(
        "--alert-threshold",
        choices=["low", "medium", "high"],
        default="medium",
        help="Alert sensitivity (low=fewer alerts, high=more alerts)"
    )
    parser.add_argument(
        "--api-url",
        default="https://churnprediction-bgio.onrender.com",
        help="API URL to monitor"
    )
    
    args = parser.parse_args()
    
    # Create and run monitoring agent
    agent = MLMonitorAgent(
        alert_threshold=args.alert_threshold,
        api_url=args.api_url
    )
    
    results = agent.run_full_check()
    
    # Exit with appropriate code
    if results["overall_status"] == "critical":
        sys.exit(2)
    elif results["overall_status"] == "degraded":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
