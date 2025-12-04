"""
Download Telco Customer Churn Dataset
Uses kagglehub to download directly without manual steps
"""

import os
import shutil

def download_dataset():
    """Download the Telco Customer Churn dataset from Kaggle."""
    
    # Target path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_path = os.path.join(base_dir, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    if os.path.exists(target_path):
        print(f"✅ Dataset already exists at {target_path}")
        return target_path
    
    print("📥 Downloading Telco Customer Churn dataset from Kaggle...")
    
    try:
        import kagglehub
        
        # Download dataset
        path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        print(f"✅ Downloaded to: {path}")
        
        # Find the CSV file
        csv_file = None
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_file = os.path.join(root, file)
                    break
        
        if csv_file:
            # Copy to our data directory
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy(csv_file, target_path)
            print(f"✅ Dataset saved to: {target_path}")
            return target_path
        else:
            print("❌ Could not find CSV file in downloaded data")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n📋 Manual download instructions:")
        print("   1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("   2. Download 'WA_Fn-UseC_-Telco-Customer-Churn.csv'")
        print(f"   3. Place it in: {os.path.dirname(target_path)}")
        return None


if __name__ == "__main__":
    download_dataset()
