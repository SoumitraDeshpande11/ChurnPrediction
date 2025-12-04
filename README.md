# MLOps Series - Part 1: End-to-End ML Pipeline

> Customer Churn Prediction System - From Notebook to Production

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What is This?

This is **Part 1** of my MLOps series where I build production-grade ML systems step by step.

**Part 1**: Deploy an end-to-end ML pipeline (You are here)  
**Part 2**: Add auto-retraining + CI/CD  
**Part 3**: Build agentic monitoring system

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Raw Data  │────▶│  Training   │────▶│   Model     │
│   (CSV)     │     │  Pipeline   │     │   (.pkl)    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │◀───▶│  FastAPI    │◀───▶│  Predictor  │
│  (Request)  │     │  Server     │     │   Module    │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Quick Start

### 1. Clone & Setup

```bash
# Clone the repo
git clone https://github.com/SoumitraDeshpande11/ChurnPrediction.git
cd ChurnPrediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place it in:

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 3. Train the Model

```bash
python src/train.py
```

This will:
- Load and preprocess the data
- Train a RandomForest classifier
- Save model, preprocessor, and metrics to `models/v1/`

### 4. Run the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API docs.

## Docker Deployment

```bash
# Build the image
docker build -t churn-predictor .

# Run the container
docker run -p 8000:8000 churn-predictor
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict/batch` | Batch predictions |
| `GET` | `/model/info` | Model metadata |
| `GET` | `/model/features` | Feature list |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Example Response

```json
{
  "churn_prediction": true,
  "churn_label": "Yes",
  "churn_probability": 0.7234,
  "retention_probability": 0.2766,
  "risk_level": "HIGH"
}
```

## Project Structure

```
mlops-series/
├── src/
│   ├── __init__.py
│   ├── train.py          # Training pipeline
│   ├── predict.py        # Prediction module
│   └── preprocess.py     # Data preprocessing
├── api/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   └── schemas.py        # Pydantic models
├── models/
│   └── v1/
│       ├── model.pkl     # Trained model
│       ├── preprocessor.pkl
│       └── metrics.json
├── data/
│   ├── raw/              # Raw datasets
│   └── processed/        # Processed data
├── tests/
│   └── test_model.py     # Unit tests
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~80% |
| Precision | ~65% |
| Recall | ~55% |
| F1 Score | ~59% |

*Actual metrics will be saved in `models/v1/metrics.json` after training.*

## Coming Next

- **Part 2**: Auto-retraining pipeline with GitHub Actions
- **Part 3**: AI agent for pipeline monitoring

## Tech Stack

- **ML**: Scikit-learn, Pandas, NumPy
- **API**: FastAPI, Uvicorn, Pydantic
- **Deployment**: Docker, Render/Railway
- **Testing**: Pytest

## License

MIT License - feel free to use this for learning!

---

**GitHub**: [github.com/SoumitraDeshpande11/ChurnPrediction](https://github.com/SoumitraDeshpande11/ChurnPrediction)
