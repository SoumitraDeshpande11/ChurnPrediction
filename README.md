# MLOps Series: End-to-End ML Pipeline

> Customer Churn Prediction System - From Notebook to Production

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green.svg)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What is This?

This is my MLOps series where I build production-grade ML systems step by step.

**Part 1**: Deploy an end-to-end ML pipeline  
**Part 2**: Add auto-retraining + CI/CD (Current)  
**Part 3**: Build agentic monitoring system

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │           GitHub Actions CI/CD           │
                    │  ┌─────────┐ ┌─────────┐ ┌─────────────┐ │
                    │  │  Test   │→│  Build  │→│   Deploy    │ │
                    │  └─────────┘ └─────────┘ └─────────────┘ │
                    └──────────────────┬───────────────────────┘
                                       │
    ┌──────────────────────────────────┼──────────────────────────────────┐
    │                                  ▼                                  │
    │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
    │  │   Raw Data  │────▶│  Training   │────▶│   Model     │           │
    │  │   (CSV)     │     │  Pipeline   │     │   (v1, v2)  │           │
    │  └─────────────┘     └─────────────┘     └──────┬──────┘           │
    │         │                   ▲                   │                   │
    │         │                   │                   ▼                   │
    │         │            ┌──────┴──────┐     ┌─────────────┐           │
    │         └───────────▶│  Retrain    │     │  Predictor  │           │
    │         (new data)   │  Pipeline   │     │   Module    │           │
    │                      └─────────────┘     └──────┬──────┘           │
    │                                                 │                   │
    └─────────────────────────────────────────────────┼───────────────────┘
                                                      │
                    ┌─────────────────────────────────┼─────────────────┐
                    │                                 ▼                 │
                    │  ┌─────────────┐     ┌─────────────────────────┐  │
                    │  │   Client    │◀───▶│   FastAPI + Render      │  │
                    │  │  (Request)  │     │   (Production API)      │  │
                    │  └─────────────┘     └─────────────────────────┘  │
                    └───────────────────────────────────────────────────┘
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
ChurnPrediction/
├── .github/
│   └── workflows/
│       ├── ci-cd.yml         # CI/CD pipeline
│       └── retrain.yml       # Scheduled retraining
├── src/
│   ├── __init__.py
│   ├── train.py              # Training pipeline
│   ├── predict.py            # Prediction module
│   └── preprocess.py         # Data preprocessing
├── api/
│   ├── __init__.py
│   ├── main.py               # FastAPI application
│   └── schemas.py            # Pydantic models
├── scripts/
│   ├── download_data.py      # Dataset downloader
│   └── retrain.py            # Auto-retraining script
├── models/
│   └── v1/                   # Versioned models
│       ├── model.pkl
│       ├── preprocessor.pkl
│       └── metrics.json
├── data/
│   ├── raw/                  # Raw datasets
│   └── processed/            # Processed data + hashes
├── tests/
│   └── test_model.py         # Comprehensive tests
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

## CI/CD Pipeline

On every push to `main`:

1. **Test** - Run pytest on all tests
2. **Build** - Build Docker image
3. **Deploy** - Trigger Render deployment

```bash
# Trigger manually
gh workflow run ci-cd.yml
```

## Auto-Retraining

The retraining pipeline:

1. Detects new data (via file hash)
2. Validates data quality
3. Trains new model
4. Compares metrics with current model
5. Promotes only if better

```bash
# Manual retrain
python scripts/retrain.py

# Force retrain (even without new data)
python scripts/retrain.py --force
```

Scheduled to run weekly via GitHub Actions.

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~77% |
| Precision | ~56% |
| Recall | ~71% |
| F1 Score | ~62% |

*Metrics saved in `models/v1/metrics.json` after training.*

## Coming Next

- **Part 3**: AI agent for pipeline monitoring with intelligent alerting

## Tech Stack

- **ML**: Scikit-learn, Pandas, NumPy
- **API**: FastAPI, Uvicorn, Pydantic
- **CI/CD**: GitHub Actions
- **Deployment**: Docker, Render
- **Testing**: Pytest

## License

MIT License - feel free to use this for learning!

---

**GitHub**: [github.com/SoumitraDeshpande11/ChurnPrediction](https://github.com/SoumitraDeshpande11/ChurnPrediction)
