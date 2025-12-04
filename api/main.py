"""
FastAPI Application for Customer Churn Prediction
MLOps Series - Part 1

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    CustomerInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from src.predict import ChurnPredictor


# Global predictor instance
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI app.
    Loads model on startup.
    """
    global predictor
    print("Starting up Customer Churn Prediction API...")
    
    try:
        predictor = ChurnPredictor()
        print("Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("   API will start but predictions won't work until model is trained.")
        predictor = None
    
    yield
    
    print("Shutting down API...")


# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
    ## MLOps Series - Part 1
    
    End-to-end ML pipeline for predicting customer churn.
    
    ### Features:
    - **Single Prediction**: Predict churn for one customer
    - **Batch Prediction**: Predict churn for multiple customers
    - **Model Info**: Get model metadata and metrics
    - **Health Check**: Verify API and model status
    
    ### Built with:
    - Python + FastAPI
    - Scikit-learn (RandomForest)
    - Telco Customer Churn Dataset
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint."""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check API and model health status.
    """
    return HealthResponse(
        status="healthy" if predictor else "degraded",
        model_loaded=predictor is not None,
        model_version="v1",
        api_version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_churn(customer: CustomerInput):
    """
    Predict churn probability for a single customer.
    
    Returns:
    - **churn_prediction**: Boolean indicating if customer will churn
    - **churn_probability**: Probability of churning (0-1)
    - **risk_level**: LOW, MEDIUM, or HIGH risk category
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first with 'python src/train.py'"
        )
    
    try:
        # Convert Pydantic model to dict
        customer_data = customer.model_dump()
        
        # Make prediction
        result = predictor.predict(customer_data)
        
        return PredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn probability for multiple customers.
    
    Useful for batch processing and bulk analysis.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Convert all customers to dicts
        customers_data = [c.model_dump() for c in request.customers]
        
        # Make predictions
        results = predictor.predict_batch(customers_data)
        predictions = [PredictionResponse(**r) for r in results]
        
        # Count risk levels
        risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for pred in predictions:
            risk_counts[pred.risk_level] += 1
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            high_risk_count=risk_counts["HIGH"],
            medium_risk_count=risk_counts["MEDIUM"],
            low_risk_count=risk_counts["LOW"]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns model type, features, and performance metrics.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )
    
    try:
        info = predictor.get_model_info()
        return ModelInfoResponse(**info)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.get("/model/features", tags=["Model"])
async def get_feature_list():
    """
    Get list of features expected by the model.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    return {
        "features": predictor.preprocessor.feature_columns,
        "categorical": predictor.preprocessor.categorical_columns,
        "numerical": predictor.preprocessor.numerical_columns
    }


# For running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
