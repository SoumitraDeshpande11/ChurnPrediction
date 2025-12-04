"""
Pydantic Schemas for API Request/Response
MLOps Series - Part 1
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ContractType(str, Enum):
    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"


class PaymentMethodType(str, Enum):
    ELECTRONIC_CHECK = "Electronic check"
    MAILED_CHECK = "Mailed check"
    BANK_TRANSFER = "Bank transfer (automatic)"
    CREDIT_CARD = "Credit card (automatic)"


class InternetServiceType(str, Enum):
    DSL = "DSL"
    FIBER_OPTIC = "Fiber optic"
    NO = "No"


class YesNo(str, Enum):
    YES = "Yes"
    NO = "No"


class YesNoNA(str, Enum):
    YES = "Yes"
    NO = "No"
    NO_SERVICE = "No internet service"


class CustomerInput(BaseModel):
    """Input schema for customer churn prediction."""
    
    gender: str = Field(..., description="Customer gender", examples=["Male", "Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Is senior citizen (0 or 1)")
    Partner: YesNo = Field(..., description="Has partner")
    Dependents: YesNo = Field(..., description="Has dependents")
    tenure: int = Field(..., ge=0, description="Months with company")
    PhoneService: YesNo = Field(..., description="Has phone service")
    MultipleLines: str = Field(..., description="Has multiple lines")
    InternetService: InternetServiceType = Field(..., description="Internet service type")
    OnlineSecurity: str = Field(..., description="Has online security")
    OnlineBackup: str = Field(..., description="Has online backup")
    DeviceProtection: str = Field(..., description="Has device protection")
    TechSupport: str = Field(..., description="Has tech support")
    StreamingTV: str = Field(..., description="Has streaming TV")
    StreamingMovies: str = Field(..., description="Has streaming movies")
    Contract: ContractType = Field(..., description="Contract type")
    PaperlessBilling: YesNo = Field(..., description="Uses paperless billing")
    PaymentMethod: PaymentMethodType = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges in $")
    TotalCharges: float = Field(..., ge=0, description="Total charges in $")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for churn prediction."""
    
    churn_prediction: bool = Field(..., description="Will customer churn (True/False)")
    churn_label: str = Field(..., description="Churn label (Yes/No)")
    churn_probability: float = Field(..., description="Probability of churning")
    retention_probability: float = Field(..., description="Probability of staying")
    risk_level: str = Field(..., description="Risk level (LOW/MEDIUM/HIGH)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "churn_prediction": True,
                    "churn_label": "Yes",
                    "churn_probability": 0.7234,
                    "retention_probability": 0.2766,
                    "risk_level": "HIGH"
                }
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    customers: List[CustomerInput]


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_loaded: bool
    model_version: str
    api_version: str
    
    model_config = {"protected_namespaces": ()}


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    model_type: str
    model_dir: str
    feature_count: int
    features: List[str]
    metrics: Optional[dict] = None
    
    model_config = {"protected_namespaces": ()}
