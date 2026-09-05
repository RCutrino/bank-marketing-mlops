from pydantic import BaseModel, Field
from typing import Literal


class ClientFeatures(BaseModel):
    """Input features for bank marketing prediction (without duration)."""

    age: int = Field(..., ge=18, le=100, example=41)
    job: str = Field(..., example="management")
    marital: str = Field(..., example="married")
    education: str = Field(..., example="tertiary")
    default: str = Field(..., example="no")
    balance: float = Field(..., example=1500.0)
    housing: str = Field(..., example="yes")
    loan: str = Field(..., example="no")
    contact: str = Field(..., example="cellular")
    day: int = Field(..., ge=1, le=31, example=15)
    month: str = Field(..., example="may")
    campaign: int = Field(..., ge=1, example=2)
    pdays: int = Field(..., example=-1)
    previous: int = Field(..., ge=0, example=0)
    poutcome: str = Field(..., example="unknown")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 41,
                "job": "management",
                "marital": "married",
                "education": "tertiary",
                "default": "no",
                "balance": 1500.0,
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "day": 15,
                "month": "may",
                "campaign": 2,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown",
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = subscribe, 0 = not subscribe")
    probability: float = Field(..., description="Probability of subscribing")
    threshold: float = Field(..., description="Decision threshold used")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scenario: str