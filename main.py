from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from models import load_models

app = FastAPI(title="Forecast API with XGBoost & ARIMA")

# -------------------
# Request/Response Models
# -------------------
class PredictRequest(BaseModel):
    model: str
    features: Optional[List[List[float]]] = None  # For XGBoost
    steps: Optional[int] = None                  # For ARIMA

class PredictResponse(BaseModel):
    predictions: List[float]

# -------------------
# Load models
# -------------------
models = load_models()
print("Loaded models:", list(models.keys()))

# -------------------
# Prediction helper
# -------------------
def predict_with_model(model, features=None, steps=None):
    # ARIMA
    if "ARIMA" in str(type(model)):
        if not steps:
            raise HTTPException(status_code=400, detail="ARIMA requires 'steps' to forecast")
        start = len(model.data.endog)
        end = start + steps - 1
        preds = model.predict(start=start, end=end).tolist()
    # XGBoost / scikit-learn
    else:
        if features is None:
            raise HTTPException(status_code=400, detail="Feature-based models require 'features'")
        X = np.array(features)
        if X.shape[1] != model.n_features_in_:
            raise HTTPException(
                status_code=400,
                detail=f"Feature shape mismatch: expected {model.n_features_in_}, got {X.shape[1]}"
            )
        preds = model.predict(X).tolist()
    
    return preds

# -------------------
# Endpoints
# -------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model_key = req.model.lower()
    if model_key not in models:
        raise HTTPException(status_code=400, detail="Model not found")
    
    preds = predict_with_model(
        models[model_key],
        features=req.features,
        steps=req.steps
    )

    return PredictResponse(predictions=preds)

@app.get("/")
def root():
    return {"msg": "API is live", "models": list(models.keys())}
