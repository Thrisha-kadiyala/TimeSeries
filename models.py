# models.py
import joblib
import numpy as np

def load_models():
    models = {}
    try:
        models["arima"] = joblib.load("arima_model.pkl")
    except Exception:
        models["arima"] = None
    try:
        models["xgb"] = joblib.load("xgb_model.pkl")
    except Exception:
        models["xgb"] = None
    return models

def predict_with_model(model, features):
    """
    Predict only for new data. No y_true is needed.
    """
    X = np.array(features)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if model is None:
        preds = [0] * len(X)
    elif hasattr(model, "predict"):
        preds = model.predict(X).tolist()
    else:
        preds = X.mean(axis=1).tolist()

    return preds

