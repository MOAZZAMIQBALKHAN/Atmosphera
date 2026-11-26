# model.py
# Prediction module — loads model.pkl and exposes predict_temperature()
# Imported by app.py; do NOT run this file directly.

import joblib
import os
import numpy as np

# ── Path to the saved model ──────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

# ── Load model once at import time (cached for the lifetime of the Flask app) ─
_model = None

def _load_model():
    """Lazily loads (and caches) the trained Random Forest model."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "model.pkl not found. "
                "Please run  'python train_model.py'  first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_temperature(humidity: float,
                        pressure: float,
                        wind_speed: float) -> float:
    """
    Predict temperature (°C) from live weather features.

    Parameters
    ----------
    humidity   : relative humidity in %
    pressure   : atmospheric pressure in hPa
    wind_speed : wind speed in m/s

    Returns
    -------
    Predicted temperature rounded to 1 decimal place.
    """
    model  = _load_model()
    X      = np.array([[humidity, pressure, wind_speed]])
    result = model.predict(X)
    return round(float(result[0]), 1)
