# train_model.py
# Run this file ONCE to train and save the ML model to model.pkl
# Usage: python train_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_and_save_model():
    print("=" * 50)
    print("  WEATHER PREDICTION MODEL TRAINER")
    print("=" * 50)

    # ── 1. Load dataset ──────────────────────────────────
    csv_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
    print(f"\n[1/4] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[df['temperature'] > 20]
    print(f"      Rows: {len(df)}  |  Columns: {list(df.columns)}")

    # ── 2. Prepare features & target ────────────────────
    print("\n[2/4] Preparing features...")
    features = ['humidity', 'pressure', 'wind_speed']
    target   = 'temperature'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"      Training samples : {len(X_train)}")
    print(f"      Testing  samples : {len(X_test)}")

    # ── 3. Train Random Forest Regressor ────────────────
    print("\n[3/4] Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    print(f"      MAE (Mean Absolute Error) : {mae:.2f} °C")
    print(f"      R² Score                  : {r2:.4f}")

    # ── 4. Save model ────────────────────────────────────
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    print(f"\n[4/4] Saving model to: {model_path}")
    joblib.dump(model, model_path)
    print("\n✅  Model saved successfully!")
    print("=" * 50)
    print("  You can now run:  python app.py")
    print("=" * 50)

if __name__ == '__main__':
    train_and_save_model()
