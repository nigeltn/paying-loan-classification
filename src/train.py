import os
import pandas as pd
import xgboost as xgb
import joblib
import time
from typing import Any
from config import config


def train_sklearn_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series, save_path: str) -> Any:
    """Trains a scikit-learn compatible model and saves it."""
    model_name = model.__class__.__name__
    print(f"--- Training {model_name} Model ---")

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    print(f"Training complete in {end_time - start_time:.2f} seconds.")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

    return model


def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, save_path: str) -> Any:
    """Trains an XGBoost model and saves it."""
    dtrain = xgb.DMatrix(X_train, label=y_train)

    print("--- Training XGBoost Model ---")
    start_time = time.time()
    model = xgb.train(config.XGB_PARAMS, dtrain, config.XGB_STEPS)
    end_time = time.time()

    print(f"Training complete in {end_time - start_time:.2f} seconds.")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

    return model