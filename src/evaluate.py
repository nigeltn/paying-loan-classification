import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict


def evaluate_model(model: Any, model_name: str, X_test: pd.DataFrame, y_test: pd.Series, figure_path: str) -> Dict[
    str, float]:
    """Evaluates a model's performance and plots its confusion matrix."""
    print(f"\n--- Evaluating {model_name} ---")

    if isinstance(model, xgb.Booster):
        dtest = xgb.DMatrix(X_test, label=y_test)
        y_pred_prob = model.predict(dtest)
        y_pred = (y_pred_prob > 0.5).astype(int)
    else: 
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path)
    plt.show()

    return {"accuracy": accuracy, "f1_score": f1}