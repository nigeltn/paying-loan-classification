import pandas as pd
from typing import List
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from config import config


def select_features(df: pd.DataFrame) -> List[str]:
    """
    Performs a two-stage feature selection process:
    1. Correlation-based filtering.
    2. Recursive Feature Elimination (RFE).
    """
    # Stage 1: Correlation-based filtering
    print(f"Selecting top {config.TOP_FEATURES_CORR} features by correlation...")
    correlation_with_label = df.corr()[config.TARGET_VARIABLE].abs().sort_values(ascending=False)
    top_corr_features = correlation_with_label.index[:config.TOP_FEATURES_CORR]

    X_temp = df[top_corr_features].drop(config.TARGET_VARIABLE, axis=1)
    y_temp = df[config.TARGET_VARIABLE]

    # Stage 2: RFE
    print(f"Running RFE to select top {config.TOP_FEATURES_RFE} features...")
    estimator = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)
    rfe = RFE(estimator, n_features_to_select=config.TOP_FEATURES_RFE)
    rfe.fit(X_temp, y_temp)

    selected_features: List[str] = list(X_temp.columns[rfe.support_])
    print("Final selected features:")
    print(selected_features)

    return selected_features