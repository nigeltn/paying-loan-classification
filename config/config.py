from typing import List, Dict, Any

DATA_PATH: str = "data/raw/raw_dataset.csv"

MODEL_PATHS: Dict[str, str] = {
    "DecisionTree": "models/decision_tree.joblib",
    "RandomForest": "models/random_forest.joblib",
    "XGBoost": "models/xgboost_model.joblib"
}

FIGURE_PATHS: Dict[str, str] = {
    "DecisionTree": "reports/figures/cm_decision_tree.png",
    "RandomForest": "reports/figures/cm_random_forest.png",
    "XGBoost": "reports/figures/cm_xgboost.png"
}

# --- Feature Engineering Parameters ---
TARGET_VARIABLE: str = "label"
FEATURES_TO_DROP: List[str] = ["customer_id"]
TOP_FEATURES_CORR: int = 50
TOP_FEATURES_RFE: int = 25

# --- Data Processing Parameters ---
TEST_SIZE: float = 0.3
RANDOM_STATE: int = 42

# --- Model Parameters ---
# Decision Tree
DT_PARAMS: Dict[str, Any] = {
    'max_depth': 10,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE
}

# Random Forest
RF_PARAMS: Dict[str, Any] = {
    'n_estimators': 200,
    'max_depth': 10,
    'random_state': RANDOM_STATE
}

# XGBoost
XGB_PARAMS: Dict[str, Any] = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.01,
    'max_depth': 5,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'random_state': RANDOM_STATE,
    'device': 'cpu',
}
XGB_STEPS: int = 1500
