from typing import List, Dict, Any

# --- File Paths ---
DATA_PATH: str = "data/raw/raw_dataset.csv"

# Paths for saving models and figures
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
TOP_FEATURES_CORR: int = 5
TOP_FEATURES_RFE: int = 2

# --- Data Processing Parameters ---
TEST_SIZE: float = 0.3
RANDOM_STATE: int = 42

# --- Model Parameters ---
# Decision Tree
DT_PARAMS: Dict[str, Any] = {
    'max_depth': 7,
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
    'eta': 0.07,
    'objective': 'multi:softprob',
    'num_class': 2,
    'device': 'cuda',
    'max_depth': 10,
    'verbosity': 1,
    'num_parallel_tree': 20
}
XGB_STEPS: int = 400