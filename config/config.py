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
    'eta': 0.07,
    'objective': 'multi:softprob',  
    'num_class': 2,
    'device': 'cpu',
    'max_depth': 10,
    'num_parallel_tree': 5,
    'scale_pos_weight': 4.584297520661157,   
    'random_state': RANDOM_STATE,
}
XGB_STEPS: int = 500
