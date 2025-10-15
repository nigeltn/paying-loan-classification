import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from config import config
from src.data_processing import balance_data
from src.feature_engineering import select_features
from src.train import train_sklearn_model, train_xgboost_model
from src.evaluate import evaluate_model


def run_pipeline():
    """Runs the full model comparison pipeline."""
    # 1. Data Loading and Feature Engineering
    df = pd.read_csv(config.DATA_PATH)
    selected_features = select_features(df)

    X = df[selected_features]
    y = df[config.TARGET_VARIABLE]

    # 2. Data Balancing and Splitting
    X_balanced, y_balanced = balance_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    # 3. Define Models from your notebook
    models_to_train = {
        "DecisionTree": DecisionTreeClassifier(**config.DT_PARAMS),
        "RandomForest": RandomForestClassifier(**config.RF_PARAMS)
    }

    results = {}

    # 4. Train and Evaluate scikit-learn models
    for name, model in models_to_train.items():
        trained_model = train_sklearn_model(model, X_train, y_train, config.MODEL_PATHS[name])
        results[name] = evaluate_model(trained_model, name, X_test, y_test, config.FIGURE_PATHS[name])

    # 5. Train and Evaluate XGBoost separately
    xgb_model = train_xgboost_model(X_train, y_train, config.MODEL_PATHS["XGBoost"])
    results["XGBoost"] = evaluate_model(xgb_model, "XGBoost", X_test, y_test, config.FIGURE_PATHS["XGBoost"])

    # 6. Summarize and display results
    results_df = pd.DataFrame(results).T.sort_values(by='f1_score', ascending=False)
    print("\n\n--- Model Comparison Summary ---")
    print(results_df)


if __name__ == '__main__':
    run_pipeline()