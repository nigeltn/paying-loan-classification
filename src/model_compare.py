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
    df = pd.read_csv(config.DATA_PATH)

    X = df.drop(columns=[config.TARGET_VARIABLE])
    y = df[config.TARGET_VARIABLE]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    selected_features = select_features(X_train, y_train)
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    X_train_bal, y_train_bal = balance_data(X_train, y_train)


    models_to_train = {
        "DecisionTree": DecisionTreeClassifier(**config.DT_PARAMS),
        "RandomForest": RandomForestClassifier(**config.RF_PARAMS)
    }

    results = {}

    for name, model in models_to_train.items():
        trained_model = train_sklearn_model(model, X_train_bal, y_train_bal, config.MODEL_PATHS[name])
        results[name] = evaluate_model(trained_model, name, X_test, y_test, config.FIGURE_PATHS[name])

    xgb_model = train_xgboost_model(X_train_bal, y_train_bal, config.MODEL_PATHS["XGBoost"])
    results["XGBoost"] = evaluate_model(xgb_model, "XGBoost", X_test, y_test, config.FIGURE_PATHS["XGBoost"])

    results_df = pd.DataFrame(results).T.sort_values(by='f1_score', ascending=False)
    print("\n\n--- Model Comparison Summary ---")
    print(results_df)


if __name__ == '__main__':
    run_pipeline()