import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import xgboost as xgb

from config import config
from src.data_processing import balance_data
from src.feature_engineering import select_features
from src.train import train_sklearn_model, train_xgboost_model
from src.evaluate import evaluate_model


def run_pipeline():
    """Runs the full model comparison pipeline using K-Fold Cross-Validation."""
    df = pd.read_csv(config.DATA_PATH)

    X = df.drop(columns=[config.TARGET_VARIABLE])
    y = df[config.TARGET_VARIABLE]

    # Step 1: Create a final, hold-out test set. This data will NOT be used for CV.
    # We'll use it once at the very end to evaluate our best chosen model.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    # --- Define Models ---
    models_to_train = {
        "DecisionTree": DecisionTreeClassifier(**config.DT_PARAMS),
        "RandomForest": RandomForestClassifier(**config.RF_PARAMS),
        "XGBoost": "xgboost"  # Using a placeholder for our custom XGBoost logic
    }

    # --- K-Fold Cross-Validation Setup ---
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    cv_results = {}

    for name, model in models_to_train.items():
        print(f"\n--- Cross-Validating {name} ---")
        fold_f1_scores = []

        # Step 2: Loop through each fold. `skf.split` provides indices for train/validation sets.
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
            print(f"  -- Fold {fold + 1}/{N_SPLITS} --")
            X_train_fold, y_train_fold = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
            X_val_fold, y_val_fold = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]

            selected_features = select_features(X_train_fold, y_train_fold)
            X_train_fold = X_train_fold[selected_features]
            X_val_fold = X_val_fold[selected_features]  # Apply same features to validation set

            X_train_bal, y_train_bal = balance_data(X_train_fold, y_train_fold)

            if name == "XGBoost":
                dtrain = xgb.DMatrix(X_train_bal, label=y_train_bal)
                trained_model = xgb.train(config.XGB_PARAMS, dtrain, config.XGB_STEPS)
            else:
                trained_model = model.fit(X_train_bal, y_train_bal)

            if name == "XGBoost":
                dval = xgb.DMatrix(X_val_fold)
                y_pred_prob = trained_model.predict(dval)
                y_pred = (y_pred_prob > 0.5).astype(int)
            else:
                y_pred = trained_model.predict(X_val_fold)

            f1 = f1_score(y_val_fold, y_pred)
            fold_f1_scores.append(f1)
            print(f"  Fold F1 Score: {f1:.4f}")

        cv_results[name] = {
            'mean_f1': np.mean(fold_f1_scores),
            'std_f1': np.std(fold_f1_scores)
        }

    # --- Display Final Cross-Validation Results ---
    results_df = pd.DataFrame(cv_results).T.sort_values(by='mean_f1', ascending=False)
    print("\n\n--- Cross-Validation Summary ---")
    print(" (Mean F1 score and standard deviation across all folds)")
    print(results_df)


if __name__ == '__main__':
    run_pipeline()