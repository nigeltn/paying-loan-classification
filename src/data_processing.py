import pandas as pd
from typing import Tuple
from imblearn.over_sampling import SMOTE
from config import config

def load_and_clean_data(path: str) -> pd.DataFrame:
    """Loads and performs initial cleaning on the dataset."""
    df = pd.read_csv(path)
    if config.FEATURES_TO_DROP:
        df = df.drop(columns=config.FEATURES_TO_DROP, errors='ignore')
    return df

def balance_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Balances the dataset using SMOTE."""
    smote = SMOTE(sampling_strategy='minority', random_state=config.RANDOM_STATE)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print(f"Data balanced with SMOTE. Original size: {len(y)}, New size: {len(y_balanced)}")
    return X_balanced, y_balanced