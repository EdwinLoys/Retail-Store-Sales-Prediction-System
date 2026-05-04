import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

TARGET = "sales"


def prepare_xy(df: pd.DataFrame):
    X = df.drop(columns=[TARGET, "date"], errors="ignore")
    y = df[TARGET]
    return X, y


def train(df: pd.DataFrame, model_path: Path) -> dict:
    """Train a regression model and persist it."""
    X, y = prepare_xy(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return {"mae": mae, "r2": r2, "model_path": str(model_path)}
