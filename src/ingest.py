import pandas as pd
from pathlib import Path


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Read the raw CSV and return a DataFrame."""
    return pd.read_csv(csv_path, parse_dates=["date"])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning steps (missing values, dtype fixes, etc.)."""
    df = df.copy()
    # Drop rows without target
    df = df.dropna(subset=["sales"])
    # Fill missing numeric features with median
    numeric_cols = df.select_dtypes("number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    # Ensure categorical columns are strings
    cat_cols = ["store_id", "product_category"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("unknown")
    return df
