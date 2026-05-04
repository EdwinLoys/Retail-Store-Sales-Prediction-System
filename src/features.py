import pandas as pd


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple time‑based features."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One‑hot encode low‑cardinality categoricals."""
    df = df.copy()
    cat_cols = ["store_id", "product_category"]
    existing = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing, drop_first=True)
    return df
