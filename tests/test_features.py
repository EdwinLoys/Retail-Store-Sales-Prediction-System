import pandas as pd
from src.features import add_date_features, encode_categoricals

def test_add_date_features():
    df = pd.DataFrame({"date": pd.to_datetime(["2023-03-15"])})
    out = add_date_features(df)
    assert out["year"].iloc[0] == 2023
    assert out["month"].iloc[0] == 3
    assert out["day_of_week"].iloc[0] == 2  # Wednesday
    assert out["is_month_end"].iloc[0] == 0

def test_encode_categoricals():
    df = pd.DataFrame({
        "store_id": ["A", "B"],
        "product_category": ["X", "Y"],
        "sales": [10, 20],
    })
    out = encode_categoricals(df)
    # Should create one‑hot columns, dropping first of each
    assert "store_id_B" in out.columns
    assert "product_category_Y" in out.columns
    # Original categorical columns removed
    assert "store_id" not in out.columns
