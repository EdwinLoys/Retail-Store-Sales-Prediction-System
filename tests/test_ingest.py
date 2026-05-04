import pandas as pd
from pathlib import Path

from src.ingest import load_raw, clean_data

def test_load_raw(tmp_path: Path):
    csv = tmp_path / "sample.csv"
    csv.write_text("date,sales,store_id\n2023-01-01,100,1\n")
    df = load_raw(csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert "date" in df.columns

def test_clean_data():
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        "sales": [100, None],
        "store_id": [1, None],
    })
    cleaned = clean_data(raw)
    # row with missing sales should be dropped
    assert cleaned.shape[0] == 1
    assert cleaned.iloc[0]["store_id"] == "1"
