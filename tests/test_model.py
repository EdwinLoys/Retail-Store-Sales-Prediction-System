import pandas as pd
from pathlib import Path
from src.model import train

def test_train(tmp_path: Path):
    # Minimal dataset
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
        "sales": [100, 150, 130, 170],
        "store_id": [1, 1, 2, 2],
    })
    # add dummy features so the model has something to train on
    df["month"] = df["date"].dt.month
    model_path = tmp_path / "model.joblib"
    metrics = train(df, model_path)
    assert model_path.is_file()
    assert metrics["mae"] >= 0
