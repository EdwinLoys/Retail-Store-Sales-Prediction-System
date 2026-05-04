#!/usr/bin/env python
import argparse
from pathlib import Path

from src.ingest import load_raw, clean_data
from src.features import add_date_features, encode_categoricals
from src.model import train


def main(csv_path: Path, model_path: Path):
    # 1️⃣ Load & clean
    raw = load_raw(csv_path)
    clean = clean_data(raw)

    # 2️⃣ Feature engineering
    feats = add_date_features(clean)
    feats = encode_categoricals(feats)

    # 3️⃣ Train / evaluate
    metrics = train(feats, model_path)
    print("✅ Training complete")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R² : {metrics['r2']:.3f}")
    print(f"Model saved to {metrics['model_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail sales forecasting")
    parser.add_argument(
        "--data", type=Path, required=True, help="Path to raw CSV file"
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/sales_gbr.joblib")
    )
    args = parser.parse_args()
    main(args.data, args.model)
