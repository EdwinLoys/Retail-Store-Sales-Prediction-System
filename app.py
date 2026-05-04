#!/usr/bin/env python
"""Entry point for deployment platforms that expect a callable ``app``.
It simply runs the same training pipeline as ``run.py`` but exposes a
``main`` function that can be used as a console script via ``pyproject.toml``.
"""
import argparse
from pathlib import Path

from src.ingest import load_raw, clean_data
from src.features import add_date_features, encode_categoricals
from src.model import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail sales forecasting entry point")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the raw CSV file containing sales data",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/sales_gbr.joblib"),
        help="Where to persist the trained model",
    )
    args = parser.parse_args()

    # Load and clean data
    raw = load_raw(args.data)
    clean = clean_data(raw)

    # Feature engineering
    feats = add_date_features(clean)
    feats = encode_categoricals(feats)

    # Train and persist the model
    metrics = train(feats, args.model)
    print("✅ Training complete")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R² : {metrics['r2']:.3f}")
    print(f"Model saved to {metrics['model_path']}")


if __name__ == "__main__":
    main()
