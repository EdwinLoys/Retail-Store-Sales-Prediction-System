# Retail Store Sales Prediction System

A small end‑to‑end machine‑learning project that predicts daily sales for a retail store.

## 📚 Overview
* **Language**: Python 3.11+
* **Core libraries**: pandas, numpy, scikit‑learn, matplotlib, seaborn
* **Model**: GradientBoostingRegressor (regression) – easy to replace with any other estimator.
* **Features**: basic date‑based features, one‑hot encoded categorical columns (store, product category).
* **Tests**: pytest suite covering data loading, cleaning, feature engineering and model training.

## 🗂 Project structure
```
Retail-Store-Sales-Prediction-System/
│
├─ data/
│   ├─ raw/          # place your original CSV(s) here
│   └─ processed/    # optional cleaned files
│
├─ src/               # source code
│   ├─ __init__.py
│   ├─ ingest.py      # loading & cleaning
│   ├─ features.py    # feature engineering helpers
│   ├─ model.py       # training / evaluation / model persistence
│   └─ visualisation.py
│
├─ tests/             # pytest test suite
│   ├─ __init__.py
│   ├─ test_ingest.py
│   ├─ test_features.py
│   └─ test_model.py
│
├─ notebooks/         # optional exploratory notebooks
│
├─ .gitignore
├─ requirements.txt   # Python dependencies
├─ run.py             # CLI entry point
└─ README.md
```

## 🚀 Quick start
1. **Clone the repo** (already done) and `cd` into it.
2. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Prepare your data** – put a CSV file with at least the columns `date` (YYYY‑MM‑DD), `sales` (numeric target), `store_id` and optionally `product_category` into `data/raw/`.
4. **Run the training pipeline**:
   ```bash
   python run.py --data data/raw/your_file.csv
   ```
   The script will:
   * load and clean the data,
   * add date‑based features and one‑hot encode categoricals,
   * train a GradientBoostingRegressor,
   * print MAE and R² metrics,
   * save the model to `models/sales_gbr.joblib`.
5. **Run the test suite**:
   ```bash
   pytest -q
   ```
   All tests should pass.

## 🧪 Testing & debugging
* Use `pytest --cov=src` for a coverage report.
* Insert `import pdb; pdb.set_trace()` in any function to start an interactive debugger.
* The `visualisation.py` module contains a helper to plot feature importance after training:
  ```python
  from src.model import train
  from src.visualisation import plot_feature_importance
  # after training
  plot_feature_importance(model, feature_names=list(X.columns))
  ```

## 📦 Packaging (optional)
If you want to install the package locally:
```bash
pip install -e .
```
Then you can import the modules as `import src.ingest` etc.

## 🤝 Contributing
Feel free to open issues or PRs. Follow the existing coding style (`black`, `flake8`).

---
*Created by Edwin Loys – Azure Cloud Engineer & .NET Backend Developer*