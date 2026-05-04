import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_feature_importance(model, feature_names):
    """Bar plot of feature importances for tree‑based models."""
    importance = model.feature_importances_
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importance})
    df_imp = df_imp.sort_values("importance", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=df_imp)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()
