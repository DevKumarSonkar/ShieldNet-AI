import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-blocking backend suitable for scripts and server environments
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 console output for Windows paths
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def load_model_and_features(model_path="models/advanced_model.joblib", features_path="models/feature_names.joblib"):
    """Loads the trained tree-based model and feature names list."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'. Run step3_train_models.py first.")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature names file not found at '{features_path}'. Run step3_train_models.py first.")

    print(f"[*] Loading model from: {os.path.abspath(model_path)}")
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features


def extract_feature_importance(model, feature_names):
    """
    Extracts Gini feature importance scores from a tree-based ensemble
    and returns a sorted DataFrame.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"Model {type(model).__name__} does not have 'feature_importances_' attribute.")

    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    fi_df["Importance_Pct"] = fi_df["Importance"] * 100
    fi_df["Cumulative_Importance"] = fi_df["Importance_Pct"].cumsum()
    return fi_df


def plot_top_features(fi_df, top_n=15, output_path="models/top_15_feature_importance.png"):
    """
    Plots a modern, publication-quality horizontal bar chart
    of the Top N Most Important Features using Seaborn and Matplotlib.
    """
    top_df = fi_df.head(top_n).copy()
    # Invert order for horizontal bar chart (highest importance at top)
    plot_df = top_df.iloc[::-1].reset_index(drop=True)

    plt.figure(figsize=(11, 7))
    palette = sns.color_palette("crest_r", n_colors=top_n)

    bars = plt.barh(
        plot_df["Feature"],
        plot_df["Importance_Pct"],
        color=palette,
        edgecolor="black",
        linewidth=0.8,
        height=0.68
    )

    # Annotate exact percentage and decimal importance at the end of each bar
    max_val = plot_df["Importance_Pct"].max()
    for bar in bars:
        width = bar.get_width()
        y_pos = bar.get_y() + bar.get_height() / 2
        plt.text(
            width + (max_val * 0.012),
            y_pos,
            f"{width:.2f}% ({width/100:.4f})",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#222222"
        )

    plt.title(f"Top {top_n} Most Important Features for Network Intrusion Detection (NSL-KDD)", fontsize=13, fontweight="bold", pad=14)
    plt.xlabel("Gini Feature Importance (%)", fontsize=11, fontweight="bold", labelpad=8)
    plt.ylabel("Network Traffic Feature", fontsize=11, fontweight="bold", labelpad=8)
    plt.xlim(0, max_val * 1.18)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    # Style borders
    sns.despine(top=True, right=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[+] Feature importance plot saved to: {os.path.abspath(output_path)}")
    plt.close()


def run_feature_importance():
    print("=" * 70)
    print("STEP 5: FEATURE IMPORTANCE & MODEL INTERPRETABILITY")
    print("=" * 70)

    # 1. Load Model and Features
    model, feature_names = load_model_and_features()
    model_name = type(model).__name__
    print(f"[+] Loaded {model_name} with {len(feature_names)} input features.\n")

    # 2. Extract and Sort Feature Importances
    fi_df = extract_feature_importance(model, feature_names)

    # 3. Print Top 5 Features to Console
    print("-" * 50)
    print("TOP 5 MOST IMPORTANT INTRUSION DETECTION FEATURES")
    print("-" * 50)
    top_5 = fi_df.head(5)
    for rank, row in enumerate(top_5.itertuples(), start=1):
        print(f"  {rank}. {row.Feature:<28} : {row.Importance:.6f}  ({row.Importance_Pct:.2f}%)")
    print("-" * 50)
    print(f"[*] Top 5 features account for {top_5['Importance_Pct'].sum():.2f}% of total model decisions.\n")

    # 4. Display Top 15 Table
    print("-" * 50)
    print("TOP 15 FEATURES SUMMARY TABLE")
    print("-" * 50)
    display_cols = ["Feature", "Importance", "Importance_Pct", "Cumulative_Importance"]
    formatted_top15 = fi_df.head(15)[display_cols].copy()
    formatted_top15["Importance"] = formatted_top15["Importance"].map(lambda x: f"{x:.4f}")
    formatted_top15["Importance_Pct"] = formatted_top15["Importance_Pct"].map(lambda x: f"{x:.2f}%")
    formatted_top15["Cumulative_Importance"] = formatted_top15["Cumulative_Importance"].map(lambda x: f"{x:.2f}%")
    print(formatted_top15.to_string(index=False))
    print()

    # 5. Plot and Save Horizontal Bar Chart
    print("-" * 50)
    print("GENERATING VISUALIZATION")
    print("-" * 50)
    plot_path = os.path.join("models", "top_15_feature_importance.png")
    plot_top_features(fi_df, top_n=15, output_path=plot_path)

    print("\n" + "=" * 70)
    print("STEP 5 COMPLETE")
    print("=" * 70)

    return fi_df


if __name__ == "__main__":
    fi_df = run_feature_importance()
