import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-blocking backend suitable for scripts and server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Ensure UTF-8 console output for Windows paths
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Import preprocessing pipeline
from step2_preprocess import preprocess_nsl_kdd


def load_artifacts(model_dir="models"):
    """Loads serialized models, scaler, and feature list from disk."""
    baseline_path = os.path.join(model_dir, "baseline_model.joblib")
    advanced_path = os.path.join(model_dir, "advanced_model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")

    if not os.path.exists(baseline_path) or not os.path.exists(advanced_path):
        raise FileNotFoundError(
            f"Models not found in '{model_dir}/'. Please run step3_train_models.py first."
        )

    print(f"[*] Loading models from: {os.path.abspath(model_dir)}")
    baseline_model = joblib.load(baseline_path)
    advanced_model = joblib.load(advanced_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    return baseline_model, advanced_model, scaler


def plot_confusion_matrices(cm_dict, output_path="confusion_matrices.png"):
    """
    Plots side-by-side Confusion Matrices using seaborn heatmap, annotated
    with counts, percentages, and (TN, FP, FN, TP) labels.
    """
    fig, axes = plt.subplots(1, len(cm_dict), figsize=(14, 6))
    if len(cm_dict) == 1:
        axes = [axes]

    labels = ["Normal (0)", "Intrusion (1)"]

    for ax, (model_name, cm) in zip(axes, cm_dict.items()):
        total = np.sum(cm)
        tn, fp, fn, tp = cm.ravel()

        # Build custom cell annotations: Count, Percentage, Metric quadrant
        annot_labels = np.array([
            [f"TN: {tn:,}\n({tn/total:.1%})", f"FP: {fp:,}\n({fp/total:.1%})"],
            [f"FN: {fn:,}\n({fn/total:.1%})", f"TP: {tp:,}\n({tp/total:.1%})"]
        ])

        sns.heatmap(
            cm,
            annot=annot_labels,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=1.5,
            linecolor="white",
            annot_kws={"fontsize": 11, "fontweight": "bold"}
        )

        ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_ylabel("Actual Label", fontsize=11, fontweight="bold")

    plt.suptitle("NSL-KDD Test Set - Intrusion Detection Evaluation", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save high-resolution figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[+] Confusion matrix plot saved to: {os.path.abspath(output_path)}")
    plt.close(fig)


def evaluate_nsl_kdd():
    """
    Step 4 Pipeline:
      1. Loads saved models from disk.
      2. Obtains preprocessed test data (X_test, y_test).
      3. Generates predictions and class probabilities.
      4. Calculates Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
      5. Generates detailed classification reports.
      6. Plots and saves side-by-side Confusion Matrices.
    """
    print("=" * 70)
    print("STEP 4: MODEL EVALUATION & PERFORMANCE COMPARISON")
    print("=" * 70)

    # 1. Load Saved Models
    baseline_model, advanced_model, _ = load_artifacts()

    # 2. Prepare Test Data
    _, _, X_test, y_test, _ = preprocess_nsl_kdd()
    y_test_bin = (y_test != 'normal').astype(int)

    target_names = ["Normal (0)", "Intrusion (1)"]
    models = {
        type(baseline_model).__name__: baseline_model,
        type(advanced_model).__name__: advanced_model
    }

    metrics_summary = []
    cm_dict = {}

    # 3. Evaluate each model
    for name, model in models.items():
        print("\n" + "=" * 70)
        print(f"EVALUATING MODEL: {name}")
        print("=" * 70)

        # Predictions
        y_pred = model.predict(X_test)

        # ROC-AUC if probability estimates are available
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test_bin, y_proba)
        except (AttributeError, IndexError):
            roc_auc = np.nan

        # Metric Calculations
        acc = accuracy_score(y_test_bin, y_pred)
        prec = precision_score(y_test_bin, y_pred)
        rec = recall_score(y_test_bin, y_pred)
        f1 = f1_score(y_test_bin, y_pred)

        metrics_summary.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        })

        # Display Scalar Metrics
        print(f"  • Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        print(f"  • Precision: {prec:.4f} ({prec*100:.2f}%)")
        print(f"  • Recall   : {rec:.4f} ({rec*100:.2f}%)")
        print(f"  • F1-Score : {f1:.4f} ({f1*100:.2f}%)")
        if not np.isnan(roc_auc):
            print(f"  • ROC-AUC  : {roc_auc:.4f}")

        # Classification Report
        print("\n" + "-" * 50)
        print(f"CLASSIFICATION REPORT ({name})")
        print("-" * 50)
        report = classification_report(y_test_bin, y_pred, target_names=target_names, digits=4)
        print(report)

        # Confusion Matrix
        cm = confusion_matrix(y_test_bin, y_pred)
        cm_dict[name] = cm
        tn, fp, fn, tp = cm.ravel()
        print("-" * 50)
        print(f"CONFUSION MATRIX BREAKDOWN ({name})")
        print("-" * 50)
        print(f"  • True Negatives  (TN - Normal correctly identified):    {tn:,}")
        print(f"  • False Positives (FP - Normal flagged as Attack):       {fp:,}  (False Alarm Rate: {fp/(fp+tn):.2%})")
        print(f"  • False Negatives (FN - Attack missed):                 {fn:,}")
        print(f"  • True Positives  (TP - Attack successfully blocked):   {tp:,}")

    # 4. Comparative Metrics Summary Table
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARATIVE SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(metrics_summary)
    print(summary_df.to_string(index=False))

    # 5. Plot and Save Confusion Matrices
    print("\n" + "-" * 50)
    print("VISUALIZING CONFUSION MATRICES")
    print("-" * 50)
    plot_path = os.path.join("models", "confusion_matrices.png")
    plot_confusion_matrices(cm_dict, output_path=plot_path)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return summary_df, cm_dict


if __name__ == "__main__":
    summary_df, cm_dict = evaluate_nsl_kdd()
