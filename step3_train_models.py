import os
import sys
import time
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Ensure UTF-8 output encoding for console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Import preprocessing pipeline from Step 2
from step2_preprocess import preprocess_nsl_kdd

def train_and_save_models(model_dir="models", baseline_type="decision_tree"):
    """
    Step 3 Pipeline:
      1. Preprocesses the NSL-KDD dataset into X_train, y_train, X_test, y_test.
      2. Converts raw attack strings into binary labels (0: Normal, 1: Intrusion).
      3. Initializes and trains a baseline model (DecisionTree or LogisticRegression).
      4. Initializes and trains an advanced ensemble model (RandomForestClassifier).
      5. Serializes trained models, scaler, and feature metadata using joblib.
      6. Prints confirmation messages and model artifacts status.
    """
    print("=" * 70)
    print("STEP 3: NIDS MODEL TRAINING & SERIALIZATION")
    print("=" * 70)

    # 1. Obtain Preprocessed Datasets
    X_train, y_train, X_test, y_test, scaler = preprocess_nsl_kdd()

    # 2. Encode Labels for Binary Intrusion Detection
    # 0 = Normal traffic, 1 = Attack / Intrusion
    # Crucial for NSL-KDD because test set contains 17 novel zero-day attack variants
    print("\n" + "-" * 50)
    print("TARGET LABEL ENCODING (Binary NIDS)")
    print("-" * 50)
    y_train_bin = (y_train != 'normal').astype(int)
    y_test_bin = (y_test != 'normal').astype(int)

    train_dist = y_train_bin.value_counts().to_dict()
    print(f"Training Labels  -> Normal (0): {train_dist.get(0, 0):,}, Intrusion (1): {train_dist.get(1, 0):,}")
    test_dist = y_test_bin.value_counts().to_dict()
    print(f"Testing Labels   -> Normal (0): {test_dist.get(0, 0):,}, Intrusion (1): {test_dist.get(1, 0):,}\n")

    # 3. Model Configuration
    # Baseline Model
    if baseline_type == "logistic_regression":
        baseline_name = "LogisticRegression"
        baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        baseline_name = "DecisionTreeClassifier"
        baseline_model = DecisionTreeClassifier(random_state=42, max_depth=15)

    # Advanced Model
    advanced_name = "RandomForestClassifier"
    advanced_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    # 4. Train Models
    print("-" * 50)
    print("MODEL TRAINING IN PROGRESS")
    print("-" * 50)

    # Train Baseline
    print(f"[*] Training Baseline ({baseline_name})...")
    start_time = time.time()
    baseline_model.fit(X_train, y_train_bin)
    baseline_duration = time.time() - start_time
    print(f"[OK] {baseline_name} trained successfully in {baseline_duration:.2f} seconds.")

    # Train Advanced
    print(f"\n[*] Training Advanced ({advanced_name})...")
    start_time = time.time()
    advanced_model.fit(X_train, y_train_bin)
    advanced_duration = time.time() - start_time
    print(f"[OK] {advanced_name} trained successfully in {advanced_duration:.2f} seconds.\n")

    # 5. Save Models and Metadata to Disk
    os.makedirs(model_dir, exist_ok=True)
    baseline_path = os.path.join(model_dir, "baseline_model.joblib")
    advanced_path = os.path.join(model_dir, "advanced_model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    features_path = os.path.join(model_dir, "feature_names.joblib")

    print("-" * 50)
    print(f"SERIALIZING ARTIFACTS TO '{model_dir}/'")
    print("-" * 50)

    joblib.dump(baseline_model, baseline_path)
    print(f"[+] Saved baseline model: {os.path.abspath(baseline_path)}")

    joblib.dump(advanced_model, advanced_path)
    print(f"[+] Saved advanced model: {os.path.abspath(advanced_path)}")

    joblib.dump(scaler, scaler_path)
    print(f"[+] Saved StandardScaler: {os.path.abspath(scaler_path)}")

    joblib.dump(list(X_train.columns), features_path)
    print(f"[+] Saved feature list:   {os.path.abspath(features_path)}\n")

    # 6. Final Confirmation
    print("=" * 70)
    print("CONFIRMATION: TRAINING & ARTIFACT PERSISTENCE COMPLETE!")
    print("=" * 70)
    print(f"1. Baseline Model ({baseline_name}) -> Ready for evaluation")
    print(f"2. Advanced Model ({advanced_name}) -> Ready for evaluation")
    print(f"3. Trained on {X_train.shape[0]:,} records across {X_train.shape[1]} features.")
    print("=" * 70)

    return {
        "baseline_model": baseline_model,
        "advanced_model": advanced_model,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train_bin,
        "X_test": X_test,
        "y_test": y_test_bin
    }

if __name__ == "__main__":
    trained_artifacts = train_and_save_models()
