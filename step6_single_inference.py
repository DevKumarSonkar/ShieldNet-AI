import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

# Ensure UTF-8 console output for Windows paths
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Import columns and path resolver
from step1_load_data import COLUMNS, resolve_dataset_path


def load_inference_artifacts(model_dir="models", model_name="advanced_model.joblib"):
    """
    Loads serialized model, StandardScaler, and training feature schema.
    """
    model_path = os.path.join(model_dir, model_name)
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    features_path = os.path.join(model_dir, "feature_names.joblib")

    for path, desc in [
        (model_path, "Trained Model"),
        (scaler_path, "StandardScaler"),
        (features_path, "Feature Names List")
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{desc} not found at '{path}'. Run step3_train_models.py first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)

    return model, scaler, feature_names


def preprocess_single_sample(sample_series_or_df, scaler, feature_names):
    """
    Preprocesses a single network connection sample to strictly match the 122-feature training schema:
      1. Isolates feature columns (drops 'attack' and 'difficulty_level' if present).
      2. Performs One-Hot Encoding on categorical attributes ('protocol_type', 'service', 'flag').
      3. Reindexes to match the exact 122 training columns, filling missing categories with 0.0.
      4. Standardizes numerical features using the fitted StandardScaler.
    """
    if isinstance(sample_series_or_df, pd.Series):
        sample_df = pd.DataFrame([sample_series_or_df])
    else:
        sample_df = sample_series_or_df.copy()

    # Drop target and metadata columns if present
    drop_cols = [c for c in ['attack', 'difficulty_level'] if c in sample_df.columns]
    features_df = sample_df.drop(columns=drop_cols)

    cat_cols = ['protocol_type', 'service', 'flag']
    num_cols = [c for c in features_df.columns if c not in cat_cols]

    # 1. One-hot encode the categorical columns
    encoded_df = pd.get_dummies(features_df, columns=cat_cols, dtype=float)

    # 2. Reindex against the exact 122 training columns (fill unseen categories with 0.0)
    aligned_df = encoded_df.reindex(columns=feature_names, fill_value=0.0)

    # 3. Standardize the continuous numerical columns using the saved scaler
    aligned_df[num_cols] = scaler.transform(aligned_df[num_cols])

    return aligned_df


def predict_connection(sample_df, model, scaler, feature_names):
    """
    Performs inference on a single incoming network flow.
    Returns: prediction (0 or 1), predicted_label, and class probabilities.
    """
    processed_vector = preprocess_single_sample(sample_df, scaler, feature_names)

    # Make prediction
    pred_code = model.predict(processed_vector)[0]
    pred_label = "Intrusion (Attack)" if pred_code == 1 else "Normal Traffic"

    # Prediction confidence
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(processed_vector)[0]
        prob_normal = probabilities[0]
        prob_intrusion = probabilities[1]
        confidence = max(prob_normal, prob_intrusion)
    else:
        prob_normal = np.nan
        prob_intrusion = np.nan
        confidence = 1.0

    return {
        "pred_code": int(pred_code),
        "pred_label": pred_label,
        "confidence": float(confidence),
        "prob_normal": float(prob_normal),
        "prob_intrusion": float(prob_intrusion),
        "processed_vector": processed_vector
    }


def run_single_sample_inference(sample_type="random", index=None, model_type="advanced"):
    """
    End-to-end single-sample demonstration.
    """
    print("=" * 75)
    print("STEP 6: REAL-TIME SINGLE-SAMPLE NIDS INFERENCE")
    print("=" * 75)

    # 1. Load Artifacts
    model_file = "advanced_model.joblib" if model_type == "advanced" else "baseline_model.joblib"
    model, scaler, feature_names = load_inference_artifacts(model_name=model_file)
    print(f"[+] Loaded Model: {type(model).__name__}")
    print(f"[+] Expected Input Features: {len(feature_names)} columns")

    # 2. Select a Sample from KDDTest+.txt
    test_path = resolve_dataset_path("KDDTest+.txt")
    test_df = pd.read_csv(test_path, header=None, names=COLUMNS)

    if index is not None and 0 <= index < len(test_df):
        sample = test_df.iloc[[index]].copy()
        sample_idx = index
    elif sample_type == "normal":
        sample = test_df[test_df['attack'] == 'normal'].sample(1, random_state=np.random.randint(0, 10000)).copy()
        sample_idx = sample.index[0]
    elif sample_type == "attack":
        sample = test_df[test_df['attack'] != 'normal'].sample(1, random_state=np.random.randint(0, 10000)).copy()
        sample_idx = sample.index[0]
    else:
        sample = test_df.sample(1, random_state=np.random.randint(0, 10000)).copy()
        sample_idx = sample.index[0]

    actual_attack = sample['attack'].values[0]
    actual_category = "Normal Traffic" if actual_attack == "normal" else f"Attack / Intrusion ({actual_attack})"

    # 3. Display Raw Network Flow Attributes
    print("\n" + "-" * 75)
    print(f"RAW INCOMING NETWORK CONNECTION (Sample Index #{sample_idx:,})")
    print("-" * 75)
    display_keys = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'logged_in', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate'
    ]
    for k in display_keys:
        print(f"  • {k:<28} : {sample[k].values[0]}")
    print(f"  • {'[Ground Truth Attack Label]':<28} : {actual_attack}")

    # 4. Run Preprocessing and Model Inference
    result = predict_connection(sample, model, scaler, feature_names)

    # 5. Display Prediction Results
    print("\n" + "=" * 75)
    print("INFERENCE DETECTION RESULT")
    print("=" * 75)

    is_correct = (result['pred_code'] == 0 and actual_attack == 'normal') or \
                 (result['pred_code'] == 1 and actual_attack != 'normal')

    status_tag = "[ALERT: INTRUSION DETECTED]" if result['pred_code'] == 1 else "[ALLOWED: NORMAL TRAFFIC]"
    print(f"  Status        : {status_tag}")
    print(f"  Prediction    : {result['pred_label']}")
    print(f"  Confidence    : {result['confidence'] * 100:.2f}%")
    print(f"  Probability   : Normal: {result['prob_normal']*100:.2f}%  |  Intrusion: {result['prob_intrusion']*100:.2f}%")
    print(f"  Ground Truth  : {actual_category}")
    print(f"  Verdict       : {'✓ CORRECT PREDICTION' if is_correct else '✗ MISCLASSIFICATION'}")
    print("=" * 75)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-sample NIDS Inference")
    parser.add_argument("--sample-type", choices=["random", "normal", "attack"], default="random",
                        help="Filter sample by traffic type")
    parser.add_argument("--index", type=int, default=None, help="Specific row index from test dataset")
    parser.add_argument("--model", choices=["advanced", "baseline"], default="advanced", help="Model to use")
    args = parser.parse_args()

    run_single_sample_inference(sample_type=args.sample_type, index=args.index, model_type=args.model)
