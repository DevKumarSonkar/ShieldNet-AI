import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ensure UTF-8 output encoding for console (handles non-ASCII directory paths on Windows)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Import the 43 standard columns and path resolver from Step 1
from step1_load_data import COLUMNS, resolve_dataset_path

# ---------------------------------------------------------
# Attack Taxonomy Mapping (NSL-KDD 5-Class Categorization)
# ---------------------------------------------------------
ATTACK_MAPPING = {
    # Denial of Service (DoS)
    'neptune': 'DoS', 'smurf': 'DoS', 'pod': 'DoS', 'teardrop': 'DoS',
    'land': 'DoS', 'back': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
    'processtable': 'DoS', 'mailbomb': 'DoS',
    # Probe / Surveillance
    'satan': 'Probe', 'ipsweep': 'Probe', 'portsweep': 'Probe',
    'nmap': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    # Remote to Local (R2L)
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
    'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L',
    'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    # User to Root (U2R)
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
    # Normal Traffic
    'normal': 'Normal'
}


def load_raw_datasets(train_file="KDDTrain+.txt", test_file="KDDTest+.txt"):
    """
    Loads both training and testing raw datasets using the 43 standard column headers.
    """
    # Accept common alternative spelling like KDTest+.txt
    try:
        train_path = resolve_dataset_path(train_file)
    except FileNotFoundError:
        train_path = resolve_dataset_path("KDDTrain+.txt")

    try:
        test_path = resolve_dataset_path(test_file)
    except FileNotFoundError:
        # Fallback if named KDTest+.txt
        test_path = resolve_dataset_path("KDTest+.txt")

    print(f"Loading Training Set from: {os.path.abspath(train_path)}")
    train_df = pd.read_csv(train_path, header=None, names=COLUMNS)

    print(f"Loading Testing Set from:  {os.path.abspath(test_path)}")
    test_df = pd.read_csv(test_path, header=None, names=COLUMNS)

    return train_df, test_df


def preprocess_nsl_kdd(train_file="KDDTrain+.txt", test_file="KDDTest+.txt"):
    """
    Step 2 Pipeline:
      1. Loads train and test datasets.
      2. Separates target ('attack') and metadata ('difficulty_level').
      3. One-Hot Encodes categorical features (protocol_type, service, flag) across
         both sets to ensure identical feature schemas.
      4. Fits StandardScaler ONLY on training numerical data to avoid data leakage,
         then transforms both train and test sets.
      5. Splits back into (X_train, y_train) and (X_test, y_test).
    """
    print("=" * 70)
    print("STEP 2: DATA PREPROCESSING & FEATURE PIPELINE")
    print("=" * 70)

    # 1. Load Datasets
    train_df, test_df = load_raw_datasets(train_file, test_file)
    n_train = len(train_df)
    n_test = len(test_df)
    print(f"\n[+] Raw Train Shape: {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")
    print(f"[+] Raw Test Shape:  {test_df.shape[0]:,} rows x {test_df.shape[1]} columns\n")

    # 2. Extract Target Labels and Metadata
    # 'attack' is the target; 'difficulty_level' is a study score, not a live network feature
    y_train = train_df['attack'].copy()
    y_test = test_df['attack'].copy()

    drop_cols = ['attack', 'difficulty_level']
    features_train = train_df.drop(columns=drop_cols)
    features_test = test_df.drop(columns=drop_cols)

    # Identify categorical and numerical columns
    cat_cols = ['protocol_type', 'service', 'flag']
    num_cols = [c for c in features_train.columns if c not in cat_cols]

    print("-" * 50)
    print("CATEGORICAL & NUMERICAL FEATURE IDENTIFICATION")
    print("-" * 50)
    print(f"Categorical features to One-Hot Encode ({len(cat_cols)}): {cat_cols}")
    print(f"Numerical features to Scale ({len(num_cols)}): {len(num_cols)} features\n")

    # 3. Handle Categorical Encoding with One-Hot Encoding
    # Combine feature sets temporarily to align all possible categories (e.g. services present in train but absent in test)
    print("-" * 50)
    print("ONE-HOT ENCODING (Column Alignment)")
    print("-" * 50)
    combined_features = pd.concat([features_train, features_test], axis=0, ignore_index=True)
    combined_encoded = pd.get_dummies(combined_features, columns=cat_cols, dtype=float)

    total_encoded_features = combined_encoded.shape[1]
    dummy_cols_count = total_encoded_features - len(num_cols)
    print(f"[+] Expanded 3 categorical features into {dummy_cols_count} binary indicator columns.")
    print(f"[+] Total aligned feature columns: {total_encoded_features}\n")

    # Split combined features back into train and test
    X_train = combined_encoded.iloc[:n_train].copy().reset_index(drop=True)
    X_test = combined_encoded.iloc[n_train:].copy().reset_index(drop=True)

    # 4. Feature Scaling (Preventing Data Leakage)
    print("-" * 50)
    print("NUMERICAL SCALING (StandardScaler)")
    print("-" * 50)
    print("[*] Fitting StandardScaler ONLY on X_train numerical features...")
    scaler = StandardScaler()
    
    # Fit on train, transform train
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    # Transform test using train parameters
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    print("[OK] Scaling completed without data leakage.\n")

    # 5. Verification & Shapes
    print("=" * 70)
    print("PREPROCESSING SUMMARY & FINAL SHAPES")
    print("=" * 70)
    print(f"X_train shape: {X_train.shape}  |  y_train shape: {y_train.shape}")
    print(f"X_test shape : {X_test.shape}   |  y_test shape : {y_test.shape}")
    
    # Assert column consistency
    assert (X_train.columns == X_test.columns).all(), "Error: Feature column mismatch between Train and Test!"
    print(f"[OK] Exact feature column alignment verified: {X_train.shape[1]} identical columns.")
    print("=" * 70)

    return X_train, y_train, X_test, y_test, scaler


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, scaler = preprocess_nsl_kdd()
