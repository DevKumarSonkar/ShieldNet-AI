import os
import sys

# Ensure UTF-8 output encoding for console (handles non-ASCII directory paths on Windows)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd

# ---------------------------------------------------------
# Step 1: Define the 43 Standard Column Headers for NSL-KDD
# ---------------------------------------------------------
# NSL-KDD has 41 network traffic features + 1 attack label + 1 difficulty level
COLUMNS = [
    # 1-4: Basic features of individual TCP connections
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    # 10-22: Content features within a connection suggested by domain knowledge
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    # 23-31: Traffic features computed using a 2-second time window
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    # 32-41: Host-based traffic features computed using historical connection windows
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    # 42-43: Target label and difficulty level
    'attack',
    'difficulty_level'
]


def resolve_dataset_path(filename="KDDTrain+.txt"):
    """
    Locates the dataset file, handling both flat and nested folder structures.
    """
    candidates = [
        os.path.join("dataset", filename, filename),
        os.path.join("dataset", filename),
        filename
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Could not locate '{filename}' in candidate paths: {candidates}")


def load_and_inspect_dataset(data_path=None):
    """
    Loads KDDTrain+.txt, assigns column headers, checks missing values,
    and displays basic dataset statistics and class distribution.
    """
    if data_path is None:
        data_path = resolve_dataset_path("KDDTrain+.txt")

    print("=" * 70)
    print("STEP 1: LOADING & INSPECTING NSL-KDD DATASET")
    print("=" * 70)
    print(f"Loading dataset from: {os.path.abspath(data_path)}\n")

    # Load dataset with standard column names
    df = pd.read_csv(data_path, header=None, names=COLUMNS)

    # 1. Basic Shape and Preview
    print(f"[+] Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"    (Total Features: {len(COLUMNS) - 2}, Target: 1, Difficulty: 1)\n")

    # 2. Check for Missing Values
    null_counts = df.isnull().sum()
    total_missing = null_counts.sum()
    print("-" * 50)
    print("MISSING VALUES CHECK")
    print("-" * 50)
    if total_missing == 0:
        print("[OK] No missing values detected across all 43 columns.\n")
    else:
        print(f"[!] Warning: Found {total_missing} missing values:")
        print(null_counts[null_counts > 0])
        print()

    # 3. Basic Dataset Info
    print("-" * 50)
    print("DATASET INFO")
    print("-" * 50)
    print(f"Memory Usage: {df.memory_usage().sum() / (1024 ** 2):.2f} MB")
    print("Data Types Breakdown:")
    print(df.dtypes.value_counts().to_string())
    print()

    # Categorical and numerical columns count
    cat_cols = df.select_dtypes(include=['object', 'string', 'str']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}")
    print(f"Numerical Columns ({len(num_cols)}): {len(num_cols)} columns\n")

    # 4. Attack Class Distribution
    print("-" * 50)
    print("ATTACK CLASS DISTRIBUTION (DETAILED)")
    print("-" * 50)
    attack_counts = df['attack'].value_counts()
    attack_pct = df['attack'].value_counts(normalize=True) * 100
    dist_df = pd.DataFrame({'Count': attack_counts, 'Percentage (%)': attack_pct.round(2)})
    print(dist_df.to_string())
    print()

    # Binary distribution (normal vs attack)
    print("-" * 50)
    print("BINARY DISTRIBUTION (Normal vs Intrusion)")
    print("-" * 50)
    binary_label = df['attack'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
    bin_counts = binary_label.value_counts()
    bin_pct = binary_label.value_counts(normalize=True) * 100
    bin_df = pd.DataFrame({'Count': bin_counts, 'Percentage (%)': bin_pct.round(2)})
    print(bin_df.to_string())
    print()

    # 5. First 5 Rows Preview
    print("-" * 50)
    print("FIRST 5 ROWS PREVIEW")
    print("-" * 50)
    print(df.head(5)[['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'attack', 'difficulty_level']])
    print("=" * 70)

    return df


if __name__ == "__main__":
    df = load_and_inspect_dataset()
