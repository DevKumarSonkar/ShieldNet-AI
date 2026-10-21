# 🛡️ Network Watcher: Machine Learning Network Intrusion Detection System (NIDS)

A production-grade, end-to-end Network Intrusion Detection System (NIDS) trained on the **NSL-KDD benchmark dataset**. The system detects malicious traffic patterns, alerts on zero-day attacks, extracts key feature importances, and provides real-time inference via a **Streamlit Security Operations Center (SOC) dashboard** and containerized Docker environment.

---

## 📌 Table of Contents
- [Project Architecture](#-project-architecture)
- [Dataset Overview (NSL-KDD)](#-dataset-overview-nsl-kdd)
- [Performance Benchmarks](#-performance-benchmarks)
- [Project Structure](#-project-structure)
- [Local Installation & Pipeline Execution](#-local-installation--pipeline-execution)
- [Running the Streamlit Dashboard](#-running-the-streamlit-dashboard)
- [Docker Containerization](#-docker-containerization)
- [License](#-license)

---

## 🏗️ Project Architecture

The pipeline is organized into modular, reproducible stages:

```
[KDDTrain+.txt & KDDTest+.txt]
             │
             ▼
   Step 1: Data Ingestion & Schema Assignment (43 standard NSL-KDD columns)
             │
             ▼
   Step 2: Preprocessing & Column Alignment (122 one-hot features, leak-free StandardScaler)
             │
             ▼
   Step 3: Model Training (DecisionTree Baseline & RandomForest Ensemble)
             │
             ▼
   Step 4: Benchmark Evaluation (Unseen test set with 17 zero-day attack vectors)
             │
             ▼
   Step 5: Feature Importance & Interpretability (Top 15 ranking & Gini scores)
             │
             ▼
   Step 6: Real-time Single-Packet Inference Engine
             │
             ▼
   Step 7: Streamlit SOC Dashboard (Interactive traffic inspector & alerting)
             │
             ▼
   Step 8: Production Containerization (Docker & Docker Compose)
```

---

## 📊 Dataset Overview (NSL-KDD)

The **NSL-KDD** dataset solves several inherent issues of the legacy KDD'99 dataset by removing redundant records to ensure classifiers are not biased toward frequent flows.

- **Training Records (`KDDTrain+.txt`)**: `125,973` connections (53.46% Normal, 46.54% Intrusion).
- **Testing Records (`KDDTest+.txt`)**: `22,544` connections.
- **Novel Zero-Day Threat Evaluation**: The test set contains **17 distinct attack types** (such as `apache2`, `mscan`, `snmpguess`, `saint`, `httptunnel`, `mailbomb`, `udpstorm`) that **never appear in the training dataset**, providing a rigorous benchmark for true behavioral anomaly detection.

### Feature Representation
- **Raw Connection Features**: 41 network traffic features (3 categorical: `protocol_type`, `service`, `flag`; 38 continuous numerical features).
- **Target Label**: `attack` (mapped to binary: `0 = Normal`, `1 = Intrusion / Anomaly`).
- **One-Hot Encoded Schema**: Categorical expansion yields **122 unified feature columns** strictly aligned across train and test sets.

---

## 📈 Performance Benchmarks

Evaluated against the **22,544 unseen test connections** in `KDDTest+.txt`:

| Metric | Baseline (`DecisionTreeClassifier`) | Advanced (`RandomForestClassifier`) |
| :--- | :---: | :---: |
| **Accuracy** | 78.31% | 76.97% |
| **Precision** | 95.83% | **96.72%** |
| **Recall** | 64.71% | 61.64% |
| **F1-Score** | 77.25% | 75.29% |
| **ROC-AUC** | 0.7236 | **0.9614** |
| **False Alarms (FP)** | 361 (3.72%) | **268 (2.76%)** |
| **Attacks Blocked (TP)** | 8,304 | 7,910 |

> **Key ML Insight**: Random Forest achieves **96.72% precision** with a **False Alarm Rate (FAR) of only 2.76%**, minimizing alert fatigue in production SOC workflows. The **0.9614 ROC-AUC** demonstrates superior probability ranking across novel zero-day exploits.

### Top 5 Predictive Features
1. **`src_bytes` (15.92%)**: Outgoing payload volume.
2. **`dst_bytes` (8.44%)**: Inbound payload volume.
3. **`flag_SF` (6.94%)**: Normal TCP 3-way handshake completion.
4. **`dst_host_srv_count` (6.02%)**: Destination port service connection density.
5. **`logged_in` (5.62%)**: Authentication success indicator.

---

## 📁 Project Structure

```
Network-Watcher/
├── dataset/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── models/
│   ├── baseline_model.joblib
│   ├── advanced_model.joblib
│   ├── scaler.joblib
│   ├── feature_names.joblib
│   ├── confusion_matrices.png
│   └── top_15_feature_importance.png
├── step1_load_data.py            # Step 1: Ingestion & 43-column schema assignment
├── step2_preprocess.py           # Step 2: One-Hot alignment & StandardScaler
├── step3_train_models.py         # Step 3: Train & serialize models
├── step4_evaluate_models.py      # Step 4: Metrics, reports & confusion matrix
├── step5_feature_importance.py   # Step 5: Gini importance & ranking bar chart
├── step6_single_inference.py     # Step 6: Single-sample CLI inference engine
├── app.py                        # Step 7: Streamlit SOC web dashboard
├── Dockerfile                    # Step 8: Production Docker container
├── docker-compose.yml            # Step 8: Multi-container orchestration
├── .dockerignore                 # Step 8: Docker build context exclusions
├── requirements.txt              # Production dependency specifications
└── README.md                     # Documentation
```

---

## 💻 Local Installation & Pipeline Execution

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/Network-Watcher.git
cd Network-Watcher

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Step-by-Step Pipeline
```bash
# Step 1: Load and inspect the NSL-KDD dataset
python step1_load_data.py

# Step 2: Run preprocessing & generate (X_train, y_train, X_test, y_test)
python step2_preprocess.py

# Step 3: Train baseline & advanced models and save to models/
python step3_train_models.py

# Step 4: Evaluate test set performance and generate confusion matrix
python step4_evaluate_models.py

# Step 5: Extract and visualize top 15 feature importances
python step5_feature_importance.py

# Step 6: Run single-sample inference CLI
python step6_single_inference.py --sample-type attack
python step6_single_inference.py --sample-type normal
```

---

## 🌐 Running the Streamlit Dashboard

Launch the interactive dashboard:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to inspect network flows, test attack presets (`Neptune DoS`, `Smurf ICMP`, `Apache2 Zero-Day`), and view live security alerts.

---

## 🐳 Docker Containerization

### Build & Run with Docker CLI
```bash
# 1. Build the Docker container image
docker build -t network-watcher-nids:latest .

# 2. Run container in detached mode mapping port 8501
docker run -d -p 8501:8501 --name nids-dashboard network-watcher-nids:latest

# 3. View container logs
docker logs -f nids-dashboard

# 4. Stop container
docker stop nids-dashboard
```

### Build & Run with Docker Compose
```bash
# Start container in background
docker compose up -d

# Stop container
docker compose down
```

Access the containerized dashboard at **`http://localhost:8501`**.

---

## 📜 License
This project is licensed under the Apache 2.0 License.
