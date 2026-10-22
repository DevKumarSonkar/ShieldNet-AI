# ShieldNet AI — Network Watcher

ShieldNet AI is a Streamlit network monitoring dashboard with live Scapy capture, PCAP analysis, explainable baseline flow detection, optional compatible machine-learning artifacts, safe local traffic simulation, and auditable mitigation/reporting.

## Current detection status

The dashboard extracts nine flow-window features: packet count, total bytes, average packet size, throughput, TCP SYN/ACK/RST ratios, unique destination ports, and average inter-arrival time. Flow windows are two seconds and grouped by source IP.

The checked-in `models/advanced_model.joblib`, scaler, and feature schema were built for the separate NSL-KDD connection dataset (122 encoded input columns). They do **not** describe these live packet features, so ShieldNet deliberately does not use them for live inference. Unless compatible live artifacts exist, the dashboard labels detection **Baseline rules**. Those heuristics are not trained ML predictions.

To enable live hybrid inference, add trusted artifacts to `models/`:

- `live_feature_names.joblib`: the ordered list matching `FEATURE_NAMES` in `shieldnet.py`.
- `live_random_forest.joblib`: a fitted scikit-learn `RandomForestClassifier` trained on the nine flow features and binary labels (`0` benign, `1` threat).
- `live_isolation_forest.joblib`: an optional fitted scikit-learn `IsolationForest` trained on the same feature order.

The runtime checks artifact types and feature counts. Loading joblib files can execute code; only load model files from a trusted source. No training is performed by the dashboard.

## Windows setup

Open PowerShell in the project directory and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .streamlit\config.toml.example .streamlit\config.toml
streamlit run app.py --server.address 127.0.0.1
```

Then open <http://localhost:8501>.

For live capture on Windows, install Npcap and start PowerShell as Administrator if the selected adapter requires elevation. Capture can still be unavailable due to adapter permissions, driver setup, or other software using the adapter; the dashboard displays the capture error and remains usable for PCAP analysis. Do not run elevated unless live capture or a confirmed firewall rule requires it.

Linux capture commonly requires root or packet-capture capabilities and libpcap. The optional Linux firewall integration requires root and a preconfigured nftables `inet filter input` chain. Firewall integration only adds a rule after the user checks the confirmation control and the process passes the administrator check. Rule removal is not automated.

Docker starts the dashboard but does not grant packet-capture privileges or firewall access. Use host installation for live monitoring unless you deliberately configure a trusted, least-privilege container environment.

## Dashboard controls

- **Start monitoring / Stop** runs capture in a background thread. Scapy capture supports TCP, UDP, and ICMP over IPv4/IPv6.
- **Analyze uploaded capture** accepts PCAP/PCAPNG and uses the same feature and detection pipeline.
- **Detection sensitivity** adjusts the alert threshold. The score remains normalized from 0 to 100.
- **Auto-add sources to deny cache** records eligible public source addresses in this process's in-memory cache. This is not an OS/network packet filter and is not persisted.
- **Block IP Now** adds a validated public IP to that cache. Loopback, reserved, link-local, multicast, unspecified, and private/non-global addresses are protected by default.
- **Add OS firewall rule** is separate, requires a second explicit confirmation, validates the address, checks administrator privileges, and invokes the platform command without a shell. No forged TCP resets are sent.
- **Safe local simulation** requires explicit initiation and sends only bounded raw TCP SYN or UDP probes to `127.0.0.1` (maximum 32 packets, 10 packets per second, 5 seconds).
- The incident log can be exported as CSV/JSON. Session totals and mitigation actions can be downloaded as TXT/PDF.

## Existing NSL-KDD pipeline

The original staged pipeline remains available:

```powershell
python step1_load_data.py
python step2_preprocess.py
python step3_train_models.py
python step4_evaluate_models.py
python step5_feature_importance.py
python step6_single_inference.py --sample-type attack
```

Those steps train/evaluate the dataset-specific models used by the legacy single-sample inference script. They do not train a live-flow detector. Review the metric outputs produced by your own run before using them; historic figures in earlier documentation are not a guarantee of current results.

## Tests

Run unit tests from the project root:

```powershell
python -m pytest -q
```

Tests cover flow features, threat scoring, mitigation validation/confirmation safeguards, local-only simulation bounds, model compatibility fallback, and report generation. They do not perform live capture or change real firewall rules.

## Configuration and logs

Copy `.streamlit/config.toml.example` to `.streamlit/config.toml` for local Streamlit settings. No credentials or secrets are required. Operational diagnostics use Python logging; incident exports are generated on demand and no packet payloads are retained in the incident records.
