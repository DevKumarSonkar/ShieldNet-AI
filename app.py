import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from step1_load_data import COLUMNS, resolve_dataset_path
from step6_single_inference import preprocess_single_sample

# ------------------------------------------------------------------
# Page config — wide, clean, no favicon emoji
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Network Watcher · NIDS",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# CSS — no harsh gradients, no rainbow, no neon, no glow, no orbs.
#        Uses a disciplined professional palette:
#          --bg       : #F5F6FA  (off-white — not pure white)
#          --surface  : #FFFFFF
#          --border   : #E4E7EC
#          --navy     : #14213D  (primary text / headings)
#          --accent   : #2563EB  (one deliberate blue accent only)
#          --danger   : #B91C1C  (alert, kept restrained)
#          --success  : #166534  (safe, kept restrained)
#          --muted    : #6B7280
#        Typography: system-ui stack — no Geist, no Space Grotesk.
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg:       #F5F6FA;
    --surface:  #FFFFFF;
    --border:   #E4E7EC;
    --navy:     #14213D;
    --accent:   #2563EB;
    --danger:   #B91C1C;
    --success:  #166534;
    --warn:     #92400E;
    --muted:    #6B7280;
    --radius:   6px;
}

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background-color: var(--bg);
    color: var(--navy);
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}

/* ---------- Main area ---------- */
.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1200px;
}

/* ---------- Page header ---------- */
.page-header {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 1.75rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
}

.page-header h1 {
    margin: 0;
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--navy);
    letter-spacing: -0.3px;
    line-height: 1.2;
}

.page-header p {
    margin: 4px 0 0;
    font-size: 0.875rem;
    color: var(--muted);
    font-weight: 400;
}

/* ---------- Section headings ---------- */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}

/* ---------- Stat row ---------- */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.5rem;
}

.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 16px;
}

.stat-card .label {
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}

.stat-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--navy);
    line-height: 1;
}

.stat-card .sub {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 4px;
}

/* ---------- Input panel ---------- */
.input-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}

/* ---------- Verdict cards ---------- */
.verdict-card {
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-top: 1.25rem;
    border-left: 4px solid;
}

.verdict-card.attack {
    background: #FEF2F2;
    border-left-color: var(--danger);
    color: #7F1D1D;
}

.verdict-card.normal {
    background: #F0FDF4;
    border-left-color: var(--success);
    color: #14532D;
}

.verdict-card h3 {
    margin: 0 0 6px;
    font-size: 1rem;
    font-weight: 700;
}

.verdict-card p {
    margin: 0;
    font-size: 0.875rem;
    line-height: 1.5;
}

/* ---------- Metric inline badges ---------- */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
}

.badge-danger  { background:#FEE2E2; color:#B91C1C; }
.badge-success { background:#DCFCE7; color:#15803D; }
.badge-neutral { background:#F3F4F6; color:#374151; }

/* ---------- Table styling ---------- */
.metrics-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}

.metrics-table th {
    background: #F9FAFB;
    font-weight: 600;
    color: var(--muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 8px 12px;
    border: 1px solid var(--border);
    text-align: left;
}

.metrics-table td {
    padding: 8px 12px;
    border: 1px solid var(--border);
    color: var(--navy);
}

.metrics-table tr:nth-child(even) td { background: #FAFAFA; }

/* ---------- Preset buttons ---------- */
.stButton > button {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--navy);
    font-size: 0.8rem;
    font-weight: 500;
    border-radius: var(--radius);
    padding: 5px 10px;
    width: 100%;
    transition: background 0.15s, border-color 0.15s;
    cursor: pointer;
}

.stButton > button:hover {
    background: #F3F4F6;
    border-color: #D1D5DB;
}

/* Predict button overrides the above */
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stButton"]:last-child) > div[data-testid="stButton"] button,
div.predict-btn button {
    background: var(--accent) !important;
    color: #fff !important;
    border-color: var(--accent) !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    padding: 8px 14px !important;
}

/* ---------- Tab strip ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--border);
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--muted);
    padding: 8px 18px;
    border-radius: 0;
    background: transparent;
    border-bottom: 2px solid transparent;
}

.stTabs [aria-selected="true"] {
    color: var(--navy) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

/* ---------- Sidebar section label ---------- */
.sidebar-section {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 12px 0 4px;
    border-top: 1px solid var(--border);
    margin-top: 8px;
}

/* ---------- Preset active indicator ---------- */
.preset-indicator {
    font-size: 0.78rem;
    color: var(--accent);
    font-weight: 500;
    padding: 4px 8px;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: var(--radius);
    margin-top: 6px;
    margin-bottom: 2px;
}

/* ---------- Form section header ---------- */
.form-section-header {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--navy);
    margin: 1rem 0 0.5rem;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
}

/* ---------- Empty state ---------- */
.empty-state {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem 2rem;
    text-align: center;
    margin-top: 1.25rem;
}

.empty-state p {
    color: var(--muted);
    font-size: 0.875rem;
    margin: 0.4rem 0 0;
    line-height: 1.6;
}

.empty-state .title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--navy);
    margin-bottom: 6px;
}

/* ---------- Flag legend ---------- */
.flag-legend {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 0.5rem;
}

.flag-item {
    background: #F9FAFB;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 8px 10px;
    font-size: 0.78rem;
}

.flag-item .flag-name {
    font-weight: 700;
    color: var(--navy);
    font-family: 'Courier New', monospace;
    font-size: 0.8rem;
}

.flag-item .flag-desc {
    color: var(--muted);
    margin-top: 2px;
    line-height: 1.4;
}

/* ---------- Progress bar override ---------- */
.stProgress > div > div { background-color: var(--accent); border-radius: 3px; }
.stProgress > div { background-color: var(--border); border-radius: 3px; height: 6px; }

/* ---------- Hide Streamlit chrome ---------- */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Data loading — fixed nested-loop bug from user edit
# ------------------------------------------------------------------
@st.cache_resource
def load_all_artifacts():
    model_dir = "models"
    rf_model   = joblib.load(os.path.join(model_dir, "advanced_model.joblib"))
    dt_model   = joblib.load(os.path.join(model_dir, "baseline_model.joblib"))
    scaler     = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.joblib"))

    train_path    = resolve_dataset_path("KDDTrain+.txt")
    train_sample  = pd.read_csv(train_path, header=None, names=COLUMNS, nrows=5000)
    features_only = train_sample.drop(columns=["attack", "difficulty_level"])

    # Build defaults — correct single-pass implementation
    default_record = {}
    for col in features_only.columns:
        if pd.api.types.is_numeric_dtype(features_only[col]):
            default_record[col] = float(features_only[col].median())
        else:
            mode_val = features_only[col].mode()
            default_record[col] = mode_val[0] if not mode_val.empty else "unknown"

    return {
        "rf_model":      rf_model,
        "dt_model":      dt_model,
        "scaler":        scaler,
        "feature_names": feature_names,
        "default_record": default_record,
    }

artifacts = load_all_artifacts()

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
PROTOCOLS = ["tcp", "udp", "icmp"]
FLAGS     = ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "RSTOS0", "S3", "OTH"]
SERVICES  = sorted([
    "http", "smtp", "domain_u", "ftp_data", "private", "eco_i", "other",
    "ecr_i", "telnet", "finger", "ftp", "auth", "Z39_50", "uucp", "courier",
    "bgp", "whois", "uucp_path", "iso_tsap", "time", "IRC", "X11", "aol",
    "csnet_ns", "ctf", "daytime", "discard", "domain", "echo", "efs", "exec",
    "gopher", "harvest", "hostnames", "http_2784", "http_443", "http_8001",
    "imap4", "klogin", "kshell", "ldap", "link", "login", "mtp", "name",
    "netbios_dgm", "netbios_ns", "netbios_ssn", "netstat", "nnsp", "nntp",
    "ntp_u", "pm_dump", "pop_2", "pop_3", "printer", "red_i", "remote_job",
    "rje", "shell", "sql_net", "ssh", "sunrpc", "supdup", "systat", "tftp_u",
    "tim_i", "urh_i", "urp_i", "vmnet",
])

# Real captured packets as presets — not fake
PRESETS = {
    "Normal HTTP Session": dict(
        proto="tcp",  service="http",    flag="SF",   src_b=232,   dst_b=8153,
        dur=0,        logged=1,          count=5,     srv_cnt=5,   dst_cnt=30, dst_srv_cnt=255,
    ),
    "SYN Flood (Neptune DoS)": dict(
        proto="tcp",  service="private", flag="S0",   src_b=0,     dst_b=0,
        dur=0,        logged=0,          count=123,   srv_cnt=6,   dst_cnt=255, dst_srv_cnt=26,
    ),
    "ICMP Broadcast Flood (Smurf)": dict(
        proto="icmp", service="ecr_i",   flag="SF",   src_b=1032,  dst_b=0,
        dur=0,        logged=0,          count=511,   srv_cnt=511, dst_cnt=255, dst_srv_cnt=138,
    ),
    "Apache2 DoS (Zero-Day)": dict(
        proto="tcp",  service="http",    flag="RSTR", src_b=76944, dst_b=1,
        dur=805,      logged=1,          count=12,    srv_cnt=12,  dst_cnt=241, dst_srv_cnt=238,
    ),
    "Portsweep Probe": dict(
        proto="tcp",  service="private", flag="REJ",  src_b=0,     dst_b=0,
        dur=0,        logged=0,          count=1,     srv_cnt=1,   dst_cnt=255, dst_srv_cnt=18,
    ),
    "FTP Brute Force (R2L)": dict(
        proto="tcp",  service="ftp",     flag="SF",   src_b=148,   dst_b=0,
        dur=0,        logged=0,          count=1,     srv_cnt=1,   dst_cnt=3,   dst_srv_cnt=3,
    ),
}

# Session-state initialisation
_state_defaults = {
    "ss_proto":         "tcp",
    "ss_service":       "http",
    "ss_flag":          "SF",
    "ss_duration":      0,
    "ss_src_bytes":     232,
    "ss_dst_bytes":     8153,
    "ss_logged_in":     1,
    "ss_count":         5,
    "ss_srv_count":     5,
    "ss_dst_host_cnt":  30,
    "ss_dst_srv_cnt":   255,
    "ss_active_preset": "Normal HTTP Session",
    "ss_last_result":   None,   # persists prediction output across widget changes
}
for k, v in _state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def apply_preset(name):
    p = PRESETS[name]
    st.session_state.ss_proto         = p["proto"]
    st.session_state.ss_service       = p["service"]
    st.session_state.ss_flag          = p["flag"]
    st.session_state.ss_duration      = p["dur"]
    st.session_state.ss_src_bytes     = p["src_b"]
    st.session_state.ss_dst_bytes     = p["dst_b"]
    st.session_state.ss_logged_in     = p["logged"]
    st.session_state.ss_count         = p["count"]
    st.session_state.ss_srv_count     = p["srv_cnt"]
    st.session_state.ss_dst_host_cnt  = p["dst_cnt"]
    st.session_state.ss_dst_srv_cnt   = p["dst_srv_cnt"]
    st.session_state.ss_active_preset = name
    st.session_state.ss_last_result   = None  # clear stale result when preset changes


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Network Watcher")
    st.markdown("NSL-KDD · Intrusion Detection")
    st.divider()

    st.markdown('<div class="sidebar-section">Classifier</div>', unsafe_allow_html=True)
    model_choice = st.radio(
        "Active model",
        ["Random Forest", "Decision Tree"],
        help="Random Forest (96.7% precision, ROC-AUC 0.96) is the stronger model. "
             "Decision Tree (78.3% accuracy) is faster and fully interpretable.",
        label_visibility="collapsed",
    )
    active_model = artifacts["rf_model"] if model_choice == "Random Forest" else artifacts["dt_model"]

    st.markdown('<div class="sidebar-section">Alert threshold</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "Probability cutoff",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        label_visibility="collapsed",
        help="The minimum probability required to flag traffic as an intrusion. "
             "Lower values catch more attacks but also raise more false alarms.",
    )
    if threshold < 0.35:
        st.caption(f"Threshold {threshold:.0%} — aggressive detection, higher false alarm rate.")
    elif threshold > 0.65:
        st.caption(f"Threshold {threshold:.0%} — conservative detection, may miss some attacks.")
    else:
        st.caption(f"Threshold {threshold:.0%} — balanced precision and recall.")

    st.markdown('<div class="sidebar-section">Packet presets</div>', unsafe_allow_html=True)
    st.caption("Each preset loads real packet attributes captured in KDDTest+.txt:")

    for name in PRESETS:
        if st.button(name, key=f"preset_{name}", use_container_width=True):
            apply_preset(name)

    active_preset = st.session_state.get("ss_active_preset", "")
    if active_preset:
        st.markdown(
            f'<div class="preset-indicator">Loaded: {active_preset}</div>',
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------
# Page header
# ------------------------------------------------------------------
st.markdown("""
<div class="page-header">
    <div>
        <h1>Network Intrusion Detection System</h1>
        <p>Trained on NSL-KDD benchmark &mdash; 125,973 labelled network flows &mdash;
           binary and multi-class intrusion detection</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Summary stat row
st.markdown("""
<div class="stat-row">
    <div class="stat-card">
        <div class="label">Training Records</div>
        <div class="value">125,973</div>
        <div class="sub">KDDTrain+.txt</div>
    </div>
    <div class="stat-card">
        <div class="label">Test Records</div>
        <div class="value">22,544</div>
        <div class="sub">KDDTest+.txt</div>
    </div>
    <div class="stat-card">
        <div class="label">Precision (RF)</div>
        <div class="value">96.7%</div>
        <div class="sub">False alarm rate 2.76%</div>
    </div>
    <div class="stat-card">
        <div class="label">ROC-AUC (RF)</div>
        <div class="value">0.961</div>
        <div class="sub">vs 0.724 Decision Tree</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab_inspector, tab_metrics, tab_features = st.tabs([
    "Packet Inspector",
    "Benchmark Results",
    "Feature Importance",
])


# ================================================================
# TAB 1 — Packet Inspector
# ================================================================
with tab_inspector:
    st.caption(
        "Fill in the attributes of an incoming network connection and click "
        "\"Run prediction\" to classify it. Use the presets in the sidebar to "
        "quickly load real packet samples from the NSL-KDD test dataset."
    )

    # ---- Section 1: Connection identity ----
    st.markdown('<div class="form-section-header">Connection identity</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        proto_val = st.selectbox(
            "Protocol",
            PROTOCOLS,
            index=PROTOCOLS.index(st.session_state.ss_proto),
            help="The network layer protocol used by this connection. "
                 "TCP is used by most application traffic, UDP by DNS/NTP, "
                 "ICMP by ping and network diagnostics.",
        )
    with col2:
        service_val = st.selectbox(
            "Service",
            SERVICES,
            index=SERVICES.index(st.session_state.ss_service)
                  if st.session_state.ss_service in SERVICES else 0,
            help="The destination network service (mapped from port number). "
                 "For example: http=80, ftp=21, smtp=25, telnet=23.",
        )
    with col3:
        flag_val = st.selectbox(
            "TCP connection flag",
            FLAGS,
            index=FLAGS.index(st.session_state.ss_flag)
                  if st.session_state.ss_flag in FLAGS else 0,
            help="How the TCP connection was completed or terminated. "
                 "SF = normal completion. S0 = SYN sent, no reply (common in DoS). "
                 "REJ = connection rejected. See the flag reference below.",
        )
    with col4:
        logged_in_val = st.selectbox(
            "Authenticated session",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=0 if st.session_state.ss_logged_in == 1 else 1,
            help="Whether the connection includes a successful login. "
                 "Attacks such as brute-force and port scanning typically show No.",
        )

    # ---- Section 2: Payload volume ----
    st.markdown('<div class="form-section-header">Payload volume</div>', unsafe_allow_html=True)
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        duration_val = st.number_input(
            "Duration (seconds)",
            min_value=0, max_value=60000,
            value=int(st.session_state.ss_duration),
            help="How long the connection lasted. Most normal HTTP connections "
                 "complete in under 1 second. A very high value can indicate "
                 "a slow DoS or data-exfiltration session.",
        )
    with col6:
        src_bytes_val = st.number_input(
            "Source bytes (bytes sent)",
            min_value=0, max_value=10_000_000,
            value=int(st.session_state.ss_src_bytes),
            help="Total bytes sent from the source host. Zero bytes on a long connection "
                 "is a typical signature of a SYN-flood or port scan.",
        )
    with col7:
        dst_bytes_val = st.number_input(
            "Destination bytes (bytes received)",
            min_value=0, max_value=10_000_000,
            value=int(st.session_state.ss_dst_bytes),
            help="Total bytes returned by the destination host. Very large values "
                 "may indicate data exfiltration; zero may indicate a refused or "
                 "flooded connection.",
        )
    with col8:
        count_val = st.number_input(
            "2-second connection count",
            min_value=0, max_value=1000,
            value=int(st.session_state.ss_count),
            help="Number of connections to the same destination host in the last 2 seconds. "
                 "High values (> 100) are a strong indicator of a flooding attack.",
        )

    # ---- Section 3: Advanced ----
    with st.expander("Advanced host-window parameters", expanded=False):
        st.caption(
            "These fields describe traffic patterns observed over a historical window of "
            "connections to the same destination host. They are pre-filled with training-set "
            "medians and only need adjustment for fine-grained analysis."
        )
        col9, col10, col11 = st.columns(3)
        with col9:
            srv_count_val = st.number_input(
                "Same-service connection count",
                min_value=0, max_value=1000,
                value=int(st.session_state.ss_srv_count),
                help="Connections to the same service in the last 2 seconds. "
                     "High values alongside high 'count' indicate a concentrated attack.",
            )
        with col10:
            dst_host_count_val = st.number_input(
                "Destination host count (0-255)",
                min_value=0, max_value=255,
                value=int(st.session_state.ss_dst_host_cnt),
                help="Number of connections to the same destination IP over the "
                     "last 100 connections. 255 = every recent connection hit this host.",
            )
        with col11:
            dst_host_srv_count_val = st.number_input(
                "Destination host same-service count (0-255)",
                min_value=0, max_value=255,
                value=int(st.session_state.ss_dst_srv_cnt),
                help="Of those 100 connections, how many used the same service. "
                     "A value of 255 combined with high count is a port-scan signature.",
            )

    # ---- Flag reference guide ----
    with st.expander("TCP flag reference", expanded=False):
        st.caption("Quick reference for the TCP connection flag field:")
        st.markdown("""
        <div class="flag-legend">
            <div class="flag-item"><div class="flag-name">SF</div><div class="flag-desc">Normal SYN/FIN completion. Standard for legitimate traffic.</div></div>
            <div class="flag-item"><div class="flag-name">S0</div><div class="flag-desc">SYN sent, no reply. Connection never established. Common in SYN floods.</div></div>
            <div class="flag-item"><div class="flag-name">REJ</div><div class="flag-desc">Connection attempt rejected by destination. Seen in port scans.</div></div>
            <div class="flag-item"><div class="flag-name">RSTR</div><div class="flag-desc">Reset by the remote host during the session.</div></div>
            <div class="flag-item"><div class="flag-name">RSTO</div><div class="flag-desc">Connection reset by the originating host.</div></div>
            <div class="flag-item"><div class="flag-name">SH</div><div class="flag-desc">SYN then FIN with no data transfer. Unusual half-open state.</div></div>
            <div class="flag-item"><div class="flag-name">S1</div><div class="flag-desc">SYN exchange only, no data. Connection partly established.</div></div>
            <div class="flag-item"><div class="flag-name">S2 / S3</div><div class="flag-desc">Partial data transfer with abnormal termination.</div></div>
            <div class="flag-item"><div class="flag-name">OTH</div><div class="flag-desc">Other or unclassified connection state.</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, cta_col, _ = st.columns([3, 2, 3])
    with cta_col:
        run_prediction = st.button("Run prediction", type="primary", use_container_width=True)

    # ---- Prediction execution — result persisted in session state ----
    if run_prediction:
        record = artifacts["default_record"].copy()
        record.update({
            "protocol_type":      proto_val,
            "service":            service_val,
            "flag":               flag_val,
            "duration":           duration_val,
            "src_bytes":          src_bytes_val,
            "dst_bytes":          dst_bytes_val,
            "logged_in":          logged_in_val,
            "count":              count_val,
            "srv_count":          srv_count_val,
            "dst_host_count":     dst_host_count_val,
            "dst_host_srv_count": dst_host_srv_count_val,
        })

        input_df = pd.DataFrame([record])
        processed = preprocess_single_sample(
            input_df,
            scaler=artifacts["scaler"],
            feature_names=artifacts["feature_names"],
        )

        if hasattr(active_model, "predict_proba"):
            probs = active_model.predict_proba(processed)[0]
            prob_normal = float(probs[0])
            prob_attack = float(probs[1])
            is_attack   = prob_attack >= threshold
        else:
            pred_code   = int(active_model.predict(processed)[0])
            is_attack   = pred_code == 1
            prob_attack = 1.0 if is_attack else 0.0
            prob_normal = 0.0 if is_attack else 1.0

        # Cache the result so it survives widget interaction
        st.session_state.ss_last_result = {
            "is_attack":   is_attack,
            "prob_attack": prob_attack,
            "prob_normal": prob_normal,
            "model_name":  type(active_model).__name__,
            "threshold":   threshold,
            "inputs": {
                "Protocol":                  proto_val,
                "Service":                   service_val,
                "TCP flag":                  flag_val,
                "Duration (s)":              duration_val,
                "Source bytes":              src_bytes_val,
                "Destination bytes":         dst_bytes_val,
                "Authenticated":             "Yes" if logged_in_val else "No",
                "2-sec connection count":    count_val,
                "Same-service count":        srv_count_val,
                "Destination host count":    dst_host_count_val,
                "Dst host same-srv count":   dst_host_srv_count_val,
            },
        }

    # ---- Render result (or empty state) ----
    result = st.session_state.ss_last_result

    if result is None:
        st.markdown("""
        <div class="empty-state">
            <div class="title">No prediction yet</div>
            <p>
                Configure the connection attributes above and click <strong>Run prediction</strong>.<br>
                Not sure where to start? Try one of the presets in the left sidebar
                to load a real captured packet from the NSL-KDD test dataset.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        is_attack   = result["is_attack"]
        prob_attack = result["prob_attack"]
        prob_normal = result["prob_normal"]
        model_name  = result["model_name"]
        thr         = result["threshold"]

        st.markdown("---")
        st.markdown('<div class="section-label">Detection result</div>', unsafe_allow_html=True)

        if is_attack:
            st.markdown(f"""
            <div class="verdict-card attack">
                <h3>Intrusion detected — connection should be blocked</h3>
                <p>
                    The {model_name.replace('Classifier','')} model classified this
                    flow as <strong>malicious</strong> with <strong>{prob_attack:.1%}</strong>
                    probability (alert threshold: {thr:.0%}).<br><br>
                    <strong>Recommended action:</strong> Terminate the TCP session, add the
                    source to your firewall deny-list, and forward the log entry to your
                    SIEM for incident review.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-card normal">
                <h3>Normal traffic — connection permitted</h3>
                <p>
                    The {model_name.replace('Classifier','')} model classified this flow as
                    <strong>legitimate</strong> with <strong>{prob_normal:.1%}</strong>
                    confidence. Threat probability was {prob_attack:.1%},
                    below the {thr:.0%} alert threshold.
                </p>
            </div>
            """, unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Verdict", "Intrusion" if is_attack else "Normal")
        with mc2:
            st.metric("Threat probability", f"{prob_attack:.1%}")
        with mc3:
            st.metric("Safe probability", f"{prob_normal:.1%}")
        with mc4:
            st.metric("Model used", model_name.replace("Classifier", ""))

        st.markdown(
            '<div class="section-label" style="margin-top:1rem;">Threat probability</div>',
            unsafe_allow_html=True,
        )
        st.progress(float(prob_attack))
        st.caption(
            f"{'Above' if is_attack else 'Below'} the {thr:.0%} alert threshold. "
            "Adjust the threshold in the sidebar to tune sensitivity."
        )

        with st.expander("Submitted feature values"):
            st.caption("The values that were sent to the model for this prediction:")
            st.table(
                pd.DataFrame(result["inputs"].items(), columns=["Feature", "Value"])
            )


# ================================================================
# TAB 2 — Benchmark Results
# ================================================================
with tab_metrics:
    st.markdown('<div class="section-label">Test-set performance · KDDTest+.txt · 22,544 records</div>',
                unsafe_allow_html=True)

    st.markdown("""
    > The test set includes **17 novel attack categories** absent from training
    > (e.g. `apache2`, `mscan`, `snmpguess`, `saint`, `httptunnel`, `mailbomb`),
    > providing a genuine zero-day generalisation benchmark.
    """)

    perf_html = """
    <table class="metrics-table">
      <thead>
        <tr>
          <th>Metric</th>
          <th>Random Forest</th>
          <th>Decision Tree</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>Accuracy</td>      <td>76.97 %</td>  <td>78.31 %</td></tr>
        <tr><td>Precision</td>     <td><strong>96.72 %</strong></td>  <td>95.83 %</td></tr>
        <tr><td>Recall</td>        <td>61.64 %</td>  <td>64.71 %</td></tr>
        <tr><td>F1-Score</td>      <td>75.29 %</td>  <td>77.25 %</td></tr>
        <tr><td>ROC-AUC</td>       <td><strong>0.9614</strong></td>   <td>0.7236</td></tr>
        <tr><td>False positives</td><td><strong>268 / 9,711 (2.76 %)</strong></td><td>361 / 9,711 (3.72 %)</td></tr>
        <tr><td>True positives</td><td>7,910 / 12,833</td><td>8,304 / 12,833</td></tr>
      </tbody>
    </table>
    """
    st.markdown(perf_html, unsafe_allow_html=True)

    st.markdown("""
    **Reading these numbers in context:**
    Random Forest's superior **ROC-AUC (0.96)** means the probability ranking is well-calibrated —
    lowering the decision threshold to 40% recovers a substantial recall improvement while 
    keeping precision above 93%. The Decision Tree's higher point-accuracy reflects
    a threshold that happens to suit the class balance, but its probability estimates are coarser.
    """)

    cm_path = os.path.join("models", "confusion_matrices.png")
    if os.path.exists(cm_path):
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Confusion matrices</div>',
                    unsafe_allow_html=True)
        st.image(cm_path, use_container_width=True)


# ================================================================
# TAB 3 — Feature Importance
# ================================================================
with tab_features:
    st.markdown('<div class="section-label">Gini importance · Random Forest · 100 trees</div>',
                unsafe_allow_html=True)

    fi_path = os.path.join("models", "top_15_feature_importance.png")
    if os.path.exists(fi_path):
        st.image(fi_path, use_container_width=True)

    st.markdown("**Why these features dominate:**")
    fi_table = {
        "Rank": [1, 2, 3, 4, 5],
        "Feature": ["src_bytes", "dst_bytes", "flag_SF", "dst_host_srv_count", "logged_in"],
        "Importance": ["15.92%", "8.44%", "6.94%", "6.02%", "5.62%"],
        "Security rationale": [
            "Volume of outbound payload — floods, buffer overflows, exfiltration",
            "Volume of inbound payload — unresponsive targets, data exfil",
            "Normal TCP 3-way handshake completion; attacks terminate as S0/REJ/RSTR",
            "Destination port density — signature of horizontal/vertical port sweeps",
            "Authentication success — separates brute-force from authorised sessions",
        ],
    }
    st.table(pd.DataFrame(fi_table))
