from __future__ import annotations

import io
import ipaddress
import json
import logging
import os
import random
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure local project modules are discoverable regardless of working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from step1_load_data import COLUMNS, resolve_dataset_path
from step2_preprocess import ATTACK_MAPPING

try:
    from scapy.all import IP, TCP, UDP, ICMP, conf, get_if_list, sniff
    SCAPY_AVAILABLE = True
except Exception as _scapy_err:
    SCAPY_AVAILABLE = False
    logging.warning("Scapy packet capture unavailable: %s", _scapy_err)

try:
    from shieldnet import (
        Incident,
        MitigationManager,
        RuntimeState,
        build_session_report,
        report_pdf,
        run_local_simulation,
        write_incidents_csv,
        write_incidents_json,
    )
    SHIELDNET_AVAILABLE = True
except Exception as _exc:
    SHIELDNET_AVAILABLE = False
    logging.warning("ShieldNet hardware capture module unavailable: %s", _exc)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Cyber SOC Aesthetics
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ShieldNet AI | Security Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Modern Cyber SOC Theme Styles */
:root {
    --soc-bg: #0b0f19;
    --soc-surface: #111827;
    --soc-card: #141c2e;
    --soc-card-border: #1e293b;
    --soc-border-active: #38bdf8;
    --soc-cyan: #06b6d4;
    --soc-blue: #38bdf8;
    --soc-emerald: #10b981;
    --soc-amber: #f59e0b;
    --soc-rose: #f43f5e;
    --soc-red: #ef4444;
    --soc-text: #f8fafc;
    --soc-muted: #94a3b8;
}

.main .block-container {
    max-width: 1540px;
    padding-top: 1.1rem;
    padding-bottom: 2.5rem;
}

/* Header Banner */
.soc-banner {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(15, 23, 42, 0.88) 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.soc-banner h1 {
    margin: 0;
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 12px;
}

.soc-banner p {
    margin: 4px 0 0 0;
    color: #94a3b8;
    font-size: 0.92rem;
}

/* Metric Cards */
.soc-card {
    background: #141c2e;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: transform 0.2s ease, border-color 0.2s ease;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}
.soc-card:hover {
    border-color: rgba(56, 189, 248, 0.5);
    transform: translateY(-2px);
}
.soc-card-title {
    font-size: 0.76rem;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 6px;
}
.soc-card-value {
    font-size: 2.1rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    line-height: 1.1;
    color: #f8fafc;
}
.soc-card-delta {
    font-size: 0.8rem;
    margin-top: 6px;
    font-weight: 500;
}

/* Status Badges */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 5px 13px;
    border-radius: 9999px;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-secure {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.4);
}
.badge-caution {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.4);
}
.badge-critical {
    background: rgba(239, 68, 68, 0.2);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.5);
    box-shadow: 0 0 12px rgba(239, 68, 68, 0.35);
}

/* Pulsing Status Dot */
.pulse-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 8px;
    animation: soc-pulse 1.8s infinite;
}
.pulse-green { background-color: #10b981; box-shadow: 0 0 8px #10b981; }
.pulse-yellow { background-color: #f59e0b; box-shadow: 0 0 8px #f59e0b; }
.pulse-red { background-color: #ef4444; box-shadow: 0 0 10px #ef4444; }

@keyframes soc-pulse {
    0% { transform: scale(0.9); opacity: 0.75; }
    50% { transform: scale(1.2); opacity: 1; }
    100% { transform: scale(0.9); opacity: 0.75; }
}

/* Anomaly Callout Box */
.anomaly-callout {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-left: 5px solid #ef4444;
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 12px;
    margin-bottom: 14px;
}
.anomaly-callout.clean {
    border-color: rgba(16, 185, 129, 0.35);
    border-left: 5px solid #10b981;
}

.mono {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# IP Geolocation Engine (Suspicious IP Origin Resolver)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def resolve_ip_geo(ip_str: str) -> dict[str, Any]:
    """
    Resolves geographic location (Country, City, Coordinates, Flag, ISP)
    for suspicious source IPs with automatic caching and offline fallbacks.
    """
    clean_ip = ip_str.strip().split(" ")[0].split(":")[0]

    # Check for private, loopback, or reserved network addresses
    try:
        ip_obj = ipaddress.ip_address(clean_ip)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved or ip_obj.is_link_local:
            return {
                "ip": clean_ip,
                "country": "Internal Network",
                "country_code": "LAN",
                "city": "Local Subnet (RFC 1918)",
                "lat": 20.0,
                "lon": 0.0,
                "isp": "Local Enterprise Intranet",
                "org": "Internal Infrastructure",
                "flag": "🏠",
                "is_local": True,
            }
    except Exception:
        pass

    # Online Geolocation lookup via ip-api.com (Fast, free, keyless JSON endpoint)
    try:
        url = f"http://ip-api.com/json/{clean_ip}?fields=status,country,countryCode,regionName,city,lat,lon,isp,org,as,query"
        req = urllib.request.Request(url, headers={"User-Agent": "ShieldNet-SOC/2.0"})
        with urllib.request.urlopen(req, timeout=1.8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("status") == "success":
                cc = data.get("countryCode", "UN")
                flag_emoji = "".join(chr(127397 + ord(c)) for c in cc.upper()) if len(cc) == 2 else "🌐"
                return {
                    "ip": clean_ip,
                    "country": data.get("country", "Unknown"),
                    "country_code": cc,
                    "city": data.get("city", "Unknown City"),
                    "lat": float(data.get("lat", 0.0)),
                    "lon": float(data.get("lon", 0.0)),
                    "isp": data.get("isp", "Unknown ISP"),
                    "org": data.get("org", data.get("as", "Autonomous System")),
                    "flag": flag_emoji,
                    "is_local": False,
                }
    except Exception:
        pass

    # Deterministic fallback for simulated or offline IPs so map never breaks
    hash_val = abs(hash(clean_ip)) % 1000
    pseudo_locations = [
        ("United States", "US", "Ashburn", 39.0438, -77.4874, "DigitalOcean C2 Hosting", "🇺🇸"),
        ("Germany", "DE", "Frankfurt", 50.1109, 8.6821, "Hetzner Tor Exit Relay", "🇩🇪"),
        ("Netherlands", "NL", "Amsterdam", 52.3676, 4.9041, "Bulletproof Data Center", "🇳🇱"),
        ("Russia", "RU", "Moscow", 55.7558, 37.6173, "Rostelecom Autonomous Net", "🇷🇺"),
        ("China", "CN", "Shanghai", 31.2304, 121.4737, "China Telecom Backbone", "🇨🇳"),
        ("India", "IN", "Mumbai", 19.0760, 72.8777, "Bharti Enterprise Gateway", "🇮🇳"),
        ("Brazil", "BR", "São Paulo", -23.5505, -46.6333, "Claro Brasil Hosting", "🇧🇷"),
        ("United Kingdom", "GB", "London", 51.5074, -0.1278, "BT Public Subnet", "🇬🇧"),
        ("Singapore", "SG", "Singapore", 1.3521, 103.8198, "Zenlayer Cloud Node", "🇸🇬"),
    ]
    loc = pseudo_locations[hash_val % len(pseudo_locations)]
    return {
        "ip": clean_ip,
        "country": loc[0],
        "country_code": loc[1],
        "city": loc[2],
        "lat": loc[3],
        "lon": loc[4],
        "isp": loc[5],
        "org": "Threat Telemetry Network",
        "flag": loc[6],
        "is_local": False,
    }


# -----------------------------------------------------------------------------
# Cached Artifact Loaders & Preprocessing Pipeline
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Security Intelligence Engine...")
def load_nids_models(model_dir: str = "models"):
    """Loads trained NIDS models, StandardScaler, and 122-feature schema."""
    adv_path = os.path.join(model_dir, "advanced_model.joblib")
    base_path = os.path.join(model_dir, "baseline_model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    features_path = os.path.join(model_dir, "feature_names.joblib")

    for path, name in [
        (adv_path, "advanced_model.joblib"),
        (scaler_path, "scaler.joblib"),
        (features_path, "feature_names.joblib"),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing required artifact '{name}' in '{model_dir}/'.")

    adv_model = joblib.load(adv_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)

    base_model = None
    if os.path.isfile(base_path):
        try:
            base_model = joblib.load(base_path)
        except Exception:
            base_model = None

    return {
        "advanced_model": adv_model,
        "baseline_model": base_model,
        "scaler": scaler,
        "feature_names": feature_names,
    }


@st.cache_data(show_spinner="Indexing Threat Intelligence Baseline...")
def load_telemetry_dataset():
    """Loads KDDTest+.txt for simulation with synthetic fallback."""
    try:
        path = resolve_dataset_path("KDDTest+.txt")
        df = pd.read_csv(path, header=None, names=COLUMNS)
        return df
    except Exception as exc:
        logging.warning("Could not load KDDTest+.txt (%s). Generating synthetic fallback flows.", exc)
        rows = []
        for i in range(150):
            is_atk = i % 3 != 0
            rows.append({
                "duration": 0,
                "protocol_type": random.choice(["tcp", "udp", "icmp"]),
                "service": random.choice(["http", "private", "smtp", "domain_u", "ftp"]),
                "flag": "S0" if is_atk and (i % 2 == 0) else "SF",
                "src_bytes": 0 if is_atk else random.randint(150, 4500),
                "dst_bytes": 0 if is_atk else random.randint(200, 8000),
                "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
                "num_failed_logins": 1 if is_atk and (i % 5 == 0) else 0,
                "logged_in": 0 if is_atk else 1,
                "num_compromised": 0, "root_shell": 0, "su_attempted": 0,
                "num_root": 0, "num_file_creations": 0, "num_shells": 0,
                "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0,
                "is_guest_login": 0,
                "count": random.randint(80, 450) if is_atk else random.randint(1, 15),
                "srv_count": random.randint(50, 300) if is_atk else random.randint(1, 12),
                "serror_rate": 1.0 if (is_atk and i % 2 == 0) else 0.0,
                "srv_serror_rate": 1.0 if (is_atk and i % 2 == 0) else 0.0,
                "rerror_rate": 0.5 if (is_atk and i % 4 == 0) else 0.0,
                "srv_rerror_rate": 0.5 if (is_atk and i % 4 == 0) else 0.0,
                "same_srv_rate": 0.1 if (is_atk and i % 3 == 0) else 1.0,
                "diff_srv_rate": 0.7 if (is_atk and i % 3 == 0) else 0.0,
                "srv_diff_host_rate": 0.0,
                "dst_host_count": random.randint(150, 255) if is_atk else random.randint(5, 50),
                "dst_host_srv_count": random.randint(5, 40) if is_atk else random.randint(80, 255),
                "dst_host_same_srv_rate": 0.08 if is_atk else 0.95,
                "dst_host_diff_srv_rate": 0.65 if is_atk else 0.02,
                "dst_host_same_src_port_rate": 0.01,
                "dst_host_srv_diff_host_rate": 0.0,
                "dst_host_serror_rate": 1.0 if (is_atk and i % 2 == 0) else 0.0,
                "dst_host_srv_serror_rate": 1.0 if (is_atk and i % 2 == 0) else 0.0,
                "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
                "attack": "neptune" if is_atk else "normal",
                "difficulty_level": 21,
            })
        return pd.DataFrame(rows)


def preprocess_traffic_samples(sample_series_or_df, scaler, feature_names):
    """Aligns any traffic row strictly to the 122-feature training schema."""
    if isinstance(sample_series_or_df, pd.Series):
        sample_df = pd.DataFrame([sample_series_or_df])
    else:
        sample_df = sample_series_or_df.copy()

    drop_cols = [c for c in ["attack", "difficulty_level"] if c in sample_df.columns]
    features_df = sample_df.drop(columns=drop_cols)

    cat_cols = ["protocol_type", "service", "flag"]
    num_cols = getattr(scaler, "feature_names_in_", None)
    if num_cols is None:
        num_cols = [c for c in features_df.columns if c not in cat_cols]

    encoded_df = pd.get_dummies(features_df, columns=cat_cols, dtype=float)
    aligned_df = encoded_df.reindex(columns=feature_names, fill_value=0.0)
    aligned_df[num_cols] = scaler.transform(aligned_df[num_cols])

    return aligned_df


# -----------------------------------------------------------------------------
# Anomaly Parameter Breakdown & Threat Classification
# -----------------------------------------------------------------------------
def analyze_anomaly_parameters(raw_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluates raw connection attributes to identify triggered security parameters."""
    triggered = []

    count = raw_dict.get("count", 0)
    if count >= 60:
        triggered.append({
            "param": "count",
            "name": "Connection Surge Volume",
            "value": f"{count:,} conn/2s",
            "benchmark": "< 15 conn/2s",
            "severity": "CRITICAL" if count > 200 else "HIGH",
            "desc": "Extreme connection frequency directed to target host within 2s window.",
        })

    serror = raw_dict.get("serror_rate", 0.0)
    dst_serror = raw_dict.get("dst_host_serror_rate", 0.0)
    if serror >= 0.35 or dst_serror >= 0.40:
        triggered.append({
            "param": "serror_rate",
            "name": "SYN Handshake Error Rate",
            "value": f"{max(serror, dst_serror) * 100:.1f}%",
            "benchmark": "< 5.0%",
            "severity": "CRITICAL",
            "desc": "Pervasive half-open SYN handshakes without response acknowledgment.",
        })

    flag = str(raw_dict.get("flag", "SF")).upper()
    if flag in ["S0", "REJ", "RSTO", "RSTOS0", "SH", "RSTR"]:
        triggered.append({
            "param": "flag",
            "name": "Anomalous TCP State Flag",
            "value": flag,
            "benchmark": "SF (Established)",
            "severity": "HIGH",
            "desc": f"Connection terminated in abnormal status '{flag}' indicating reset or dropped handshake.",
        })

    diff_srv = raw_dict.get("diff_srv_rate", 0.0)
    dst_diff_srv = raw_dict.get("dst_host_diff_srv_rate", 0.0)
    if diff_srv >= 0.35 or dst_diff_srv >= 0.35:
        triggered.append({
            "param": "diff_srv_rate",
            "name": "Service Port Dispersal",
            "value": f"{max(diff_srv, dst_diff_srv) * 100:.1f}%",
            "benchmark": "< 8.0%",
            "severity": "HIGH",
            "desc": "Connections traversing multiple distinct services, characteristic of port scanning.",
        })

    rerror = raw_dict.get("rerror_rate", 0.0)
    if rerror >= 0.30:
        triggered.append({
            "param": "rerror_rate",
            "name": "Connection Rejection Rate",
            "value": f"{rerror * 100:.1f}%",
            "benchmark": "< 3.0%",
            "severity": "MEDIUM",
            "desc": "Target ports rejecting connection requests (RST reply packets).",
        })

    src_bytes = raw_dict.get("src_bytes", 0)
    dst_bytes = raw_dict.get("dst_bytes", 0)
    if src_bytes == 0 and dst_bytes == 0 and count >= 15:
        triggered.append({
            "param": "payload",
            "name": "Zero-Byte Payload Exchange",
            "value": "0 B / 0 B",
            "benchmark": "> 0 B",
            "severity": "MEDIUM",
            "desc": "Asymmetric zero-payload packets sent purely for network exhaustion or reconnaissance.",
        })

    failed_logins = raw_dict.get("num_failed_logins", 0)
    if failed_logins >= 1:
        triggered.append({
            "param": "num_failed_logins",
            "name": "Failed Authentication Attempts",
            "value": f"{failed_logins} failed",
            "benchmark": "0",
            "severity": "HIGH",
            "desc": "Credential validation failure indicating dictionary or brute-force assault.",
        })

    root_shell = raw_dict.get("root_shell", 0)
    su_attempt = raw_dict.get("su_attempted", 0)
    if root_shell == 1 or su_attempt == 1:
        triggered.append({
            "param": "root_shell",
            "name": "Root Privilege Execution",
            "value": "ROOT / SU INVOKED",
            "benchmark": "0",
            "severity": "CRITICAL",
            "desc": "Host execution attempting to spawn superuser or administrative shell context.",
        })

    hot = raw_dict.get("hot", 0)
    if hot >= 2:
        triggered.append({
            "param": "hot",
            "name": "Sensitive System File Access",
            "value": f"{hot} hot indicators",
            "benchmark": "0",
            "severity": "HIGH",
            "desc": "Access or execution touching privileged directory paths or scripts.",
        })

    dst_host_count = raw_dict.get("dst_host_count", 0)
    if dst_host_count >= 220:
        triggered.append({
            "param": "dst_host_count",
            "name": "Host Connection Saturation",
            "value": f"{dst_host_count}/255",
            "benchmark": "< 100",
            "severity": "LOW",
            "desc": "Target host historical window approaching maximum connection tracking capacity.",
        })

    return triggered


def classify_network_threat(raw_dict: dict[str, Any], ground_truth: str = "normal") -> dict[str, Any]:
    """Classifies attack category and prescribes actionable firewall mitigations."""
    category = "Normal"
    threat_title = "Normal Traffic Flow"
    severity = "INFORMATIONAL"
    mitigation = "Permit Flow (Status: ALLOWED)"

    if ground_truth and ground_truth != "normal":
        family = ATTACK_MAPPING.get(ground_truth.lower(), "Zero-Day Anomaly")
        gt_clean = ground_truth.replace("_", " ").title()
        if family == "DoS":
            category = "DoS"
            threat_title = f"DoS / {gt_clean} Flood"
            severity = "CRITICAL"
            mitigation = "Drop Packet & Enforce Rate Limiting (5 req/s) on Subnet"
        elif family == "Probe":
            category = "Probe"
            threat_title = f"Reconnaissance Probe ({gt_clean})"
            severity = "HIGH"
            mitigation = "Blacklist Source IP via OS Firewall & Drop ICMP/SYN Probes"
        elif family == "R2L":
            category = "R2L"
            threat_title = f"Brute-Force & R2L Exploit ({gt_clean})"
            severity = "HIGH"
            mitigation = "Enforce IP Jail (Fail2ban), Revoke Token & Reset Session"
        elif family == "U2R":
            category = "U2R"
            threat_title = f"Privilege Escalation U2R ({gt_clean})"
            severity = "CRITICAL"
            mitigation = "Isolate Host Subnet, Terminate Rogue Process & Dump Memory"
        else:
            category = "Zero-Day"
            threat_title = f"Novel Zero-Day Threat ({gt_clean})"
            severity = "CRITICAL"
            mitigation = "Quarantine Source Flow & Trigger Deep Packet Inspection"
        return {"category": category, "threat_title": threat_title, "severity": severity, "mitigation": mitigation}

    # Metadata Heuristics Classification
    root_shell = raw_dict.get("root_shell", 0)
    su_attempt = raw_dict.get("su_attempted", 0)
    failed_logins = raw_dict.get("num_failed_logins", 0)
    diff_srv = max(raw_dict.get("diff_srv_rate", 0.0), raw_dict.get("dst_host_diff_srv_rate", 0.0))
    rerror = raw_dict.get("rerror_rate", 0.0)
    serror = max(raw_dict.get("serror_rate", 0.0), raw_dict.get("dst_host_serror_rate", 0.0))
    count = raw_dict.get("count", 0)
    flag = str(raw_dict.get("flag", "SF")).upper()

    if root_shell == 1 or su_attempt == 1:
        category = "U2R"
        threat_title = "Privilege Escalation / Root Exploit"
        severity = "CRITICAL"
        mitigation = "Isolate Host Subnet, Terminate Rogue Process & Dump Memory"
    elif failed_logins >= 1 or (raw_dict.get("service") in ["ftp", "telnet", "ssh"] and raw_dict.get("logged_in") == 0 and count >= 5):
        category = "R2L"
        threat_title = "Authentication Brute-Force / R2L Attack"
        severity = "HIGH"
        mitigation = "Enforce IP Jail (Fail2ban), Revoke Token & Reset Session"
    elif diff_srv >= 0.35 or rerror >= 0.30:
        category = "Probe"
        threat_title = "Reconnaissance / Port Scanning Probe"
        severity = "HIGH"
        mitigation = "Blacklist Source IP via OS Firewall & Drop ICMP/SYN Probes"
    elif serror >= 0.35 or count >= 60 or flag in ["S0", "REJ"]:
        category = "DoS"
        threat_title = "DoS / Neptune SYN Flood Attack"
        severity = "CRITICAL"
        mitigation = "Drop Packet & Enforce Rate Limiting (5 req/s) on Subnet"
    else:
        category = "Zero-Day"
        threat_title = "Unclassified Statistical Anomaly"
        severity = "MEDIUM"
        mitigation = "Quarantine Flow & Dispatch DPI Inspection"

    return {"category": category, "threat_title": threat_title, "severity": severity, "mitigation": mitigation}


def port_to_service(port: int) -> str:
    """Maps transport layer port numbers to NSL-KDD service schema names."""
    port_map = {
        80: "http", 443: "http_443", 53: "domain_u", 21: "ftp", 20: "ftp_data",
        22: "ssh", 23: "telnet", 25: "smtp", 110: "pop_3", 143: "imap4",
        123: "ntp_u", 445: "netbios_ssn", 137: "netbios_ns", 138: "netbios_dgm",
        139: "netbios_ssn", 3306: "sqlattack", 5432: "sqlattack", 8080: "http",
        6667: "IRC", 6000: "X11", 79: "finger", 7: "echo",
    }
    return port_map.get(port, "private")


def synthesize_profile(raw_dict: dict[str, Any], idx: int, is_threat: bool) -> tuple[str, str]:
    """Generates realistic SOC source and destination profiles."""
    service = raw_dict.get("service", "http")
    port_map = {"http": 80, "smtp": 25, "domain_u": 53, "ftp": 21, "telnet": 23, "private": 8080}
    dport = port_map.get(service, 443)

    if not is_threat:
        src = f"192.168.1.{15 + (idx % 220)}"
        dst = f"10.0.1.10:{dport} [Web-Cluster-01]"
    else:
        flag = raw_dict.get("flag", "SF")
        if flag == "S0" or raw_dict.get("count", 0) > 80:
            src = f"185.220.101.{(idx % 240) + 1}"
            dst = f"10.0.1.10:{dport} [Web-DMZ]"
        elif raw_dict.get("diff_srv_rate", 0) > 0.3:
            src = f"194.26.29.{(idx % 240) + 1}"
            dst = f"10.0.1.0/24 [Core-Subnet]"
        elif raw_dict.get("num_failed_logins", 0) > 0:
            src = f"212.102.40.{(idx % 240) + 1}"
            dst = f"10.0.1.25:{dport} [Auth-Gateway]"
        else:
            src = f"45.142.214.{(idx % 240) + 1}"
            dst = f"10.0.1.50:{dport} [Internal-App]"

    return src, dst


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def init_session_state():
    if "soc_telemetry" not in st.session_state:
        st.session_state.soc_telemetry = []
    if "soc_incidents" not in st.session_state:
        st.session_state.soc_incidents = []
    if "blocked_ips" not in st.session_state:
        st.session_state.blocked_ips = set()
    if "total_inspected" not in st.session_state:
        st.session_state.total_inspected = 0
    if "normal_count" not in st.session_state:
        st.session_state.normal_count = 0
    if "threat_count" not in st.session_state:
        st.session_state.threat_count = 0
    if "mitigated_count" not in st.session_state:
        st.session_state.mitigated_count = 0
    if "auto_stream" not in st.session_state:
        st.session_state.auto_stream = False
    if "traffic_source_mode" not in st.session_state:
        st.session_state.traffic_source_mode = "📡 Live Physical Network Sniffer"
    if "defense_stance" not in st.session_state:
        st.session_state.defense_stance = "🛡️ Balanced Protection (Recommended)"
    if "stream_scenario" not in st.session_state:
        st.session_state.stream_scenario = "Mixed NSL-KDD Traffic"
    if "latest_flow_analysis" not in st.session_state:
        st.session_state.latest_flow_analysis = None
    if "live_sniff_active" not in st.session_state:
        st.session_state.live_sniff_active = False
    if "live_packet_count" not in st.session_state:
        st.session_state.live_packet_count = 0
    if "shieldnet_runtime" not in st.session_state and SHIELDNET_AVAILABLE:
        st.session_state.shieldnet_runtime = RuntimeState()

init_session_state()


# -----------------------------------------------------------------------------
# Core Ingestion & Telemetry Processing
# -----------------------------------------------------------------------------
def ingest_telemetry_packet(raw_row: pd.Series, artifacts: dict[str, Any], sensitivity: float = 0.50, real_src: str | None = None, real_dst: str | None = None):
    """
    Ingests network traffic through the 122-feature pipeline, predicts anomalies,
    resolves geolocation for suspicious IPs, and logs actionable SOC incidents.
    """
    model = artifacts["advanced_model"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]

    aligned_vec = preprocess_traffic_samples(raw_row, scaler, feature_names)

    pred_code = int(model.predict(aligned_vec)[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(aligned_vec)[0]
        prob_normal = float(probs[0])
        prob_intrusion = float(probs[1])
    else:
        prob_intrusion = 1.0 if pred_code == 1 else 0.0
        prob_normal = 1.0 - prob_intrusion

    is_anomaly = (prob_intrusion >= sensitivity) or (pred_code == 1)
    anomaly_score = int(round(prob_intrusion * 100))

    raw_dict = raw_row.to_dict()
    ground_truth = str(raw_dict.get("attack", "normal"))
    idx = st.session_state.total_inspected + 1

    if real_src and real_dst:
        src_profile = real_src
        dst_profile = real_dst
        src_ip_clean = real_src.split(":")[0]
    else:
        src_profile, dst_profile = synthesize_profile(raw_dict, idx, is_anomaly)
        src_ip_clean = src_profile.split(" ")[0]

    is_blocked = src_ip_clean in st.session_state.blocked_ips
    triggered_params = analyze_anomaly_parameters(raw_dict) if is_anomaly else []
    threat_info = classify_network_threat(raw_dict, ground_truth if is_anomaly else "normal")

    # Geolocation lookup for suspicious or active IP
    geo = resolve_ip_geo(src_ip_clean)

    now_iso = datetime.now(timezone.utc).strftime("%H:%M:%S")

    flow_record = {
        "index": idx,
        "timestamp": now_iso,
        "source": src_profile,
        "source_ip": src_ip_clean,
        "destination": dst_profile,
        "protocol": str(raw_dict.get("protocol_type", "tcp")).upper(),
        "service": str(raw_dict.get("service", "http")),
        "flag": str(raw_dict.get("flag", "SF")),
        "bytes": f"{int(raw_dict.get('src_bytes', 0)):,} B",
        "pred_label": "MALICIOUS" if is_anomaly else "NORMAL",
        "is_threat": is_anomaly,
        "threat_score": anomaly_score,
        "confidence": max(prob_normal, prob_intrusion),
        "threat_title": threat_info["threat_title"] if is_anomaly else "Normal Traffic",
        "category": threat_info["category"] if is_anomaly else "Legitimate",
        "severity": threat_info["severity"] if is_anomaly else "INFORMATIONAL",
        "mitigation_status": "Blocked" if is_blocked else ("Detected" if is_anomaly else "Allowed"),
        "raw_attributes": raw_dict,
        "triggered_params": triggered_params,
        "recommended_mitigation": threat_info["mitigation"] if is_anomaly else "None",
        "geo": geo,
    }

    st.session_state.total_inspected += 1
    if is_anomaly:
        st.session_state.threat_count += 1
        incident_id = f"INC-{1000 + len(st.session_state.soc_incidents)}"
        incident = {
            "incident_id": incident_id,
            "timestamp": now_iso,
            "source_ip": src_ip_clean,
            "source_profile": f"{geo['flag']} {src_profile} ({geo['city']}, {geo['country_code']})",
            "country": geo["country"],
            "city": geo["city"],
            "country_flag": geo["flag"],
            "isp": geo["isp"],
            "lat": geo["lat"],
            "lon": geo["lon"],
            "destination": dst_profile,
            "protocol_service": f"{flow_record['protocol']} / {flow_record['service']}",
            "threat_type": threat_info["threat_title"],
            "category": threat_info["category"],
            "severity": threat_info["severity"],
            "confidence": f"{anomaly_score}%",
            "triggered_summary": ", ".join([p["name"] for p in triggered_params[:2]]) or "Statistical Deviation",
            "recommended_mitigation": threat_info["mitigation"],
            "mitigation_status": "Mitigated / Blocked" if is_blocked else "Pending Action",
            "raw_attributes": raw_dict,
            "triggered_params": triggered_params,
        }
        st.session_state.soc_incidents.insert(0, incident)
        st.session_state.soc_incidents = st.session_state.soc_incidents[:500]
    else:
        st.session_state.normal_count += 1

    st.session_state.soc_telemetry.insert(0, flow_record)
    st.session_state.soc_telemetry = st.session_state.soc_telemetry[:200]
    st.session_state.latest_flow_analysis = flow_record


def select_scenario_sample(test_df: pd.DataFrame, scenario: str) -> pd.Series:
    """Selects an appropriate sample row according to the simulation scenario."""
    if test_df.empty:
        raise ValueError("Telemetry dataset is empty.")

    if scenario == "Normal Workload Stream":
        subset = test_df[test_df["attack"] == "normal"]
        return subset.sample(1).iloc[0] if not subset.empty else test_df.sample(1).iloc[0]
    elif scenario == "DoS / Neptune SYN Flood":
        subset = test_df[test_df["attack"].isin(["neptune", "smurf", "pod", "teardrop", "back", "apache2", "processtable"])]
        return subset.sample(1).iloc[0] if not subset.empty else test_df.sample(1).iloc[0]
    elif scenario == "Reconnaissance / Port Scanning":
        subset = test_df[test_df["attack"].isin(["satan", "ipsweep", "portsweep", "nmap", "mscan", "saint"])]
        return subset.sample(1).iloc[0] if not subset.empty else test_df.sample(1).iloc[0]
    elif scenario == "Brute-Force & R2L Exploit":
        subset = test_df[test_df["attack"].isin(["guess_passwd", "ftp_write", "warezmaster", "warezclient", "snmpguess"])]
        return subset.sample(1).iloc[0] if not subset.empty else test_df.sample(1).iloc[0]
    elif scenario == "Zero-Day Attack Variant":
        novel_attacks = ["saint", "mscan", "apache2", "processtable", "mailbomb", "httptunnel"]
        subset = test_df[test_df["attack"].isin(novel_attacks)]
        return subset.sample(1).iloc[0] if not subset.empty else test_df.sample(1).iloc[0]
    else:
        return test_df.sample(1).iloc[0]


def process_real_network_batch(artifacts: dict[str, Any], sensitivity: float = 0.50):
    """
    Captures a real slice of live network packets using Scapy,
    groups them into flow windows, and runs them through the NIDS ML model.
    """
    if not SCAPY_AVAILABLE:
        return

    try:
        packets = sniff(timeout=1.2, count=35, store=True)
        if not packets:
            return

        st.session_state.live_packet_count += len(packets)
        flows = {}

        for p in packets:
            if IP in p:
                src = p[IP].src
                dst = p[IP].dst
                proto = "tcp" if TCP in p else ("udp" if UDP in p else "icmp")
                dport = p[TCP].dport if TCP in p else (p[UDP].dport if UDP in p else 80)
                size = len(p)
                is_syn = bool(p[TCP].flags & 0x02) if TCP in p else False
                is_ack = bool(p[TCP].flags & 0x10) if TCP in p else False
                is_rst = bool(p[TCP].flags & 0x04) if TCP in p else False

                if src not in flows:
                    flows[src] = {
                        "pkts": 0, "bytes": 0, "syn": 0, "ack": 0, "rst": 0,
                        "dports": set(), "proto": proto, "dst": dst, "dport": dport,
                    }
                f = flows[src]
                f["pkts"] += 1
                f["bytes"] += size
                if is_syn: f["syn"] += 1
                if is_ack: f["ack"] += 1
                if is_rst: f["rst"] += 1
                f["dports"].add(dport)

        for src, f in flows.items():
            serror = f["syn"] / f["pkts"] if f["pkts"] else 0.0
            rerror = f["rst"] / f["pkts"] if f["pkts"] else 0.0
            diff_srv = len(f["dports"]) / f["pkts"] if f["pkts"] else 0.0
            dominant_service = port_to_service(f["dport"])

            flag_label = "S0" if serror > 0.5 else ("REJ" if rerror > 0.5 else "SF")

            row_dict = {
                "duration": 0, "protocol_type": f["proto"], "service": dominant_service,
                "flag": flag_label, "src_bytes": f["bytes"], "dst_bytes": 0,
                "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
                "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
                "root_shell": 0, "su_attempted": 0, "num_root": 0,
                "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
                "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
                "count": f["pkts"], "srv_count": f["pkts"],
                "serror_rate": serror, "srv_serror_rate": serror,
                "rerror_rate": rerror, "srv_rerror_rate": rerror,
                "same_srv_rate": max(0.0, 1.0 - diff_srv),
                "diff_srv_rate": diff_srv, "srv_diff_host_rate": 0.0,
                "dst_host_count": min(255, f["pkts"] * 3),
                "dst_host_srv_count": min(255, f["pkts"]),
                "dst_host_same_srv_rate": max(0.0, 1.0 - diff_srv),
                "dst_host_diff_srv_rate": diff_srv,
                "dst_host_same_src_port_rate": 0.0, "dst_host_srv_diff_host_rate": 0.0,
                "dst_host_serror_rate": serror, "dst_host_srv_serror_rate": serror,
                "dst_host_rerror_rate": rerror, "dst_host_srv_rerror_rate": rerror,
                "attack": "normal", "difficulty_level": 21,
            }
            real_dst_str = f"{f['dst']}:{f['dport']}"
            ingest_telemetry_packet(pd.Series(row_dict), artifacts, sensitivity, real_src=src, real_dst=real_dst_str)
    except Exception as exc:
        logging.warning("Live packet capture slice failed: %s", exc)


# -----------------------------------------------------------------------------
# Dynamic Network Health Index & KPI Metric Computation
# -----------------------------------------------------------------------------
def compute_network_health():
    """Calculates Network Health Index (0-100%) factoring in active threat mitigation."""
    recent = st.session_state.soc_telemetry[:60]
    if not recent:
        return 100.0, "SECURE · NOMINAL", "badge-secure", "pulse-green"

    window_size = len(recent)
    unmitigated_threats = sum(1 for f in recent if f["is_threat"] and f["source_ip"] not in st.session_state.blocked_ips)
    mitigated_threats = sum(1 for f in recent if f["is_threat"] and f["source_ip"] in st.session_state.blocked_ips)

    effective_load = unmitigated_threats + (0.25 * mitigated_threats)
    threat_ratio = effective_load / window_size
    health = max(0.0, min(100.0, 100.0 * (1.0 - (1.6 * threat_ratio))))
    health = round(health, 1)

    if health >= 85.0:
        return health, "SECURE · OPTIMAL", "badge-secure", "pulse-green"
    elif health >= 60.0:
        return health, "CAUTION · ELEVATED RISK", "badge-caution", "pulse-yellow"
    else:
        return health, "CRITICAL ALERT · INTRUSION ACTIVE", "badge-critical", "pulse-red"


# -----------------------------------------------------------------------------
# Load ML Artifacts & Dataset
# -----------------------------------------------------------------------------
try:
    artifacts = load_nids_models()
    test_dataset = load_telemetry_dataset()
except Exception as e:
    st.error(f"⚠️ Critical initialization failure: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# Sidebar: User-Friendly Security Controls & Abstraction
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ ShieldNet Security")
    st.caption("AI-Powered Network Intrusion Prevention System")

    st.markdown("---")
    # Requirement 1: User-friendly Abstraction over AI model parameters
    st.markdown("### 🔒 Defense Stance")
    stance_options = [
        "🛡️ Balanced Protection (Recommended)",
        "⚡ High Vigilance (Proactive Guard)",
        "🚨 Strict Lockdown (Maximum Vigilance)",
        "🛠️ Custom Security Policy",
    ]
    selected_stance = st.selectbox(
        "Operational Mode",
        stance_options,
        index=0,
        help="Preconfigured security presets that abstract complex machine learning thresholds.",
    )
    st.session_state.defense_stance = selected_stance

    # Map intuitive stance to detection threshold
    if selected_stance == "🛡️ Balanced Protection (Recommended)":
        sensitivity = 0.50
        st.success("Nominal vigilance. Ideal balance between threat mitigation and zero false alarms.")
    elif selected_stance == "⚡ High Vigilance (Proactive Guard)":
        sensitivity = 0.35
        st.warning("Elevated sensitivity. Detects subtle reconnaissance probes and credential spraying.")
    elif selected_stance == "🚨 Strict Lockdown (Maximum Vigilance)":
        sensitivity = 0.20
        st.error("Zero-tolerance defense. Aggressively flags statistical anomalies and unverified handshakes.")
    else:
        # Custom Advanced Expander
        with st.expander("⚙️ Advanced Model Hyperparameters", expanded=True):
            sensitivity = st.slider(
                "Internal Decision Threshold",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
                help="Direct control over Random Forest probabilistic decision boundary.",
            )
            st.caption(f"Active Features: **{len(artifacts['feature_names'])} Aligned Inputs**")
            st.caption("StandardScaler: **38 Scaled Dimensions**")

    st.markdown("---")
    # Requirement 2: Real Network Live Packet Ingestion Toggle
    st.markdown("### 📡 Traffic Ingestion Engine")
    source_mode = st.radio(
        "Capture Source",
        ["📡 Live Physical Network Sniffer", "🧪 Simulation & Stress Test"],
        index=0 if SCAPY_AVAILABLE else 1,
        help="Switch between sniffing real incoming network packets on your physical interface or replaying stress test profiles.",
    )
    st.session_state.traffic_source_mode = source_mode

    if source_mode == "📡 Live Physical Network Sniffer":
        if not SCAPY_AVAILABLE:
            st.warning("Npcap or Scapy is not available. Please install Npcap on Windows or run as Administrator.")
        else:
            iface_name = getattr(conf, "iface", "Primary Adapter")
            st.info(f"Interface: `{iface_name}`")
            live_active = st.toggle("Active Network Sniffing 🟢", value=st.session_state.live_sniff_active)
            st.session_state.live_sniff_active = live_active
            st.caption(f"Real Packets Sniffed: **{st.session_state.live_packet_count:,}**")
            if st.button("Capture One Slice ⚡", use_container_width=True):
                process_real_network_batch(artifacts, sensitivity)
                st.rerun()
    else:
        # Simulation controls
        stream_scenario = st.selectbox(
            "Attack Simulation Profile",
            [
                "Mixed NSL-KDD Traffic",
                "Normal Workload Stream",
                "DoS / Neptune SYN Flood",
                "Reconnaissance / Port Scanning",
                "Brute-Force & R2L Exploit",
                "Zero-Day Attack Variant",
            ],
            index=0,
        )
        st.session_state.stream_scenario = stream_scenario

        auto_toggle = st.toggle("Continuous Background Replay (2s)", value=st.session_state.auto_stream)
        st.session_state.auto_stream = auto_toggle

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("Step Flow ⏭️", use_container_width=True):
                sample = select_scenario_sample(test_dataset, st.session_state.stream_scenario)
                ingest_telemetry_packet(sample, artifacts, sensitivity)
                st.rerun()
        with col_b2:
            if st.button("Burst (10) ⚡", use_container_width=True):
                for _ in range(10):
                    sample = select_scenario_sample(test_dataset, st.session_state.stream_scenario)
                    ingest_telemetry_packet(sample, artifacts, sensitivity)
                st.rerun()

        if st.button("🚨 Simulate Attack Wave (25)", type="primary", use_container_width=True):
            for _ in range(25):
                sample = select_scenario_sample(test_dataset, "DoS / Neptune SYN Flood")
                ingest_telemetry_packet(sample, artifacts, sensitivity)
            st.rerun()

    st.markdown("---")
    st.markdown("### 🛑 One-Click Active Defense")
    if st.button("Mitigate All Detected Attackers", use_container_width=True):
        added = 0
        for inc in st.session_state.soc_incidents:
            if inc["source_ip"] not in st.session_state.blocked_ips:
                st.session_state.blocked_ips.add(inc["source_ip"])
                inc["mitigation_status"] = "Mitigated / Blocked"
                added += 1
        st.session_state.mitigated_count = len(st.session_state.blocked_ips)
        st.success(f"Added {added} unique IPs to active firewall deny cache.")
        st.rerun()

    if st.button("Clear Telemetry & Incident Logs", use_container_width=True):
        st.session_state.soc_telemetry.clear()
        st.session_state.soc_incidents.clear()
        st.session_state.blocked_ips.clear()
        st.session_state.total_inspected = 0
        st.session_state.normal_count = 0
        st.session_state.threat_count = 0
        st.session_state.mitigated_count = 0
        st.session_state.latest_flow_analysis = None
        st.session_state.live_packet_count = 0
        st.rerun()


# -----------------------------------------------------------------------------
# Background Live Sniffer or Stream Fragment Worker
# -----------------------------------------------------------------------------
@st.fragment(run_every="2s")
def background_capture_worker():
    """Background execution tick for either real network capture or simulation."""
    if st.session_state.traffic_source_mode == "📡 Live Physical Network Sniffer" and st.session_state.live_sniff_active:
        process_real_network_batch(artifacts, sensitivity)
    elif st.session_state.traffic_source_mode == "🧪 Simulation & Stress Test" and st.session_state.auto_stream:
        sample = select_scenario_sample(test_dataset, st.session_state.stream_scenario)
        ingest_telemetry_packet(sample, artifacts, sensitivity)


background_capture_worker()


# -----------------------------------------------------------------------------
# Main Header Banner & Top KPI Cards
# -----------------------------------------------------------------------------
health_score, health_status, health_badge_cls, health_pulse_cls = compute_network_health()

st.markdown(f"""
<div class="soc-banner">
    <div>
        <h1>🛡️ ShieldNet AI · Network Anomaly Operations Center</h1>
        <p>Production NIDS Engine · {st.session_state.traffic_source_mode} · {st.session_state.defense_stance}</p>
    </div>
    <div style="text-align:right">
        <span class="badge {health_badge_cls}">
            <span class="pulse-dot {health_pulse_cls}"></span>{health_status}
        </span>
        <div style="font-size:0.75rem; color:#94a3b8; margin-top:6px; font-family:monospace">
            DEFENSE STANCE: {selected_stance.split(' ')[1].upper()}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

total = st.session_state.total_inspected
normal = st.session_state.normal_count
threats = st.session_state.threat_count
blocked = len(st.session_state.blocked_ips)
normal_pct = f"{(normal / total * 100):.1f}%" if total > 0 else "100.0%"
mitigated_pct = f"{(blocked / max(threats, 1) * 100):.1f}%" if threats > 0 else "100.0%"

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(f"""
    <div class="soc-card" style="border-top:3px solid {'#10b981' if health_score >= 85 else ('#f59e0b' if health_score >= 60 else '#ef4444')}">
        <div class="soc-card-title">Network Health Index</div>
        <div class="soc-card-value" style="color:{'#10b981' if health_score >= 85 else ('#f59e0b' if health_score >= 60 else '#ef4444')}">{health_score}%</div>
        <div class="soc-card-delta" style="color:#94a3b8">Sliding 60-Flow Window</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="soc-card" style="border-top:3px solid #38bdf8">
        <div class="soc-card-title">Total Packets Inspected</div>
        <div class="soc-card-value">{total:,}</div>
        <div class="soc-card-delta" style="color:#38bdf8">122-Feature Aligned</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="soc-card" style="border-top:3px solid #10b981">
        <div class="soc-card-title">Normal Traffic Flows</div>
        <div class="soc-card-value">{normal:,}</div>
        <div class="soc-card-delta" style="color:#10b981">{normal_pct} of Inspected</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="soc-card" style="border-top:3px solid #f43f5e">
        <div class="soc-card-title">Active Threats Detected</div>
        <div class="soc-card-value" style="color:#f43f5e">{threats:,}</div>
        <div class="soc-card-delta" style="color:#f43f5e">{threats} Anomaly Incidents</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="soc-card" style="border-top:3px solid #a855f7">
        <div class="soc-card-title">Attacker IPs Blacklisted</div>
        <div class="soc-card-value" style="color:#a855f7">{blocked:,}</div>
        <div class="soc-card-delta" style="color:#a855f7">{mitigated_pct} Mitigated</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Operations Center Tab Navigation
# -----------------------------------------------------------------------------
tab_live, tab_map, tab_incidents, tab_explain, tab_capture = st.tabs([
    "🛰️ Real-Time Stream & Anomaly Monitor",
    "🌍 Global Threat Origin Map",
    "🚨 Actionable SOC Incident Center",
    "🧠 AI Explainability & Architecture",
    "📁 Forensic PCAP Analyzer",
])


# =============================================================================
# TAB 1: Real-Time Stream Simulator & Telemetry Monitor
# =============================================================================
with tab_live:
    col_chart1, col_chart2 = st.columns([1, 2])

    with col_chart1:
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            number={"suffix": "%", "font": {"size": 34, "color": "#f8fafc"}},
            title={"text": "Network Health Gauge", "font": {"size": 15, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                "bar": {"color": "#10b981" if health_score >= 85 else ("#f59e0b" if health_score >= 60 else "#ef4444"), "thickness": 0.28},
                "bgcolor": "#1e293b",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 60], "color": "rgba(239, 68, 68, 0.2)"},
                    {"range": [60, 85], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [85, 100], "color": "rgba(16, 185, 129, 0.2)"},
                ],
                "threshold": {"line": {"color": "#f8fafc", "width": 3}, "thickness": 0.75, "value": health_score},
            },
        ))
        gauge_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=240,
            margin=dict(l=20, r=20, t=35, b=10),
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col_chart2:
        if st.session_state.soc_telemetry:
            rev_telemetry = list(reversed(st.session_state.soc_telemetry[:40]))
            timeline_df = pd.DataFrame([{
                "Packet #": item["index"],
                "Anomaly Score": item["threat_score"],
                "Verdict": item["pred_label"],
                "Threat Type": item["threat_title"],
            } for item in rev_telemetry])

            line_fig = px.line(
                timeline_df,
                x="Packet #",
                y="Anomaly Score",
                color="Verdict",
                color_discrete_map={"NORMAL": "#10b981", "MALICIOUS": "#ef4444"},
                markers=True,
                hover_data=["Threat Type"],
                title="Live Anomaly Probability Sequence",
            )
            line_fig.add_hline(
                y=int(sensitivity * 100),
                line_dash="dot",
                line_color="#f59e0b",
                annotation_text=f"Sensitivity ({int(sensitivity * 100)}%)",
                annotation_position="bottom right",
            )
            line_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=240,
                margin=dict(l=10, r=10, t=35, b=10),
                yaxis=dict(range=[0, 105], gridcolor="#1e293b", title="Threat Score (%)"),
                xaxis=dict(gridcolor="#1e293b", title="Packet Index"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(line_fig, use_container_width=True)
        else:
            st.info("Awaiting telemetry flow... Sniff packets or click 'Step Flow ⏭️'.")

    st.markdown("---")

    # Active Packet Deep-Dive
    latest = st.session_state.latest_flow_analysis
    if latest:
        st.subheader("🔍 Active Packet Deep-Dive & Geographic Forensics")
        box_class = "clean" if not latest["is_threat"] else "danger"
        geo = latest.get("geo", {})

        col_flow_meta, col_flow_verdict = st.columns([3, 2])

        with col_flow_meta:
            st.markdown(f"""
            <div class="anomaly-callout {box_class}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; font-size:1.05rem; color:{'#10b981' if not latest['is_threat'] else '#ef4444'}">
                        {'✅ Normal Legitimate Connection' if not latest['is_threat'] else '🚨 Intrusive Network Flow Detected'}
                    </span>
                    <span class="badge {'badge-secure' if not latest['is_threat'] else 'badge-critical'}">
                        {latest['category']} · {latest['severity']}
                    </span>
                </div>
                <div style="margin-top:8px; font-size:0.88rem; color:#cbd5e1; display:grid; grid-template-columns:1fr 1fr; gap:6px;">
                    <div>• <b>Source IP:</b> <span class="mono">{geo.get('flag', '')} {latest['source']}</span></div>
                    <div>• <b>Location:</b> <span class="mono">{geo.get('city', 'Local')}, {geo.get('country', 'Internal')}</span></div>
                    <div>• <b>Provider / ISP:</b> <span class="mono">{geo.get('isp', 'Local Subnet')}</span></div>
                    <div>• <b>Destination:</b> <span class="mono">{latest['destination']}</span></div>
                    <div>• <b>Protocol / Service:</b> <span class="mono">{latest['protocol']} / {latest['service']}</span></div>
                    <div>• <b>TCP Flag / Payload:</b> <span class="mono">{latest['flag']} ({latest['bytes']})</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_flow_verdict:
            score = latest["threat_score"]
            st.metric("Threat Confidence Score", f"{score}%", delta=f"{'Intrusion Alert' if latest['is_threat'] else 'Nominal'}", delta_color="inverse" if latest['is_threat'] else "normal")
            st.caption(f"Recommended Mitigation: **{latest['recommended_mitigation']}**")

        if latest["is_threat"]:
            st.markdown("#### ⚡ Triggered Anomaly Signatures")
            if latest["triggered_params"]:
                p_cols = st.columns(min(len(latest["triggered_params"]), 3))
                for idx, param in enumerate(latest["triggered_params"][:3]):
                    with p_cols[idx]:
                        st.markdown(f"""
                        <div style="background:#0f172a; border:1px solid rgba(239, 68, 68, 0.4); border-radius:8px; padding:12px; margin-bottom:10px;">
                            <div style="font-size:0.75rem; color:#f87171; font-weight:700; text-transform:uppercase">{param['name']}</div>
                            <div style="font-size:1.25rem; font-weight:800; color:#f8fafc; font-family:monospace; margin:4px 0;">{param['value']}</div>
                            <div style="font-size:0.75rem; color:#94a3b8">Normal: {param['benchmark']} · <b>{param['severity']}</b></div>
                            <div style="font-size:0.75rem; color:#cbd5e1; margin-top:6px">{param['desc']}</div>
                        </div>
                        """, unsafe_allow_html=True)

            if latest["source_ip"] not in st.session_state.blocked_ips:
                if st.button(f"🛡️ Apply Firewall Drop to {latest['source_ip']}", type="primary"):
                    st.session_state.blocked_ips.add(latest["source_ip"])
                    st.success(f"Blacklisted {latest['source_ip']} in session deny cache.")
                    st.rerun()

    st.markdown("---")

    # Live-Scrolling Telemetry Table with Geo Flags
    st.subheader("📋 Real-Time Network Packet Telemetry Stream")
    if st.session_state.soc_telemetry:
        display_data = [{
            "Packet #": item["index"],
            "Timestamp": item["timestamp"],
            "Origin Location": f"{item.get('geo', {}).get('flag', '')} {item.get('geo', {}).get('country_code', 'LAN')}",
            "Source IP": item["source_ip"],
            "Target": item["destination"],
            "Protocol": item["protocol"],
            "Service": item["service"],
            "Flag": item["flag"],
            "Payload": item["bytes"],
            "Threat Score": f"{item['threat_score']}%",
            "Verdict": item["pred_label"],
            "Classification": item["threat_title"],
            "Mitigation": item["mitigation_status"],
        } for item in st.session_state.soc_telemetry[:25]]

        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("No network flows captured yet in this session.")


# =============================================================================
# TAB 2: Global Threat Origin Map (Requirement 3: Geographical Location)
# =============================================================================
with tab_map:
    st.subheader("🌍 Global Threat Origin & Geospatial Intelligence")
    st.caption("Visualizing the geographic origins of suspicious and malicious network traffic across the world.")

    threat_incidents = [i for i in st.session_state.soc_incidents if i.get("lat") and i.get("lon")]

    if threat_incidents:
        map_df = pd.DataFrame(threat_incidents)

        # Plotly Scattergeo World Map
        map_fig = go.Figure(go.Scattergeo(
            lon=map_df["lon"],
            lat=map_df["lat"],
            text=map_df.apply(lambda r: f"<b>{r['country_flag']} {r['source_ip']}</b><br>Location: {r['city']}, {r['country']}<br>Threat: {r['threat_type']}<br>ISP: {r['isp']}<br>Severity: {r['severity']}", axis=1),
            hoverinfo="text",
            mode="markers",
            marker=dict(
                size=14,
                color=map_df["severity"].map({"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#f59e0b", "INFORMATIONAL": "#38bdf8"}).fillna("#ef4444"),
                line=dict(width=1.5, color="#f8fafc"),
                opacity=0.88,
            ),
        ))

        map_fig.update_layout(
            geo=dict(
                bgcolor="rgba(0,0,0,0)",
                showland=True,
                landcolor="#151d30",
                showocean=True,
                oceancolor="#0b0f19",
                showlakes=True,
                lakecolor="#0b0f19",
                showcountries=True,
                countrycolor="#1e293b",
                countrywidth=1,
                coastlinecolor="#334155",
                projection_type="equirectangular",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
            margin=dict(l=0, r=0, t=10, b=0),
        )

        st.plotly_chart(map_fig, use_container_width=True)

        col_m1, col_m2 = st.columns([1, 1])

        with col_m1:
            st.markdown("#### 🏆 Top Threat Origin Countries")
            country_counts = map_df.groupby(["country", "country_flag"]).size().reset_index(name="Incident Count").sort_values("Incident Count", ascending=False)
            country_counts["Origin"] = country_counts["country_flag"] + " " + country_counts["country"]
            st.dataframe(country_counts[["Origin", "Incident Count"]], use_container_width=True, hide_index=True)

        with col_m2:
            st.markdown("#### 🏢 Top Malicious ISPs & Networks")
            isp_counts = map_df["isp"].value_counts().reset_index()
            isp_counts.columns = ["Internet Service Provider / Autonomous System", "Threat Count"]
            st.dataframe(isp_counts.head(7), use_container_width=True, hide_index=True)

    else:
        st.info("No threat incidents recorded with geographical coordinates yet. Ingest simulated attacks or sniff incoming traffic to populate the threat map.")


# =============================================================================
# TAB 3: Actionable SOC Incident Center
# =============================================================================
with tab_incidents:
    st.subheader("🚨 Actionable SOC Incident Management Center")
    st.caption("Persistent audit trail of all detected anomalies with geographic origins and firewall mitigations.")

    incidents = st.session_state.soc_incidents

    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
    with f_col1:
        cat_filter = st.selectbox("Filter Category", ["All Categories", "DoS", "Probe", "R2L", "U2R", "Zero-Day"])
    with f_col2:
        status_filter = st.selectbox("Mitigation Status", ["All Statuses", "Pending Action", "Mitigated / Blocked"])
    with f_col3:
        st.write("")
        st.write("")
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("Mitigate All Pending Incidents 🛡️", use_container_width=True):
                count_mit = 0
                for inc in incidents:
                    if inc["mitigation_status"] == "Pending Action":
                        inc["mitigation_status"] = "Mitigated / Blocked"
                        st.session_state.blocked_ips.add(inc["source_ip"])
                        count_mit += 1
                st.session_state.mitigated_count = len(st.session_state.blocked_ips)
                st.success(f"Successfully mitigated {count_mit} pending security incidents.")
                st.rerun()
        with col_act2:
            if incidents:
                inc_csv = pd.DataFrame(incidents).to_csv(index=False).encode("utf-8")
                st.download_button("Export Incidents (CSV) 📥", inc_csv, "soc_incidents.csv", "text/csv", use_container_width=True)

    filtered_incidents = incidents
    if cat_filter != "All Categories":
        filtered_incidents = [i for i in filtered_incidents if i["category"] == cat_filter]
    if status_filter != "All Statuses":
        filtered_incidents = [i for i in filtered_incidents if i["mitigation_status"] == status_filter]

    if filtered_incidents:
        inc_df = pd.DataFrame([{
            "Incident ID": i["incident_id"],
            "Timestamp": i["timestamp"],
            "Origin": f"{i.get('country_flag', '')} {i.get('country', 'Internal')}",
            "Source IP": i["source_ip"],
            "ISP / Host": i.get("isp", "Local Network"),
            "Target": i["destination"],
            "Threat Classification": i["threat_type"],
            "Severity": i["severity"],
            "Confidence": i["confidence"],
            "Triggered Signatures": i["triggered_summary"],
            "Recommended Firewall Mitigation": i["recommended_mitigation"],
            "Status": i["mitigation_status"],
        } for i in filtered_incidents])

        st.dataframe(inc_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            cat_counts = pd.Series([i["category"] for i in incidents]).value_counts().reset_index()
            cat_counts.columns = ["Attack Family", "Count"]
            donut_fig = px.pie(
                cat_counts, names="Attack Family", values="Count", hole=0.45,
                color="Attack Family",
                color_discrete_map={"DoS": "#ef4444", "Probe": "#f59e0b", "R2L": "#3b82f6", "U2R": "#a855f7", "Zero-Day": "#ec4899"},
                title="Threat Category Breakdown",
            )
            donut_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260)
            st.plotly_chart(donut_fig, use_container_width=True)

        with c_chart2:
            sev_counts = pd.Series([i["severity"] for i in incidents]).value_counts().reset_index()
            sev_counts.columns = ["Severity Level", "Incidents"]
            bar_fig = px.bar(
                sev_counts, x="Severity Level", y="Incidents", color="Severity Level",
                color_discrete_map={"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#f59e0b", "LOW": "#38bdf8"},
                title="Incident Severity Distribution",
            )
            bar_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260)
            st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.info("No actionable incidents match the current filter selection.")


# =============================================================================
# TAB 4: AI Explainability & Architecture (Abstracted from Main View)
# =============================================================================
with tab_explain:
    st.subheader("🧠 122-Feature ML Pipeline & Explainability Engine")
    st.caption("Technical architectural audit of preprocessing transformations and model feature importance.")

    col_fi, col_pipe = st.columns([3, 2])

    with col_fi:
        st.markdown("#### 🏆 Top 15 Model Feature Importances (Gini Index)")
        model = artifacts["advanced_model"]
        fnames = artifacts["feature_names"]

        if hasattr(model, "feature_importances_"):
            fi_df = pd.DataFrame({
                "Feature": fnames,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False).head(15)

            fi_fig = px.bar(
                fi_df.iloc[::-1], x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Blues",
                title="Top Predictive Features in Random Forest",
            )
            fi_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=450, coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=35, b=10),
                xaxis=dict(gridcolor="#1e293b", title="Importance Score"),
                yaxis=dict(gridcolor="#1e293b"),
            )
            st.plotly_chart(fi_fig, use_container_width=True)
        else:
            st.info("Feature importance is not available for this model type.")

    with col_pipe:
        st.markdown("#### ⚙️ Feature Schema Specifications")
        st.markdown(f"""
        - **Total Model Features**: `{len(artifacts['feature_names'])} columns`
        - **Categorical Columns**: `protocol_type`, `service`, `flag`
        - **Numerical Features**: `{len(artifacts['scaler'].feature_names_in_)} scaled dimensions`
        - **Zero-Day Resilience**: Dynamic 0.0 filling for unseen service categories
        - **Leakage Prevention**: Scaler fitted strictly on train partition
        """)

        st.markdown("#### 🔬 Domain Importance Highlights")
        st.markdown("""
        1. **src_bytes & dst_bytes**: Detects asymmetric volume anomalies and 0-byte SYN floods.
        2. **flag_SF & flag_S0**: Distinguishes completed TCP handshakes from half-open attacks.
        3. **same_srv_rate & diff_srv_rate**: Core signatures for reconnaissance port sweeps.
        4. **dst_host_srv_serror_rate**: Host-level concentration of half-open connection errors.
        """)


# =============================================================================
# TAB 5: Forensic PCAP Analyzer
# =============================================================================
with tab_capture:
    st.subheader("📁 PCAP Forensic Capture Analysis")
    st.caption("Upload raw packet capture (.pcap or .pcapng) files for automated anomaly extraction.")

    if not SHIELDNET_AVAILABLE:
        st.warning("PCAP forensic engine requires Scapy.")
    else:
        state: RuntimeState = st.session_state.shieldnet_runtime
        pcap_file = st.file_uploader("Upload .pcap or .pcapng", type=["pcap", "pcapng"])
        if pcap_file and st.button("Process PCAP Capture", use_container_width=True):
            try:
                state.process_pcap(pcap_file, sensitivity)
                st.success("PCAP analysis completed.")
            except Exception as exc:
                st.error(f"PCAP parsing error: {exc}")

        st.markdown("---")
        st.markdown("### 🧪 Safe Localhost Probe Simulation")
        sim_choice = st.selectbox("Simulation Vector", ["Port scan", "SYN burst", "UDP traffic"])
        sim_confirm = st.checkbox("I confirm sending bounded local packets to 127.0.0.1 (Max 32 packets)")
        if st.button("Run Local Simulation", disabled=not sim_confirm):
            try:
                sim_res = run_local_simulation(sim_choice, port_count=8, rate_per_second=3, duration_seconds=2)
                st.success(f"Transmitted {sim_res['packets_sent']} probes to {sim_res['target']}.")
            except Exception as exc:
                st.error(f"Simulation failed: {exc}")
