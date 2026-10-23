import os
import sys
import pytest
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import (
    load_nids_models,
    preprocess_traffic_samples,
    analyze_anomaly_parameters,
    classify_network_threat,
)


def test_model_artifacts_loading():
    artifacts = load_nids_models("models")
    assert artifacts["advanced_model"] is not None
    assert artifacts["scaler"] is not None
    assert len(artifacts["feature_names"]) == 122
    assert artifacts["scaler"].n_features_in_ == 38


def test_122_feature_alignment_pipeline():
    artifacts = load_nids_models("models")
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]

    # Raw connection sample
    raw_sample = pd.Series({
        "duration": 0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "S0",
        "src_bytes": 0,
        "dst_bytes": 0,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 150, "srv_count": 150,
        "serror_rate": 1.0, "srv_serror_rate": 1.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 255, "dst_host_srv_count": 10,
        "dst_host_same_srv_rate": 0.04, "dst_host_diff_srv_rate": 0.06,
        "dst_host_same_src_port_rate": 0.0, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 1.0, "dst_host_srv_serror_rate": 1.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
        "attack": "neptune", "difficulty_level": 20,
    })

    aligned_vec = preprocess_traffic_samples(raw_sample, scaler, feature_names)
    assert aligned_vec.shape == (1, 122)
    assert list(aligned_vec.columns) == feature_names

    # Predict with model
    pred = artifacts["advanced_model"].predict(aligned_vec)[0]
    assert pred in [0, 1]


def test_anomaly_parameters_breakdown():
    syn_flood_sample = {
        "count": 220,
        "serror_rate": 1.0,
        "dst_host_serror_rate": 1.0,
        "flag": "S0",
        "src_bytes": 0,
        "dst_bytes": 0,
    }
    triggered = analyze_anomaly_parameters(syn_flood_sample)
    params = [t["param"] for t in triggered]
    assert "count" in params
    assert "serror_rate" in params
    assert "flag" in params
    assert "payload" in params


def test_threat_classification_and_mitigation():
    dos_threat = classify_network_threat({"count": 200, "flag": "S0"}, ground_truth="neptune")
    assert dos_threat["category"] == "DoS"
    assert "Rate Limiting" in dos_threat["mitigation"]

    probe_threat = classify_network_threat({"diff_srv_rate": 0.8}, ground_truth="portsweep")
    assert probe_threat["category"] == "Probe"
    assert "Blacklist" in probe_threat["mitigation"]

    r2l_threat = classify_network_threat({"num_failed_logins": 3}, ground_truth="guess_passwd")
    assert r2l_threat["category"] == "R2L"
    assert "Fail2ban" in r2l_threat["mitigation"]


def test_ip_geolocation_resolution():
    from app import resolve_ip_geo

    # Test local subnet
    lan_geo = resolve_ip_geo("192.168.1.105")
    assert lan_geo["is_local"] is True
    assert lan_geo["country_code"] == "LAN"
    assert lan_geo["flag"] == "🏠"

    # Test public/simulated IP
    pub_geo = resolve_ip_geo("8.8.8.8")
    assert pub_geo["is_local"] is False
    assert "lat" in pub_geo and "lon" in pub_geo
    assert len(pub_geo["country"]) > 0


def test_port_to_service_mapping():
    from app import port_to_service

    assert port_to_service(80) == "http"
    assert port_to_service(443) in ["http", "http_443"]
    assert port_to_service(53) == "domain_u"
    assert port_to_service(21) == "ftp"
    assert port_to_service(22) == "ssh"
    assert port_to_service(99999) == "private"


def test_copilot_investigation():
    from soc_copilot import generate_forensic_investigation, answer_copilot_query

    mock_incident = {
        "incident_id": "INC-1001",
        "timestamp": "14:20:00",
        "source_ip": "185.220.101.44",
        "destination": "10.0.1.10:80",
        "protocol_service": "TCP / http",
        "threat_type": "DoS / Neptune Flood",
        "category": "DoS",
        "severity": "CRITICAL",
        "confidence": "99%",
        "country": "Germany",
        "city": "Frankfurt",
        "isp": "Hetzner Tor Exit",
        "country_flag": "🇩🇪",
        "raw_attributes": {"count": 250, "serror_rate": 1.0, "flag": "S0"},
        "triggered_params": [{"name": "SYN Handshake Error Rate"}],
    }

    inv = generate_forensic_investigation(mock_incident)
    assert "TCP SYN" in inv["playbook_title"]
    assert "T1498" in inv["mitre_technique"]
    assert "netsh" in inv["commands"]["windows_netsh"]
    assert "iptables" in inv["commands"]["linux_iptables"]
    assert "EXECUTIVE INCIDENT BRIEFING" in inv["executive_briefing"]

    # Test query handling
    ans_why = answer_copilot_query("Why was this detected?", mock_incident, [mock_incident])
    assert "SYN Handshake Error Rate" in ans_why
    assert "DoS" in ans_why

    ans_block = answer_copilot_query("How do I block this?", mock_incident, [mock_incident])
    assert "netsh" in ans_block or "iptables" in ans_block

