import csv
import ctypes
import io
import json
import platform
import subprocess
from types import SimpleNamespace
import pytest
from scapy.all import IP, TCP, PcapWriter

from shieldnet import (
    FEATURE_NAMES, Incident, MitigationManager, PacketRecord, build_session_report,
    RuntimeState, detect_threat, extract_features, load_live_models, report_pdf,
    run_local_simulation, write_incidents_csv, write_incidents_json,
)


def test_feature_extraction_uses_packet_flags_ports_and_timing():
    packets = [
        PacketRecord(1.0, "8.8.8.8", "1.1.1.1", "TCP", 60, 80, tcp_syn=True),
        PacketRecord(1.5, "8.8.8.8", "1.1.1.1", "TCP", 100, 443, tcp_ack=True),
        PacketRecord(2.0, "8.8.8.8", "1.1.1.1", "UDP", 40, 53),
    ]
    got = extract_features(packets, window_seconds=2)
    assert set(got) == set(FEATURE_NAMES)
    assert got["packet_count"] == 3
    assert got["total_bytes"] == 200
    assert got["average_packet_size"] == pytest.approx(200 / 3)
    assert got["throughput_bps"] == 800
    assert got["tcp_syn_ratio"] == pytest.approx(0.5)
    assert got["tcp_ack_ratio"] == pytest.approx(0.5)
    assert got["unique_destination_ports"] == 3
    assert got["average_inter_arrival_seconds"] == pytest.approx(0.5)


def test_scoring_is_normalized_and_detects_port_sweep():
    features = {name: 0.0 for name in FEATURE_NAMES}
    features.update(packet_count=12, unique_destination_ports=10, tcp_syn_ratio=1.0)
    result = detect_threat(features, sensitivity=0.8)
    assert 0 <= result["score"] <= 100
    assert result["is_threat"] is True
    assert result["attack_type"] == "Port scan"
    assert result["mode"] == "Baseline rules"


def test_mitigation_rejects_protected_and_invalid_ips():
    manager = MitigationManager()
    for value in ("127.0.0.1", "::1", "169.254.1.2", "224.0.0.1", "192.168.1.4", "1.2.3.4;whoami"):
        with pytest.raises(ValueError):
            manager.ban(value)
    ip = manager.ban("8.8.8.8")
    assert manager.is_banned(ip)
    with pytest.raises(PermissionError, match="confirmation"):
        manager.firewall_block("8.8.8.8", confirmed=False)


def test_firewall_command_is_argument_list_and_shell_is_disabled(monkeypatch):
    calls = {}
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(shell32=SimpleNamespace(IsUserAnAdmin=lambda: 1)), raising=False)

    def fake_run(args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    MitigationManager().firewall_block("8.8.8.8", confirmed=True)
    assert isinstance(calls["args"], list)
    assert "remoteip=8.8.8.8" in calls["args"]
    assert calls["kwargs"]["shell"] is False


def test_legacy_dataset_models_are_not_misapplied_to_live_features():
    rf, isolation, label = load_live_models()
    assert rf is None and isolation is None
    assert "Baseline rules" in label


def test_report_exports_include_incidents_and_actions():
    incident = Incident("2026-01-01T00:00:00Z", "8.8.8.8", "Port scan", "High", 91, "Detected")
    csv_data = write_incidents_csv([incident]).decode("utf-8")
    assert next(csv.DictReader(io.StringIO(csv_data)))["source_ip"] == "8.8.8.8"
    data = json.loads(write_incidents_json([incident]))
    assert data[0]["threat_score"] == 91
    report = build_session_report({"packets_analyzed": 12}, [incident], [{"timestamp": "now", "ip": "8.8.8.8", "action": "In-memory block"}])
    assert "Port scan" in report and "In-memory block" in report and "Packets Analyzed: 12" in report
    pdf = report_pdf(report)
    assert pdf.startswith(b"%PDF")


def test_simulation_is_fixed_to_loopback_and_bounded(monkeypatch):
    sent_to = []

    def fake_send(packet, **kwargs):
        sent_to.append(packet[IP].dst)

    monkeypatch.setattr("scapy.all.send", fake_send)
    result = run_local_simulation("SYN burst", port_count=4, rate_per_second=10, duration_seconds=0.5)
    assert result["target"] == "127.0.0.1"
    assert result["packets_sent"] <= 4
    assert all(address == "127.0.0.1" for address in sent_to)
    with pytest.raises(ValueError):
        run_local_simulation("UDP traffic", port_count=33)


def test_pcap_analysis_splits_records_into_two_second_windows():
    packets = []
    for timestamp, port in ((1000.1, 80), (1002.1, 443)):
        packet = IP(src="8.8.8.8", dst="1.1.1.1") / TCP(sport=12345, dport=port, flags="S")
        packet.time = timestamp
        packets.append(packet)
    file_obj = io.BytesIO()
    writer = PcapWriter(file_obj, sync=True)
    for packet in packets:
        writer.write(packet)
    payload = file_obj.getvalue()
    writer.close()
    file_obj = io.BytesIO(payload)

    state = RuntimeState()
    state.process_pcap(file_obj)
    snap = state.snapshot()
    assert snap["packet_count"] == 2
    assert len(snap["throughput_history"]) == 2
