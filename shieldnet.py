"""Safe packet-flow analytics, optional inference, mitigation and reporting."""
from __future__ import annotations

import csv
import ipaddress
import json
import logging
import os
import platform
import socket
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOG = logging.getLogger("shieldnet")
FEATURE_NAMES = (
    "packet_count", "total_bytes", "average_packet_size", "throughput_bps",
    "tcp_syn_ratio", "tcp_ack_ratio", "tcp_rst_ratio",
    "unique_destination_ports", "average_inter_arrival_seconds",
)


@dataclass
class PacketRecord:
    timestamp: float
    source_ip: str
    destination_ip: str
    protocol: str
    size: int
    destination_port: int | None = None
    tcp_syn: bool = False
    tcp_ack: bool = False
    tcp_rst: bool = False


@dataclass
class Incident:
    timestamp: str
    source_ip: str
    attack_type: str
    severity: str
    threat_score: int
    mitigation_status: str = "Observed"
    detection_mode: str = "Baseline rules"


def packet_to_record(packet: Any) -> PacketRecord | None:
    """Convert an IPv4/IPv6 Scapy packet to a small, serializable record."""
    try:
        from scapy.layers.inet import IP, TCP, UDP, ICMP
        from scapy.layers.inet6 import IPv6
        layer = packet.getlayer(IP) or packet.getlayer(IPv6)
        if layer is None:
            return None
        proto = "TCP" if packet.haslayer(TCP) else "UDP" if packet.haslayer(UDP) else "ICMP" if packet.haslayer(ICMP) else None
        if proto is None:
            return None
        tcp = packet.getlayer(TCP)
        return PacketRecord(
            timestamp=float(getattr(packet, "time", time.time())),
            source_ip=str(layer.src), destination_ip=str(layer.dst), protocol=proto,
            size=int(len(packet)),
            destination_port=int(packet[UDP].dport if packet.haslayer(UDP) else tcp.dport) if proto in ("TCP", "UDP") else None,
            tcp_syn=bool(tcp and int(tcp.flags) & 0x02),
            tcp_ack=bool(tcp and int(tcp.flags) & 0x10),
            tcp_rst=bool(tcp and int(tcp.flags) & 0x04),
        )
    except Exception:
        LOG.debug("Could not decode packet", exc_info=True)
        return None


def extract_features(packets: Iterable[PacketRecord], window_seconds: float = 2.0) -> dict[str, float]:
    rows = sorted(list(packets), key=lambda p: p.timestamp)
    if not rows:
        return {name: 0.0 for name in FEATURE_NAMES}
    count = len(rows)
    total = sum(max(0, int(p.size)) for p in rows)
    tcp = [p for p in rows if p.protocol == "TCP"]
    gaps = [max(0.0, b.timestamp - a.timestamp) for a, b in zip(rows, rows[1:])]
    return {
        "packet_count": float(count), "total_bytes": float(total),
        "average_packet_size": total / count,
        "throughput_bps": total * 8.0 / max(float(window_seconds), 0.001),
        "tcp_syn_ratio": sum(p.tcp_syn for p in tcp) / len(tcp) if tcp else 0.0,
        "tcp_ack_ratio": sum(p.tcp_ack for p in tcp) / len(tcp) if tcp else 0.0,
        "tcp_rst_ratio": sum(p.tcp_rst for p in tcp) / len(tcp) if tcp else 0.0,
        "unique_destination_ports": float(len({p.destination_port for p in rows if p.destination_port is not None})),
        "average_inter_arrival_seconds": sum(gaps) / len(gaps) if gaps else 0.0,
    }


def _probability(model: Any, vector: np.ndarray) -> float | None:
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vector)[0]
            classes = list(getattr(model, "classes_", range(len(probs))))
            idx = classes.index(1) if 1 in classes else len(probs) - 1
            return float(np.clip(probs[idx], 0, 1))
        pred = int(model.predict(vector)[0])
        return float(pred == 1)
    except Exception:
        LOG.warning("Inference failed; using baseline rules for this window", exc_info=True)
        return None


def load_live_models(model_dir: str | Path = "models") -> tuple[Any | None, Any | None, str]:
    """Load only artifacts explicitly trained for the nine live-flow features."""
    try:
        import joblib
        root = Path(model_dir)
        schema_path = root / "live_feature_names.joblib"
        rf_path = root / "live_random_forest.joblib"
        iso_path = root / "live_isolation_forest.joblib"
        if not schema_path.is_file() or joblib.load(schema_path) != list(FEATURE_NAMES):
            return None, None, "Baseline rules (no compatible live-flow model artifacts)"
        rf = joblib.load(rf_path) if rf_path.is_file() else None
        iso = joblib.load(iso_path) if iso_path.is_file() else None
        if rf is not None:
            from sklearn.ensemble import RandomForestClassifier
            valid_rf = (isinstance(rf, RandomForestClassifier)
                        and getattr(rf, "n_features_in_", None) == len(FEATURE_NAMES)
                        and len(getattr(rf, "estimators_", [])) > 0
                        and {0, 1}.issubset(set(getattr(rf, "classes_", []))))
            if not valid_rf:
                rf = None
        if iso is not None:
            from sklearn.ensemble import IsolationForest
            valid_iso = (isinstance(iso, IsolationForest)
                         and getattr(iso, "n_features_in_", None) == len(FEATURE_NAMES)
                         and len(getattr(iso, "estimators_", [])) > 0)
            if not valid_iso:
                iso = None
        if rf is None and iso is None:
            return None, None, "Baseline rules (no valid live-flow models)"
        return rf, iso, "Hybrid ML" if rf is not None and iso is not None else "Random Forest" if rf is not None else "Isolation Forest"
    except Exception:
        LOG.warning("Live model artifacts unavailable", exc_info=True)
        return None, None, "Baseline rules (artifact validation failed)"


def detect_threat(features: dict[str, float], sensitivity: float = 0.65,
                  random_forest: Any = None, isolation_forest: Any = None) -> dict[str, Any]:
    """Return a normalized 0-100 score; heuristic thresholds are labelled baseline."""
    sensitivity = float(np.clip(sensitivity, 0.05, 0.99))
    count = features["packet_count"]
    ports = features["unique_destination_ports"]
    syn = features["tcp_syn_ratio"]
    rst = features["tcp_rst_ratio"]
    rate = features["throughput_bps"]
    baseline = max(
        min(count / 100.0, 1.0) * 0.32,
        min(ports / 20.0, 1.0) * 0.90,
        (syn * 0.72 if count >= 8 else 0.0),
        (rst * 0.50 if count >= 10 else 0.0),
        min(rate / 2_000_000.0, 1.0) * 0.55,
    )
    vector = np.asarray([[features[name] for name in FEATURE_NAMES]], dtype=float)
    rf_score = _probability(random_forest, vector) if random_forest is not None else None
    iso_score = None
    if isolation_forest is not None:
        try:
            raw = float(isolation_forest.decision_function(vector)[0])
            iso_score = float(np.clip(0.5 - raw, 0.0, 1.0))
        except Exception:
            LOG.warning("Isolation Forest inference failed", exc_info=True)
    available = [v for v in (rf_score, iso_score) if v is not None]
    score = (0.55 * baseline + 0.45 * float(np.mean(available))) if available else baseline
    score = int(round(np.clip(score, 0.0, 1.0) * 100))
    threshold = int(round((1.0 - sensitivity) * 55 + 30))
    is_threat = score >= threshold
    attack_type = "Port scan" if ports >= 8 else "SYN burst" if syn >= 0.65 and count >= 8 else "High-volume traffic" if rate >= 1_000_000 else "Anomalous flow"
    mode = "Hybrid ML" if rf_score is not None and iso_score is not None else "Random Forest" if rf_score is not None else "Isolation Forest" if iso_score is not None else "Baseline rules"
    return {"score": score, "threshold": threshold, "is_threat": is_threat,
            "attack_type": attack_type, "mode": mode, "baseline_score": round(baseline * 100, 2)}


class MitigationManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._banned: set[str] = set()
        self.actions: list[dict[str, str]] = []

    @staticmethod
    def validate_ip(value: str, allow_private: bool = False) -> str:
        try:
            addr = ipaddress.ip_address(str(value).strip())
        except ValueError as exc:
            raise ValueError("Enter a valid IP address") from exc
        if addr.is_loopback or addr.is_unspecified or addr.is_multicast or addr.is_link_local or addr.is_reserved:
            raise ValueError("Loopback, unspecified, multicast, link-local and reserved addresses are protected")
        if not allow_private and not addr.is_global:
            raise ValueError("Private/non-global addresses are protected by default")
        return str(addr)

    def ban(self, value: str, allow_private: bool = False) -> str:
        ip = self.validate_ip(value, allow_private=allow_private)
        with self._lock:
            self._banned.add(ip)
            self.actions.append({"timestamp": datetime.now(timezone.utc).isoformat(), "ip": ip, "action": "In-memory block"})
        LOG.info("Added %s to in-memory block cache", ip)
        return ip

    def is_banned(self, value: str) -> bool:
        with self._lock:
            return value in self._banned

    def snapshot(self) -> tuple[list[dict[str, str]], int]:
        with self._lock:
            return list(self.actions), len(self._banned)

    def firewall_block(self, value: str, confirmed: bool, allow_private: bool = False) -> str:
        if not confirmed:
            raise PermissionError("Explicit firewall confirmation is required")
        ip = self.validate_ip(value, allow_private=allow_private)
        system = platform.system()
        if system == "Windows":
            import ctypes
            if not ctypes.windll.shell32.IsUserAnAdmin():
                raise PermissionError("Run the dashboard as Administrator to change Windows Firewall rules")
            command = ["netsh", "advfirewall", "firewall", "add", "rule", f"name=ShieldNet-Block-{ip}", "dir=in", "action=block", f"remoteip={ip}"]
        elif system == "Linux":
            if os.geteuid() != 0:
                raise PermissionError("Run with administrator privileges to change Linux firewall rules")
            family = "ip6" if ":" in ip else "ip"
            command = ["nft", "add", "rule", "inet", "filter", "input", family, "saddr", ip, "drop"]
        else:
            raise OSError(f"Firewall integration is unsupported on {system}")
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False, shell=False)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "Firewall command failed").strip())
        with self._lock:
            self.actions.append({"timestamp": datetime.now(timezone.utc).isoformat(), "ip": ip, "action": f"{system} firewall block"})
        LOG.info("Firewall block added for %s", ip)
        return f"{system} firewall rule added for {ip}"


def write_incidents_csv(incidents: Iterable[Incident | dict[str, Any]]) -> bytes:
    rows = [asdict(item) if isinstance(item, Incident) else dict(item) for item in incidents]
    fields = list(Incident.__dataclass_fields__)
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_incidents_json(incidents: Iterable[Incident | dict[str, Any]]) -> bytes:
    rows = [asdict(item) if isinstance(item, Incident) else dict(item) for item in incidents]
    return json.dumps(rows, indent=2, ensure_ascii=False).encode("utf-8")


def build_session_report(stats: dict[str, Any], incidents: Iterable[Incident | dict[str, Any]], actions: Iterable[dict[str, Any]]) -> str:
    rows = [asdict(item) if isinstance(item, Incident) else dict(item) for item in incidents]
    lines = ["ShieldNet AI Session Report", "=" * 28, f"Generated: {datetime.now(timezone.utc).isoformat()}", "", "Traffic totals"]
    lines += [f"{key.replace('_', ' ').title()}: {value}" for key, value in stats.items()]
    lines += ["", f"Detections ({len(rows)})"]
    lines += [f"{r.get('timestamp')} | {r.get('source_ip')} | {r.get('attack_type')} | {r.get('severity')} | score {r.get('threat_score')} | {r.get('mitigation_status')}" for r in rows]
    lines += ["", "Mitigation actions"]
    lines += [f"{a.get('timestamp')} | {a.get('ip')} | {a.get('action')}" for a in actions]
    return "\n".join(lines) + "\n"


def report_pdf(text: str) -> bytes:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        import io
    except ImportError as exc:
        raise RuntimeError("Install reportlab to generate PDF reports") from exc
    output = io.BytesIO()
    page = canvas.Canvas(output, pagesize=letter)
    width, height = letter
    y = height - 48
    for line in text.splitlines():
        if y < 48:
            page.showPage()
            y = height - 48
        page.drawString(42, y, line[:110])
        y -= 15
    page.save()
    return output.getvalue()


class RuntimeState:
    """Thread-safe session state shared by a background capture worker and UI."""
    def __init__(self):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.active = False
        self.sensitivity = 0.65
        self.auto_block = False
        self.error = ""
        self.interface = ""
        self.packet_count = 0
        self.total_bytes = 0
        self.pps_history: list[dict[str, float]] = []
        self.throughput_history: list[dict[str, float]] = []
        self.score_history: list[dict[str, Any]] = []
        self.incidents: list[Incident] = []
        self._window: dict[str, list[PacketRecord]] = defaultdict(list)
        self.window_started = time.monotonic()
        self.mitigation = MitigationManager()
        self.rf_model, self.iso_model, self.model_label = load_live_models()

    def ingest(self, record: PacketRecord, sensitivity: float = 0.65, auto_block: bool = False):
        with self.lock:
            self.packet_count += 1
            self.total_bytes += max(0, record.size)
            self._window[record.source_ip].append(record)

    def flush_window(self, sensitivity: float = 0.65, auto_block: bool = False):
        with self.lock:
            groups, self._window = self._window, defaultdict(list)
            start, self.window_started = self.window_started, time.monotonic()
            elapsed = max(self.window_started - start, 0.001)
        self._analyze_groups(groups, elapsed, sensitivity, auto_block)

    def _analyze_groups(self, groups: dict[str, list[PacketRecord]], elapsed: float,
                        sensitivity: float, auto_block: bool = False):
        with self.lock:
            total_packets = sum(len(v) for v in groups.values())
            self.pps_history.append({"timestamp": time.time(), "packets_per_second": total_packets / elapsed})
            self.pps_history = self.pps_history[-300:]
            window_bytes = sum(p.size for rows in groups.values() for p in rows)
            self.throughput_history.append({"timestamp": time.time(), "kbps": window_bytes * 8 / elapsed / 1000})
            self.throughput_history = self.throughput_history[-300:]
        for source, packets in groups.items():
            features = extract_features(packets, elapsed)
            result = detect_threat(features, sensitivity, self.rf_model, self.iso_model)
            with self.lock:
                self.score_history.append({"timestamp": time.time(), "score": result["score"], "source_ip": source})
                self.score_history = self.score_history[-300:]
                if result["is_threat"]:
                    status = "Detected"
                    if auto_block:
                        try:
                            self.mitigation.ban(source)
                            status = "Added to in-memory deny cache"
                        except ValueError:
                            status = "Detected; source protected from blocking"
                    self.incidents.append(Incident(datetime.now(timezone.utc).isoformat(), source, result["attack_type"], "High" if result["score"] >= 75 else "Medium", result["score"], status, result["mode"]))
                    self.incidents = self.incidents[-1000:]

    def snapshot(self) -> dict[str, Any]:
        actions, banned_count = self.mitigation.snapshot()
        with self.lock:
            return {"active": self.active, "error": self.error, "interface": self.interface,
                    "packet_count": self.packet_count, "total_bytes": self.total_bytes,
                    "pps_history": list(self.pps_history), "throughput_history": list(self.throughput_history), "score_history": list(self.score_history),
                    "incidents": [Incident(**asdict(item)) for item in self.incidents], "actions": actions,
                    "banned_count": banned_count, "model_label": self.model_label}

    def update_settings(self, sensitivity: float, auto_block: bool):
        with self.lock:
            self.sensitivity = float(np.clip(sensitivity, 0.05, 0.99))
            self.auto_block = bool(auto_block)

    def process_pcap(self, file_obj: Any, sensitivity: float = 0.65):
        with self.lock:
            if self.active:
                raise RuntimeError("Stop live monitoring before analyzing an uploaded PCAP")
        try:
            from scapy.utils import PcapReader
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            groups: dict[str, list[PacketRecord]] = defaultdict(list)
            active_window = None
            for packet in PcapReader(file_obj):
                record = packet_to_record(packet)
                if record:
                    window = int(record.timestamp // 2.0)
                    if active_window is not None and window != active_window:
                        self._analyze_groups(groups, 2.0, sensitivity)
                        groups = defaultdict(list)
                    active_window = window
                    groups[record.source_ip].append(record)
                    with self.lock:
                        self.packet_count += 1
                        self.total_bytes += max(0, record.size)
            if groups:
                self._analyze_groups(groups, 2.0, sensitivity)
        except Exception as exc:
            LOG.exception("PCAP processing failed")
            raise RuntimeError(f"Could not read PCAP: {exc}") from exc

    def start_capture(self, interface: str | None, sensitivity: float = 0.65, auto_block: bool = False):
        with self.lock:
            if self.active:
                return
            self.stop_event.clear()
            self.active = True
            self.error = ""
            self.interface = interface or "default"
            self.sensitivity = float(np.clip(sensitivity, 0.05, 0.99))
            self.auto_block = bool(auto_block)
            self.thread = threading.Thread(target=self._capture_worker, args=(interface,), daemon=True, name="shieldnet-capture")
            self.thread.start()

    def stop_capture(self):
        self.stop_event.set()
        thread = self.thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        with self.lock:
            self.active = False
            self.thread = None

    def _capture_worker(self, interface: str | None):
        try:
            from scapy.all import sniff
            last_flush = time.monotonic()
            while not self.stop_event.is_set():
                packets = sniff(iface=interface or None, timeout=0.5, store=True, filter="tcp or udp or icmp or icmp6")
                for packet in packets:
                    record = packet_to_record(packet)
                    if record:
                        self.ingest(record)
                if time.monotonic() - last_flush >= 2.0:
                    with self.lock:
                        sensitivity, auto_block = self.sensitivity, self.auto_block
                    self.flush_window(sensitivity, auto_block)
                    last_flush = time.monotonic()
            with self.lock:
                sensitivity, auto_block = self.sensitivity, self.auto_block
            self.flush_window(sensitivity, auto_block)
        except Exception as exc:
            LOG.exception("Capture failed")
            with self.lock:
                self.error = f"Capture error: {exc}. Check interface selection and capture permissions/Npcap."
        finally:
            with self.lock:
                self.active = False


def run_local_simulation(kind: str, port_count: int = 8, rate_per_second: int = 5, duration_seconds: float = 2.0) -> dict[str, Any]:
    """Generate small localhost-only traffic. No user-supplied target is accepted."""
    if kind not in {"Port scan", "SYN burst", "UDP traffic"}:
        raise ValueError("Unknown simulation")
    port_count = int(port_count)
    rate_per_second = int(rate_per_second)
    duration_seconds = float(duration_seconds)
    if not 1 <= port_count <= 32 or not 1 <= rate_per_second <= 10 or not 0.5 <= duration_seconds <= 5:
        raise ValueError("Simulation bounds are 1–32 ports, 1–10 packets/second and 0.5–5 seconds")
    target = "127.0.0.1"
    ports = list(range(1, port_count + 1))
    deadline = time.monotonic() + duration_seconds
    sent = 0
    interval = 1.0 / rate_per_second
    while sent < min(port_count, int(rate_per_second * duration_seconds)) and time.monotonic() < deadline:
        port = ports[sent % len(ports)]
        if kind in {"SYN burst", "Port scan"}:
            from scapy.all import IP, TCP, send
            send(IP(dst=target) / TCP(dport=port, flags="S"), count=1, inter=0, verbose=False)
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.sendto(b"ShieldNet local UDP simulation", (target, port))
            finally:
                sock.close()
        sent += 1
        if sent < min(port_count, int(rate_per_second * duration_seconds)):
            time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
    return {"kind": kind, "target": target, "packets_sent": sent, "duration_seconds": duration_seconds}

