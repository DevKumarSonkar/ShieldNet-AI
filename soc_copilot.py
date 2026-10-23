"""AI-Powered SOC Copilot & Incident Forensics Engine for ShieldNet AI."""
from __future__ import annotations

import json
from typing import Any


def generate_forensic_investigation(incident: dict[str, Any]) -> dict[str, Any]:
    """
    Synthesizes deep threat intelligence, root-cause forensic rationale,
    tailored multi-platform firewall commands, and executive briefing notes.
    """
    category = incident.get("category", "Zero-Day")
    threat_type = incident.get("threat_type", "Network Intrusion")
    src_ip = incident.get("source_ip", "0.0.0.0")
    dst = incident.get("destination", "10.0.1.10:80")
    proto_srv = incident.get("protocol_service", "TCP / http")
    confidence = incident.get("confidence", "95%")
    severity = incident.get("severity", "HIGH")
    country = incident.get("country", "Unknown")
    city = incident.get("city", "Unknown City")
    isp = incident.get("isp", "Unknown Provider")
    flag = incident.get("country_flag", "🌐")
    triggered = incident.get("triggered_params", [])
    raw = incident.get("raw_attributes", {})

    count = raw.get("count", 0)
    serror = raw.get("serror_rate", 0.0)
    tcp_flag = raw.get("flag", "SF")
    payload = raw.get("src_bytes", 0)

    # 1. Attack Playbook & Root-Cause Forensic Analysis
    if category == "DoS":
        playbook_title = "TCP SYN Resource Exhaustion (State Exhaustion Flood)"
        attack_intent = (
            f"Adversary initiated a volumetric half-open connection burst targeting `{dst}`. "
            f"By transmitting TCP SYN packets with abnormal state flags (`{tcp_flag}`) and zero completion acknowledgments "
            f"({serror * 100:.1f}% SYN error rate), the target host's TCP transmission control block (TCB) queue "
            f"and connection backlog table are filled to maximum capacity, causing resource starvation for legitimate traffic."
        )
        mitre_technique = "T1498.001 - Network Denial of Service: Direct Network Flood"
        cve_relevance = "Affects unhardened TCP listeners, HTTP/HTTPS web application servers, and DMZ endpoints."
        immediate_action = "Drop incoming SYN packets from source IP and activate SYN Proxy / TCP Cookie filtering."

    elif category == "Probe":
        playbook_title = "Reconnaissance & Boundary Service Mapping"
        attack_intent = (
            f"Adversary executed systematic boundary surveillance across destination `{dst}`. "
            f"The scanner evaluated response variance across services and transport ports. The high error/reset rate "
            f"indicates automated scanner sweeps (e.g., Nmap, Shodan, or Masscan) enumerating listening daemon versions "
            f"and firewall boundary filtering policies prior to targeted vulnerability exploitation."
        )
        mitre_technique = "T1595.001 - Active Scanning: Scanning IP Blocks & Port Enumeration"
        cve_relevance = "Pre-attack reconnaissance targeting unauthenticated management ports and exposed daemons."
        immediate_action = "Blacklist remote IP at perimeter edge firewall and drop ICMP echo / TCP SYN probe responses."

    elif category == "R2L":
        playbook_title = "Unauthorized Remote Authentication & Infiltration"
        attack_intent = (
            f"Adversary targeted administrative/remote access service `{proto_srv}` on `{dst}`. "
            f"The detection signature identifies repeated failed credential authentication validations and unauthorized "
            f"session negotiation attempts without completing authentication, indicating dictionary attacks, credential stuffing, "
            f"or password spraying against exposed remote interfaces."
        )
        mitre_technique = "T1110.001 - Brute Force: Password Spraying & Credential Guessing"
        cve_relevance = "Exposed SSH (22), FTP (21), Telnet (23), or database ports without rate-limiting."
        immediate_action = "Enforce IP Jail jail-time via Fail2ban, revoke active access tokens, and enforce Multi-Factor Authentication."

    elif category == "U2R":
        playbook_title = "Privilege Escalation & Unauthorized Root Execution"
        attack_intent = (
            f"Host telemetry detected anomalous system execution attempting to elevate unprivileged session context "
            f"to superuser/root authority (`root_shell` or `su` invocation). The anomalous source payload volume indicates "
            f"buffer overflow exploitation or binary vulnerability hijacking targeting local daemons."
        )
        mitre_technique = "T1068 - Exploitation for Privilege Escalation"
        cve_relevance = "Local kernel vulnerabilities, setuid/setgid binary misconfigurations, and vulnerable service daemons."
        immediate_action = "CRITICAL: Immediately isolate host subnet, terminate rogue process threads, and dump memory forensics."

    else:
        playbook_title = "Zero-Day Anomaly & Statistical Variance"
        attack_intent = (
            f"Connection vector against `{dst}` deviated significantly from historical baseline parameters across "
            f"high-dimensional feature distributions. The interaction of traffic volume, packet sizing, and connection timing "
            f"triggered algorithmic intrusion confidence ({confidence}) without matching known heuristic signatures."
        )
        mitre_technique = "T1203 - Exploitation for Client Execution / Unclassified Novel Vector"
        cve_relevance = "Novel zero-day network vulnerability variants or protocol fuzzing sweeps."
        immediate_action = "Route traffic through deep packet inspection (DPI) quarantine sandbox and capture full PCAP stream."

    # 2. Multi-Platform Firewall Command Generator
    mitigation_commands = {
        "windows_netsh": f'netsh advfirewall firewall add rule name="ShieldNet-Block-{src_ip}" dir=in action=block remoteip={src_ip}',
        "linux_iptables": f"iptables -I INPUT -s {src_ip} -j DROP",
        "linux_nftables": f"nft add rule inet filter input ip saddr {src_ip} drop",
        "linux_fail2ban": f"fail2ban-client set <JAIL_NAME> banip {src_ip}",
        "nginx_block": f"deny {src_ip};",
        "cisco_acl": f"access-list 100 deny ip host {src_ip} any",
    }

    # 3. Plain-English Executive Summary
    executive_briefing = (
        f"**EXECUTIVE INCIDENT BRIEFING [{incident.get('incident_id', 'INC-NIDS')}]:**\n\n"
        f"At `{incident.get('timestamp')}`, ShieldNet AI detected a **{severity}** security anomaly originating from "
        f"**{flag} {src_ip} ({city}, {country})** affiliated with **{isp}**. "
        f"The adversary initiated **{threat_type}** against production endpoint `{dst}` ({proto_srv}) with a model confidence of **{confidence}**.\n\n"
        f"**Root Cause:** {attack_intent}\n\n"
        f"**Impact Assessment:** Risk of service degradation, credential compromise, or host compromise. "
        f"Immediate enforcement of perimeter filtering is strongly recommended."
    )

    return {
        "playbook_title": playbook_title,
        "mitre_technique": mitre_technique,
        "cve_relevance": cve_relevance,
        "attack_intent": attack_intent,
        "immediate_action": immediate_action,
        "commands": mitigation_commands,
        "executive_briefing": executive_briefing,
    }


def answer_copilot_query(query: str, incident: dict[str, Any] | None, recent_incidents: list[dict[str, Any]]) -> str:
    """Answers operator queries in real time regarding threats and network security."""
    q_lower = query.lower()

    if incident is None and recent_incidents:
        incident = recent_incidents[0]

    if not incident:
        return "No incidents have been detected yet in this monitoring session. Ingest incoming traffic or simulate an attack to analyze forensic telemetry."

    category = incident.get("category", "General")
    src_ip = incident.get("source_ip", "Unknown")
    threat = incident.get("threat_type", "Anomaly")
    country = incident.get("country", "Unknown")
    isp = incident.get("isp", "Unknown Provider")
    dst = incident.get("destination", "Target Endpoint")
    flag = incident.get("country_flag", "🌐")

    if "why" in q_lower or "reason" in q_lower or "detect" in q_lower:
        triggered_names = [p["name"] for p in incident.get("triggered_params", [])]
        signatures = ", ".join(triggered_names) if triggered_names else "High-dimensional feature correlation"
        return (
            f"**Why was `{src_ip}` flagged?**\n\n"
            f"The Random Forest model evaluated 122 input features and determined a **{incident.get('confidence', '90%')}** probability of malicious intent. "
            f"The primary triggered parameters were: **{signatures}**.\n\n"
            f"This profile matches known `{category}` attack vectors designed to degrade or compromise `{dst}`."
        )

    elif "how" in q_lower or "block" in q_lower or "mitigat" in q_lower or "firewall" in q_lower:
        return (
            f"**Recommended Mitigation for `{src_ip}`:**\n\n"
            f"1. **Perimeter Firewall**: Execute OS firewall drop immediately.\n"
            f"   - **Windows**: `netsh advfirewall firewall add rule name=\"Block-{src_ip}\" dir=in action=block remoteip={src_ip}`\n"
            f"   - **Linux**: `iptables -I INPUT -s {src_ip} -j DROP`\n"
            f"2. **Session Reset**: Sever any half-open TCP connection states.\n"
            f"3. **Rate Limiting**: Enforce a SYN rate-limit of 5 connections/second from subnet `{src_ip.rsplit('.', 1)[0]}.0/24`."
        )

    elif "vulnerab" in q_lower or "risk" in q_lower or "impact" in q_lower:
        if category == "DoS":
            return f"**Vulnerability Impact:** The target `{dst}` risks connection starvation, thread pool exhaustion in web servers (Nginx/IIS/Apache), and unresponsive application gateways."
        elif category == "Probe":
            return f"**Vulnerability Impact:** The adversary has mapped open ports and service banners on `{dst}`. High risk of follow-up exploit payloads targeted at identified daemon versions."
        elif category == "R2L":
            return f"**Vulnerability Impact:** Exposed authentication services on `{dst}` risk credential discovery. Accounts without MFA or lockout thresholds are acutely vulnerable."
        else:
            return f"**Vulnerability Impact:** Elevated risk of host infiltration or privilege compromise. Immediate network isolation of `{dst}` is recommended."

    elif "origin" in q_lower or "where" in q_lower or "location" in q_lower or "geo" in q_lower:
        return (
            f"**Geospatial Intelligence for `{src_ip}`:**\n\n"
            f"- **Country:** {flag} {country}\n"
            f"- **City:** {incident.get('city', 'Unknown')}\n"
            f"- **Provider / ISP:** {isp}\n"
            f"- **Coordinates:** Latitude `{incident.get('lat', 'N/A')}`, Longitude `{incident.get('lon', 'N/A')}`\n\n"
            f"Traffic originating from this hosting provider or autonomous system has exhibited aggressive `{threat}` signatures."
        )

    else:
        return (
            f"**SOC Copilot Analysis Summary for `{src_ip}` ({threat}):**\n\n"
            f"- **Origin:** {flag} {country} ({isp})\n"
            f"- **Target:** `{dst}`\n"
            f"- **Threat Family:** `{category}` (Severity: **{incident.get('severity')}**)\n"
            f"- **Recommendation:** {incident.get('recommended_mitigation')}\n\n"
            f"You can ask me: *'Why was this detected?'*, *'How do I block this IP?'*, *'What is our vulnerability risk?'*, or *'Show geographic origin'*"
        )
