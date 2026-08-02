"""
report_generator.py
----------------------
Renders the results of a log analysis run (alerts + risk profiles) into a
human-readable text report and a machine-readable JSON report, so the tool
is usable both by a human analyst and by downstream automation (e.g. piping
into a SIEM or ticketing system).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import List

from detectors import Alert
from risk_scorer import RiskProfile


def generate_text_report(alerts: List[Alert], profiles: List[RiskProfile],
                          total_events: int, log_path: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "=" * 60,
        "  SECURITY LOG ANALYSIS REPORT",
        "=" * 60,
        f"Log file analyzed : {log_path}",
        f"Generated         : {ts}",
        f"Total events      : {total_events}",
        f"Total alerts      : {len(alerts)}",
        f"Source IPs flagged: {len(profiles)}",
        "",
    ]

    if not profiles:
        lines.append("No suspicious activity detected. No source IPs exceeded any detection threshold.")
    else:
        lines.append("RISK RANKING (highest risk first)")
        lines.append("-" * 60)
        for p in profiles:
            lines.append(f"[{p.risk_level}] {p.source_ip}  (score={p.risk_score}, {p.alert_count} alert(s))")
            lines.append(f"    Alert types: {', '.join(p.alert_types)}")
        lines.append("")
        lines.append("ALERT DETAIL")
        lines.append("-" * 60)
        for a in sorted(alerts, key=lambda x: SEVERITY_ORDER.get(x.severity, 0), reverse=True):
            lines.append(f"[{a.severity.upper()}] {a.alert_type} — {a.source_ip}")
            lines.append(f"    {a.description}")
            lines.append(f"    Window: {a.window_start} -> {a.window_end}")
            lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}


def generate_json_report(alerts: List[Alert], profiles: List[RiskProfile],
                          total_events: int, log_path: str) -> str:
    payload = {
        "log_file": log_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_events": total_events,
        "total_alerts": len(alerts),
        "risk_profiles": [asdict(p) for p in profiles],
        "alerts": [asdict(a) for a in alerts],
    }
    return json.dumps(payload, indent=2)
