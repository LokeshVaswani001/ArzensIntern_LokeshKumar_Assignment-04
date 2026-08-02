"""
detectors.py
--------------
Detection logic for the security log analyzer.

Combines simple, explainable rule-based detectors (thresholds on counts
within a time window) with a lightweight statistical anomaly score (z-score
on per-source event rate), reflecting the kind of "AI-assisted" hybrid
approach discussed in the accompanying research report: rules give
predictable, auditable coverage of known attack patterns, while the
statistical layer flags unusual volume even when it doesn't match a
predefined rule.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List

from log_parser import LogEvent

# --- Configurable detection thresholds ---
BRUTE_FORCE_FAILURE_THRESHOLD = 5      # failed logins from one IP...
BRUTE_FORCE_WINDOW_SECONDS = 60        # ...within this many seconds
PORT_SCAN_EVENT_THRESHOLD = 10         # port_scan events from one IP...
PORT_SCAN_WINDOW_SECONDS = 30          # ...within this many seconds
Z_SCORE_ALERT_THRESHOLD = 3.0          # statistical anomaly cutoff


@dataclass
class Alert:
    alert_type: str
    severity: str          # "low", "medium", "high"
    source_ip: str
    description: str
    event_count: int
    window_start: str
    window_end: str


def _events_by_ip(events: List[LogEvent]) -> Dict[str, List[LogEvent]]:
    grouped: Dict[str, List[LogEvent]] = defaultdict(list)
    for e in events:
        grouped[e.source_ip].append(e)
    for ip in grouped:
        grouped[ip].sort(key=lambda e: e.timestamp)
    return grouped


def detect_brute_force(events: List[LogEvent]) -> List[Alert]:
    """Flag any source IP with >= BRUTE_FORCE_FAILURE_THRESHOLD failed
    login attempts within a BRUTE_FORCE_WINDOW_SECONDS sliding window."""
    alerts = []
    by_ip = _events_by_ip([e for e in events if e.event_type == "login" and e.status == "failure"])

    for ip, ip_events in by_ip.items():
        window = timedelta(seconds=BRUTE_FORCE_WINDOW_SECONDS)
        start_idx = 0
        for end_idx in range(len(ip_events)):
            while ip_events[end_idx].timestamp - ip_events[start_idx].timestamp > window:
                start_idx += 1
            count = end_idx - start_idx + 1
            if count >= BRUTE_FORCE_FAILURE_THRESHOLD:
                alerts.append(Alert(
                    alert_type="brute_force_login",
                    severity="high",
                    source_ip=ip,
                    description=(
                        f"{count} failed login attempts from {ip} within "
                        f"{BRUTE_FORCE_WINDOW_SECONDS}s (target user(s): "
                        f"{', '.join(sorted(set(e.username for e in ip_events[start_idx:end_idx+1])))})"
                    ),
                    event_count=count,
                    window_start=ip_events[start_idx].timestamp.isoformat(),
                    window_end=ip_events[end_idx].timestamp.isoformat(),
                ))
                break  # one alert per IP is enough; avoid duplicate overlapping alerts
    return alerts


def detect_port_scan(events: List[LogEvent]) -> List[Alert]:
    """Flag any source IP with >= PORT_SCAN_EVENT_THRESHOLD port_scan
    events within a PORT_SCAN_WINDOW_SECONDS sliding window."""
    alerts = []
    by_ip = _events_by_ip([e for e in events if e.event_type == "port_scan"])

    for ip, ip_events in by_ip.items():
        window = timedelta(seconds=PORT_SCAN_WINDOW_SECONDS)
        start_idx = 0
        for end_idx in range(len(ip_events)):
            while ip_events[end_idx].timestamp - ip_events[start_idx].timestamp > window:
                start_idx += 1
            count = end_idx - start_idx + 1
            if count >= PORT_SCAN_EVENT_THRESHOLD:
                alerts.append(Alert(
                    alert_type="port_scan",
                    severity="medium",
                    source_ip=ip,
                    description=f"{count} port-probe events from {ip} within {PORT_SCAN_WINDOW_SECONDS}s",
                    event_count=count,
                    window_start=ip_events[start_idx].timestamp.isoformat(),
                    window_end=ip_events[end_idx].timestamp.isoformat(),
                ))
                break
    return alerts


def detect_statistical_anomalies(events: List[LogEvent]) -> List[Alert]:
    """
    Flag source IPs whose total event count is a statistical outlier
    relative to the population of all source IPs seen in this log
    (z-score based). This complements the rule-based detectors above by
    catching unusually high-volume sources that don't match a specific
    known pattern.
    """
    by_ip = _events_by_ip(events)
    counts = {ip: len(evts) for ip, evts in by_ip.items()}

    if len(counts) < 2:
        return []  # not enough data points for a meaningful z-score

    values = list(counts.values())
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    if stdev == 0:
        return []

    alerts = []
    for ip, count in counts.items():
        z = (count - mean) / stdev
        if z >= Z_SCORE_ALERT_THRESHOLD:
            ip_events = by_ip[ip]
            alerts.append(Alert(
                alert_type="statistical_volume_anomaly",
                severity="low",
                source_ip=ip,
                description=(
                    f"{ip} generated {count} events (z-score={z:.2f} vs. "
                    f"population mean={mean:.1f}, stdev={stdev:.1f}) — "
                    f"unusually high activity volume"
                ),
                event_count=count,
                window_start=ip_events[0].timestamp.isoformat(),
                window_end=ip_events[-1].timestamp.isoformat(),
            ))
    return alerts


def run_all_detectors(events: List[LogEvent]) -> List[Alert]:
    """Run all detectors and return a combined, de-duplicated alert list."""
    alerts: List[Alert] = []
    alerts.extend(detect_brute_force(events))
    alerts.extend(detect_port_scan(events))
    alerts.extend(detect_statistical_anomalies(events))
    return alerts
