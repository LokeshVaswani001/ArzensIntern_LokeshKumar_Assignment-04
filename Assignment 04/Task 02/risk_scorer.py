"""
risk_scorer.py
----------------
Combines detector alerts into a single, human-interpretable risk score per
source IP, so an analyst can triage by priority rather than reading every
individual alert.

Scoring is a simple, transparent weighted sum (not a black-box model) —
appropriate for a small automation tool where analysts need to trust and
audit exactly why a score was assigned, per the "Ethical and Responsible AI
Usage" principle of explainability discussed in the research report.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from detectors import Alert

SEVERITY_WEIGHTS = {"low": 1, "medium": 3, "high": 5}


@dataclass
class RiskProfile:
    source_ip: str
    risk_score: int
    risk_level: str
    alert_count: int
    alert_types: List[str]


def _risk_level_for_score(score: int) -> str:
    if score >= 8:
        return "CRITICAL"
    if score >= 5:
        return "HIGH"
    if score >= 3:
        return "MEDIUM"
    return "LOW"


def compute_risk_profiles(alerts: List[Alert]) -> List[RiskProfile]:
    """Aggregate alerts by source IP into a sorted list of RiskProfiles,
    highest risk first."""
    by_ip: Dict[str, List[Alert]] = defaultdict(list)
    for alert in alerts:
        by_ip[alert.source_ip].append(alert)

    profiles = []
    for ip, ip_alerts in by_ip.items():
        score = sum(SEVERITY_WEIGHTS[a.severity] for a in ip_alerts)
        profiles.append(RiskProfile(
            source_ip=ip,
            risk_score=score,
            risk_level=_risk_level_for_score(score),
            alert_count=len(ip_alerts),
            alert_types=sorted(set(a.alert_type for a in ip_alerts)),
        ))

    profiles.sort(key=lambda p: p.risk_score, reverse=True)
    return profiles
