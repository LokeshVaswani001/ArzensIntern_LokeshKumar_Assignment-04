"""
log_parser.py
---------------
Parses raw authentication/access log lines into structured LogEvent
records. Supports a simple, common log line format and is designed to be
extended with additional parsers (e.g. syslog, JSON logs) without changing
the rest of the pipeline.

Expected line format:
    <ISO8601 timestamp> <source_ip> <event_type> <username> <status> [<extra>]

Example:
    2026-08-02T03:14:05Z 192.168.1.45 login alice failure
    2026-08-02T03:14:07Z 192.168.1.45 login alice failure
    2026-08-02T09:02:11Z 10.0.0.12 login bob success
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

VALID_EVENT_TYPES = {"login", "file_access", "port_scan", "sudo"}
VALID_STATUSES = {"success", "failure"}


class LogParseError(ValueError):
    """Raised when a log line cannot be parsed into a valid LogEvent."""


@dataclass
class LogEvent:
    timestamp: datetime
    source_ip: str
    event_type: str
    username: str
    status: str
    extra: Optional[str] = None
    raw_line: str = ""


def parse_line(line: str, line_number: int = -1) -> Optional[LogEvent]:
    """
    Parse a single raw log line into a LogEvent.

    Returns None for blank lines or comment lines (starting with '#'),
    and raises LogParseError for malformed non-blank lines so the caller
    can decide how to handle bad data (skip, log, or abort).
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    parts = stripped.split()
    if len(parts) < 5:
        raise LogParseError(
            f"Line {line_number}: expected at least 5 whitespace-separated "
            f"fields (timestamp, source_ip, event_type, username, status), "
            f"got {len(parts)}: {stripped!r}"
        )

    ts_str, source_ip, event_type, username, status = parts[:5]
    extra = " ".join(parts[5:]) if len(parts) > 5 else None

    try:
        timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError as e:
        raise LogParseError(f"Line {line_number}: invalid timestamp {ts_str!r} ({e})") from e

    if event_type not in VALID_EVENT_TYPES:
        raise LogParseError(
            f"Line {line_number}: unknown event_type {event_type!r}; "
            f"expected one of {sorted(VALID_EVENT_TYPES)}"
        )
    if status not in VALID_STATUSES:
        raise LogParseError(
            f"Line {line_number}: unknown status {status!r}; "
            f"expected one of {sorted(VALID_STATUSES)}"
        )
    if not source_ip:
        raise LogParseError(f"Line {line_number}: empty source_ip")

    return LogEvent(
        timestamp=timestamp, source_ip=source_ip, event_type=event_type,
        username=username, status=status, extra=extra, raw_line=stripped,
    )


def parse_log_file(path: str, strict: bool = False) -> List[LogEvent]:
    """
    Parse an entire log file into a list of LogEvent objects.

    If strict=True, the first malformed line raises LogParseError.
    If strict=False (default), malformed lines are logged as warnings and
    skipped, so a single bad line doesn't abort analysis of an otherwise
    valid log file.
    """
    events: List[LogEvent] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    event = parse_line(line, line_number)
                    if event is not None:
                        events.append(event)
                except LogParseError as e:
                    if strict:
                        raise
                    logger.warning("Skipping malformed line: %s", e)
    except FileNotFoundError:
        raise FileNotFoundError(f"Log file not found: {path}")
    except PermissionError:
        raise PermissionError(f"Permission denied reading log file: {path}")

    logger.info("Parsed %d valid events from %s", len(events), path)
    return events
