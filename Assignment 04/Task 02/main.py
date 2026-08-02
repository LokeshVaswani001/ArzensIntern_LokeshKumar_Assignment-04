#!/usr/bin/env python3
"""
main.py
---------
Security Log Analyzer — CLI entry point.

An AI-assisted security automation tool that parses authentication/access
logs, detects suspicious behavior (brute-force logins, port scans, and
statistically anomalous activity volume), scores each source IP by risk,
and generates a human-readable and machine-readable report.

Usage:
    python main.py --log-file sample_logs/suspicious_activity.log
    python main.py --log-file mylog.log --output-json report.json --output-text report.txt
    python main.py --log-file mylog.log --strict   # abort on first malformed line

Exit codes:
    0 = ran successfully, no alerts
    1 = ran successfully, alerts found
    2 = error (bad arguments, missing file, parse failure in --strict mode)

Author: Lokesh Kumar
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from detectors import run_all_detectors
from log_parser import LogParseError, parse_log_file
from report_generator import generate_json_report, generate_text_report
from risk_scorer import compute_risk_profiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("security_log_analyzer")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Security Log Analyzer — detects brute-force logins, port "
                    "scans, and statistical volume anomalies in authentication/access logs.",
    )
    parser.add_argument("--log-file", required=True, help="Path to the log file to analyze")
    parser.add_argument("--output-text", default=None, help="Path to write the human-readable text report (default: print to stdout)")
    parser.add_argument("--output-json", default=None, help="Path to write the machine-readable JSON report")
    parser.add_argument("--strict", action="store_true", help="Abort on the first malformed log line instead of skipping it")
    return parser.parse_args(argv)


def validate_inputs(args: argparse.Namespace) -> None:
    """Fail fast with a clear error message for common input problems,
    rather than letting a cryptic exception surface later in the pipeline."""
    log_path = Path(args.log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {args.log_file}")
    if not log_path.is_file():
        raise ValueError(f"Log file path is not a regular file: {args.log_file}")
    if log_path.stat().st_size == 0:
        logger.warning("Log file is empty: %s — report will show zero events.", args.log_file)

    for out_path_str in (args.output_text, args.output_json):
        if out_path_str:
            out_path = Path(out_path_str)
            if out_path.parent != Path("") and not out_path.parent.exists():
                raise FileNotFoundError(f"Output directory does not exist: {out_path.parent}")


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        validate_inputs(args)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Input validation failed: %s", e)
        return 2

    try:
        events = parse_log_file(args.log_file, strict=args.strict)
    except LogParseError as e:
        logger.error("Strict parsing failed: %s", e)
        return 2
    except (FileNotFoundError, PermissionError) as e:
        logger.error("Could not read log file: %s", e)
        return 2
    except Exception as e:  # unexpected errors are still caught, logged, and reported cleanly
        logger.error("Unexpected error while parsing log file: %s", e)
        return 2

    if not events:
        logger.warning("No valid events parsed from %s.", args.log_file)

    alerts = run_all_detectors(events)
    profiles = compute_risk_profiles(alerts)

    text_report = generate_text_report(alerts, profiles, total_events=len(events), log_path=args.log_file)
    if args.output_text:
        Path(args.output_text).write_text(text_report, encoding="utf-8")
        logger.info("Wrote text report to %s", args.output_text)
    else:
        print(text_report)

    if args.output_json:
        json_report = generate_json_report(alerts, profiles, total_events=len(events), log_path=args.log_file)
        Path(args.output_json).write_text(json_report, encoding="utf-8")
        logger.info("Wrote JSON report to %s", args.output_json)

    return 1 if alerts else 0


if __name__ == "__main__":
    sys.exit(main())
