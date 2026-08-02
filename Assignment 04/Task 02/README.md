# Security Log Analyzer

An AI-assisted security automation tool that parses authentication and
access logs, detects suspicious behavior, scores risk per source IP, and
generates human- and machine-readable reports.

Built for Task 02 — Practical Security Automation Project (AI, Automation
& Security Engineering track).

## What it does

1. **Parses** raw log lines into structured events (`log_parser.py`).
2. **Detects** three kinds of suspicious behavior (`detectors.py`):
   - **Brute-force logins**: ≥5 failed login attempts from one IP within 60 seconds.
   - **Port scans**: ≥10 port-probe events from one IP within 30 seconds.
   - **Statistical volume anomalies**: any source IP whose total event count is a
     z-score ≥3.0 outlier relative to all other sources in the log — this
     is the "AI-assisted" layer that catches unusual volume even when it
     doesn't match a predefined rule.
3. **Scores risk** per source IP as a transparent, auditable weighted sum
   of alert severities (`risk_scorer.py`) — not a black-box score, so an
   analyst can see exactly why an IP was ranked the way it was.
4. **Reports** results as readable text and structured JSON (`report_generator.py`).

## Project structure

```
security_log_analyzer/
├── main.py                 # CLI entry point — orchestrates the pipeline
├── log_parser.py           # Raw log line -> LogEvent
├── detectors.py             # LogEvent list -> Alert list
├── risk_scorer.py           # Alert list -> per-IP RiskProfile
├── report_generator.py      # Alerts + profiles -> text/JSON report
└── sample_logs/
    └── suspicious_activity.log   # Sample log with embedded attack patterns
```

Each module has a single responsibility and depends only on the modules
below it in the pipeline — `main.py` is the only file that imports from
all of them, so any module can be tested or reused independently.

## Usage

```bash
# Basic run — prints a text report to stdout
python main.py --log-file sample_logs/suspicious_activity.log

# Save both a text and a JSON report
python main.py --log-file sample_logs/suspicious_activity.log \
    --output-text report.txt --output-json report.json

# Strict mode: abort on the first malformed log line instead of skipping it
python main.py --log-file sample_logs/suspicious_activity.log --strict
```

**Exit codes:** `0` = ran successfully, no alerts. `1` = ran successfully,
alerts found. `2` = an error occurred (bad arguments, missing file, or a
parse failure in `--strict` mode) — useful for scripting/CI integration.

## Log format expected

```
<ISO8601 timestamp> <source_ip> <event_type> <username> <status> [<extra>]
```
`event_type` ∈ {`login`, `file_access`, `port_scan`, `sudo`}, `status` ∈
{`success`, `failure`}. Lines starting with `#` are treated as comments and
skipped; blank lines are skipped.

Example:
```
2026-08-02T00:20:00Z 203.0.113.77 login admin failure
2026-08-02T00:20:05Z 203.0.113.77 login admin failure
```

## Input validation & error handling

- Missing/unreadable log file, non-existent output directory, and empty
  log files are all caught with clear, specific error messages before any
  processing begins (`validate_inputs()` in `main.py`).
- Malformed individual log lines are, by default, logged as warnings and
  skipped rather than crashing the whole run — a single bad line in a
  10,000-line production log shouldn't prevent analysis of the other
  9,999. `--strict` mode is available when the caller wants the opposite
  behavior (e.g. validating a log source's format compliance).
- All file I/O is wrapped in explicit `try/except` blocks distinguishing
  `FileNotFoundError` from `PermissionError` from other unexpected errors,
  so the failure message tells the user what actually went wrong.

## Testing

The tool was tested against:
- A realistic sample log (`sample_logs/suspicious_activity.log`) containing
  normal background traffic, an embedded brute-force burst, an embedded
  port-scan burst, and one intentionally malformed line — all correctly
  detected/handled.
- A non-existent file path (correctly rejected with exit code 2).
- An empty log file (correctly handled with a "no events" report, exit code 0).
- Strict mode against the malformed line (correctly aborts with exit code 2).
- A clean log with no suspicious activity (correctly reports zero alerts, exit code 0).

## Extending this tool

- Add a new detector: write a function in `detectors.py` that takes
  `List[LogEvent]` and returns `List[Alert]`, then add it to
  `run_all_detectors()`.
- Add a new log format: write a new parser function in `log_parser.py`
  that returns `LogEvent` objects; the rest of the pipeline is
  format-agnostic.
- Add a new output format (e.g. CSV, Slack webhook): add a function to
  `report_generator.py` following the existing `generate_*_report()` pattern.
