#!/usr/bin/env python3
"""
Extract TERR.Spacing values from an ArduPilot DataFlash .BIN log.

Usage:
  python3 terrain_log_spacing_check.py /path/to/log.BIN [expected_spacing_m]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


def add_local_pymavlink_to_path(repo_root: Path) -> None:
    pymavlink_parent = repo_root / "modules" / "mavlink"
    sys.path.insert(0, str(pymavlink_parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check TERR.Spacing in DataFlash logs")
    parser.add_argument("log", help="Path to DataFlash .BIN log")
    parser.add_argument(
        "expected_spacing",
        nargs="?",
        type=int,
        help="Optional expected spacing in meters (e.g. 100 or 30)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]

    if not log_path.exists():
        print(f"Missing log file: {log_path}")
        return 2

    add_local_pymavlink_to_path(repo_root)
    try:
        from pymavlink import mavutil  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"Failed to import pymavlink: {exc}")
        print("Tip: run this from inside an ArduPilot checkout with modules initialized.")
        return 2

    mlog = mavutil.mavlink_connection(str(log_path))
    spacing_counts: Counter[int] = Counter()
    terr_count = 0

    while True:
        msg = mlog.recv_match(blocking=False)
        if msg is None:
            break
        if msg.get_type() != "TERR":
            continue
        spacing = getattr(msg, "Spacing", None)
        if spacing is None:
            spacing = getattr(msg, "spacing", None)
        if spacing is None:
            continue
        terr_count += 1
        spacing_counts[int(spacing)] += 1

    print(f"Log: {log_path}")
    print(f"TERR messages: {terr_count}")
    if not spacing_counts:
        print("No TERR.Spacing values found.")
        return 1

    print("TERR spacing values:")
    for spacing in sorted(spacing_counts):
        print(f"  {spacing} m -> {spacing_counts[spacing]} messages")

    if args.expected_spacing is not None:
        mismatches = terr_count - spacing_counts[args.expected_spacing]
        print(f"Expected spacing: {args.expected_spacing} m")
        print(f"Messages not matching expected: {mismatches}")
        if mismatches > 0:
            print("RESULT: FAIL")
            return 1

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
