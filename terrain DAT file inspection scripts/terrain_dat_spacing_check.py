#!/usr/bin/env python3
"""
Check ArduPilot terrain .DAT tile spacing and block integrity.

Usage:
  python3 terrain_dat_spacing_check.py /path/to/NxxExxx.DAT [expected_spacing_m]
"""

from __future__ import annotations

import argparse
import binascii
import struct
from collections import Counter
from pathlib import Path

TERRAIN_GRID_FORMAT_VERSION = 1
IO_BLOCK_SIZE = 2048
IO_BLOCK_DATA_SIZE = 1821


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ArduPilot terrain .DAT spacing")
    parser.add_argument("dat", help="Path to .DAT file (not .gz)")
    parser.add_argument(
        "expected_spacing",
        nargs="?",
        type=int,
        help="Optional expected spacing in meters (e.g. 100 or 30)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dat_path = Path(args.dat).expanduser().resolve()

    if dat_path.suffix.lower() == ".gz":
        print("Input looks gzip-compressed (.gz). Please decompress to .DAT first.")
        return 2
    if not dat_path.exists():
        print(f"Missing file: {dat_path}")
        return 2

    data = dat_path.read_bytes()
    file_size = len(data)
    if file_size == 0:
        print("File is empty.")
        return 2

    if file_size % IO_BLOCK_SIZE == 0:
        block_size = IO_BLOCK_SIZE
    elif file_size % IO_BLOCK_DATA_SIZE == 0:
        block_size = IO_BLOCK_DATA_SIZE
    else:
        print(
            f"Unsupported file length ({file_size}). Expected multiple of {IO_BLOCK_SIZE} "
            f"(padded) or {IO_BLOCK_DATA_SIZE} (unpadded)."
        )
        return 2

    total_blocks = 0
    valid_blocks = 0
    empty_blocks = 0
    header_errors = 0
    crc_errors = 0
    spacing_mismatch_blocks = 0
    spacing_counts: Counter[int] = Counter()

    for pos in range(0, file_size, block_size):
        block = data[pos : pos + block_size]
        if len(block) < IO_BLOCK_DATA_SIZE:
            header_errors += 1
            break
        total_blocks += 1

        try:
            bitmap, lat, lon, crc, version, spacing = struct.unpack_from("<QiiHHH", block, 0)
        except struct.error:
            header_errors += 1
            continue

        if bitmap == 0 and lat == 0 and lon == 0 and crc == 0 and version == 0 and spacing == 0:
            empty_blocks += 1
            continue

        if version != TERRAIN_GRID_FORMAT_VERSION:
            header_errors += 1
            continue

        crc_data = bytearray(block[:IO_BLOCK_DATA_SIZE])
        crc_data[16:18] = b"\x00\x00"
        calc_crc = binascii.crc_hqx(crc_data, 0)
        if calc_crc != crc:
            crc_errors += 1
            continue

        valid_blocks += 1
        spacing_counts[spacing] += 1
        if args.expected_spacing is not None and spacing != args.expected_spacing:
            spacing_mismatch_blocks += 1

    unique_spacings = sorted(spacing_counts.keys())

    print(f"File: {dat_path}")
    print(f"Size: {file_size} bytes")
    print(f"Block layout: {block_size}-byte blocks")
    print(f"Total blocks: {total_blocks}")
    print(f"Valid blocks: {valid_blocks}")
    print(f"Empty blocks: {empty_blocks}")
    print(f"Header errors: {header_errors}")
    print(f"CRC errors: {crc_errors}")

    if not unique_spacings:
        print("Spacing values: none (no valid terrain blocks)")
    else:
        print("Spacing values:")
        for spacing in unique_spacings:
            print(f"  {spacing} m -> {spacing_counts[spacing]} blocks")

    if args.expected_spacing is not None:
        print(f"Expected spacing: {args.expected_spacing} m")
        print(f"Blocks not matching expected: {spacing_mismatch_blocks}")

    ok = True
    if valid_blocks == 0:
        ok = False
    if header_errors > 0 or crc_errors > 0:
        ok = False
    if len(unique_spacings) > 1:
        ok = False
    if args.expected_spacing is not None and spacing_mismatch_blocks > 0:
        ok = False

    if ok:
        print("RESULT: PASS")
        return 0
    print("RESULT: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
