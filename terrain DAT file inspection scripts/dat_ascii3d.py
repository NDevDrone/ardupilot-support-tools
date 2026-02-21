#!/usr/bin/env python3
"""Render ArduPilot DAT terrain as readable ASCII height/shade maps."""

from __future__ import annotations

import argparse
import binascii
import shutil
import struct
from collections import defaultdict
from pathlib import Path

IO_BLOCK_SIZE = 2048
IO_BLOCK_DATA_SIZE = 1821
TERRAIN_GRID_FORMAT_VERSION = 1
GRID_N = 28
GRID_E = 32
BLOCK_SPACING_N = 24
BLOCK_SPACING_E = 28
H_RAW_COUNT = GRID_N * GRID_E
RAMP_HEIGHT = " .:-=+*#%@"
RAMP_SHADE = " .,:;irsXA253hMHGS#9B&@"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render DAT terrain as ASCII")
    p.add_argument("dat", help="Path to .DAT (decompressed)")
    p.add_argument("--w", type=int, default=84, help="Output width in chars")
    p.add_argument("--h", type=int, default=28, help="Output height in chars")
    p.add_argument("--mode", choices=["both", "height", "shade"], default="both")
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive pan mode (arrow keys move view, q quits)",
    )
    p.add_argument(
        "--fit-terminal",
        action="store_true",
        help="Clamp render width so ASCII rows fit current terminal width",
    )
    p.add_argument(
        "--frame",
        action="store_true",
        help="Draw |...| frame around rows to make width and trailing spaces visible",
    )
    return p.parse_args()


def iter_valid_blocks(data: bytes):
    if len(data) % IO_BLOCK_SIZE == 0:
        block_size = IO_BLOCK_SIZE
    elif len(data) % IO_BLOCK_DATA_SIZE == 0:
        block_size = IO_BLOCK_DATA_SIZE
    else:
        raise ValueError("Unsupported DAT file size.")

    for pos in range(0, len(data), block_size):
        block = data[pos : pos + block_size]
        if len(block) < IO_BLOCK_DATA_SIZE:
            continue
        bitmap, lat, lon, crc, version, spacing = struct.unpack_from("<QiiHHH", block, 0)
        if bitmap == 0 and lat == 0 and lon == 0 and crc == 0 and version == 0 and spacing == 0:
            continue
        if version != TERRAIN_GRID_FORMAT_VERSION:
            continue

        crc_data = bytearray(block[:IO_BLOCK_DATA_SIZE])
        crc_data[16:18] = b"\x00\x00"
        if binascii.crc_hqx(crc_data, 0) != crc:
            continue

        heights = struct.unpack_from("<" + str(H_RAW_COUNT) + "h", block, 22)
        grid_idx_n, grid_idx_e = struct.unpack_from("<HH", block, 22 + H_RAW_COUNT * 2)
        yield grid_idx_n, grid_idx_e, spacing, heights


def build_height_grid(data: bytes):
    sums = defaultdict(float)
    counts = defaultdict(int)
    spacing = None
    blocks = 0

    for grid_idx_n, grid_idx_e, sp, heights in iter_valid_blocks(data):
        blocks += 1
        if spacing is None:
            spacing = sp
        for n in range(GRID_N):
            row_base = n * GRID_E
            gn = grid_idx_n * BLOCK_SPACING_N + n
            for e in range(GRID_E):
                ge = grid_idx_e * BLOCK_SPACING_E + e
                key = (gn, ge)
                sums[key] += heights[row_base + e]
                counts[key] += 1

    if not sums:
        raise ValueError("No valid terrain points found in DAT.")

    ns = sorted({k[0] for k in sums})
    es = sorted({k[1] for k in sums})
    n_map = {v: i for i, v in enumerate(ns)}
    e_map = {v: i for i, v in enumerate(es)}
    grid = [[0.0 for _ in es] for _ in ns]
    for (n, e), total in sums.items():
        grid[n_map[n]][e_map[e]] = total / counts[(n, e)]
    return grid, spacing, blocks


def resize_nearest(grid, out_h: int, out_w: int):
    h = len(grid)
    w = len(grid[0])
    out_h = max(1, min(out_h, h))
    out_w = max(1, min(out_w, w))
    out = []
    for oy in range(out_h):
        iy = int(round(oy * (h - 1) / max(1, out_h - 1)))
        row = []
        for ox in range(out_w):
            ix = int(round(ox * (w - 1) / max(1, out_w - 1)))
            row.append(grid[iy][ix])
        out.append(row)
    return out


def quantile(sorted_vals, q: float):
    idx = int(round((len(sorted_vals) - 1) * q))
    return sorted_vals[max(0, min(len(sorted_vals) - 1, idx))]


def map_to_ascii(matrix, ramp: str, robust: bool):
    flat = [v for row in matrix for v in row]
    sorted_vals = sorted(flat)
    if robust and len(sorted_vals) > 20:
        lo = quantile(sorted_vals, 0.02)
        hi = quantile(sorted_vals, 0.98)
    else:
        lo = sorted_vals[0]
        hi = sorted_vals[-1]
    span = max(1e-9, hi - lo)

    lines = []
    for row in matrix:
        chars = []
        for v in row:
            t = (v - lo) / span
            if t < 0:
                t = 0
            elif t > 1:
                t = 1
            i = int(round(t * (len(ramp) - 1)))
            chars.append(ramp[i])
        lines.append("".join(chars))
    return lines


def hillshade(matrix):
    h = len(matrix)
    w = len(matrix[0])
    # Light from NW, slight upward component.
    lx, ly, lz = -0.65, -0.65, 0.4
    ln = (lx * lx + ly * ly + lz * lz) ** 0.5
    lx, ly, lz = lx / ln, ly / ln, lz / ln

    out = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        ym = max(0, y - 1)
        yp = min(h - 1, y + 1)
        for x in range(w):
            xm = max(0, x - 1)
            xp = min(w - 1, x + 1)
            dzdx = (matrix[y][xp] - matrix[y][xm]) * 0.5
            dzdy = (matrix[yp][x] - matrix[ym][x]) * 0.5
            nx, ny, nz = -dzdx, -dzdy, 1.0
            nn = (nx * nx + ny * ny + nz * nz) ** 0.5
            nx, ny, nz = nx / nn, ny / nn, nz / nn
            s = nx * lx + ny * ly + nz * lz
            out[y][x] = max(0.0, s)
    return out


def effective_width(requested_w: int, fit_terminal: bool, frame: bool) -> int:
    if not fit_terminal:
        return max(1, requested_w)
    cols = shutil.get_terminal_size(fallback=(120, 40)).columns
    reserve = 2 if frame else 0
    max_w = max(8, cols - reserve)
    return max(1, min(requested_w, max_w))


def emit_ascii_block(lines: list[str], width: int, frame: bool) -> None:
    if frame:
        print("+" + ("-" * width) + "+")
    for line in lines:
        padded = (line[:width]).ljust(width)
        if frame:
            print("|" + padded + "|")
        else:
            print(padded)
    if frame:
        print("+" + ("-" * width) + "+")


def run_interactive(lines: list[str], title: str) -> None:
    import curses

    if not lines:
        return

    map_h = len(lines)
    map_w = max(len(ln) for ln in lines)
    padded = [ln.ljust(map_w) for ln in lines]

    def ui(stdscr):
        y_off = 0
        x_off = 0
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)

        while True:
            stdscr.erase()
            max_y, max_x = stdscr.getmaxyx()
            view_w = max(8, max_x - 2)
            view_h = max(4, max_y - 4)
            max_x_off = max(0, map_w - view_w)
            max_y_off = max(0, map_h - view_h)
            x_off = max(0, min(x_off, max_x_off))
            y_off = max(0, min(y_off, max_y_off))

            header = (
                f"{title} | map {map_w}x{map_h} | view {view_w}x{view_h} | "
                f"off x={x_off}/{max_x_off} y={y_off}/{max_y_off}"
            )
            stdscr.addnstr(0, 0, header, max_x - 1)
            stdscr.addnstr(1, 0, "+" + ("-" * view_w) + "+", max_x - 1)
            for row in range(view_h):
                src_row = y_off + row
                line = padded[src_row][x_off : x_off + view_w] if src_row < map_h else " " * view_w
                stdscr.addnstr(2 + row, 0, "|" + line.ljust(view_w) + "|", max_x - 1)
            stdscr.addnstr(2 + view_h, 0, "+" + ("-" * view_w) + "+", max_x - 1)
            stdscr.addnstr(
                3 + view_h,
                0,
                "Arrows pan | h/j/k/l pan | H/J/K/L fast pan | q quit",
                max_x - 1,
            )
            stdscr.refresh()

            key = stdscr.getch()
            if key in (ord("q"), ord("Q")):
                break
            if key in (curses.KEY_LEFT, ord("h")):
                x_off -= 1
            elif key in (curses.KEY_RIGHT, ord("l")):
                x_off += 1
            elif key in (curses.KEY_UP, ord("k")):
                y_off -= 1
            elif key in (curses.KEY_DOWN, ord("j")):
                y_off += 1
            elif key == ord("H"):
                x_off -= 8
            elif key == ord("L"):
                x_off += 8
            elif key == ord("K"):
                y_off -= 4
            elif key == ord("J"):
                y_off += 4

    curses.wrapper(ui)


def main() -> int:
    a = parse_args()
    dat_path = Path(a.dat).expanduser().resolve()
    data = dat_path.read_bytes()
    grid, spacing, blocks = build_height_grid(data)

    if a.interactive:
        sampled = resize_nearest(grid, a.h, a.w)
        if a.mode == "shade":
            lines = map_to_ascii(hillshade(sampled), RAMP_SHADE, robust=False)
            title = f"{dat_path.name} shade {len(sampled[0])}x{len(sampled)}"
        else:
            lines = map_to_ascii(sampled, RAMP_HEIGHT, robust=True)
            title = f"{dat_path.name} height {len(sampled[0])}x{len(sampled)}"
        run_interactive(lines, title)
        return 0

    out_w = effective_width(a.w, a.fit_terminal, a.frame)
    sampled = resize_nearest(grid, a.h, out_w)
    flat = [v for row in sampled for v in row]
    print(f"file={dat_path.name} spacing={spacing}m blocks={blocks}")
    print(f"grid_points={len(grid)}x{len(grid[0])} render={len(sampled)}x{len(sampled[0])}")
    print(f"elev_min={min(flat):.1f}m elev_max={max(flat):.1f}m")
    print()

    if a.mode in ("both", "height"):
        print("HEIGHT (low=' ' high='@'):")
        emit_ascii_block(map_to_ascii(sampled, RAMP_HEIGHT, robust=True), out_w, a.frame)
        print()

    if a.mode in ("both", "shade"):
        print("HILLSHADE (shape emphasis):")
        shaded = hillshade(sampled)
        emit_ascii_block(map_to_ascii(shaded, RAMP_SHADE, robust=False), out_w, a.frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
