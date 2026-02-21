#!/usr/bin/env python3
"""Create simple PNG terrain previews from an ArduPilot DAT file."""

from __future__ import annotations

import argparse
import binascii
import struct
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IO_BLOCK_SIZE = 2048
IO_BLOCK_DATA_SIZE = 1821
TERRAIN_GRID_FORMAT_VERSION = 1
GRID_N = 28
GRID_E = 32
BLOCK_SPACING_N = 24
BLOCK_SPACING_E = 28
H_RAW_COUNT = GRID_N * GRID_E


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make terrain PNG previews from DAT")
    p.add_argument("dat", help="Path to .DAT")
    p.add_argument("--outdir", default="input", help="Output directory")
    return p.parse_args()


def iter_valid_blocks(data: bytes):
    if len(data) % IO_BLOCK_SIZE == 0:
        block_size = IO_BLOCK_SIZE
    elif len(data) % IO_BLOCK_DATA_SIZE == 0:
        block_size = IO_BLOCK_DATA_SIZE
    else:
        raise ValueError("Unsupported DAT file size")

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


def build_grid(dat_path: Path):
    data = dat_path.read_bytes()
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
        raise ValueError("No valid terrain points found")

    ns = sorted({k[0] for k in sums})
    es = sorted({k[1] for k in sums})
    n_map = {v: i for i, v in enumerate(ns)}
    e_map = {v: i for i, v in enumerate(es)}
    arr = np.zeros((len(ns), len(es)), dtype=np.float32)
    for (n, e), total in sums.items():
        arr[n_map[n], e_map[e]] = total / counts[(n, e)]
    return arr, spacing, blocks


def hillshade(arr: np.ndarray) -> np.ndarray:
    dy, dx = np.gradient(arr.astype(np.float64))
    nx = -dx
    ny = -dy
    nz = np.ones_like(arr, dtype=np.float64)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nx /= norm
    ny /= norm
    nz /= norm
    lx, ly, lz = -0.65, -0.65, 0.4
    ln = (lx * lx + ly * ly + lz * lz) ** 0.5
    lx, ly, lz = lx / ln, ly / ln, lz / ln
    shade = nx * lx + ny * ly + nz * lz
    return np.clip(shade, 0.0, 1.0)


def save_plots(arr: np.ndarray, dat_name: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(dat_name).stem
    p1 = outdir / f"{stem}_heightmap.png"
    p2 = outdir / f"{stem}_hillshade.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, cmap="terrain", origin="lower")
    ax.set_title(f"{dat_name} heightmap")
    ax.set_xlabel("east index")
    ax.set_ylabel("north index")
    fig.colorbar(im, ax=ax, label="meters")
    fig.tight_layout()
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    sh = hillshade(arr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(sh, cmap="gray", origin="lower")
    ax.set_title(f"{dat_name} hillshade")
    ax.set_xlabel("east index")
    ax.set_ylabel("north index")
    fig.tight_layout()
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    return p1, p2


def main() -> int:
    a = parse_args()
    dat_path = Path(a.dat).expanduser().resolve()
    outdir = Path(a.outdir).expanduser().resolve()
    arr, spacing, blocks = build_grid(dat_path)
    p1, p2 = save_plots(arr, dat_path.name, outdir)
    print(f"file={dat_path.name} spacing={spacing}m blocks={blocks}")
    print(f"grid={arr.shape[0]}x{arr.shape[1]} elev_min={arr.min():.1f}m elev_max={arr.max():.1f}m")
    print(f"wrote={p1}")
    print(f"wrote={p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
