#!/usr/bin/env python3
from __future__ import annotations

import argparse
import binascii
import re
import struct
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
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
    p = argparse.ArgumentParser(description='Plot DAT tile with geo axes and marker')
    p.add_argument('dat', help='Path to .DAT')
    p.add_argument('--marker-lat', type=float, required=True)
    p.add_argument('--marker-lon', type=float, required=True)
    p.add_argument('--marker-label', default='City')
    p.add_argument('--zoom-deg', type=float, default=None, help='Optional +/- degree window around marker')
    p.add_argument('--render-spacing-m', type=float, default=None, help='Optional coarse render spacing (meters)')
    p.add_argument('--out', required=True)
    return p.parse_args()


def parse_tile_bounds(name: str):
    m = re.search(r'([NS])(\d{2})([EW])(\d{3})', name)
    if not m:
        raise ValueError('Could not parse tile name from filename')
    ns, lat_s, ew, lon_s = m.groups()
    lat0 = int(lat_s)
    lon0 = int(lon_s)
    if ns == 'S':
        lat0 = -lat0
    if ew == 'W':
        lon0 = -lon0
    # Tile naming: N45 => [45,46), S36 => [-36,-35)
    return float(lat0), float(lat0 + 1), float(lon0), float(lon0 + 1)


def iter_valid_blocks(data: bytes):
    if len(data) % IO_BLOCK_SIZE == 0:
        block_size = IO_BLOCK_SIZE
    elif len(data) % IO_BLOCK_DATA_SIZE == 0:
        block_size = IO_BLOCK_DATA_SIZE
    else:
        raise ValueError('Unsupported DAT file size')

    for pos in range(0, len(data), block_size):
        block = data[pos:pos + block_size]
        if len(block) < IO_BLOCK_DATA_SIZE:
            continue
        bitmap, lat, lon, crc, version, spacing = struct.unpack_from('<QiiHHH', block, 0)
        if bitmap == 0 and lat == 0 and lon == 0 and crc == 0 and version == 0 and spacing == 0:
            continue
        if version != TERRAIN_GRID_FORMAT_VERSION:
            continue
        crc_data = bytearray(block[:IO_BLOCK_DATA_SIZE])
        crc_data[16:18] = b'\x00\x00'
        if binascii.crc_hqx(crc_data, 0) != crc:
            continue
        heights = struct.unpack_from('<' + str(H_RAW_COUNT) + 'h', block, 22)
        grid_idx_n, grid_idx_e = struct.unpack_from('<HH', block, 22 + H_RAW_COUNT * 2)
        yield grid_idx_n, grid_idx_e, spacing, heights


def build_grid(dat_path: Path):
    data = dat_path.read_bytes()
    sums = defaultdict(float)
    counts = defaultdict(int)
    spacing = None

    for grid_idx_n, grid_idx_e, sp, heights in iter_valid_blocks(data):
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
        raise ValueError('No valid terrain points found')

    ns = sorted({k[0] for k in sums})
    es = sorted({k[1] for k in sums})
    n_map = {v: i for i, v in enumerate(ns)}
    e_map = {v: i for i, v in enumerate(es)}
    arr = np.zeros((len(ns), len(es)), dtype=np.float32)
    for (n, e), total in sums.items():
        arr[n_map[n], e_map[e]] = total / counts[(n, e)]
    return arr, spacing


def downsample_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    h, w = arr.shape
    h2 = h // factor
    w2 = w // factor
    if h2 < 1 or w2 < 1:
        return arr
    trimmed = arr[: h2 * factor, : w2 * factor]
    return trimmed.reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def main() -> int:
    a = parse_args()
    dat_path = Path(a.dat).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    arr, spacing = build_grid(dat_path)
    effective_spacing = spacing
    if a.render_spacing_m is not None and a.render_spacing_m > spacing:
        factor = max(1, int(round(a.render_spacing_m / spacing)))
        arr = downsample_mean(arr, factor)
        effective_spacing = spacing * factor
    lat0, lat1, lon0, lon1 = parse_tile_bounds(dat_path.name)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(
        arr,
        cmap='terrain',
        origin='lower',
        extent=[lon0, lon1, lat0, lat1],
        aspect='auto',
    )
    ax.scatter([a.marker_lon], [a.marker_lat], s=70, c='red', edgecolors='black', linewidths=0.9, zorder=5)
    ax.text(a.marker_lon + 0.015, a.marker_lat + 0.01, a.marker_label, color='red', fontsize=10, weight='bold')

    ax.set_title(f'{dat_path.name} (data {spacing}m, render ~{int(effective_spacing)}m)')
    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    ax.grid(color='white', alpha=0.22, linewidth=0.5)
    if a.zoom_deg is not None and a.zoom_deg > 0:
        x0 = max(lon0, a.marker_lon - a.zoom_deg)
        x1 = min(lon1, a.marker_lon + a.zoom_deg)
        y0 = max(lat0, a.marker_lat - a.zoom_deg)
        y1 = min(lat1, a.marker_lat + a.zoom_deg)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    fig.colorbar(im, ax=ax, label='Terrain altitude (m)')
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f'Wrote: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
