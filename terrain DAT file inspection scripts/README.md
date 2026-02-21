# Terrain Spacing Test Kit

Required in `input/`: one `*.DAT` (decompressed from `.DAT.gz`), optional `*.BIN`.
Needs: `python3`; `lua` only for Lua check; `matplotlib` + `numpy` for PNG previews.

1. DAT check (Python): `python3 terrain_dat_spacing_check.py input/FILE.DAT 30`
Checks DAT block CRC/header validity and spacing.

2. DAT check (Lua, optional cross-check): `lua terrain_dat_spacing_check_desktop.lua input/FILE.DAT 30`
Same DAT spacing/integrity check in Lua.

3. Log check (Python, optional): `python3 terrain_log_spacing_check.py input/FILE.BIN 30`
Checks `TERR.Spacing` values in the log.

4. PNG terrain preview (Python): `python3 dat_preview_png.py input/FILE.DAT --outdir input`
Creates:
- `input/FILE_heightmap.png` (color elevation map)
- `input/FILE_hillshade.png` (shape/ridges)

5. ASCII preview (optional): `python3 dat_ascii3d.py input/FILE.DAT --w 90 --h 30 --mode both`
Prints text height + hillshade views in terminal.
