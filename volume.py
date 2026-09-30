"""Tile a [64,H,W] luminance volume through blur-core-v1 (time axis only) and stitch.

Exactness: the operator acts on each pixel's time column independently and keeps
each column's sum, so tiling loses nothing. Atlas rescales output relative to
each tile's own maximum, so every output column is renormalised to its known
input sum, which also removes any per-tile gain (no seams).
"""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from progress import progbar

BLOCK = 64                 # time steps = 6 qubits (see emulator.py)
TILE = 36                  # 64*36*36 = 83k values
MAX_BODY = 1_000_000       # request cap is 1_048_576 B (HTTP 413)
# The binding limit is the RESULT: returned as floats (~19.4 B/value) and Atlas fails jobs whose
# payload exceeds ~2 MB (TMPRL1103). Measured: 32k values ok; 122k and 200k values failed.
MAX_RESULT_VALUES = 90_000


def luma(frames: np.ndarray) -> np.ndarray:
    """[T,H,W,3] uint8 -> [T,H,W] float32 (Rec.709 on the encoded values)."""
    f = frames.astype(np.float32)
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def plan_tiles(h: int, w: int, tile: int = TILE):
    """Rectangles (y0,y1,x0,x1) covering the frame; edge tiles are smaller."""
    ys = list(range(0, h, tile)) + [h]
    xs = list(range(0, w, tile)) + [w]
    return [(ys[i], ys[i + 1], xs[j], xs[j + 1])
            for i in range(len(ys) - 1) for j in range(len(xs) - 1)]


def is_static(tile: np.ndarray, eps: float = 1.0) -> bool:
    """No pixel changes by more than ``eps`` levels: blur would leave it unchanged."""
    return bool((tile.max(0) - tile.min(0)).max() <= eps)


def tile_motion(sub: np.ndarray) -> float:
    """[T,h,w] -> a single non-negative 'how much does this tile change over time' score:
    per-pixel temporal std, averaged over the tile (grey levels)."""
    return float(sub.astype(np.float64).std(axis=0).mean())


def motion_factor(motion: float, motion_ref: float, floor: float) -> float:
    """0..1, saturating: ``floor`` at ~no motion, 1.0 at/above ``motion_ref``.

    Absolute threshold, not relative to this clip's own busiest tile - a uniformly calm clip's
    busiest tile should still look calm, not get maxed-out ghosting relative to itself."""
    return floor + (1 - floor) * min(1.0, motion / motion_ref)


def split_tiles(q: np.ndarray, tile: int = TILE, skip_static: float = 1.0):
    """-> (tiles needing Atlas, static tiles that blur leaves unchanged)."""
    todo, static = [], []
    for rect in plan_tiles(q.shape[1], q.shape[2], tile):
        y0, y1, x0, x1 = rect
        sub = q[:, y0:y1, x0:x1]
        (static if is_static(sub, skip_static) or sub.sum() == 0 else todo).append(rect)
    return todo, static


def make_params(tile_vals: np.ndarray, strength: float, reach: float, style: str) -> dict:
    return {"values": tile_vals.tolist(), "axes": [0], "strength": [float(strength)],
            "reach": float(reach), "style": style}


def renorm_columns(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Scale each time column of ``out`` so its sum equals that of ``ref``."""
    so, sr = out.sum(0), ref.sum(0)
    scale = np.divide(sr, so, out=np.zeros_like(sr), where=so > 0)
    return out * scale[None]


def tile_strength_reach(sub: np.ndarray, strength: float, reach: float, adaptive: bool,
                        motion_ref: float, motion_floor: float) -> tuple[float, float]:
    """The (strength, reach) an individual tile actually gets: unchanged unless ``adaptive``,
    in which case both are scaled by that tile's own motion_factor. Shared by blur_via_atlas,
    plan_jobs and blur_tiled so their cache keys/job counts always agree."""
    if not adaptive:
        return strength, reach
    f = motion_factor(tile_motion(sub), motion_ref, motion_floor)
    return strength * f, reach * f


def blur_via_atlas(atlas, vol: np.ndarray, strength: float, reach: float = 0.0,
                   style: str = "x", tile: int = TILE, workers: int = 4,
                   skip_static: float = 1.0, log=print, progress=True,
                   adaptive: bool = False, motion_ref: float = 40.0, motion_floor: float = 0.15):
    """vol: [64,H,W] non-negative. Returns (float32 blurred volume, list of Atlas jobs).

    ``progress``: True creates a private progress bar sized to this call's own tile count;
    pass a shared ``progbar`` instance (e.g. from timesmear.py, sized to a whole clip's tile
    count) to advance one bar across several calls; False disables it (e.g. in tests).

    ``adaptive``: each tile gets its OWN strength/reach, scaled by how much that tile actually
    changes (tile_strength_reach) instead of the one clip-wide (strength, reach) - default off,
    so the plain call is unchanged.
    """
    if vol.shape[0] != BLOCK:
        raise ValueError(f"need exactly {BLOCK} frames per block, got {vol.shape[0]}")
    q = np.rint(np.clip(vol, 0, 255)).astype(np.int64)
    out = np.zeros(vol.shape, np.float64)
    todo, static = split_tiles(q, tile, skip_static)
    tiles = todo + static
    for (y0, y1, x0, x1) in static:
        out[:, y0:y1, x0:x1] = q[:, y0:y1, x0:x1]          # blur is the identity here
    log(f"{len(tiles)} tiles: {len(todo)} via Atlas, {len(tiles) - len(todo)} static (skipped)")
    bar = (progress if isinstance(progress, progbar)
          else (progbar(len(todo)) if progress and todo else None))

    def one(t):
        y0, y1, x0, x1 = t
        sub = q[:, y0:y1, x0:x1]
        s, r = tile_strength_reach(sub, strength, reach, adaptive, motion_ref, motion_floor)
        params = make_params(sub, s, r, style)
        size = len(json.dumps(params, separators=(",", ":")))
        if size > MAX_BODY:
            raise ValueError(f"tile {t} payload {size} B exceeds {MAX_BODY}; lower tile size")
        if sub.size > MAX_RESULT_VALUES:
            raise ValueError(f"tile {t}: {sub.size} values would exceed Atlas's ~2 MB result "
                             f"limit (max {MAX_RESULT_VALUES}); lower tile size")
        job = atlas.run("blur-core-v1", params,
                        label=f"timesmear s={s:.3f} r={r:.3f} tile={t}")
        res = np.asarray(job["result"]["output"], dtype=np.float64)
        if res.shape != sub.shape:
            raise ValueError(f"tile {t}: Atlas returned {res.shape}, expected {sub.shape}")
        return t, renorm_columns(res, sub.astype(np.float64)), job

    jobs = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for t, res, job in pool.map(one, todo):
            y0, y1, x0, x1 = t
            out[:, y0:y1, x0:x1] = res
            jobs.append(job)
            if bar:
                bar.upd()
    return out.astype(np.float32), jobs


def plan_jobs(vol: np.ndarray, strength: float, reach: float, style: str, jobs_dir,
              tile: int = TILE, skip_static: float = 1.0, adaptive: bool = False,
              motion_ref: float = 40.0, motion_floor: float = 0.15) -> tuple[int, int]:
    """(tiles that need Atlas, of which already cached) - no network, no key needed."""
    from pathlib import Path

    from atlas import Atlas
    q = np.rint(np.clip(vol, 0, 255)).astype(np.int64)
    todo, _ = split_tiles(q, tile, skip_static)
    cached = 0
    for (y0, y1, x0, x1) in todo:
        sub = q[:, y0:y1, x0:x1]
        s, r = tile_strength_reach(sub, strength, reach, adaptive, motion_ref, motion_floor)
        sha = Atlas.cache_key("blur-core-v1", make_params(sub, s, r, style), None)
        rec = Path(jobs_dir) / f"{sha}.json"
        cached += rec.exists() and "result" in json.loads(rec.read_text())
    return len(todo), cached


def blur_tiled(blur_fn, vol: np.ndarray, strength: float, reach: float, style: str,
              tile: int = TILE, adaptive: bool = False, motion_ref: float = 40.0,
              motion_floor: float = 0.15, skip_static: float = 1.0) -> np.ndarray:
    """Local-engine (numpy/qiskit) equivalent of blur_via_atlas's per-tile strength: calls
    ``blur_fn`` (e.g. emulator.blur_volume or qiskit_engine.blur_volume, UNCHANGED - neither
    module needs to know about tiling) once per tile instead of once for the whole volume.
    Only used when ``adaptive`` is requested - the default (non-adaptive) path calls blur_fn
    directly on the whole window and never goes through this function."""
    q = np.rint(np.clip(vol, 0, 255)).astype(np.int64)
    out = np.zeros(vol.shape, np.float32)
    todo, static = split_tiles(q, tile, skip_static)
    for (y0, y1, x0, x1) in static:
        out[:, y0:y1, x0:x1] = q[:, y0:y1, x0:x1]
    for (y0, y1, x0, x1) in todo:
        sub = q[:, y0:y1, x0:x1]
        s, r = tile_strength_reach(sub, strength, reach, adaptive, motion_ref, motion_floor)
        res = blur_fn(sub, s, r, style)
        out[:, y0:y1, x0:x1] = renorm_columns(res.astype(np.float64), sub.astype(np.float64))
    return out
