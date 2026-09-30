"""Turn a saved echo (Atlas output minus source, at working resolution) into pixels, and
persist/reload that echo as "QC data" so new looks can be produced later without recomputing it
(no new Atlas jobs, no emulator run).

E = Q - V is the *echo*: light Atlas moved to a time it was not there (E > 0) and light it took
away (E < 0). Static regions give E = 0 exactly. Every look upsamples E (and, where needed, V)
from the working resolution to the full frame size once per chunk (done by the caller, timesmear.py)
and receives the already-upsampled arrays.

All the per-pixel math here runs on the GPU (PyTorch/CUDA) - at full source resolution this is a
genuinely heavy, memory-bandwidth-bound elementwise workload, and this project assumes an NVIDIA
GPU is present (same policy as NVENC in video.py: no CPU fallback). Callers may pass plain numpy
arrays (auto-uploaded) or, for the hot path, already-GPU tensors from ``upsample``/``to_gpu`` to
avoid re-uploading the same data for every look.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda"
LOOKS = ("faithful", "ghost", "quantum", "echo")


def to_gpu(arr) -> torch.Tensor:
    """numpy array or tensor -> float32 tensor on DEVICE."""
    if torch.is_tensor(arr):
        return arr.to(DEVICE, dtype=torch.float32, non_blocking=True)
    return torch.from_numpy(np.ascontiguousarray(arr, np.float32)).to(DEVICE, non_blocking=True)


def upsample(vol, h: int, w: int) -> torch.Tensor:
    """[T,h0,w0] (numpy or tensor) -> [T,h,w] float32 GPU tensor, bilinear."""
    t = to_gpu(vol).unsqueeze(1)                                    # [T,1,h0,w0]
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t.squeeze(1)                                             # [T,h,w]


def echo_mask(e_up_frame, gamma: float = 0.5) -> np.ndarray:
    """[H,W] echo of ONE frame (numpy or tensor) -> [H,W] uint8 mask for
    secondary_engines.spatial_blur: that frame's own peak |echo| -> 255 (full spatial blur
    there), 0 -> 0 (untouched).

    ``gamma`` < 1 spreads mid-range echo toward higher mask values instead of leaving only the
    single peak pixel near 255: |echo| varies smoothly, so without this most 'active' pixels sit
    well below the frame's one peak and the mask (hence the visible blend, since blur-v1's own
    blend weight IS the mask value) averages only a few percent even in genuinely moving regions
    - confirmed on real footage (mean mask value 4.7% of 255 at gamma=1, 17.6% at gamma=0.5).
    All-zero input (nothing moved) returns an all-zero mask regardless of gamma."""
    mag = to_gpu(e_up_frame).abs()
    peak = float(mag.max())
    if peak <= 0:
        return np.zeros(tuple(mag.shape), np.uint8)
    normalized = (mag / peak).clamp(0, 1) ** gamma
    return (normalized * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()


def iridescent(x: torch.Tensor, phase: float = 0.0) -> torch.Tensor:
    """x in [0,1] -> RGB [...,3] in [0,1]: thin-film-like cosine palette (hue rolls with x)."""
    offsets = torch.tensor([0.0, 0.33, 0.67], device=DEVICE)
    ang = 2 * math.pi * (1.1 * x.unsqueeze(-1) + phase + offsets)
    return 0.5 + 0.5 * torch.cos(ang)


# Smooth black -> colour -> near-white ramps for the 'echo' look: added light glows warm (ember ->
# gold -> white-hot), removed light glows cool (deep blue -> cyan -> pale). Plain per-pixel lookup,
# no spatial mixing, so a static pixel still renders as exact black - see test_pipeline.py.
_WARM = torch.tensor([(0.00, 0.00, 0.00, 0.00), (0.35, 0.55, 0.05, 0.00),
                      (0.70, 1.00, 0.55, 0.05), (1.00, 1.00, 0.92, 0.65)], device=DEVICE)
_COOL = torch.tensor([(0.00, 0.00, 0.00, 0.00), (0.35, 0.02, 0.12, 0.45),
                      (0.70, 0.05, 0.55, 0.95), (1.00, 0.70, 0.95, 1.00)], device=DEVICE)


def _ramp(m: torch.Tensor, stops: torch.Tensor) -> torch.Tensor:
    """m in [0,1] -> RGB via piecewise-linear interpolation between colour ``stops``
    (rows of (position, r, g, b))."""
    pos, rgb = stops[:, 0].contiguous(), stops[:, 1:]
    idx = torch.clamp(torch.searchsorted(pos, m, right=True) - 1, 0, len(pos) - 2)
    p0, p1 = pos[idx], pos[idx + 1]
    t = torch.clamp((m - p0) / torch.clamp(p1 - p0, min=1e-6), 0, 1).unsqueeze(-1)
    return rgb[idx] * (1 - t) + rgb[idx + 1] * t


def render_look(name: str, frames, echo_up, v_up=None, gain: float = None,
                dark: float = 0.25) -> np.ndarray:
    """frames: [T,H,W,3] source at full resolution. echo_up, v_up: [T,H,W], already upsampled to
    H x W. Each may be numpy or an already-GPU tensor. v_up is only needed by 'quantum'. Returns
    uint8 numpy [T,H,W,3] - only the final result leaves the GPU."""
    src = to_gpu(frames) / 255.0
    e = to_gpu(echo_up) / 255.0
    if name == "faithful":
        # No invented colour: the echo (grey levels) is added to the source's OWN channels.
        # A colour video keeps its own colours - only luminance is echoed, not hue.
        g = 1.0 if gain is None else gain
        out = src + g * e.unsqueeze(-1)
    elif name == "quantum":
        # The engine's own output (V + E), decoupled from the source: what Atlas actually returned.
        if v_up is None:
            raise ValueError("look 'quantum' needs v_up")
        q = torch.clamp((to_gpu(v_up) + to_gpu(echo_up)) / 255.0, 0, 1)
        out = q.unsqueeze(-1).expand(*q.shape, 3)
    elif name == "echo":
        # Only the echo: no source pixels at all. Added light glows warm (ember/gold/white-hot),
        # removed light glows cool (deep blue/cyan/pale); black = unchanged (static regions). A
        # gentle gamma spreads out the mid-strength range instead of a hard on/off between black
        # and a saturated colour. Pure per-pixel lookup - no blur/glow, so a static pixel is exact
        # black regardless of a moving neighbour (test_pipeline.py checks this).
        g = 4.0 if gain is None else gain
        m = torch.clamp(e.abs() * g, 0, 1) ** 0.6
        out = _ramp(m * (e > 0), _WARM) + _ramp(m * (e < 0), _COOL)
    elif name == "ghost":
        # Artistic composite: added light rendered as a thin-film-iridescent glow over the source.
        g = 3.0 if gain is None else gain
        pos = torch.clamp(e * g, 0, 1)
        neg = torch.clamp(-e * g, 0, 1)
        out = src * (1 - dark * neg.unsqueeze(-1)) + iridescent(pos) * pos.unsqueeze(-1)
    else:
        raise ValueError(f"unknown look {name!r}; choices: {LOOKS}")
    return (torch.clamp(out, 0, 1) * 255 + 0.5).to(torch.uint8).contiguous().cpu().numpy()


# ---------------------------------------------------------------- QC persistence
def save_qc(out_stem: str, echo: np.ndarray, meta: dict) -> str:
    """Persist the quantum-computed echo (the expensive result) + its parameters, so any number
    of looks can be produced from it later without another Atlas call. Returns the qc.json path."""
    np.save(f"{out_stem}.echo.npy", np.asarray(echo, dtype=np.float32))
    path = f"{out_stem}.qc.json"
    Path(path).write_text(json.dumps(meta, indent=1))
    return path


def load_qc(qc_path) -> tuple[np.ndarray, dict]:
    """Accepts either a '<stem>.qc.json' path or the bare stem."""
    stem = str(qc_path)[: -len(".qc.json")] if str(qc_path).endswith(".qc.json") else str(qc_path)
    meta = json.loads(Path(f"{stem}.qc.json").read_text())
    echo = np.load(f"{stem}.echo.npy")
    return echo, meta
