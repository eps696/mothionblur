"""The two optional Atlas engines layered on top of the core temporal effect (blur-core-v1,
handled by atlas.py/volume.py): blur-v1 for spatial blur (--spatial) and retrocausal-echo-v1 for
audio (--echo-audio). Both are asset-upload/file-output round trips, unlike blur-core-v1's inline
JSON - see atlas.py's upload_bytes/find_output for the shared plumbing.
"""
import io

import numpy as np
from PIL import Image

from atlas import find_output


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def spatial_blur(atlas, frame_rgb: np.ndarray, mask_gray: np.ndarray, strength: float,
                 reach: float, style: str = "rx") -> np.ndarray:
    """Atlas blur-v1: a genuinely different quantum engine from blur-core-v1 - it blurs an image
    SPATIALLY via interference (not blur-core-v1's time-axis rotation), and its own blend contract
    (`C = m*C_blurred + (1-m)*C_original`) already does exactly the masking this needs - no new
    blend math here. See calibration.py's ``blur_v1`` probe for how that contract was verified
    against a real job before this was written (checkerboard image, half-black/half-white mask:
    the black half came back byte-identical, the white half visibly blurred).

    frame_rgb: [H,W,3] uint8. mask_gray: [H,W] uint8 (white = blur here, black = untouched).
    Returns [H,W,3] uint8: the same frame, spatially quantum-blurred where the mask says to."""
    image_id = atlas.upload_bytes(_png_bytes(frame_rgb), "frame.png", "image/png")
    mask_id = atlas.upload_bytes(_png_bytes(mask_gray), "mask.png", "image/png")
    job = atlas.run("blur-v1", {"strength": float(strength), "reach": float(reach), "style": style},
                    input_files={"image": image_id, "mask": mask_id},
                    label=f"spatial blur s={strength:.2f} r={reach:.2f} {style}")
    result = np.asarray(Image.open(io.BytesIO(atlas.download(find_output(job)["url"]))).convert("RGB"))
    if result.shape != frame_rgb.shape:
        raise ValueError(f"blur-v1 returned shape {result.shape}, expected {frame_rgb.shape}")
    return result


def echo_audio_params(strength: float, reach: float) -> dict:
    """Video strength/reach -> retrocausal-echo-v1 params (first-pass mapping, see the design
    doc): ``depth`` (echo taps, 1-32) is the audio analogue of ``reach`` - more taps, farther
    echoes. ``feedback`` (regeneration) is the analogue of ``strength``. ``decay`` slows as
    strength rises, so a stronger visual effect leaves a more persistent audio trail. ``mix``
    keeps some dry signal so the original audio is never fully lost."""
    return {
        "depth": max(1, min(32, round(4 + 28 * reach))),
        "feedback": min(0.9, 0.6 * strength),        # must stay < 1 (API: exclusiveMaximum)
        "decay": max(0.0, 0.98 - 0.2 * strength),
        "mix": 0.5,
    }


def echo_audio(atlas, audio_wav: bytes, strength: float, reach: float) -> bytes:
    """Atlas retrocausal-echo-v1: a THIRD distinct quantum engine (a quantum delay line) - echoes
    the clip's own audio in time, parameters mirroring the video so the soundtrack pre-/post-echoes
    in step with what's on screen. audio_wav: source audio as WAV bytes (video.extract_audio).
    Returns processed WAV bytes."""
    audio_id = atlas.upload_bytes(audio_wav, "audio.wav", "audio/wav")
    params = echo_audio_params(strength, reach)
    job = atlas.run("retrocausal-echo-v1", params, input_files={"audio": audio_id},
                    label=f"echo audio s={strength:.2f} r={reach:.2f} depth={params['depth']}")
    return atlas.download(find_output(job)["url"])
