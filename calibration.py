"""Dev-only calibration tools for Atlas engines. NOT part of the render pipeline; kept for
provenance and reproducibility of the constants/limits used elsewhere in this project.
Spends a few real, small Atlas jobs.

  python calibration.py impulse [strength] [reach] [style]   blur-core-v1 time-axis impulse probe;
                                                             regenerates tests/fixtures/atlas_probe.json
  python calibration.py scale T H W                          blur-core-v1 payload-size / runtime
                                                             probe (finds the limits in volume.py)
  python calibration.py blur_v1 [strength] [reach] [style]   blur-v1 spatial-blur + mask probe
                                                             (de-risks secondary_engines.py before it's
                                                             wired into the main pipeline)
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from atlas import Atlas, find_output

T = 64
FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "atlas_probe.json"


def probe_grid() -> np.ndarray:
    """[T,4]: impulse@32, impulse@10, constant, step@32 - same grid emulator.py's test fixture uses."""
    g = np.zeros((T, 4))
    g[32, 0] = 1.0
    g[10, 1] = 1.0
    g[:, 2] = 0.5
    g[32:, 3] = 1.0
    return g


def cmd_impulse(argv):
    strength = float(argv[0]) if argv else 0.3
    reach = float(argv[1]) if len(argv) > 1 else 0.0
    style = argv[2] if len(argv) > 2 else "x"
    atlas = Atlas()
    params = {"values": probe_grid().tolist(), "axes": [0], "strength": [strength],
              "reach": reach, "style": style}
    job = atlas.run("blur-core-v1", params, label=f"calibration impulse s={strength} r={reach} {style}")
    out = np.asarray(job["result"]["output"], dtype=float)
    print(f"job {job['job_id']} cached={job['cached']} seconds={job.get('seconds')} shape={out.shape}")
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    for c, name in enumerate(("impulse@32", "impulse@10", "constant", "step@32")):
        col = out[:, c]
        top = sorted(int(t) for t in np.argsort(col)[::-1][:8])
        print(f"\n{name}: sum={col.sum():.3f} max={col.max():.3f}")
        print("  top t:", top, "values:", np.round(col[top], 3))

    fixtures = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else []
    fixtures = [f for f in fixtures if not (f["strength"] == strength and f["reach"] == reach and f["style"] == style)]
    fixtures.append({"job_id": job["job_id"], "strength": strength, "reach": reach, "style": style,
                     "output": np.round(out, 6).tolist()})
    FIXTURE.write_text(json.dumps(fixtures))
    print(f"\nupdated {FIXTURE} ({len(fixtures)} fixtures)")


def cmd_scale(argv):
    t, h, w = (int(x) for x in argv[:3])
    rng = np.random.default_rng(0)
    vol = np.round(rng.random((t, h, w)) * 255).astype(int)   # ints keep JSON small
    params = {"values": vol.tolist(), "axes": [0], "strength": [0.3], "reach": 0.5,
              "style": "x", "max_qubits": 24}
    print(f"volume {t}x{h}x{w} = {vol.size} values, payload ~{len(json.dumps(params)) / 1e6:.2f} MB")
    t0 = time.time()
    job = Atlas().run("blur-core-v1", params, label=f"calibration scale {t}x{h}x{w}", timeout=1200)
    out = np.asarray(job["result"]["output"], dtype=float)
    print(f"job {job['job_id']} cached={job['cached']} server_seconds={job.get('seconds')} "
          f"wall={time.time() - t0:.1f}s out_shape={out.shape} max={out.max():.3f} "
          f"result_bytes~{len(json.dumps(job['result']))} ({len(json.dumps(job['result'])) / out.size:.1f} B/value)")


def cmd_blur_v1(argv):
    """De-risk secondary_engines.py: real blur-v1 job, checkerboard image + a half-black/half-white
    mask, so the documented mask contract (black=unchanged, white=fully blurred) is directly
    checked against real output, not assumed."""
    strength = float(argv[0]) if argv else 0.6
    reach = float(argv[1]) if len(argv) > 1 else 0.3
    style = argv[2] if len(argv) > 2 else "rx"
    size = 128
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    img = np.zeros((size, size, 3), np.uint8)
    img[..., 0] = (x // 16 % 2) * 255
    img[..., 1] = (y // 16 % 2) * 255
    img[..., 2] = 128
    mask = np.zeros((size, size), np.uint8)
    mask[:, size // 2:] = 255                       # right half = full effect, left = untouched

    def png_bytes(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    atlas = Atlas()
    image_id = atlas.upload_bytes(png_bytes(img), "probe.png", "image/png")
    mask_id = atlas.upload_bytes(png_bytes(mask), "mask.png", "image/png")
    params = {"strength": strength, "reach": reach, "style": style}
    t0 = time.time()
    job = atlas.run("blur-v1", params, input_files={"image": image_id, "mask": mask_id},
                    label=f"calibration blur_v1 s={strength} r={reach} {style}")
    print(f"job {job['job_id']} cached={job['cached']} seconds={job.get('seconds')} "
          f"wall={time.time() - t0:.1f}s")
    print("outputs:", json.dumps(job.get("outputs"), indent=1)[:800])
    output = find_output(job)
    data = atlas.download(output["url"])
    result = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    print(f"result shape={result.shape} dtype={result.dtype}")

    left_diff = np.abs(result[:, :size // 2].astype(int) - img[:, :size // 2].astype(int)).mean()
    right_diff = np.abs(result[:, size // 2:].astype(int) - img[:, size // 2:].astype(int)).mean()
    print(f"mean abs diff from source: LEFT (mask=0, expect ~0)={left_diff:.2f}  "
          f"RIGHT (mask=255, expect >0)={right_diff:.2f}")

    out_path = Path(__file__).parent / "work" / "calibration_blur_v1.png"
    out_path.parent.mkdir(exist_ok=True)
    Image.fromarray(result).save(out_path)
    print(f"saved {out_path}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("impulse", "scale", "blur_v1"):
        raise SystemExit(__doc__)
    {"impulse": cmd_impulse, "scale": cmd_scale, "blur_v1": cmd_blur_v1}[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
