"""Quantum time-smear: blur a video along TIME, any length, on any of three interchangeable
engines that implement the identical model (see emulator.py's module docstring for the model
itself):

  --backend numpy   (default) hand-derived NumPy simulation of the rotation - free, instant.
  --backend qiskit  the SAME rotation built and run as a real qiskit.QuantumCircuit - free, fast.
  --backend atlas   real Moth Atlas blur-core-v1 jobs - costs credits, guarded by --max-jobs.

  python timesmear.py plan     in.mp4                 count Atlas jobs needed, no spending
  python timesmear.py render   in.mp4 out [--backend X]  -> out_<look>.mp4 + out.echo.npy/out.qc.json
  python timesmear.py relook   out.qc.json out2         new look(s) from saved QC data, free
  python timesmear.py validate in.mp4 --backend X       X vs the numpy baseline, agreement report
  python timesmear.py synth    clip.mp4                 write the synthetic test clip

The quantum step only ever sees LUMINANCE (Rec.709), at --qwidth pixels wide, regardless of
whether the source is colour or black-and-white: a colour source keeps its own colours, only its
brightness is echoed (see looks.py). The one exception is the 'quantum' look, which shows the
engine's raw grayscale output on its own.

Any length: the clip is cut into overlapping 64-frame windows (hop --hop, default 32; see
plan_windows/window_weights). Each window is blurred along time; the echo (Q - V) of overlapping
windows is crossfaded with sin^2 weights that sum to 1. This softens window boundaries but does
not remove them: an echo stays mostly (~90%) inside its own window, ~10% spills across
(tests/test_pipeline.py).

render's expensive part (the echo) is saved once (<out>.echo.npy + <out>.qc.json) and every look
is a cheap re-composite from it. Use ``relook`` later to try new looks/gains without recomputing
anything.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import emulator
import looks
import qiskit_engine
import secondary_engines
import video
import volume
from progress import progbar

WORK = Path(__file__).parent / "work"
BLOCK = volume.BLOCK
ENGINES = {"numpy": emulator, "qiskit": qiskit_engine}    # local, free engines; "atlas" is separate


# ---------------------------------------------------------------- windows
def plan_windows(n: int, hop: int = BLOCK // 2, block: int = BLOCK) -> list[int]:
    """Window start frames covering [0, n): every frame lies in >= 1 window, interior in 2."""
    if n <= block:
        return [0]
    starts = list(range(0, n - block + 1, hop))
    if starts[-1] + block < n:
        starts.append(n - block)              # last window aligned to the end
    return starts


def window_weights(block: int = BLOCK) -> np.ndarray:
    """sin^2 ramp: w(i) + w(i + block/2) == 1, so hop = block/2 windows crossfade exactly."""
    return np.sin(np.pi * (np.arange(block) + 0.5) / block) ** 2


def _padded(v: np.ndarray) -> np.ndarray:
    n = len(v)
    return v if n >= BLOCK else np.concatenate([v, np.repeat(v[-1:], BLOCK - n, 0)], 0)


# ---------------------------------------------------------------- compute (Atlas or emulator)
def load_v(args) -> tuple[np.ndarray, float]:
    """Working-resolution luminance volume [N,h,w] and the source fps."""
    small, fps = video.read_frames(args.src, start=args.start, count=args.frames, width=args.qwidth)
    if len(small) == 0:
        raise SystemExit("no frames read (check --start/--frames)")
    return volume.luma(small), fps


def blur_windowed(v: np.ndarray, strength: float, reach: float, style: str,
                  hop: int = BLOCK // 2, backend: str = "numpy", workers: int = 4,
                  atlas=None, bar=None, adaptive: bool = False, motion_ref: float = 40.0,
                  motion_floor: float = 0.15):
    """Pure array computation, no file I/O: the windowed/crossfaded blur of a whole clip's
    working-resolution volume, on whichever engine ``backend`` names. v: [N,h,w] float, any N.
    Returns (echo [N,h,w] float32, jobs, agreement-with-numpy stats dict or None).

    ``adaptive``: each tile gets its own strength/reach scaled by how much it actually moves
    (volume.tile_strength_reach) instead of one clip-wide (strength, reach) - default off, the
    non-adaptive path (both here and in the agreement cross-check below) is unchanged."""
    n = len(v)
    padded = _padded(v)
    starts = plan_windows(len(padded), hop)
    wgt = window_weights()[:, None, None].astype(np.float32)
    acc = np.zeros(padded.shape, np.float32)
    wsum = np.zeros((len(padded), 1, 1), np.float32)
    log = (lambda msg: print("\n" + msg)) if bar else print

    def local_blur(engine_fn, win):
        if adaptive:
            return volume.blur_tiled(engine_fn, win, strength, reach, style, adaptive=True,
                                     motion_ref=motion_ref, motion_floor=motion_floor).astype(np.float32)
        return engine_fn(np.rint(win), strength, reach, style).astype(np.float32)

    jobs, err2, echo2, cnt = [], 0.0, 0.0, 0
    for st in starts:
        win = padded[st:st + BLOCK]
        if backend == "atlas":
            q, j = volume.blur_via_atlas(atlas, win, strength, reach, style, workers=workers,
                                         log=log, progress=(bar or False), adaptive=adaptive,
                                         motion_ref=motion_ref, motion_floor=motion_floor)
            jobs += j
        else:
            q = local_blur(ENGINES[backend].blur_volume, win)
        if backend != "numpy":                      # cross-check against the free/fast baseline,
                                                     # same adaptive setting so the comparison is fair
            qe = local_blur(emulator.blur_volume, win)
            err2 += float(((q - qe) ** 2).sum()); echo2 += float(((qe - win) ** 2).sum()); cnt += win.size
        acc[st:st + BLOCK] += wgt * (q - win)
        wsum[st:st + BLOCK] += wgt
    echo = (acc / np.maximum(wsum, 1e-6))[:n]
    stats = ({"rms": round((err2 / cnt) ** 0.5, 4), "echo_rms": round((echo2 / cnt) ** 0.5, 2)}
            if cnt else None)
    return echo, jobs, stats


def compute_echo(args, backend: str):
    """CLI-facing wrapper: reads the source, sizes a shared progress bar (atlas backend only),
    calls blur_windowed, packages provenance. Returns (echo [N,h,w] float32, meta, v [N,h,w])."""
    v, fps = load_v(args)
    n = len(v)
    padded = _padded(v)
    starts = plan_windows(len(padded), args.hop)

    atlas, bar = None, None
    if backend == "atlas":
        from atlas import Atlas
        atlas = Atlas(WORK)
        total_tiles = sum(len(volume.split_tiles(
            np.rint(np.clip(padded[s:s + BLOCK], 0, 255)).astype(np.int64))[0]) for s in starts)
        bar = progbar(total_tiles) if total_tiles else None

    echo, jobs, stats = blur_windowed(v, args.strength, args.reach, args.style, args.hop,
                                      backend, args.workers, atlas, bar, args.adaptive,
                                      args.motion_ref, args.motion_floor)
    meta = {"src": str(args.src), "start": args.start, "frames": n, "fps": fps, "qwidth": args.qwidth,
            "strength": args.strength, "reach": args.reach, "style": args.style, "hop": args.hop,
            "backend": backend, "n_jobs": len(jobs),
            "adaptive": args.adaptive, "motion_ref": args.motion_ref, "motion_floor": args.motion_floor,
            "jobs": [{"job_id": j["job_id"], "seconds": j.get("seconds")} for j in jobs]}
    if stats:
        meta["emulator_agreement"] = stats
    return echo, meta, v


# ---------------------------------------------------------------- output
def looks_list(args) -> list[str]:
    if args.look == "all":
        return list(looks.LOOKS)
    names = [x.strip() for x in args.look.split(",")]
    if bad := [x for x in names if x not in looks.LOOKS]:
        raise SystemExit(f"unknown look(s) {bad}; choices: {looks.LOOKS}")
    return names


def stack_selection(which: list[str]) -> list[str]:
    """Looks to place in the combined --stack video: every requested look except 'quantum' - it's
    grayscale at the working resolution, not the full-resolution colour the others are, so it
    doesn't belong beside them in one showcase reel."""
    return [name for name in which if name != "quantum"]


SPATIAL_LOOKS = ("faithful", "ghost")    # 'echo'/'quantum' show the raw temporal signal untouched


def write_looks(echo: np.ndarray, meta: dict, which: list[str], out_stem: str, args, tag: str,
                v_full: np.ndarray = None) -> None:
    """v_full: in-memory working-res volume [N,h,w] (render/preview, already computed) or None
    (relook: decoded here, once for the whole clip, only if the 'quantum' look needs it).

    --stack-only skips the individual <out>_<look>.mp4 files entirely and also drops 'quantum'
    (the stack never includes it) - real work saved, not just fewer files written.

    --fps-div N keeps only every Nth frame (by GLOBAL index, so it stays aligned across chunk
    boundaries regardless of N) and writes at fps/N, so real-time duration is unchanged. This is
    purely a rendering economy: the echo itself is always computed at the source's native frame
    rate (decimating beforehand would change which frames land in which 64-frame window and
    distort the echo pattern itself, not just its output cost).

    --spatial adds a SECOND, different Atlas engine (blur-v1: spatial interference blur, not the
    time-axis rotation blur-core-v1 does) once per chunk, masked by that chunk's own |echo| - see
    secondary_engines.py. It only touches 'faithful'/'ghost' (SPATIAL_LOOKS); 'echo'/'quantum' stay the
    raw signal. Held per-chunk (one Atlas job per chunk, not per frame) for cost control: computed
    once from the chunk's middle frame as a pixel DELTA, then added to every frame in the chunk -
    a deliberate, disclosed approximation (see the plan doc), not per-frame physical accuracy."""
    src, start, fps, qwidth = meta["src"], meta["start"], meta["fps"], meta["qwidth"]
    n = len(echo)
    info = video.probe(src)
    stacked = stack_selection(which)
    write_individually = not args.stack_only
    needed = (set(which) if write_individually else set()) | (set(stacked) if args.stack else set())
    if v_full is None and "quantum" in needed:
        small, _ = video.read_frames(src, start=start, count=n, width=qwidth)
        v_full = volume.luma(small)
    w_out = info["width"] * (2 if args.split else 1)
    out_fps = fps / args.fps_div
    audio_path = src if start == 0 else None
    if args.echo_audio and start == 0:
        wav = video.extract_audio(src)
        if wav is None:
            print("--echo-audio: source has no audio track, using the plain (silent) video")
        else:
            from atlas import Atlas
            processed = secondary_engines.echo_audio(Atlas(WORK), wav, meta["strength"], meta["reach"])
            audio_path = f"{out_stem}.echo_audio.wav"
            Path(audio_path).write_bytes(processed)
            print(f"echo audio (retrocausal-echo-v1): {audio_path}")
    writers = {name: video.VideoWriter(f"{out_stem}_{name}.mp4", w_out, info["height"], out_fps,
                                       audio_from=audio_path)
              for name in which} if write_individually else {}
    stack_writer = None
    if args.stack and stacked:
        stack_writer = video.VideoWriter(f"{out_stem}_stack.mp4", info["width"] * (1 + len(stacked)),
                                         info["height"], out_fps, audio_from=audio_path)
    spatial_targets = needed & set(SPATIAL_LOOKS)
    spatial_atlas, spatial_jobs = None, 0
    if args.spatial and spatial_targets:
        from atlas import Atlas
        spatial_atlas = Atlas(WORK)
    start_msg = f"encoding {len(writers)} look(s)"
    if stack_writer:
        start_msg += " + stack"
    if args.fps_div > 1:
        start_msg += f" at 1/{args.fps_div} fps ({out_fps:.1f} of {fps:.1f})"
    if spatial_atlas is not None:
        start_msg += (f" + spatial blur-v1 on ({'/'.join(sorted(spatial_targets))}), "
                     "up to 1 extra Atlas job per chunk")
    else:
        start_msg += " (local video I/O - no Atlas/emulator/qiskit calls happen here)"
    print(start_msg)
    bar = progbar(math.ceil(n / BLOCK))
    done = 0
    for chunk in video.iter_frames(src, chunk=BLOCK, start=start, count=n):
        m = len(chunk)
        keep = np.arange(done, done + m) % args.fps_div == 0          # global index -> aligned
        if keep.any():
            chunk = chunk[keep]
            e_gpu = looks.to_gpu(echo[done:done + m][keep])            # decimate BEFORE upsampling
            v_gpu = (looks.to_gpu(v_full[done:done + m][keep]) if "quantum" in needed else None)
            chunk_gpu = looks.to_gpu(chunk)                            # upload once, shared by all looks
            e_up = looks.upsample(e_gpu, info["height"], info["width"])
            v_up = looks.upsample(v_gpu, info["height"], info["width"]) if v_gpu is not None else None
            results = {name: looks.render_look(name, chunk_gpu, e_up, v_up, args.gain, args.dark)
                      for name in needed}
            if spatial_atlas is not None:
                mid = len(chunk) // 2
                # Mean over pixels that actually show SOME echo, not the whole frame - most of a
                # typical frame is static background (echo=0), which would otherwise dilute this
                # toward ~0 regardless of how dramatic the moving parts are.
                mag = e_up.abs()
                active = mag > 1.0
                mean_echo_frac = (min(1.0, float(mag[active].mean()) / args.spatial_echo_ref)
                                  if active.any() else 0.0)
                if mean_echo_frac > 0.03:                       # skip near-zero jobs, save credits
                    mask = looks.echo_mask(e_up[mid], args.spatial_mask_gamma)
                    s = args.spatial_strength * mean_echo_frac
                    r = args.spatial_reach * mean_echo_frac
                    rep = chunk[mid]
                    blurred = secondary_engines.spatial_blur(spatial_atlas, rep, mask, s, r, args.spatial_style)
                    delta = blurred.astype(np.int16) - rep.astype(np.int16)
                    spatial_jobs += 1
                    for name in spatial_targets:
                        results[name] = np.clip(results[name].astype(np.int16) + delta[None],
                                                0, 255).astype(np.uint8)
            for name, w in writers.items():
                res = results[name]
                w.write(np.concatenate([chunk, res], axis=2) if args.split else res)
            if stack_writer:
                stack_writer.write(np.concatenate([chunk] + [results[name] for name in stacked], axis=2))
        done += m
        bar.upd()
    for w in writers.values():
        w.close()
    msg = f"wrote {len(writers)} look(s) [{tag}]"
    if writers:
        msg += ": " + ", ".join(f"{out_stem}_{n}.mp4" for n in writers)
    if stack_writer:
        stack_writer.close()
        msg += f"{',' if writers else ':'} {out_stem}_stack.mp4 (source + {'+'.join(stacked)})"
    if spatial_atlas is not None:
        msg += f" [{spatial_jobs} spatial blur-v1 job(s)]"
    print(msg)


# ---------------------------------------------------------------- commands
def cmd_plan(args, quiet: bool = False) -> int:
    if not (args.adaptive or args.backend == "atlas"):
        if not quiet:
            print(f"backend={args.backend} runs locally: no Atlas jobs, no cost.")
        return 0
    v, _ = load_v(args)
    padded = _padded(v)
    starts = plan_windows(len(padded), args.hop)
    if args.adaptive and not quiet:
        q0 = np.rint(np.clip(padded[starts[0]:starts[0] + BLOCK], 0, 255)).astype(np.int64)
        todo0, _ = volume.split_tiles(q0)
        motions = sorted(volume.tile_motion(q0[:, y0:y1, x0:x1]) for y0, y1, x0, x1 in todo0)
        if motions:
            pct = 100 * volume.motion_factor(motions[-1], args.motion_ref, args.motion_floor)
            note = ("busiest tile reaches full strength" if motions[-1] >= args.motion_ref else
                    f"busiest tile only reaches ~{pct:.0f}% strength "
                    f"(set --motion-ref <= {motions[-1]:.0f} for it to saturate)")
            print(f"adaptive motion (first window, {len(motions)} non-static tiles, grey-level std): "
                  f"min={motions[0]:.1f} median={motions[len(motions) // 2]:.1f} max={motions[-1]:.1f}; "
                  f"--motion-ref {args.motion_ref}: {note}")
    if args.backend != "atlas":
        if not quiet:
            print(f"backend={args.backend} runs locally: no Atlas jobs, no cost.")
        return 0
    total = cached = 0
    for st in starts:
        t, c = volume.plan_jobs(padded[st:st + BLOCK], args.strength, args.reach, args.style, WORK / "jobs",
                                adaptive=args.adaptive, motion_ref=args.motion_ref, motion_floor=args.motion_floor)
        total += t; cached += c
    new = total - cached
    est = new * 14 / 60 / max(1, args.workers)
    if not quiet:
        print(f"{len(v)} frames -> {len(starts)} windows (hop {args.hop}): {total} Atlas jobs, "
              f"{cached} cached, {new} NEW (~{est:.1f} min with {args.workers} workers)")
    return new


def cmd_render(args):
    if args.backend == "atlas":
        new = cmd_plan(args)
        if new > args.max_jobs:
            raise SystemExit(f"{new} new jobs exceeds --max-jobs {args.max_jobs}; nothing was submitted. "
                             f"Re-run with --max-jobs {new} to proceed.")
    echo, meta, v = compute_echo(args, args.backend)
    looks.save_qc(args.out, echo, meta)
    tag = f"backend={args.backend}" + (f", {meta['n_jobs']} Atlas jobs" if args.backend == "atlas" else "")
    write_looks(echo, meta, looks_list(args), args.out, args, tag, v_full=v)
    if meta.get("emulator_agreement"):
        print(f"{args.backend} vs local numpy engine (grey levels):", json.dumps(meta["emulator_agreement"]))
    print(f"QC data saved: {args.out}.echo.npy + {args.out}.qc.json (reprocess later with 'relook')")


def cmd_relook(args):
    echo, meta = looks.load_qc(args.src)          # args.src holds the qc.json path for this command
    write_looks(echo, meta, looks_list(args), args.out, args, "relook (saved QC, no new computation)")


def cmd_validate(args):
    echo, meta, v = compute_echo(args, args.backend)
    static = (v.max(0) - v.min(0)) <= 1.0
    report = {"backend": args.backend, "frames": meta["frames"], "atlas_jobs": meta["n_jobs"],
              "vs_numpy_agreement": meta.get("emulator_agreement"),
              "static_pixel_share": float(static.mean()),
              "echo_energy_share_of_signal": float(np.abs(echo).sum() / np.abs(v).sum())}
    (WORK / "validate.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def cmd_synth(args):
    """Deterministic synthetic test clip: static textured background (must stay untouched), a
    disc crossing the frame (pre-/post-echoes), a pulsing square (pure temporal signal), an
    orbiting dot. args.src holds the output path for this command (it takes no input)."""
    w, h, t = 320, 180, args.frames
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = 46 + 30 * (y / h) + 10 * np.sin(x / 23.0)
    bg += 14 * (((x.astype(int) // 20) + (y.astype(int) // 20)) % 2)
    for cx, cy, r, a in [(70, 130, 30, 40), (250, 45, 24, 34), (190, 150, 18, 30)]:
        bg += a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r * r))
    frames = np.repeat(bg[None], t, 0)

    def paint(i, cx, cy, r, val):
        frames[i][(x - cx) ** 2 + (y - cy) ** 2 <= r * r] = val

    for i in range(t):
        u = i / (t - 1)
        paint(i, 20 + 280 * u, 90 + 35 * np.sin(2 * np.pi * u), 15, 235)             # crossing disc
        paint(i, 240 + 32 * np.cos(4 * np.pi * u), 60 + 32 * np.sin(4 * np.pi * u), 6, 250)  # orbiting dot
        if t * 0.4 <= i < t * 0.55:                                                  # pulsing square
            frames[i][(abs(x - 80) < 18) & (abs(y - 55) < 18)] = 220
    out = np.clip(frames, 0, 255).round().astype(np.uint8)[..., None].repeat(3, -1)
    video.write_video(out, args.src, args.fps)
    print(f"wrote {args.src}: {t} frames {w}x{h} @ {args.fps}fps")


# ---------------------------------------------------------------- CLI
# One flat parser: every command shares the same flags (a command just ignores what it doesn't
# need), instead of a separate sub-parser per command. `src`/`out` are reused contextually:
#   plan/render/validate: SRC is the source video.  render: OUT is the output prefix.
#   relook:  SRC is a saved <out>.qc.json, OUT the new output prefix.
#   synth:   SRC is the clip to write (no OUT).
COMMANDS = ("plan", "render", "relook", "validate", "synth")


def build_parser():
    p = argparse.ArgumentParser(prog="timesmear", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("src", help="source video, or (relook) a saved qc.json, or (synth) the output path")
    p.add_argument("out", nargs="?", default=str(WORK / "out"), help="output prefix (render/relook)")
    p.add_argument("--backend", default="atlas", choices=list(ENGINES) + ["atlas"],
                   help="numpy/qiskit: free, local. atlas: real Atlas jobs, costs credits (default: atlas)")
    p.add_argument("--strength", type=float, default=0.3)
    p.add_argument("--reach", type=float, default=0.55)
    p.add_argument("--style", default="x", choices=["x", "y", "xy", "yx"])
    p.add_argument("--start", type=int, default=0, help="first source frame to process")
    p.add_argument("--frames", type=int, default=0, help="number of frames to process (0 = all); synth: clip length")
    p.add_argument("--hop", type=int, default=BLOCK // 2,
                   help=f"window hop; must be {BLOCK // 2} (sin^2 crossfade needs half-overlap)")
    p.add_argument("--qwidth", type=int, default=160, help="quantum working width (px)")
    p.add_argument("--adaptive", action="store_true",
                   help="each tile gets its own strength/reach, scaled by how much that tile "
                        "actually moves, instead of one clip-wide value")
    p.add_argument("--motion-ref", type=float, default=40.0,
                   help="adaptive: per-pixel temporal std (grey levels) at which a tile reaches "
                        "full strength/reach")
    p.add_argument("--motion-floor", type=float, default=0.15,
                   help="adaptive: minimum strength/reach fraction even for a near-still tile")
    p.add_argument("--workers", type=int, default=4, help="parallel Atlas jobs (backend=atlas only)")
    p.add_argument("--max-jobs", type=int, default=100,
                   help="refuse to submit more NEW Atlas jobs than this (backend=atlas only)")
    p.add_argument("--look", default="all",
                   help=f"comma-separated looks to write, or 'all' (default): {', '.join(looks.LOOKS)}")
    p.add_argument("--gain", type=float, default=None, help="echo gain (per-look default if omitted)")
    p.add_argument("--dark", type=float, default=0.25, help="ghost look: darkening by removed light")
    p.add_argument("--split", action="store_true", help="source | result side by side, per look")
    p.add_argument("--stack", action="store_true",
                   help="also write <out>_stack.mp4: source + every requested look except "
                        "'quantum', side by side in one wide video")
    p.add_argument("--stack-only", action="store_true",
                   help="like --stack, but skip the individual <out>_<look>.mp4 files (and the "
                        "compositing work behind them) entirely - just the combined video")
    p.add_argument("--fps-div", type=int, default=1,
                   help="write only every Nth frame, at fps/N (e.g. 2 for half fps on a >30fps "
                        "source) - output rendering only, the echo is still computed at full "
                        "native frame rate (default 1: no change)")
    p.add_argument("--spatial", action="store_true",
                   help="also blur SPATIALLY via Atlas blur-v1 (a second, different quantum "
                        "engine), masked by that chunk's own echo magnitude - up to one extra "
                        "Atlas job per output chunk, only on the faithful/ghost looks")
    p.add_argument("--spatial-strength", type=float, default=0.85, help="blur-v1 strength (0-1) "
                   "before echo scaling")
    p.add_argument("--spatial-reach", type=float, default=0.5, help="blur-v1 reach (0-1) before "
                   "echo scaling")
    p.add_argument("--spatial-style", default="rx", choices=["rx", "ry"], help="blur-v1 style")
    p.add_argument("--spatial-echo-ref", type=float, default=14.0,
                   help="mean |echo| (grey levels) at which a chunk gets the full "
                        "--spatial-strength/--spatial-reach; below it, scaled down proportionally")
    p.add_argument("--spatial-mask-gamma", type=float, default=0.5,
                   help="<1 spreads mid-range echo toward a stronger mask instead of only the "
                        "frame's single peak pixel reaching full blend weight (1.0 = off); this "
                        "matters more than strength/reach for how visible --spatial is, since "
                        "blur-v1's blend weight IS the mask value")
    p.add_argument("--echo-audio", action="store_true",
                   help="replace the source audio with it run through Atlas retrocausal-echo-v1 "
                        "(a third, different quantum engine), parameters mirroring --strength/"
                        "--reach - one Atlas job for the whole clip, cached like any other")
    p.add_argument("--fps", type=float, default=24, help="synth only")
    return p


def main():
    args = build_parser().parse_args()
    args.stack = args.stack or args.stack_only
    if args.command != "synth" and not (0 <= args.strength <= 1 and 0 <= args.reach <= 1):
        raise SystemExit("--strength and --reach must be in [0,1]")
    if args.command != "synth" and args.hop != BLOCK // 2:
        raise SystemExit(f"--hop must be {BLOCK // 2} (the sin^2 crossfade sums to 1 only for half-overlap)")
    if args.fps_div < 1:
        raise SystemExit("--fps-div must be >= 1")
    {"plan": cmd_plan, "render": cmd_render, "relook": cmd_relook,
     "validate": cmd_validate, "synth": cmd_synth}[args.command](args)


if __name__ == "__main__":
    main()
