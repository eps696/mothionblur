# Quantum time-smear

A video effect for [Moth Hack 2026](https://hack.mothquantum.com), built on Moth's
[Atlas](https://platform.mothquantum.com) quantum-computing platform — using **three distinct
Atlas engines** across one coherent effect, plus a from-scratch NumPy model and a real Qiskit
circuit, both independently verified to reproduce Atlas's own numbers.

It gives every moving thing in a video echoes of itself at other moments in time — including
moments that haven't happened yet — optionally adds a second, spatial quantum blur exactly where
that displacement is strongest, and optionally echoes the clip's own audio with a third quantum
engine, in step with the video.

## What it does

Normal temporal effects (motion blur, trails, feedback) are a fixed, causal, forward-looking
kernel: light from frame *t* can only smear into *t+1, t+2, ...*. This instead sends each pixel's
brightness-over-time through Atlas's `blur-core-v1` engine, which mixes it via quantum
interference along a *Gray-coded* time axis. A moving object's echoes appear at specific,
non-adjacent frames on **both sides** of the real event — before it happens and after it's gone —
with a binary (Gray-code) structure to the offsets, not a smooth spread.

Concretely, for one pixel's 64-frame brightness history:

1. **Encode.** The 64 values become amplitudes on 6 qubits, addressed by the Gray code of the
   frame number (adjacent frames differ by one bit).
2. **Rotate.** Each qubit gets a small `Rx`/`Ry` rotation: `--strength` sets the angle, `--reach`
   shifts weight from low-order qubits (nearby frames) to high-order ones (far away in time).
3. **Measure.** The new probabilities become the new brightness-over-time.

Total brightness per pixel is conserved (light moves in time, it isn't created), and a pixel whose
value never changes is a fixed point of the rotation — **a static background is left exactly
alone**, only the moving parts of a scene get ghosts.

## Why this isn't "just a blur"

- **A real, undocumented use of the engine**: `blur-core-v1` blurs an arbitrary N-dimensional grid
  of numbers; a video is a 3-D grid (time, row, column), and this applies the blur along time only.
- **Non-linear**, unlike a classical kernel: blurring `a+b` measurably differs from blurring `a`
  and `b` separately (a superposition test gives an error where a linear kernel gives exactly
  zero). The nearest classical bit-mixing kernel with the same per-qubit probabilities still
  differs from the real result by roughly half the echo's own size on real footage.
- **No quantum-advantage claim.** Atlas doesn't document whether `blur-core-v1` runs on a QPU or a
  simulator; either way, this project's own re-implementations (`emulator.py`, `qiskit_engine.py`)
  reproduce its numbers to ~0.05 grey levels — a small, classically-simulable circuit. What's
  genuinely "quantum" here is the interference structure of the effect, not a hardware claim.

## How it's computed

**Resolution.** The quantum step only ever sees **luminance** (Rec.709) at a reduced working
width (`--qwidth`, default 160px) — Atlas caps a job's result payload at ~2MB (~90k values), so a
full-resolution clip is infeasible in one job. The echo (`Atlas output − source`) is computed at
that size, then upsampled (bilinear, on the GPU) and added onto the **untouched, full-detail
source frames** — faces, text, edges come from the source; only the ghost layer is soft.
**Colour is not invented**: `faithful` adds the same greyscale echo to a source's own R/G/B
equally, so hue survives and black-and-white stays black-and-white (`ghost`/`echo` use an
explicitly cosmetic palette instead).

**Tiling is exact, not approximate.** `blur-core-v1` takes an N-dimensional grid, so a
`[64 frames, 36 rows, 36 cols]` tile is one job — an 18-qubit state (6 time + 6 row + 6 column)
holding 1,296 independent pixel-columns side by side. Only the time qubits rotate, and Gray-coded
time blur acts on each pixel-column completely independently of its neighbours, so splitting a
frame into tiles (`volume.py`) gives bit-identical results to one hypothetical whole-frame job.
Atlas rescales each job's output to its own maximum, so every tile is renormalised back to its
known input sum before stitching (no seams). A tile that doesn't change by more than 1 grey level
needs no Atlas call at all.

**Any length, via sliding windows.** One Atlas call only ever covers 64 frames (6 qubits). Longer
clips are cut into overlapping 64-frame windows (hop 32) and crossfaded with `sin²`/`cos²` weights
that sum to exactly 1. This *softens* the seam between windows but doesn't eliminate it — checked
directly with an isolated flash near a window boundary, ~90% of its echo stays in its own window,
~10% spills into the neighbour, because a frame's position inside its 64-frame register changes
its Gray-code address and hence its echo pattern.

## Three interchangeable compute engines

`--backend {numpy, qiskit, atlas}` (default `numpy`, so nothing bills unless you say `atlas`)
selects what actually performs the rotation. All three implement the identical fitted model and
are verified numerically interchangeable (`tests/test_engines.py` runs the same suite against
both local engines, plus a direct cross-check on random input):

| Backend | What it is | Cost |
|---|---|---|
| `numpy` | Hand-derived reshape trick applying each qubit's 2×2 rotation to a batch of pixel-columns at once (`emulator.py`) | Free, instant |
| `qiskit` | The *same* rotation, built and run as a real `qiskit.QuantumCircuit`, unitary extracted via `Operator` and applied to every pixel-column in one batched multiply (`qiskit_engine.py`) — genuinely runs the circuit, not a relabelled copy of the NumPy path | Free, fast |
| `atlas` | Real Atlas `blur-core-v1` jobs — tiled, cached (`work/jobs/`), cost-guarded (`--max-jobs`) | Real credits |

`validate` runs any backend against the `numpy` baseline and reports the agreement.

## Features

| Flag | What it does |
|---|---|
| `--adaptive` | Each 36×36 tile gets its own strength/reach, scaled by how much that specific tile actually moves (`--motion-ref`/`--motion-floor`), instead of one clip-wide value. Lets you push the base strength higher for dramatic effect where things move, without over-ghosting where they don't. Works on all three backends (`volume.blur_tiled` for the local engines calls `emulator.py`/`qiskit_engine.py` unchanged, once per tile). Run `plan --adaptive` first — it reports the clip's actual motion range so you can calibrate `--motion-ref` before spending anything. |
| `--spatial` | A **second**, genuinely different Atlas engine (`blur-v1`, spatial interference blur instead of time-axis rotation), masked by that chunk's own echo magnitude — spatial blur lands where the temporal pass shows the most displacement. Cost-bounded to one job per 64-frame chunk (not per frame), held as a pixel delta added to every frame in that chunk. `--spatial-mask-gamma` (default 0.5) matters more than `--spatial-strength`/`--spatial-reach` for visibility, since `blur-v1`'s blend weight *is* the mask value and a frame's echo is rarely near its own peak everywhere. |
| `--echo-audio` | A **third** Atlas engine (`retrocausal-echo-v1`, a quantum delay line) echoes the clip's own audio, with `depth`/`feedback`/`decay` mapped from the video's `--reach`/`--strength` so the soundtrack pre-/post-echoes in step with the picture. One job for the whole clip. |
| `--stack` / `--stack-only` | Also (or only) write one combined video: source + every look except `quantum`, side by side. `--stack-only` skips the individual per-look files *and* the compute behind them. |
| `--fps-div N` | Write only every Nth frame, at `fps/N` — for >30fps sources. Output-only: the echo is always computed at native frame rate (decimating first would distort the echo pattern itself, not just its cost). |

## Looks

Every `render`/`relook` call writes any subset of four looks by default (composited on the GPU,
`looks.py` — the heaviest per-pixel workload in the pipeline; moving it off the CPU cut a 5-output
render from minutes to under a minute):

| Look | Shows | Colour |
|---|---|---|
| `faithful` | Source + its own echo (+ `--spatial`'s delta if set) | Real: source's own hue |
| `ghost` | Artistic: added light as an iridescent glow over the source | Invented palette |
| `quantum` | Atlas's raw output (`V + echo`) alone, decoupled from the source | Real, grayscale |
| `echo` | **Only** the echo, no source pixels: warm = light added, cool = light removed | Artistic ramp, real data |

`--gain`/`--dark` tune the composite; `--split` puts source and result side by side.

## QC data: compute once, look many times

`render` saves the expensive part once: `<out>.echo.npy` (the crossfaded echo volume) and
`<out>.qc.json` (every parameter needed to reproduce it). `relook <out>.qc.json <out2> --look ...`
re-composites **any** look — including `--stack`, `--spatial`, `--echo-audio` — from that saved
data, with no Atlas/emulator/qiskit calls for the video itself.

## Setup

- **Python 3.11+**, `pip install -r requirements.txt` (see that file's comment on installing a
  CUDA-enabled `torch` build — `pip install torch` alone can resolve to CPU-only on some
  platforms).
- **ffmpeg + ffprobe** on `PATH`.
- **An NVIDIA GPU with working CUDA** — hardcoded, no CPU fallback, for both video encoding
  (NVENC) and the GPU compositing pipeline. Check with
  `python -c "import torch; print(torch.cuda.is_available())"`.
- **A Moth Atlas API key**, only needed for `--backend atlas`, `--spatial`, or `--echo-audio`:
  `set MOTH_API_KEY=...` (or export it) before running.

```bat
pip install -r requirements.txt
set MOTH_API_KEY=your-key-here
python timesmear.py synth work\test.mp4
python timesmear.py render work\test.mp4 work\out --split          :: free preview, backend=numpy
python timesmear.py plan   work\test.mp4 --backend atlas            :: count real jobs first
python timesmear.py render work\test.mp4 work\out --backend atlas --max-jobs 200
python timesmear.py relook work\out.qc.json work\out2 --look ghost --gain 5
```

## Command-line reference

```
python timesmear.py plan     SRC              [compute opts]                 # count jobs, free
python timesmear.py render   SRC OUT          [compute opts] [look opts] [--max-jobs N]
python timesmear.py relook   OUT.qc.json OUT2 [look opts]                    # from saved QC
python timesmear.py validate SRC              [compute opts]                 # backend vs numpy
python timesmear.py synth    OUT.mp4          [--frames N] [--fps F]        # test clip
```

**Compute opts** (what gets computed — fixed once a render's QC is saved):
`--backend numpy|qiskit|atlas` `--strength` `--reach` `--style x|y|xy|yx` `--start` `--frames`
`--hop` `--qwidth` `--adaptive` `--motion-ref` `--motion-floor` `--workers`

**Look opts** (presentation — free to change any time via `relook`):
`--look all|faithful,ghost,...` `--gain` `--dark` `--split` `--stack` `--stack-only` `--fps-div`
`--spatial` `--spatial-strength` `--spatial-reach` `--spatial-style` `--spatial-echo-ref`
`--spatial-mask-gamma` `--echo-audio`

`OUT` is a **prefix**: `render clip.mp4 result` writes `result_faithful.mp4`, `result_ghost.mp4`,
`result_quantum.mp4`, `result_echo.mp4`, plus `result.echo.npy`/`result.qc.json`.

## Repository layout

| File | Role |
|---|---|
| `atlas.py` | Atlas HTTP client: job cache, content-hash asset cache, file-upload/file-output helpers |
| `emulator.py` | Local NumPy model of `blur-core-v1`'s time-axis behaviour |
| `qiskit_engine.py` | The same model as a real `qiskit.QuantumCircuit` |
| `volume.py` | Exact tiling into Atlas-sized jobs, static-tile skipping, stitching, motion-adaptive strength |
| `video.py` | ffmpeg/ffprobe I/O: probing, streamed read/write (NVENC), audio extraction |
| `looks.py` | Echo → pixels on the GPU, all four looks; QC save/load |
| `secondary_engines.py` | `blur-v1` (spatial) and `retrocausal-echo-v1` (audio) round trips |
| `progress.py` | Terminal progress bar with ETA |
| `timesmear.py` | CLI: windowing/crossfade, orchestration, all subcommands |
| `calibration.py` | Dev-only: regenerates the engine test fixture, measures Atlas's limits/contracts |
| `tests/test_engines.py` | `numpy`/`qiskit` engines vs. real Atlas jobs + vs. each other |
| `tests/test_core.py` | Tiling/stitching, motion-adaptive strength, Atlas job/asset cache |
| `tests/test_pipeline.py` | Windowing/crossfade, looks, QC persistence, echo-audio param mapping |

## Known limitations

- Simulated circuits (no quantum-advantage claim); Atlas's own backend for `blur-core-v1`/`blur-v1`
  isn't documented either way.
- Window seams are softened, not eliminated.
- `--spatial` holds one `blur-v1` result per 64-frame chunk rather than per frame — a deliberate
  cost/fidelity tradeoff. Visually this shows up as the spatial-blur pattern staying fixed for
  ~2 seconds then jumping to a new one at the next chunk — a "frozen imprint with motion on top"
  look, not a bug.
- Atlas's per-job credit cost isn't exposed by its API — `plan`'s time estimates are from measured
  job latency, not credits.
- 64-frame register is fixed by the fitted model; changing it needs re-calibration
  (`calibration.py impulse`).
- NVENC and CUDA are hardcoded with no CPU fallback (this project targets the machine it was built
  on).
