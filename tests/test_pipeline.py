"""Windowing/crossfade orchestration (timesmear.py) and presentation (looks.py): the pieces that
sit between the tiled Atlas call (test_core.py) and a finished video."""
import numpy as np
import pytest

import emulator
import looks
import timesmear as ts


def clip(n, h=12, w=14, seed=0):
    """A moving bright bar over a static background - no video file, no ffmpeg."""
    rng = np.random.default_rng(seed)
    v = np.full((n, h, w), 60.0, np.float32)
    for t in range(n):
        x = (3 * t) % (w - 3)
        v[t, 2:9, x:x + 3] = 200 + rng.integers(0, 40)
    return v


def blur(v, strength=0.3, reach=0.55, hop=32):
    echo, jobs, stats = ts.blur_windowed(v, strength, reach, "x", hop, backend="numpy")
    return echo


# ---------------------------------------------------------------- windows
@pytest.mark.parametrize("n,expected", [(10, [0]), (64, [0]), (65, [0, 1]), (68, [0, 4]),
                                        (96, [0, 32]), (100, [0, 32, 36]), (200, [0, 32, 64, 96, 128, 136])])
def test_plan_windows(n, expected):
    assert ts.plan_windows(n) == expected


@pytest.mark.parametrize("n", range(1, 400, 7))
def test_every_frame_is_covered_and_windows_fit(n):
    starts = ts.plan_windows(n)
    covered = np.zeros(max(n, 64), bool)
    for s in starts:
        assert s >= 0 and s + 64 <= max(n, 64)
        covered[s:s + 64] = True
    assert covered[:n].all()


def test_crossfade_weights_sum_to_one_and_are_positive():
    w = ts.window_weights()
    np.testing.assert_allclose(w[:32] + w[32:], 1.0, atol=1e-12)
    assert (w > 0).all()


def test_single_window_equals_plain_emulation():
    v = clip(64)
    echo = blur(v)
    q = emulator.blur_volume(np.rint(v), 0.3, 0.55)
    np.testing.assert_allclose(echo, q - v, atol=1e-3)


def test_short_clip_is_padded_and_trimmed():
    v = clip(40)
    echo = blur(v)
    assert echo.shape == v.shape and np.isfinite(echo).all()


def test_overlap_is_the_convex_blend_of_both_windows():
    v = clip(96)                                    # windows at 0 and 32
    echo = blur(v)
    e0 = emulator.blur_volume(np.rint(v[:64]), 0.3, 0.55) - v[:64]
    e1 = emulator.blur_volume(np.rint(v[32:96]), 0.3, 0.55) - v[32:96]
    w = ts.window_weights()
    t = 40                                          # inside the overlap 32..63
    expect = w[t] * e0[t] + w[t - 32] * e1[t - 32]
    wsum = w[t] + w[t - 32]
    np.testing.assert_allclose(echo[t], expect / wsum, atol=2e-3)


def flash_echo(at, n=160):
    v = np.full((n, 6, 6), 50.0, np.float32)
    v[at, 2:4, 2:4] = 220                                     # one flash, otherwise static
    return np.abs(blur(v)).sum((1, 2))


def test_crossfade_lets_echoes_cross_a_64_frame_boundary():
    """With hard 64-frame blocks a flash at 62 (or 66) has 100% of its echo before (after) frame 64.
    Overlapping windows must let a real share spill across; they soften the boundary, not remove it."""
    before, after = flash_echo(62), flash_echo(66)
    assert before[64:].sum() / before.sum() > 0.05          # post-echoes of a flash at 62 pass 64
    assert after[:64].sum() / after.sum() > 0.05            # pre-echoes of a flash at 66 pass 64
    assert before[:64].sum() > before[64:].sum()             # ...but most stays inside its window


def test_static_video_has_no_echo_at_any_length():
    v = np.full((150, 8, 8), 90.0, np.float32)
    assert np.abs(blur(v)).max() < 1e-6


# ---------------------------------------------------------------- looks
def frames_from(v, seed=0):
    """[T,h,w] gray volume -> a plausible [T,H,W,3] uint8 source at the same resolution."""
    rng = np.random.default_rng(seed)
    tint = rng.uniform(0.7, 1.3, 3).astype(np.float32)      # distinct per-channel scale = "colour"
    return np.clip(v[..., None] * tint, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name", looks.LOOKS)
def test_every_look_returns_valid_uint8_frames(name):
    v = clip(64, 20, 24)
    echo = blur(v)
    frames = frames_from(v)
    v_up = v if name == "quantum" else None
    out = looks.render_look(name, frames, echo, v_up)
    assert out.shape == frames.shape and out.dtype == np.uint8


def test_echo_look_shows_nothing_where_echo_is_zero_regardless_of_source():
    """'echo' must depict ONLY the echo: a look that peeked at the source would not be black
    over a static region, whatever colour that region is."""
    v = clip(64, 20, 24)
    echo = blur(v)
    static = np.abs(echo).max(0) < 1e-6
    assert static.any()
    bright_src = np.full((64, 20, 24, 3), 255, np.uint8)
    dark_src = np.zeros((64, 20, 24, 3), np.uint8)
    out_bright = looks.render_look("echo", bright_src, echo)
    out_dark = looks.render_look("echo", dark_src, echo)
    np.testing.assert_array_equal(out_bright[:, static], out_dark[:, static])
    assert (out_bright[:, static] == 0).all()


def test_faithful_look_preserves_colour_only_luminance_is_echoed():
    """A colour source must stay in colour: the echo is added identically to every channel,
    so per-pixel channel DIFFERENCES (the hue) survive except for clipping."""
    v = clip(64, 20, 24)
    echo = blur(v, strength=0.5, reach=0.7)
    frames = frames_from(v)
    out = looks.render_look("faithful", frames, echo, gain=1.0)
    delta = out.astype(np.int16) - frames.astype(np.int16)
    unclipped = ((out > 1) & (out < 254)).all(-1)          # same pixels, all 3 channels in range
    np.testing.assert_allclose(delta[..., 0][unclipped], delta[..., 1][unclipped], atol=1)
    np.testing.assert_allclose(delta[..., 1][unclipped], delta[..., 2][unclipped], atol=1)


def test_qc_round_trip_reproduces_all_looks_without_recomputing(tmp_path):
    v = clip(64, 20, 24)
    echo = blur(v)
    meta = {"src": "irrelevant.mp4", "start": 0, "frames": 64, "fps": 24.0, "qwidth": 24,
            "strength": 0.3, "reach": 0.55, "style": "x", "hop": 32, "backend": "numpy", "n_jobs": 0, "jobs": []}
    stem = str(tmp_path / "out")
    path = looks.save_qc(stem, echo, meta)
    assert path.endswith(".qc.json")
    echo2, meta2 = looks.load_qc(path)
    np.testing.assert_array_equal(echo2, echo)
    assert meta2 == meta
    echo3, meta3 = looks.load_qc(stem)                        # bare stem also accepted
    np.testing.assert_array_equal(echo3, echo)


def test_stack_selection_excludes_quantum():
    assert ts.stack_selection(["faithful", "ghost", "quantum", "echo"]) == ["faithful", "ghost", "echo"]
    assert ts.stack_selection(["quantum"]) == []
    assert ts.stack_selection(["ghost"]) == ["ghost"]


def test_echo_audio_params_for():
    import secondary_engines
    p0 = secondary_engines.echo_audio_params(0.0, 0.0)
    assert p0["depth"] == 4
    assert p0["feedback"] == pytest.approx(0.0)
    assert p0["decay"] == pytest.approx(0.98)
    p1 = secondary_engines.echo_audio_params(1.0, 1.0)
    assert p1["depth"] == 32
    assert p1["feedback"] == pytest.approx(0.6)
    assert p1["decay"] == pytest.approx(0.78)
    for p in (p0, p1):
        assert 0 <= p["feedback"] < 1          # API requires exclusiveMaximum 1
        assert 1 <= p["depth"] <= 32
