import json

import numpy as np
import pytest

import atlas as atlas_mod
import emulator
import volume


class FakeAtlas:
    """Behaves like Atlas.run for blur-core-v1 but computes locally, with the engine's
    per-tile global rescale imitated by a random gain."""

    def __init__(self):
        self.calls = 0
        self.rng = np.random.default_rng(0)

    def run(self, engine, params, input_files=None, **kw):
        assert engine == "blur-core-v1"
        self.calls += 1
        vals = np.asarray(params["values"], dtype=float)
        out = emulator.blur_volume(vals, params["strength"][0], params["reach"], params["style"])
        out = out * self.rng.uniform(0.8, 1.3)
        return {"job_id": f"fake-{self.calls:04d}-xxxxxxxx", "result": {"output": out.tolist()},
                "cached": False, "seconds": 0.0}


def moving_volume(h=130, w=150, seed=0):
    rng = np.random.default_rng(seed)
    v = np.full((64, h, w), 40.0)
    for t in range(64):
        x = 10 + 2 * t
        v[t, 30:60, x % (w - 20): x % (w - 20) + 20] = 200 + rng.integers(0, 40)
    return v


def test_plan_tiles_partition_the_frame_exactly():
    tiles = volume.plan_tiles(130, 150, 56)
    cover = np.zeros((130, 150), int)
    for y0, y1, x0, x1 in tiles:
        cover[y0:y1, x0:x1] += 1
    assert (cover == 1).all()


def test_tiled_result_equals_whole_volume_emulation_despite_per_tile_gain():
    v = moving_volume()
    fake = FakeAtlas()
    out, jobs = volume.blur_via_atlas(fake, v, 0.5, 0.4, log=lambda *_: None, progress=False)
    ref = emulator.blur_volume(np.rint(v), 0.5, 0.4)
    np.testing.assert_allclose(out, ref, atol=1e-3)


def test_static_tiles_are_skipped_and_returned_unchanged():
    v = moving_volume()
    v[:, :, 100:] = 90.0                     # right part fully static
    fake = FakeAtlas()
    out, _ = volume.blur_via_atlas(fake, v, 0.5, 0.4, log=lambda *_: None, progress=False)
    assert fake.calls < len(volume.plan_tiles(130, 150))
    np.testing.assert_allclose(out[:, :, 112:], 90.0)


def test_all_static_volume_makes_no_calls():
    fake = FakeAtlas()
    volume.blur_via_atlas(fake, np.full((64, 60, 60), 77.0), 0.5, 0.2, log=lambda *_: None, progress=False)
    assert fake.calls == 0


def test_worst_case_tile_fits_request_and_result_limits():
    worst = np.full((64, volume.TILE, volume.TILE), 255)
    size = len(json.dumps(volume.make_params(worst, 0.5, 0.5, "x"), separators=(",", ":")))
    assert size < volume.MAX_BODY < 1_048_576
    # Result is ~19.4 B/value (measured); Atlas fails jobs above ~2 MB of payload.
    assert worst.size <= volume.MAX_RESULT_VALUES
    assert worst.size * 20 < 2_000_000 * 0.85


def test_oversized_tile_is_refused_before_any_job():
    fake = FakeAtlas()
    with pytest.raises(ValueError, match="result"):
        volume.blur_via_atlas(fake, moving_volume(140, 140), 0.3, 0.3, tile=56,
                              log=lambda *_: None, progress=False)
    assert fake.calls == 0


def test_wrong_block_length_rejected():
    with pytest.raises(ValueError):
        volume.blur_via_atlas(FakeAtlas(), np.ones((32, 8, 8)), 0.3)


def test_renorm_keeps_input_column_sums_and_zero_columns():
    ref = np.zeros((64, 2))
    ref[:, 0] = 3.0
    out = np.ones((64, 2)) * 5
    r = volume.renorm_columns(out, ref)
    np.testing.assert_allclose(r.sum(0), [192.0, 0.0])


def test_tile_motion_and_motion_factor():
    still = np.full((64, 4, 4), 100.0)
    assert volume.tile_motion(still) == 0.0
    noisy = np.zeros((64, 1, 1))
    noisy[:, 0, 0] = np.linspace(0, 80, 64)
    assert volume.tile_motion(noisy) > 20
    assert volume.motion_factor(0, 40, 0.15) == pytest.approx(0.15)
    assert volume.motion_factor(40, 40, 0.15) == pytest.approx(1.0)
    assert volume.motion_factor(1000, 40, 0.15) == pytest.approx(1.0)   # saturates, never > 1
    mid = volume.motion_factor(20, 40, 0.15)
    assert 0.15 < mid < 1.0


def test_tile_strength_reach_scales_by_motion_when_adaptive():
    still = np.full((64, 4, 4), 100.0)
    busy = np.zeros((64, 4, 4)) + np.linspace(0, 200, 64)[:, None, None]
    assert volume.tile_strength_reach(still, 0.6, 0.5, False, 40.0, 0.15) == (0.6, 0.5)
    s_still, r_still = volume.tile_strength_reach(still, 0.6, 0.5, True, 40.0, 0.15)
    s_busy, r_busy = volume.tile_strength_reach(busy, 0.6, 0.5, True, 40.0, 0.15)
    assert s_still == pytest.approx(0.6 * 0.15) and r_still == pytest.approx(0.5 * 0.15)
    assert s_busy == pytest.approx(0.6) and r_busy == pytest.approx(0.5)   # saturated
    assert s_still < s_busy and r_still < r_busy


def _two_tile_motion_volume():
    """72x72 = exactly 2x2 tiles of 36. Top-left moves a lot, top-right barely moves, bottom
    half is exactly static (skipped either way)."""
    h = w = 72
    v = np.full((64, h, w), 100.0)
    v[:, :36, :36] += np.linspace(0, 150, 64)[:, None, None]
    v[:, :36, 36:] += np.linspace(0, 3, 64)[:, None, None]
    return v


def test_blur_via_atlas_adaptive_gives_tiles_different_strength():
    """Integration: a high-motion tile and a low-motion (but not static) tile in the SAME call
    reach Atlas with measurably different strength/reach."""
    sent = []

    class RecordingAtlas(FakeAtlas):
        def run(self, engine, params, input_files=None, **kw):
            sent.append((params["strength"][0], params["reach"]))
            return super().run(engine, params, input_files, **kw)

    volume.blur_via_atlas(RecordingAtlas(), _two_tile_motion_volume(), 0.6, 0.5,
                          log=lambda *_: None, progress=False,
                          adaptive=True, motion_ref=40.0, motion_floor=0.15)
    assert len(sent) == 2                          # only the two non-static tiles were sent
    strengths = sorted(s for s, r in sent)
    assert strengths[0] < 0.6 * 0.3                 # the barely-moving tile stayed near the floor
    assert strengths[-1] > 0.6 * 0.9                # the busy tile stayed near full strength


def test_blur_tiled_matches_whole_window_when_not_adaptive():
    """Regression safety net: tiling a local engine's call must be exact, same as it is for Atlas."""
    v = moving_volume()
    tiled = volume.blur_tiled(emulator.blur_volume, v, 0.5, 0.4, "x")
    whole = emulator.blur_volume(np.rint(v), 0.5, 0.4, "x")
    np.testing.assert_allclose(tiled, whole, atol=1e-3)


def test_blur_tiled_adaptive_modulates_per_tile():
    v = _two_tile_motion_volume()
    out = volume.blur_tiled(emulator.blur_volume, v, 0.6, 0.5, "x",
                            adaptive=True, motion_ref=40.0, motion_floor=0.15)
    busy_echo = np.abs(out[:, :36, :36] - v[:, :36, :36]).mean()
    calm_echo = np.abs(out[:, :36, 36:] - v[:, :36, 36:]).mean()
    assert busy_echo > calm_echo * 3                # visibly more echo where there's more motion


def test_job_cache_prevents_rebilling(tmp_path, monkeypatch):
    a = atlas_mod.Atlas(work=tmp_path, key=" secret-key \n")
    assert a._key == "secret-key"           # env.bat leaves trailing whitespace
    calls = []

    def fake_request(method, path, data=None, timeout=120):
        calls.append((method, path))
        if method == "POST":
            return {"job_id": "j1", "status": "queued", "submitted_at": "now"}
        if path.endswith("/status"):
            return {"status": "completed"}
        return {"result": {"output": [[1.0]]}, "outputs": []}

    monkeypatch.setattr(a, "_request", fake_request)
    p = {"values": [[1, 2]], "axes": [0], "strength": [0.3]}
    first = a.run("blur-core-v1", p)
    n = len(calls)
    second = a.run("blur-core-v1", p)
    assert n > 0 and len(calls) == n and second["cached"] and not first["cached"]
    assert second["result"] == first["result"]


def test_upload_bytes_caches_by_content_hash(tmp_path, monkeypatch):
    """Fixes the bug found while debugging --spatial: every upload used to get a fresh asset id
    even for byte-identical content, so file-based engines (blur-v1, retrocausal-echo-v1) never
    hit Atlas.run()'s job cache (which keys off input_files' asset ids) - repeat/identical spatial
    blur calls kept re-billing. This checks the upload itself is now cached by content hash."""
    a = atlas_mod.Atlas(work=tmp_path, key="k")
    calls = []

    def fake_request(method, path, data=None, timeout=120):
        calls.append((method, path))
        if path == "/assets":
            return {"asset_id": f"asset-{len(calls)}",
                    "upload": {"url": "http://example", "headers": {}, "method": "PUT"}}
        return {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b""

    monkeypatch.setattr(a, "_request", fake_request)
    monkeypatch.setattr(atlas_mod.urllib.request, "urlopen", lambda req, timeout=300: FakeResponse())

    id1 = a.upload_bytes(b"hello", "f.png", "image/png")
    n = len(calls)
    id2 = a.upload_bytes(b"hello", "f.png", "image/png")     # same bytes -> cache hit, no upload
    assert id1 == id2
    assert len(calls) == n
    id3 = a.upload_bytes(b"different bytes", "f.png", "image/png")  # different content -> fresh upload
    assert id3 != id1
    assert len(calls) > n


def test_cached_file_output_job_refreshes_its_download_url(tmp_path, monkeypatch):
    """Hit for real: --echo-audio on the same clip+params a second time (job cache hit, thanks
    to upload_bytes now reusing a stable asset id) got HTTP 403 downloading the CACHED presigned
    URL, which had already expired (~900s TTL from when it was first issued, not from the cache
    hit). A cache hit for a file-output job must re-fetch a fresh URL - but only for jobs that
    actually have file outputs; a cache hit for an inline-result job (blur-core-v1) must NOT pay
    for an extra network call it doesn't need."""
    a = atlas_mod.Atlas(work=tmp_path, key="k")
    calls = []

    def fake_request_file_job(method, path, data=None, timeout=120):
        calls.append((method, path))
        if method == "POST":
            return {"job_id": "j1", "status": "queued", "submitted_at": "now"}
        if path.endswith("/status"):
            return {"status": "completed"}
        # every /result call returns a "fresh" url distinguishable by call count
        return {"result": None, "outputs": [{"slot": "result", "url": f"https://x/{len(calls)}"}]}

    monkeypatch.setattr(a, "_request", fake_request_file_job)
    p = {"strength": 0.3}
    first = a.run("retrocausal-echo-v1", p, input_files={"audio": "asset-1"})
    first_url = first["outputs"][0]["url"]
    n = len(calls)
    second = a.run("retrocausal-echo-v1", p, input_files={"audio": "asset-1"})
    assert second["cached"] is True
    assert len(calls) == n + 1                      # exactly one extra call: the URL refresh
    assert second["outputs"][0]["url"] != first_url  # got a NEW url, not the stale cached one

    # An inline-result job (no file outputs) must NOT trigger that extra call on a cache hit.
    calls.clear()

    def fake_request_inline_job(method, path, data=None, timeout=120):
        calls.append((method, path))
        if method == "POST":
            return {"job_id": "j2", "status": "queued", "submitted_at": "now"}
        if path.endswith("/status"):
            return {"status": "completed"}
        return {"result": {"output": [1.0]}, "outputs": []}

    monkeypatch.setattr(a, "_request", fake_request_inline_job)
    a.run("blur-core-v1", {"strength": [0.3]})
    n = len(calls)
    cached = a.run("blur-core-v1", {"strength": [0.3]})
    assert cached["cached"] is True
    assert len(calls) == n                            # no extra network call for an inline job
