"""Both local engines (emulator.py: hand-derived NumPy; qiskit_engine.py: a real Qiskit circuit)
implement the identical fitted model and must be interchangeable - every test here runs against
both. atlas.py (the third engine, real Atlas jobs) is checked separately in test_core.py/manually,
since it needs a network call and a key."""
import json
from pathlib import Path

import numpy as np
import pytest

import emulator
import qiskit_engine

ENGINES = [emulator, qiskit_engine]
FIX = json.loads((Path(__file__).parent / "fixtures" / "atlas_probe.json").read_text())


def probe_grid():
    """Same [T,4] grid as calibration.py's impulse probe: impulse@32, impulse@10, constant, step@32."""
    g = np.zeros((64, 4))
    g[32, 0] = 1.0
    g[10, 1] = 1.0
    g[:, 2] = 0.5
    g[32:, 3] = 1.0
    return g


@pytest.fixture(params=ENGINES, ids=lambda m: m.__name__)
def engine(request):
    return request.param


@pytest.mark.parametrize("fx", FIX, ids=lambda f: f"s{f['strength']}_r{f['reach']}")
def test_matches_real_atlas_jobs(engine, fx):
    """Both engines reproduce real blur-core-v1 output (after per-column renorm)."""
    g = probe_grid()
    model = engine.blur_volume(g, fx["strength"], fx["reach"], fx["style"])
    atlas = np.asarray(fx["output"])
    atlas = atlas * (g.sum(0) / atlas.sum(0))  # engine's global rescale differs
    assert np.abs(model - atlas).max() < 2e-3


def test_strength_zero_is_identity(engine):
    g = np.random.default_rng(1).random((64, 5))
    np.testing.assert_allclose(engine.blur_volume(g, 0.0, 0.5), g, atol=1e-12)


def test_static_column_is_invariant_for_any_strength_and_reach(engine):
    col = np.full((64, 1), 0.7)
    for s, r in [(0.3, 0.0), (1.0, 0.0), (0.5, 1.0), (0.9, 0.4)]:
        np.testing.assert_allclose(engine.blur_volume(col, s, r), col, atol=1e-12)


def test_column_sums_are_conserved(engine):
    g = np.random.default_rng(2).random((64, 7)) * 255
    out = engine.blur_volume(g, 0.6, 0.3)
    np.testing.assert_allclose(out.sum(0), g.sum(0), rtol=1e-10)
    assert (out >= 0).all()


def test_reach_moves_weight_far_in_time(engine):
    imp = np.zeros((64, 1))
    imp[32, 0] = 1.0
    near = engine.blur_volume(imp, 0.3, 0.0)[:, 0]
    far = engine.blur_volume(imp, 0.3, 0.9)[:, 0]
    assert far[[0, 8, 63]].sum() > 20 * near[[0, 8, 63]].sum()


def test_columns_are_independent(engine):
    """Exact tiling relies on this: a column's result ignores its neighbours."""
    rng = np.random.default_rng(3)
    a, b = rng.random((64, 1)), rng.random((64, 1))
    joint = engine.blur_volume(np.hstack([a, b]), 0.5, 0.4)
    np.testing.assert_allclose(joint[:, :1], engine.blur_volume(a, 0.5, 0.4), atol=1e-12)
    np.testing.assert_allclose(joint[:, 1:], engine.blur_volume(b, 0.5, 0.4), atol=1e-12)


def test_rejects_bad_input(engine):
    with pytest.raises(ValueError):
        engine.blur_volume(np.ones((32, 2)), 0.3)
    with pytest.raises(ValueError):
        engine.blur_volume(-np.ones((64, 2)), 0.3)


def test_qiskit_and_numpy_engines_agree_on_random_input():
    """The two free/local backends must be numerically interchangeable, not just individually
    correct - this is the direct cross-check ('demonstrate quantum usage' with a real circuit
    that provably matches the fast NumPy path used for tiling/previewing)."""
    g = np.random.default_rng(4).random((64, 11)) * 200
    for strength, reach, style in [(0.3, 0.55, "x"), (0.7, 0.2, "y"), (0.5, 0.8, "xy")]:
        a = emulator.blur_volume(g, strength, reach, style)
        b = qiskit_engine.blur_volume(g, strength, reach, style)
        np.testing.assert_allclose(a, b, atol=1e-9)
