"""Local model of Atlas ``blur-core-v1`` along a 64-step time axis.

Reverse-engineered from real Atlas jobs (see tests/fixtures/atlas_probe.json,
agreement to ~1e-3): every column of pixel values over time is amplitude-encoded
(sqrt of the values) on 6 qubits addressed by the *Gray code* of the time index;
each qubit k gets Rx(theta_k) with

    theta_k = pi * strength * w_k,   w_k = (1 - reach) * W0[k] + reach

so ``reach`` interpolates from local (weights fall off ~2**-k) to all qubits
rotating equally (non-local, distant echoes). Measured probabilities are read
back and, here, rescaled so each column keeps its input sum. The engine's own
global rescale differs between tiles, so the real pipeline applies the same
per-column renormalisation to Atlas output.

This is a PREVIEW tool for tuning looks without spending credits. Final renders
always use real Atlas jobs; results from here are labelled PREVIEW-emulated.
"""
import numpy as np

N_QUBITS = 6
T = 2 ** N_QUBITS
# Fitted local weights (theta_k / (pi * strength) at reach=0) for n=6.
W0 = np.array([1.0, 0.5, 0.2424, 0.1212, 0.0606, 0.0455])
GRAY = np.array([i ^ (i >> 1) for i in range(T)])


def angles(strength: float, reach: float = 0.0) -> np.ndarray:
    return np.pi * strength * ((1.0 - reach) * W0 + reach)


def _apply(psi: np.ndarray, k: int, theta: float, style: str) -> np.ndarray:
    """Apply the gates in ``style`` to qubit k of psi[C, T]."""
    for g in style:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        m = ((c, -1j * s), (-1j * s, c)) if g == "x" else ((c, -s), (s, c))
        p = psi.reshape(psi.shape[0], -1, 2, 2 ** k)
        a, b = p[:, :, 0, :].copy(), p[:, :, 1, :].copy()
        p[:, :, 0, :] = m[0][0] * a + m[0][1] * b
        p[:, :, 1, :] = m[1][0] * a + m[1][1] * b
    return psi


def blur_time(columns: np.ndarray, strength: float, reach: float = 0.0,
              style: str = "x") -> np.ndarray:
    """columns: [C, 64] non-negative (time last). Returns same shape, column sums kept."""
    columns = np.asarray(columns, dtype=np.float64)
    if columns.ndim != 2 or columns.shape[1] != T:
        raise ValueError(f"expected [C, {T}] columns")
    if (columns < 0).any() or not set(style) <= {"x", "y"} or not style:
        raise ValueError("values must be >= 0 and style a non-empty mix of x,y")
    sums = columns.sum(axis=1, keepdims=True)
    psi = np.zeros(columns.shape, dtype=np.complex128)
    psi[:, GRAY] = np.sqrt(columns / np.maximum(sums, 1e-300))
    for k, theta in enumerate(angles(strength, reach)):
        psi = _apply(psi, k, theta, style)
    prob = np.abs(psi) ** 2
    return prob[:, GRAY] * sums


def blur_volume(vol: np.ndarray, strength: float, reach: float = 0.0,
                style: str = "x") -> np.ndarray:
    """vol: [64, ...spatial]; blur along axis 0 only."""
    t = vol.shape[0]
    flat = np.moveaxis(vol.reshape(t, -1), 0, 1)
    return np.moveaxis(blur_time(flat, strength, reach, style), 1, 0).reshape(vol.shape)
