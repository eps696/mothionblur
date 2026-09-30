"""Local model of Atlas blur-core-v1's time-axis behaviour, built and simulated with Qiskit.

Same fitted model as emulator.py (identical Gray-code encoding and rotation angles, see
emulator.angles), and a drop-in replacement for it: same blur_time/blur_volume contract, same
numbers (tests/test_engines.py runs the same test suite against both). The difference is only
*how* the rotation is computed:

- emulator.py: a hand-written reshape trick that applies each qubit's 2x2 rotation directly.
- this module: an actual `qiskit.QuantumCircuit` with .rx()/.ry() gates, whose exact unitary is
  extracted via `qiskit.quantum_info.Operator` and applied to every pixel-column's initial state.

Building the circuit and reading off its unitary IS running it through Qiskit - a circuit made of
only single-qubit gates has no entanglement to lose by doing so, and every column shares the same
circuit (columns differ only in their initial amplitudes), so building it once per call and
applying the resulting matrix to every column in one batched multiply is exact, not an
approximation - mathematically identical to calling `Statevector(col).evolve(circuit)` separately
for each of the (up to ~19,000) columns in a frame, just without doing that once per column.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from emulator import GRAY, N_QUBITS, T, angles


def build_circuit(strength: float, reach: float = 0.0, style: str = "x") -> QuantumCircuit:
    """The real Qiskit circuit: one Rx/Ry per qubit (per letter in ``style``, in order),
    with the same fitted angles emulator.py uses."""
    qc = QuantumCircuit(N_QUBITS, name=f"time_blur(s={strength},r={reach},{style})")
    for k, theta in enumerate(angles(strength, reach)):
        for gate in style:
            (qc.rx if gate == "x" else qc.ry)(theta, k)
    return qc


def blur_time(columns: np.ndarray, strength: float, reach: float = 0.0,
              style: str = "x") -> np.ndarray:
    """columns: [C, 64] non-negative (time last). Returns same shape, column sums kept."""
    columns = np.asarray(columns, dtype=np.float64)
    if columns.ndim != 2 or columns.shape[1] != T:
        raise ValueError(f"expected [C, {T}] columns")
    if (columns < 0).any() or not set(style) <= {"x", "y"} or not style:
        raise ValueError("values must be >= 0 and style a non-empty mix of x,y")
    sums = columns.sum(axis=1, keepdims=True)
    psi0 = np.zeros(columns.shape, dtype=np.complex128)
    psi0[:, GRAY] = np.sqrt(columns / np.maximum(sums, 1e-300))
    unitary = Operator(build_circuit(strength, reach, style)).data   # Qiskit's own simulation
    psi = psi0 @ unitary.T                                          # apply to every column at once
    prob = np.abs(psi) ** 2
    return prob[:, GRAY] * sums


def blur_volume(vol: np.ndarray, strength: float, reach: float = 0.0,
                style: str = "x") -> np.ndarray:
    """vol: [64, ...spatial]; blur along axis 0 only. Same contract as emulator.blur_volume."""
    t = vol.shape[0]
    flat = np.moveaxis(vol.reshape(t, -1), 0, 1)
    return np.moveaxis(blur_time(flat, strength, reach, style), 1, 0).reshape(vol.shape)
