from __future__ import annotations

import jax.numpy as jnp

from iqml.vqe.chemistry import PauliHamiltonian


def build_lstm_vqe_inputs(
    hamiltonian: PauliHamiltonian,
    depth: int,
    spacing: float,
) -> jnp.ndarray:
    """Build deterministic physical/structural inputs for LSTM-VQE layers.

    Feature columns:
    0. normalized layer index in [0, 1]
    1. bond spacing in Angstrom
    2. number of qubits / 32
    3. number of Pauli terms / 1024
    4. mean absolute Pauli coefficient
    5. L2 norm of Pauli coefficients
    6. max absolute Pauli coefficient
    """
    if depth <= 0:
        raise ValueError("depth must be positive")
    coeffs = hamiltonian.coefficient_array
    abs_coeffs = jnp.abs(coeffs)
    if coeffs.size == 0:
        raise ValueError("hamiltonian must contain at least one Pauli term")

    denom = max(depth - 1, 1)
    layer_index = jnp.arange(depth, dtype=jnp.float32) / float(denom)
    static = jnp.asarray(
        [
            float(spacing),
            hamiltonian.num_qubits / 32.0,
            len(hamiltonian.terms) / 1024.0,
            float(jnp.mean(abs_coeffs)),
            float(jnp.linalg.norm(coeffs)),
            float(jnp.max(abs_coeffs)),
        ],
        dtype=jnp.float32,
    )
    repeated = jnp.tile(static[None, :], (depth, 1))
    return jnp.concatenate([layer_index[:, None], repeated], axis=-1)
