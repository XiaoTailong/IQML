from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import tensorcircuit as tc

from iqml.quantum.circuits import QuantumCircuitConfig


@dataclass(frozen=True)
class EntanglementSummary:
    entropy_by_layer: jnp.ndarray
    early_slope: jnp.ndarray
    max_entropy: jnp.ndarray
    final_entropy: jnp.ndarray


def adjacent_layer_cosines(theta: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Cosine similarity between consecutive layer parameter vectors.

    ``theta`` may be a single sample with shape ``(depth, params)`` or a batch
    with shape ``(batch, depth, params)``.
    """
    if theta.ndim == 2:
        theta = theta[None, ...]
    if theta.ndim != 3:
        raise ValueError("theta must have shape (depth, params) or (batch, depth, params)")
    if theta.shape[1] < 2:
        return jnp.zeros((theta.shape[0], 0), dtype=jnp.float32)

    left = theta[:, :-1, :]
    right = theta[:, 1:, :]
    numerator = jnp.sum(left * right, axis=-1)
    denom = jnp.linalg.norm(left, axis=-1) * jnp.linalg.norm(right, axis=-1)
    return numerator / jnp.maximum(denom, eps)


def adjacent_layer_l2_steps(theta: jnp.ndarray) -> jnp.ndarray:
    """L2 parameter increments between consecutive circuit layers."""
    if theta.ndim == 2:
        theta = theta[None, ...]
    if theta.ndim != 3:
        raise ValueError("theta must have shape (depth, params) or (batch, depth, params)")
    if theta.shape[1] < 2:
        return jnp.zeros((theta.shape[0], 0), dtype=jnp.float32)
    return jnp.linalg.norm(theta[:, 1:, :] - theta[:, :-1, :], axis=-1)


def adjacent_layer_pearsons(theta: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Pearson correlation between consecutive layer parameter vectors."""
    if theta.ndim == 2:
        theta = theta[None, ...]
    if theta.ndim != 3:
        raise ValueError("theta must have shape (depth, params) or (batch, depth, params)")
    if theta.shape[1] < 2:
        return jnp.zeros((theta.shape[0], 0), dtype=jnp.float32)

    left = theta[:, :-1, :]
    right = theta[:, 1:, :]
    left = left - jnp.mean(left, axis=-1, keepdims=True)
    right = right - jnp.mean(right, axis=-1, keepdims=True)
    numerator = jnp.sum(left * right, axis=-1)
    denom = jnp.linalg.norm(left, axis=-1) * jnp.linalg.norm(right, axis=-1)
    return numerator / jnp.maximum(denom, eps)


def half_chain_entropy_from_state(
    state: jnp.ndarray,
    num_qubits: int,
    cut: int | None = None,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Von Neumann entropy of a pure state across a half-chain bipartition."""
    if cut is None:
        cut = num_qubits // 2
    if cut <= 0 or cut >= num_qubits:
        raise ValueError("cut must split the register into two non-empty parts")

    state = normalize_statevector(state, eps=eps)
    matrix = jnp.reshape(state, (2**cut, 2 ** (num_qubits - cut)))
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    probabilities = singular_values**2
    probabilities = probabilities / jnp.maximum(jnp.sum(probabilities), eps)
    probabilities = jnp.clip(probabilities, eps, 1.0)
    return -jnp.sum(probabilities * jnp.log2(probabilities))


def normalize_statevector(state: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Return a unit-norm flattened statevector."""
    state = jnp.ravel(jnp.asarray(state))
    norm = jnp.linalg.norm(state)
    return state / jnp.maximum(norm, eps)


def entanglement_profile(theta: jnp.ndarray, config: QuantumCircuitConfig) -> jnp.ndarray:
    """Return half-chain entropy after every layer for one circuit sample."""
    if theta.ndim != 2:
        raise ValueError("theta must have shape (depth, params) for one sample")
    circuit_type = config.circuit_type.lower()
    circuit = tc.Circuit(config.num_qubits)
    entropies = []
    for layer in range(theta.shape[0]):
        if circuit_type == "iqp":
            _apply_iqp_layer(circuit, theta[layer], config)
        elif circuit_type == "he":
            _apply_he_layer(circuit, theta[layer], config)
        else:
            raise ValueError(f"Unsupported circuit_type: {config.circuit_type}")
        entropies.append(
            half_chain_entropy_from_state(
                jnp.asarray(circuit.state()),
                num_qubits=config.num_qubits,
            )
        )
    return jnp.stack(entropies).astype(jnp.float32)


def entanglement_profiles(theta: jnp.ndarray, config: QuantumCircuitConfig) -> jnp.ndarray:
    """Vectorized entanglement profiles for a batch of circuit parameters."""
    if theta.ndim == 2:
        theta = theta[None, ...]
    if theta.ndim != 3:
        raise ValueError("theta must have shape (batch, depth, params)")
    return jax.vmap(lambda sample: entanglement_profile(sample, config))(theta)


def summarize_entanglement_profiles(
    profiles: jnp.ndarray,
    early_layers: int | None = None,
) -> EntanglementSummary:
    """Summarize entropy curves and estimate early entanglement growth speed."""
    if profiles.ndim == 1:
        profiles = profiles[None, ...]
    if profiles.ndim != 2:
        raise ValueError("profiles must have shape (batch, depth)")
    if profiles.shape[1] == 0:
        raise ValueError("profiles must contain at least one layer")

    if early_layers is None:
        early_layers = max(2, min(6, profiles.shape[1] // 3 or profiles.shape[1]))
    early_layers = max(2, min(int(early_layers), profiles.shape[1]))
    x = jnp.arange(early_layers, dtype=jnp.float32)
    centered_x = x - jnp.mean(x)
    denom = jnp.maximum(jnp.sum(centered_x**2), 1e-8)
    y = profiles[:, :early_layers]
    centered_y = y - jnp.mean(y, axis=1, keepdims=True)
    slopes = jnp.sum(centered_x[None, :] * centered_y, axis=1) / denom
    return EntanglementSummary(
        entropy_by_layer=profiles,
        early_slope=slopes.astype(jnp.float32),
        max_entropy=jnp.max(profiles, axis=1).astype(jnp.float32),
        final_entropy=profiles[:, -1].astype(jnp.float32),
    )


def _apply_iqp_layer(
    circuit: tc.Circuit,
    layer_theta: jnp.ndarray,
    config: QuantumCircuitConfig,
) -> None:
    for q in range(config.num_qubits):
        circuit.h(q)
    edge_offset = config.num_qubits
    for q in range(config.num_qubits):
        circuit.rz(q, theta=layer_theta[q])
    if config.num_qubits > 1:
        for q in range(config.num_qubits - 1):
            circuit.rzz(q, q + 1, theta=layer_theta[edge_offset + q])
        if config.entanglement == "circular" and config.num_qubits > 2:
            circuit.rzz(
                config.num_qubits - 1,
                0,
                theta=layer_theta[edge_offset + config.num_qubits - 1],
            )


def _apply_he_layer(
    circuit: tc.Circuit,
    layer_theta: jnp.ndarray,
    config: QuantumCircuitConfig,
) -> None:
    for q in range(config.num_qubits):
        circuit.ry(q, theta=layer_theta[q])
        circuit.rz(q, theta=0.5 * layer_theta[q])
    if config.num_qubits > 1:
        for q in range(config.num_qubits - 1):
            circuit.cnot(q, q + 1)
        if config.entanglement == "circular" and config.num_qubits > 2:
            circuit.cnot(config.num_qubits - 1, 0)
