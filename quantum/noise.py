from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import tensorcircuit as tc

from iqml.quantum.circuits import QuantumCircuitConfig, feature_count, parameter_count


@dataclass(frozen=True)
class MonteCarloNoiseConfig:
    noise_type: str = "depolarizing"
    probability: float = 0.0
    trajectories: int = 256
    trajectory_chunk_size: int = 64
    apply_after_each_layer: bool = True
    eps: float = 1e-12


def run_noisy_pqc_mc(
    theta: jnp.ndarray,
    circuit_config: QuantumCircuitConfig,
    noise_config: MonteCarloNoiseConfig,
    key: jax.Array,
) -> jnp.ndarray:
    """Monte Carlo statevector simulation of noisy PQC Z expectations."""
    if theta.ndim != 2:
        raise ValueError("theta must have shape (depth, params) for one sample")
    expected_params = parameter_count(circuit_config)
    if theta.shape[-1] != expected_params:
        raise ValueError(f"theta last dimension must equal {expected_params}")
    if noise_config.trajectories <= 0:
        raise ValueError("trajectories must be positive")
    if noise_config.trajectory_chunk_size <= 0:
        raise ValueError("trajectory_chunk_size must be positive")

    keys = jax.random.split(key, noise_config.trajectories)
    feature_sum = jnp.zeros(
        (feature_count(circuit_config, depth=int(theta.shape[0])),),
        dtype=jnp.float32,
    )
    chunk_size = min(noise_config.trajectory_chunk_size, noise_config.trajectories)
    for start in range(0, noise_config.trajectories, chunk_size):
        chunk_keys = keys[start : start + chunk_size]
        chunk_features = _run_noisy_trajectory_chunk(
            theta,
            circuit_config,
            noise_config,
            chunk_keys,
        )
        feature_sum = feature_sum + jnp.sum(chunk_features, axis=0)
    return (feature_sum / float(noise_config.trajectories)).astype(jnp.float32)


def run_noisy_pqc_batch_mc(
    theta: jnp.ndarray,
    circuit_config: QuantumCircuitConfig,
    noise_config: MonteCarloNoiseConfig,
    key: jax.Array,
    sample_chunk_size: int | None = None,
) -> jnp.ndarray:
    """Monte Carlo statevector noisy PQC for a batch of parameter sequences."""
    if theta.ndim != 3:
        raise ValueError("theta must have shape (batch, depth, params)")
    if sample_chunk_size is None:
        sample_chunk_size = int(theta.shape[0])
    if sample_chunk_size <= 0:
        raise ValueError("sample_chunk_size must be positive")
    keys = jax.random.split(key, theta.shape[0])
    features = []
    for start in range(0, int(theta.shape[0]), int(sample_chunk_size)):
        stop = start + int(sample_chunk_size)
        features.append(
            _run_noisy_sample_chunk(
                theta[start:stop],
                circuit_config,
                noise_config,
                keys[start:stop],
            )
        )
    return jnp.concatenate(features, axis=0).astype(jnp.float32)


@partial(jax.jit, static_argnames=("circuit_config", "noise_config"))
def _run_noisy_sample_chunk(
    theta: jnp.ndarray,
    circuit_config: QuantumCircuitConfig,
    noise_config: MonteCarloNoiseConfig,
    keys: jax.Array,
) -> jnp.ndarray:
    return jax.vmap(
        lambda sample_theta, sample_key: run_noisy_pqc_mc(
            sample_theta,
            circuit_config,
            noise_config,
            sample_key,
        )
    )(theta, keys)


@partial(jax.jit, static_argnames=("circuit_config", "noise_config"))
def _run_noisy_trajectory_chunk(
    theta: jnp.ndarray,
    circuit_config: QuantumCircuitConfig,
    noise_config: MonteCarloNoiseConfig,
    keys: jax.Array,
) -> jnp.ndarray:
    return jax.vmap(
        lambda trajectory_key: _run_noisy_trajectory(
            theta,
            circuit_config,
            noise_config,
            trajectory_key,
        )
    )(keys)


def _run_noisy_trajectory(
    theta: jnp.ndarray,
    circuit_config: QuantumCircuitConfig,
    noise_config: MonteCarloNoiseConfig,
    key: jax.Array,
) -> jnp.ndarray:
    state = _initial_state(circuit_config.num_qubits)
    layer_keys = jax.random.split(key, theta.shape[0])
    readouts = []
    readout_layers = _resolve_readout_layers(theta.shape[0], circuit_config.readout_layers)
    for layer in range(theta.shape[0]):
        state = _apply_unitary_layer(state, theta[layer], circuit_config)
        if noise_config.apply_after_each_layer:
            state = _apply_noise_layer(
                state,
                circuit_config.num_qubits,
                noise_config,
                layer_keys[layer],
            )
        if noise_config.apply_after_each_layer and layer + 1 in readout_layers:
            readouts.append(_observables_from_state(state, circuit_config))
        elif (
            not noise_config.apply_after_each_layer
            and layer + 1 in readout_layers
            and layer + 1 < theta.shape[0]
        ):
            readouts.append(_observables_from_state(state, circuit_config))
    if not noise_config.apply_after_each_layer:
        state = _apply_noise_layer(
            state,
            circuit_config.num_qubits,
            noise_config,
            key,
        )
        if theta.shape[0] in readout_layers:
            readouts.append(_observables_from_state(state, circuit_config))
    if not readouts:
        readouts.append(_observables_from_state(state, circuit_config))
    return jnp.concatenate(readouts, axis=0).astype(jnp.float32)


def _initial_state(num_qubits: int) -> jnp.ndarray:
    state = jnp.zeros((2**num_qubits,), dtype=jnp.complex64)
    return state.at[0].set(1.0 + 0.0j)


def _apply_unitary_layer(
    state: jnp.ndarray,
    layer_theta: jnp.ndarray,
    config: QuantumCircuitConfig,
) -> jnp.ndarray:
    circuit = tc.Circuit(config.num_qubits, inputs=state)
    circuit_type = config.circuit_type.lower()
    if circuit_type == "he":
        _apply_he_unitary_layer(circuit, layer_theta, config)
    elif circuit_type == "iqp":
        _apply_iqp_unitary_layer(circuit, layer_theta, config)
    else:
        raise ValueError(f"Unsupported circuit_type: {config.circuit_type}")
    return jnp.asarray(circuit.state(), dtype=jnp.complex64)


def _apply_he_unitary_layer(
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


def _apply_iqp_unitary_layer(
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


def _apply_noise_layer(
    state: jnp.ndarray,
    num_qubits: int,
    noise_config: MonteCarloNoiseConfig,
    key: jax.Array,
) -> jnp.ndarray:
    probability = float(noise_config.probability)
    if probability <= 0.0:
        return state
    keys = jax.random.split(key, num_qubits)
    for q in range(num_qubits):
        state = _apply_single_qubit_noise(
            state,
            q,
            num_qubits,
            noise_config,
            keys[q],
        )
    return state


def _apply_single_qubit_noise(
    state: jnp.ndarray,
    qubit: int,
    num_qubits: int,
    noise_config: MonteCarloNoiseConfig,
    key: jax.Array,
) -> jnp.ndarray:
    noise_type = noise_config.noise_type.lower()
    probability = jnp.asarray(noise_config.probability, dtype=jnp.float32)
    probability = jnp.clip(probability, 0.0, 1.0)

    if noise_type == "bit_flip":
        do_flip = jax.random.bernoulli(key, probability)
        return jax.lax.cond(
            do_flip,
            lambda s: _apply_single_qubit_matrix(s, _pauli_x(), qubit, num_qubits),
            lambda s: s,
            state,
        )
    if noise_type in ("phase_flip", "phase"):
        do_flip = jax.random.bernoulli(key, probability)
        return jax.lax.cond(
            do_flip,
            lambda s: _apply_single_qubit_matrix(s, _pauli_z(), qubit, num_qubits),
            lambda s: s,
            state,
        )
    if noise_type == "depolarizing":
        return _apply_depolarizing_noise(state, qubit, num_qubits, probability, key)
    if noise_type in ("amplitude_damping", "amplitude"):
        return _apply_amplitude_damping_noise(
            state,
            qubit,
            num_qubits,
            probability,
            key,
            noise_config.eps,
        )
    raise ValueError(f"Unsupported noise_type: {noise_config.noise_type}")


def _apply_depolarizing_noise(
    state: jnp.ndarray,
    qubit: int,
    num_qubits: int,
    probability: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    event = jax.random.choice(
        key,
        jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        p=jnp.asarray(
            [
                1.0 - probability,
                probability / 3.0,
                probability / 3.0,
                probability / 3.0,
            ],
            dtype=jnp.float32,
        ),
    )
    return jax.lax.switch(
        event,
        [
            lambda s: s,
            lambda s: _apply_single_qubit_matrix(s, _pauli_x(), qubit, num_qubits),
            lambda s: _apply_single_qubit_matrix(s, _pauli_y(), qubit, num_qubits),
            lambda s: _apply_single_qubit_matrix(s, _pauli_z(), qubit, num_qubits),
        ],
        state,
    )


def _apply_amplitude_damping_noise(
    state: jnp.ndarray,
    qubit: int,
    num_qubits: int,
    gamma: jnp.ndarray,
    key: jax.Array,
    eps: float,
) -> jnp.ndarray:
    excited_probability = _one_probability(state, qubit, num_qubits)
    jump_probability = jnp.clip(gamma * excited_probability, 0.0, 1.0)
    do_jump = jax.random.bernoulli(key, jump_probability)
    return jax.lax.cond(
        do_jump,
        lambda s: _normalize(
            _apply_single_qubit_matrix(s, _lowering(), qubit, num_qubits),
            eps,
        ),
        lambda s: _normalize(
            _apply_single_qubit_matrix(s, _no_jump(gamma), qubit, num_qubits),
            eps,
        ),
        state,
    )


def _apply_single_qubit_matrix(
    state: jnp.ndarray,
    matrix: jnp.ndarray,
    qubit: int,
    num_qubits: int,
) -> jnp.ndarray:
    tensor = jnp.reshape(state, (2,) * num_qubits)
    moved = jnp.moveaxis(tensor, qubit, 0)
    updated = jnp.tensordot(matrix, moved, axes=([1], [0]))
    restored = jnp.moveaxis(updated, 0, qubit)
    return jnp.reshape(restored, (-1,)).astype(jnp.complex64)


def _one_probability(state: jnp.ndarray, qubit: int, num_qubits: int) -> jnp.ndarray:
    tensor = jnp.reshape(state, (2,) * num_qubits)
    moved = jnp.moveaxis(tensor, qubit, 0)
    return jnp.sum(jnp.abs(moved[1]) ** 2).astype(jnp.float32)


def _normalize(state: jnp.ndarray, eps: float) -> jnp.ndarray:
    norm = jnp.linalg.norm(state)
    return (state / jnp.maximum(norm, eps)).astype(jnp.complex64)


def _z_expectations_from_state(state: jnp.ndarray, num_qubits: int) -> jnp.ndarray:
    probabilities = jnp.abs(state) ** 2
    basis = jnp.arange(2**num_qubits, dtype=jnp.uint32)
    values = []
    for q in range(num_qubits):
        bit = (basis >> (num_qubits - q - 1)) & jnp.asarray(1, dtype=jnp.uint32)
        z = 1.0 - 2.0 * bit.astype(jnp.float32)
        values.append(jnp.sum(probabilities * z))
    return jnp.stack(values).astype(jnp.float32)


def _x_expectations_from_state(state: jnp.ndarray, num_qubits: int) -> jnp.ndarray:
    values = []
    for q in range(num_qubits):
        updated = _apply_single_qubit_matrix(state, _pauli_x(), q, num_qubits)
        values.append(jnp.real(jnp.vdot(state, updated)))
    return jnp.stack(values).astype(jnp.float32)


def _zz_expectations_from_state(
    state: jnp.ndarray,
    config: QuantumCircuitConfig,
) -> jnp.ndarray:
    if config.num_qubits <= 1:
        return jnp.zeros((0,), dtype=jnp.float32)
    z_values = _z_signs(config.num_qubits)
    probabilities = jnp.abs(state) ** 2
    values = [
        jnp.sum(probabilities * z_values[q] * z_values[q + 1])
        for q in range(config.num_qubits - 1)
    ]
    if config.entanglement == "circular" and config.num_qubits > 2:
        values.append(jnp.sum(probabilities * z_values[config.num_qubits - 1] * z_values[0]))
    return jnp.stack(values).astype(jnp.float32) if values else jnp.zeros((0,), dtype=jnp.float32)


def _observables_from_state(
    state: jnp.ndarray,
    config: QuantumCircuitConfig,
) -> jnp.ndarray:
    pieces = []
    requested = _requested_observables(config)
    for name in ("z", "x", "zz"):
        if name not in requested:
            continue
        if name == "z":
            pieces.append(_z_expectations_from_state(state, config.num_qubits))
        elif name == "x":
            pieces.append(_x_expectations_from_state(state, config.num_qubits))
        elif name == "zz":
            pieces.append(_zz_expectations_from_state(state, config))
    if not pieces:
        raise ValueError(f"Unsupported observables specification: {config.observables!r}")
    return jnp.concatenate(pieces, axis=0).astype(jnp.float32)


def _requested_observables(config: QuantumCircuitConfig) -> set[str]:
    requested = {name.strip().lower() for name in config.observables.split(",") if name.strip()}
    return requested or {"z"}


def _resolve_readout_layers(
    depth: int,
    readout_layers: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if readout_layers is None:
        return (int(depth),)
    resolved = tuple(sorted({int(layer) for layer in readout_layers if 1 <= int(layer) <= int(depth)}))
    return resolved or (int(depth),)


def _z_signs(num_qubits: int) -> jnp.ndarray:
    basis = jnp.arange(2**num_qubits, dtype=jnp.uint32)
    values = []
    for q in range(num_qubits):
        bit = (basis >> (num_qubits - q - 1)) & jnp.asarray(1, dtype=jnp.uint32)
        values.append(1.0 - 2.0 * bit.astype(jnp.float32))
    return jnp.stack(values, axis=0).astype(jnp.float32)


def _pauli_x() -> jnp.ndarray:
    return jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex64)


def _pauli_y() -> jnp.ndarray:
    return jnp.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=jnp.complex64)


def _pauli_z() -> jnp.ndarray:
    return jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex64)


def _lowering() -> jnp.ndarray:
    return jnp.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.complex64)


def _no_jump(gamma: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(
        [[1.0, 0.0], [0.0, jnp.sqrt(jnp.maximum(1.0 - gamma, 0.0))]],
        dtype=jnp.complex64,
    )
