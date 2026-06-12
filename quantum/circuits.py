from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import tensorcircuit as tc

from iqml.quantum.backend import configure_tensorcircuit_jax

configure_tensorcircuit_jax()


@dataclass(frozen=True)
class QuantumCircuitConfig:
    num_qubits: int
    circuit_type: str = "iqp"
    observables: str = "z"
    entanglement: str = "linear"
    use_mid_measurement: bool = False
    readout_layers: tuple[int, ...] | None = None


def parameter_count(config: QuantumCircuitConfig) -> int:
    if config.circuit_type.lower() == "iqp":
        return config.num_qubits + _edge_count(config)
    return config.num_qubits


def supports_mid_measurement() -> bool:
    """Report whether the active TensorCircuit build exposes mid-measurement APIs."""
    circuit = tc.Circuit(1)
    return any(
        hasattr(circuit, name)
        for name in ("mid_measurement", "mid_measure", "cond_measurement")
    )


def run_pqc_batch(theta: jnp.ndarray, config: QuantumCircuitConfig) -> jnp.ndarray:
    """Run a batch of HE or IQP circuits and return quantum expectations."""
    if theta.ndim != 3:
        raise ValueError("theta must have shape (batch, depth, num_qubits)")
    expected_params = parameter_count(config)
    if theta.shape[-1] != expected_params:
        raise ValueError(f"theta last dimension must equal {expected_params}")

    return jax.vmap(lambda sample: run_pqc(sample, config))(theta)


def run_pqc(theta: jnp.ndarray, config: QuantumCircuitConfig) -> jnp.ndarray:
    """Run one parameterized circuit sample."""
    circuit_type = config.circuit_type.lower()
    if circuit_type == "iqp":
        return _run_iqp(theta, config)
    if circuit_type == "he":
        return _run_he(theta, config)
    raise ValueError(f"Unsupported circuit_type: {config.circuit_type}")


def _run_iqp(theta: jnp.ndarray, config: QuantumCircuitConfig) -> jnp.ndarray:
    """Run an IQP-style circuit with alternating Hadamard and diagonal blocks.

    Each diagonal block follows the standard IQP feature-map pattern:
    single-qubit Z phases plus two-qubit ZZ phases. Hadamard layers between
    diagonal blocks keep deeper circuits from collapsing into one commuting
    phase block.
    """
    c = tc.Circuit(config.num_qubits)
    readouts = []
    readout_layers = _resolve_readout_layers(theta.shape[0], config.readout_layers)

    for layer in range(theta.shape[0]):
        for q in range(config.num_qubits):
            c.h(q)
        _apply_iqp_diagonal_block(c, theta[layer], config)
        if layer + 1 in readout_layers:
            readouts.append(_observables(c, config))
        _optional_mid_measurement_hook(c, config, layer)

    if not readouts:
        readouts.append(_observables(c, config))
    return jnp.concatenate(readouts, axis=0).astype(jnp.float32)


def _apply_iqp_diagonal_block(
    circuit: tc.Circuit,
    layer_theta: jnp.ndarray,
    config: QuantumCircuitConfig,
) -> None:
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


def _run_he(theta: jnp.ndarray, config: QuantumCircuitConfig) -> jnp.ndarray:
    c = tc.Circuit(config.num_qubits)
    readouts = []
    readout_layers = _resolve_readout_layers(theta.shape[0], config.readout_layers)
    for layer in range(theta.shape[0]):
        for q in range(config.num_qubits):
            c.ry(q, theta=theta[layer, q])
            c.rz(q, theta=0.5 * theta[layer, q])
        if config.num_qubits > 1:
            for q in range(config.num_qubits - 1):
                c.cnot(q, q + 1)
            if config.entanglement == "circular" and config.num_qubits > 2:
                c.cnot(config.num_qubits - 1, 0)
        if layer + 1 in readout_layers:
            readouts.append(_observables(c, config))
        _optional_mid_measurement_hook(c, config, layer)
    if not readouts:
        readouts.append(_observables(c, config))
    return jnp.concatenate(readouts, axis=0).astype(jnp.float32)


def _optional_mid_measurement_hook(
    circuit: tc.Circuit,
    config: QuantumCircuitConfig,
    layer: int,
) -> None:
    """Reserved extension point for adaptive circuits with mid-circuit measurement.

    The initial framework keeps this disabled so the default circuit remains
    differentiable and easy to batch. Future experiments can replace this hook
    with TensorCircuit `mid_measurement` plus `conditional_gate` logic.
    """
    del circuit, layer
    if config.use_mid_measurement:
        raise NotImplementedError(
            "Mid-measurement experiments should implement this hook for the "
            "specific measurement and feed-forward policy."
        )


def _z_expectations(circuit: tc.Circuit, num_qubits: int) -> jnp.ndarray:
    values = [
        jnp.real(circuit.expectation((tc.gates.z(), [q])))
        for q in range(num_qubits)
    ]
    return jnp.stack(values).astype(jnp.float32)


def _observables(circuit: tc.Circuit, config: QuantumCircuitConfig) -> jnp.ndarray:
    pieces = []
    requested = {name.strip().lower() for name in config.observables.split(",") if name.strip()}
    if not requested:
        requested = {"z"}
    for name in ("z", "x", "zz"):
        if name not in requested:
            continue
        if name == "z":
            pieces.append(_z_expectations(circuit, config.num_qubits))
        elif name == "x":
            pieces.append(_x_expectations(circuit, config.num_qubits))
        elif name == "zz":
            pieces.append(_zz_expectations(circuit, config))
    if not pieces:
        raise ValueError(f"Unsupported observables specification: {config.observables!r}")
    return jnp.concatenate(pieces, axis=0)


def observable_count(config: QuantumCircuitConfig) -> int:
    requested = {name.strip().lower() for name in config.observables.split(",") if name.strip()}
    if not requested:
        requested = {"z"}
    count = 0
    for name in requested:
        if name in {"z", "x"}:
            count += config.num_qubits
        elif name == "zz":
            count += _edge_count(config)
        else:
            raise ValueError(f"Unsupported observable {name!r}")
    return count


def feature_count(config: QuantumCircuitConfig, depth: int | None = None) -> int:
    layers = _resolve_readout_layers(depth, config.readout_layers) if depth is not None else config.readout_layers
    num_layers = 1 if layers is None else len(layers)
    return observable_count(config) * num_layers


def _x_expectations(circuit: tc.Circuit, num_qubits: int) -> jnp.ndarray:
    values = [
        jnp.real(circuit.expectation((tc.gates.x(), [q])))
        for q in range(num_qubits)
    ]
    return jnp.stack(values).astype(jnp.float32)


def _zz_expectations(circuit: tc.Circuit, config: QuantumCircuitConfig) -> jnp.ndarray:
    if config.num_qubits <= 1:
        return jnp.zeros((0,), dtype=jnp.float32)
    values = [
        jnp.real(circuit.expectation((tc.gates.z(), [q]), (tc.gates.z(), [q + 1])))
        for q in range(config.num_qubits - 1)
    ]
    if config.entanglement == "circular" and config.num_qubits > 2:
        values.append(
            jnp.real(circuit.expectation((tc.gates.z(), [config.num_qubits - 1]), (tc.gates.z(), [0])))
        )
    return jnp.stack(values).astype(jnp.float32) if values else jnp.zeros((0,), dtype=jnp.float32)


def _edge_count(config: QuantumCircuitConfig) -> int:
    if config.num_qubits <= 1:
        return 0
    if config.entanglement == "circular" and config.num_qubits > 2:
        return config.num_qubits
    return config.num_qubits - 1


def _resolve_readout_layers(depth: int | None, readout_layers: tuple[int, ...] | None) -> tuple[int, ...] | None:
    if readout_layers is None:
        return None if depth is None else (int(depth),)
    if depth is None:
        return tuple(readout_layers)
    resolved = tuple(sorted({int(layer) for layer in readout_layers if 1 <= int(layer) <= int(depth)}))
    if not resolved:
        return (int(depth),)
    return resolved
