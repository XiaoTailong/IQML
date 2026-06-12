from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import tensorcircuit as tc

try:
    from tensorcircuit.quantum import PauliStringSum2COO
    from tensorcircuit.templates.measurements import operator_expectation
except ImportError:
    PauliStringSum2COO = None
    operator_expectation = None

from iqml.quantum.backend import configure_tensorcircuit_jax
from iqml.vqe.chemistry import PauliHamiltonian, PauliTerm

configure_tensorcircuit_jax()


@dataclass(frozen=True)
class VQEConfig:
    num_qubits: int
    depth: int
    entanglement: str = "linear"
    theta_mode: str = "residual"
    residual_scale: float = 0.1
    ansatz_type: str = "he"
    hamiltonian_gate_scale: float = 2.0
    hidden_theta_scale: float = 0.1
    hidden_theta_eps: float = 1e-6
    hidden_theta_norm: str = "rms"


@dataclass(frozen=True)
class FallbackPauliHamiltonianOperator:
    """Small operator wrapper used when TensorCircuit lacks PauliStringSum2COO."""

    terms: tuple[PauliTerm, ...]

    @property
    def data(self) -> tuple[PauliTerm, ...]:
        return self.terms


XYZ_HAMILTONIAN_CONTROL_NAMES = ("z_field", "zz", "xx", "yy")


def ansatz_parameter_shape(config: VQEConfig) -> tuple[int, ...]:
    if config.depth <= 0:
        raise ValueError("config.depth must be positive")
    if config.num_qubits <= 0:
        raise ValueError("config.num_qubits must be positive")
    if is_hardware_efficient_ansatz(config):
        return (config.depth, config.num_qubits, 2)
    if is_xyz_hamiltonian_ansatz(config):
        return (config.depth, len(XYZ_HAMILTONIAN_CONTROL_NAMES))
    raise ValueError(f"Unsupported ansatz_type {config.ansatz_type!r}")


def ansatz_layer_parameter_size(config: VQEConfig) -> int:
    shape = ansatz_parameter_shape(config)
    size = 1
    for dim in shape[1:]:
        size *= int(dim)
    return size


def ansatz_parameter_size(config: VQEConfig) -> int:
    shape = ansatz_parameter_shape(config)
    size = 1
    for dim in shape:
        size *= int(dim)
    return size


def is_hardware_efficient_ansatz(config: VQEConfig) -> bool:
    return config.ansatz_type.lower() in {"he", "hardware_efficient", "hardware-efficient"}


def is_xyz_hamiltonian_ansatz(config: VQEConfig) -> bool:
    return config.ansatz_type.lower() in {
        "xyz_hamiltonian",
        "xyz_trotter",
        "xyz_grid",
        "heisenberg_xyz",
        "hamiltonian_inspired",
        "hamiltonian-inspired",
    }


def vqe_energy(
    theta: jnp.ndarray,
    hamiltonian: PauliHamiltonian,
    config: VQEConfig,
    hamiltonian_operator: Any | None = None,
) -> jnp.ndarray:
    if is_hardware_efficient_ansatz(config):
        return he_vqe_energy(theta, hamiltonian, config, hamiltonian_operator)
    if is_xyz_hamiltonian_ansatz(config):
        return xyz_hamiltonian_vqe_energy(theta, hamiltonian, config, hamiltonian_operator)
    raise ValueError(f"Unsupported ansatz_type {config.ansatz_type!r}")


def he_vqe_energy(
    theta: jnp.ndarray,
    hamiltonian: PauliHamiltonian,
    config: VQEConfig,
    hamiltonian_operator: Any | None = None,
) -> jnp.ndarray:
    circuit = build_he_ansatz(theta, config)
    return pauli_hamiltonian_expectation(circuit, hamiltonian, hamiltonian_operator)


def xyz_hamiltonian_vqe_energy(
    theta: jnp.ndarray,
    hamiltonian: PauliHamiltonian,
    config: VQEConfig,
    hamiltonian_operator: Any | None = None,
) -> jnp.ndarray:
    circuit = build_xyz_hamiltonian_ansatz(theta, hamiltonian, config)
    return pauli_hamiltonian_expectation(circuit, hamiltonian, hamiltonian_operator)


def build_he_ansatz(theta: jnp.ndarray, config: VQEConfig) -> tc.Circuit:
    if theta.shape != (config.depth, config.num_qubits, 2):
        raise ValueError(
            "theta must have shape "
            f"({config.depth}, {config.num_qubits}, 2), got {theta.shape}"
        )

    circuit = tc.Circuit(config.num_qubits)
    for layer in range(config.depth):
        for qubit in range(config.num_qubits):
            circuit.ry(qubit, theta=theta[layer, qubit, 0])
            circuit.rz(qubit, theta=theta[layer, qubit, 1])
        _entangle(circuit, config)
    return circuit


def build_xyz_hamiltonian_ansatz(
    theta: jnp.ndarray,
    hamiltonian: PauliHamiltonian,
    config: VQEConfig,
) -> tc.Circuit:
    expected_shape = ansatz_parameter_shape(config)
    if theta.shape != expected_shape:
        raise ValueError(f"theta must have shape {expected_shape}, got {theta.shape}")
    if hamiltonian.num_qubits != config.num_qubits:
        raise ValueError("hamiltonian.num_qubits must match config.num_qubits")

    circuit = tc.Circuit(config.num_qubits)
    for layer in range(config.depth):
        apply_xyz_hamiltonian_layer(circuit, theta[layer], hamiltonian, config)
    return circuit


def apply_xyz_hamiltonian_layer(
    circuit: tc.Circuit,
    layer_theta: jnp.ndarray,
    hamiltonian: PauliHamiltonian,
    config: VQEConfig,
) -> None:
    if layer_theta.shape != (len(XYZ_HAMILTONIAN_CONTROL_NAMES),):
        raise ValueError(
            "layer_theta must have shape "
            f"({len(XYZ_HAMILTONIAN_CONTROL_NAMES)},), got {layer_theta.shape}"
        )
    scale = jnp.asarray(config.hamiltonian_gate_scale, dtype=layer_theta.dtype)
    z_control, zz_control, xx_control, yy_control = layer_theta
    for term in hamiltonian.terms:
        if not term.pauli_string:
            continue
        coeff = jnp.asarray(term.coefficient, dtype=layer_theta.dtype)
        paulis = tuple((int(qubit), str(pauli).upper()) for qubit, pauli in term.pauli_string)
        if len(paulis) == 1 and paulis[0][1] == "Z":
            circuit.rz(paulis[0][0], theta=scale * z_control * coeff)
        elif len(paulis) == 2:
            left, pauli_left = paulis[0]
            right, pauli_right = paulis[1]
            if pauli_left != pauli_right:
                raise ValueError(
                    "XYZ Hamiltonian-inspired ansatz only supports matching two-qubit Pauli terms"
                )
            if pauli_left == "Z":
                circuit.rzz(left, right, theta=scale * zz_control * coeff)
            elif pauli_left == "X":
                circuit.rxx(left, right, theta=scale * xx_control * coeff)
            elif pauli_left == "Y":
                circuit.ryy(left, right, theta=scale * yy_control * coeff)
            else:
                raise ValueError(f"Unsupported Pauli operator {pauli_left!r}")
        else:
            raise ValueError(
                "XYZ Hamiltonian-inspired ansatz supports identity, one-qubit Z, "
                "and two-qubit XX/YY/ZZ Hamiltonian terms"
            )


def pauli_hamiltonian_expectation(
    circuit: tc.Circuit,
    hamiltonian: PauliHamiltonian,
    hamiltonian_operator: Any | None = None,
) -> jnp.ndarray:
    if hamiltonian_operator is None:
        hamiltonian_operator = build_tensorcircuit_hamiltonian(hamiltonian)
    if isinstance(hamiltonian_operator, FallbackPauliHamiltonianOperator):
        expectation = sum(
            float(term.coefficient) * _pauli_expectation(circuit, term)
            for term in hamiltonian_operator.terms
        )
    else:
        if operator_expectation is None:
            raise RuntimeError("TensorCircuit operator_expectation is not available")
        expectation = operator_expectation(circuit, hamiltonian_operator)
    norm = statevector_norm(circuit.state())
    return expectation / jnp.maximum(norm, 1e-12)


def statevector_norm(state: jnp.ndarray) -> jnp.ndarray:
    state = jnp.ravel(jnp.asarray(state))
    return jnp.real(jnp.vdot(state, state))


def build_tensorcircuit_hamiltonian(hamiltonian: PauliHamiltonian) -> Any:
    if PauliStringSum2COO is None:
        return FallbackPauliHamiltonianOperator(tuple(hamiltonian.terms))
    structures, weights = pauli_hamiltonian_structures(hamiltonian)
    return PauliStringSum2COO(structures, weights)


def pauli_hamiltonian_structures(
    hamiltonian: PauliHamiltonian,
) -> tuple[list[list[int]], list[float]]:
    if hamiltonian.num_qubits <= 0:
        raise ValueError("hamiltonian.num_qubits must be positive")
    if not hamiltonian.terms:
        raise ValueError("hamiltonian must contain at least one Pauli term")

    structures: list[list[int]] = []
    weights: list[float] = []
    for term in hamiltonian.terms:
        structure = [0] * hamiltonian.num_qubits
        for qubit, pauli in term.pauli_string:
            structure[qubit] = _pauli_code(pauli)
        structures.append(structure)
        weights.append(float(term.coefficient))
    return structures, weights


def _pauli_code(pauli: str) -> int:
    pauli_upper = pauli.upper()
    if pauli_upper == "X":
        return 1
    if pauli_upper == "Y":
        return 2
    if pauli_upper == "Z":
        return 3
    raise ValueError(f"Unsupported Pauli operator {pauli!r}")


def _entangle(circuit: tc.Circuit, config: VQEConfig) -> None:
    if config.num_qubits <= 1:
        return
    for qubit in range(config.num_qubits - 1):
        circuit.cnot(qubit, qubit + 1)
    if config.entanglement == "circular" and config.num_qubits > 2:
        circuit.cnot(config.num_qubits - 1, 0)


def _pauli_expectation(circuit: tc.Circuit, term: PauliTerm) -> jnp.ndarray:
    if not term.pauli_string:
        return jnp.asarray(1.0, dtype=jnp.float32)

    operators = []
    for qubit, pauli in term.pauli_string:
        pauli_upper = pauli.upper()
        if pauli_upper == "X":
            gate = tc.gates.x()
        elif pauli_upper == "Y":
            gate = tc.gates.y()
        elif pauli_upper == "Z":
            gate = tc.gates.z()
        else:
            raise ValueError(f"Unsupported Pauli operator {pauli!r}")
        operators.append((gate, [qubit]))
    expectation = jnp.real(circuit.expectation(*operators))
    norm = statevector_norm(circuit.state())
    return expectation / jnp.maximum(norm, 1e-12)
