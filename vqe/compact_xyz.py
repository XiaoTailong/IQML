from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from iqml.vqe.chemistry import PauliHamiltonian


@dataclass(frozen=True)
class CompactXYZHamiltonian:
    num_qubits: int
    z_coeffs: jnp.ndarray
    edge_left: jnp.ndarray
    edge_right: jnp.ndarray
    zz_coeffs: jnp.ndarray
    xx_coeffs: jnp.ndarray
    yy_coeffs: jnp.ndarray


def compact_xyz_hamiltonian(hamiltonian: PauliHamiltonian) -> CompactXYZHamiltonian:
    """Extract the coefficient arrays needed by the XYZ-grid compact simulator."""
    z_coeffs = [0.0] * hamiltonian.num_qubits
    edge_terms: dict[tuple[int, int], dict[str, float]] = {}
    for term in hamiltonian.terms:
        if not term.pauli_string:
            continue
        paulis = tuple((int(qubit), str(pauli).upper()) for qubit, pauli in term.pauli_string)
        coeff = float(term.coefficient)
        if len(paulis) == 1 and paulis[0][1] == "Z":
            z_coeffs[paulis[0][0]] += coeff
            continue
        if len(paulis) == 2:
            (left, pauli_left), (right, pauli_right) = paulis
            if pauli_left != pauli_right:
                raise ValueError("Compact XYZ simulator only supports matching two-qubit Pauli terms")
            edge = tuple(sorted((left, right)))
            if edge not in edge_terms:
                edge_terms[edge] = {"X": 0.0, "Y": 0.0, "Z": 0.0}
            edge_terms[edge][pauli_left] += coeff
            continue
        raise ValueError("Compact XYZ simulator supports identity, Z, XX, YY, and ZZ terms")

    edges = sorted(edge_terms)
    return CompactXYZHamiltonian(
        num_qubits=hamiltonian.num_qubits,
        z_coeffs=jnp.asarray(z_coeffs, dtype=jnp.float64),
        edge_left=jnp.asarray([edge[0] for edge in edges], dtype=jnp.int32),
        edge_right=jnp.asarray([edge[1] for edge in edges], dtype=jnp.int32),
        zz_coeffs=jnp.asarray([edge_terms[edge]["Z"] for edge in edges], dtype=jnp.float64),
        xx_coeffs=jnp.asarray([edge_terms[edge]["X"] for edge in edges], dtype=jnp.float64),
        yy_coeffs=jnp.asarray([edge_terms[edge]["Y"] for edge in edges], dtype=jnp.float64),
    )


def compact_xyz_energy(
    theta: jnp.ndarray,
    compact: CompactXYZHamiltonian,
    *,
    hamiltonian_gate_scale: float = 2.0,
) -> jnp.ndarray:
    state = compact_xyz_state(theta, compact, hamiltonian_gate_scale=hamiltonian_gate_scale)
    return compact_xyz_expectation(state, compact)


def compact_xyz_state(
    theta: jnp.ndarray,
    compact: CompactXYZHamiltonian,
    *,
    hamiltonian_gate_scale: float = 2.0,
) -> jnp.ndarray:
    state = jnp.zeros((1 << compact.num_qubits,), dtype=jnp.complex128)
    state = state.at[0].set(jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128))
    scale = jnp.asarray(hamiltonian_gate_scale, dtype=jnp.float64)

    def layer_step(current_state: jnp.ndarray, layer_theta: jnp.ndarray):
        next_state = _apply_xyz_layer(current_state, layer_theta, compact, scale)
        return next_state, None

    state, _ = jax.lax.scan(layer_step, state, jnp.asarray(theta, dtype=jnp.float64))
    norm = jnp.sqrt(jnp.maximum(jnp.real(jnp.vdot(state, state)), 1e-24))
    return state / norm


def compact_xyz_entropy_profile(
    theta: jnp.ndarray,
    compact: CompactXYZHamiltonian,
    *,
    hamiltonian_gate_scale: float = 2.0,
) -> jnp.ndarray:
    state = jnp.zeros((1 << compact.num_qubits,), dtype=jnp.complex128)
    state = state.at[0].set(jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128))
    scale = jnp.asarray(hamiltonian_gate_scale, dtype=jnp.float64)

    def layer_step(current_state: jnp.ndarray, layer_theta: jnp.ndarray):
        next_state = _apply_xyz_layer(current_state, layer_theta, compact, scale)
        norm = jnp.sqrt(jnp.maximum(jnp.real(jnp.vdot(next_state, next_state)), 1e-24))
        next_state = next_state / norm
        entropy = _half_chain_entropy(next_state, compact.num_qubits)
        return next_state, entropy

    _, entropy = jax.lax.scan(layer_step, state, jnp.asarray(theta, dtype=jnp.float64))
    return entropy


def compact_xyz_expectation(state: jnp.ndarray, compact: CompactXYZHamiltonian) -> jnp.ndarray:
    indices = jnp.arange(state.shape[0], dtype=jnp.uint32)
    energy = jnp.asarray(0.0, dtype=jnp.float64)

    def z_step(carry: jnp.ndarray, item):
        qubit, coeff = item
        sign = _z_sign(indices, compact.num_qubits, qubit)
        term = coeff * jnp.real(jnp.vdot(state, sign.astype(state.dtype) * state))
        return carry + term, None

    energy, _ = jax.lax.scan(
        z_step,
        energy,
        (jnp.arange(compact.num_qubits, dtype=jnp.int32), compact.z_coeffs),
    )

    def edge_step(carry: jnp.ndarray, item):
        left, right, zz_coeff, xx_coeff, yy_coeff = item
        zz_sign = _z_sign(indices, compact.num_qubits, left) * _z_sign(indices, compact.num_qubits, right)
        flipped = _flip_indices(indices, compact.num_qubits, left, right)
        xx_state = state[flipped]
        yy_phase = -zz_sign.astype(state.dtype)
        yy_state = yy_phase * xx_state
        zz_energy = zz_coeff * jnp.real(jnp.vdot(state, zz_sign.astype(state.dtype) * state))
        xx_energy = xx_coeff * jnp.real(jnp.vdot(state, xx_state))
        yy_energy = yy_coeff * jnp.real(jnp.vdot(state, yy_state))
        return carry + zz_energy + xx_energy + yy_energy, None

    energy, _ = jax.lax.scan(
        edge_step,
        energy,
        (compact.edge_left, compact.edge_right, compact.zz_coeffs, compact.xx_coeffs, compact.yy_coeffs),
    )
    return energy


def _apply_xyz_layer(
    state: jnp.ndarray,
    layer_theta: jnp.ndarray,
    compact: CompactXYZHamiltonian,
    scale: jnp.ndarray,
) -> jnp.ndarray:
    z_control, zz_control, xx_control, yy_control = layer_theta
    indices = jnp.arange(state.shape[0], dtype=jnp.uint32)

    def z_step(current_state: jnp.ndarray, item):
        qubit, coeff = item
        angle = scale * z_control * coeff
        return _apply_rz(current_state, indices, compact.num_qubits, qubit, angle), None

    state, _ = jax.lax.scan(
        z_step,
        state,
        (jnp.arange(compact.num_qubits, dtype=jnp.int32), compact.z_coeffs),
    )

    def edge_step(current_state: jnp.ndarray, item):
        left, right, zz_coeff, xx_coeff, yy_coeff = item
        current_state = _apply_rzz(
            current_state,
            indices,
            compact.num_qubits,
            left,
            right,
            scale * zz_control * zz_coeff,
        )
        current_state = _apply_rxx(
            current_state,
            indices,
            compact.num_qubits,
            left,
            right,
            scale * xx_control * xx_coeff,
        )
        current_state = _apply_ryy(
            current_state,
            indices,
            compact.num_qubits,
            left,
            right,
            scale * yy_control * yy_coeff,
        )
        return current_state, None

    state, _ = jax.lax.scan(
        edge_step,
        state,
        (compact.edge_left, compact.edge_right, compact.zz_coeffs, compact.xx_coeffs, compact.yy_coeffs),
    )
    return state


def _apply_rz(
    state: jnp.ndarray,
    indices: jnp.ndarray,
    num_qubits: int,
    qubit: jnp.ndarray,
    angle: jnp.ndarray,
) -> jnp.ndarray:
    sign = _z_sign(indices, num_qubits, qubit)
    phase = jnp.exp(-0.5j * jnp.asarray(angle, dtype=jnp.float64) * sign)
    return state * phase


def _apply_rzz(
    state: jnp.ndarray,
    indices: jnp.ndarray,
    num_qubits: int,
    left: jnp.ndarray,
    right: jnp.ndarray,
    angle: jnp.ndarray,
) -> jnp.ndarray:
    sign = _z_sign(indices, num_qubits, left) * _z_sign(indices, num_qubits, right)
    phase = jnp.exp(-0.5j * jnp.asarray(angle, dtype=jnp.float64) * sign)
    return state * phase


def _apply_rxx(
    state: jnp.ndarray,
    indices: jnp.ndarray,
    num_qubits: int,
    left: jnp.ndarray,
    right: jnp.ndarray,
    angle: jnp.ndarray,
) -> jnp.ndarray:
    return _apply_two_qubit_pauli_rotation(
        state,
        indices,
        num_qubits,
        left,
        right,
        angle,
        phase=jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128),
    )


def _apply_ryy(
    state: jnp.ndarray,
    indices: jnp.ndarray,
    num_qubits: int,
    left: jnp.ndarray,
    right: jnp.ndarray,
    angle: jnp.ndarray,
) -> jnp.ndarray:
    phase = -_z_sign(indices, num_qubits, left) * _z_sign(indices, num_qubits, right)
    return _apply_two_qubit_pauli_rotation(state, indices, num_qubits, left, right, angle, phase)


def _apply_two_qubit_pauli_rotation(
    state: jnp.ndarray,
    indices: jnp.ndarray,
    num_qubits: int,
    left: jnp.ndarray,
    right: jnp.ndarray,
    angle: jnp.ndarray,
    phase: jnp.ndarray,
) -> jnp.ndarray:
    flipped = _flip_indices(indices, num_qubits, left, right)
    cos = jnp.cos(0.5 * angle).astype(jnp.complex128)
    sin = jnp.sin(0.5 * angle).astype(jnp.complex128)
    return cos * state - 1j * sin * phase.astype(jnp.complex128) * state[flipped]


def _z_sign(indices: jnp.ndarray, num_qubits: int, qubit: jnp.ndarray) -> jnp.ndarray:
    bit_position = jnp.asarray(num_qubits - 1, dtype=jnp.int32) - qubit
    bit = (indices >> bit_position.astype(jnp.uint32)) & jnp.asarray(1, dtype=jnp.uint32)
    return jnp.where(bit == 0, 1.0, -1.0)


def _flip_indices(indices: jnp.ndarray, num_qubits: int, left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    left_shift = jnp.asarray(num_qubits - 1, dtype=jnp.int32) - left
    right_shift = jnp.asarray(num_qubits - 1, dtype=jnp.int32) - right
    mask = (jnp.asarray(1, dtype=jnp.uint32) << left_shift.astype(jnp.uint32)) | (
        jnp.asarray(1, dtype=jnp.uint32) << right_shift.astype(jnp.uint32)
    )
    return indices ^ mask


def _half_chain_entropy(state: jnp.ndarray, num_qubits: int) -> jnp.ndarray:
    left_qubits = num_qubits // 2
    right_qubits = num_qubits - left_qubits
    matrix = jnp.reshape(state, (1 << left_qubits, 1 << right_qubits))
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    probabilities = jnp.real(singular_values * jnp.conj(singular_values))
    probabilities = probabilities / jnp.maximum(jnp.sum(probabilities), 1e-24)
    probabilities = jnp.clip(probabilities, 1e-24, 1.0)
    return -jnp.sum(probabilities * jnp.log2(probabilities))
