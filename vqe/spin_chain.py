from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import jax.numpy as jnp
import numpy as np

from iqml.vqe.chemistry import PauliHamiltonian, PauliTerm

SPIN_CONTROL_FEATURE_NAMES = (
    "layer_fraction",
    "anneal_smoothstep",
    "sin_pi_s",
    "cos_pi_s",
    "jzz_over_scale",
    "jxx_over_scale",
    "hx_over_scale",
    "hz_over_scale",
    "jzz2_over_scale",
    "disorder_over_scale",
    "num_qubits_over_32",
    "depth_over_128",
    "periodic_flag",
)

XYZ_GRID_CONTROL_FEATURE_NAMES = (
    "layer_fraction",
    "anneal_smoothstep",
    "sin_pi_s",
    "cos_pi_s",
    "z_field_over_scale",
    "alpha_mean_over_scale",
    "alpha_std_over_scale",
    "beta_mean_over_scale",
    "beta_std_over_scale",
    "yy_anisotropy",
    "grid_rows_over_8",
    "grid_cols_over_8",
    "num_qubits_over_32",
    "edge_density",
    "depth_over_128",
)


def build_mixed_field_ising_hamiltonian(
    *,
    num_qubits: int,
    jzz: float = 1.0,
    hx: float = 0.8,
    hz: float = 0.2,
    jxx: float = 0.0,
    jzz2: float = 0.0,
    periodic: bool = False,
    disorder_strength: float = 0.0,
    disorder_seed: int = 0,
) -> PauliHamiltonian:
    """Build a mixed-field spin-chain Hamiltonian.

    The sign convention is

    H = - sum_i Jzz_i Z_i Z_{i+1}
        - sum_i Jxx_i X_i X_{i+1}
        - sum_i hx X_i
        - sum_i hz_i Z_i
        - sum_i Jzz2_i Z_i Z_{i+2}.

    A nonzero ``disorder_strength`` deterministically modulates the ZZ couplings
    and local longitudinal fields. This gives a physically meaningful family of
    controllable many-body Hamiltonians without introducing chemistry
    dependencies.
    """
    if num_qubits <= 1:
        raise ValueError("num_qubits must be greater than 1")
    if disorder_strength < 0:
        raise ValueError("disorder_strength must be non-negative")

    zz_mod, z_mod, xx_mod, zz2_mod = _deterministic_disorder(
        num_qubits=num_qubits,
        strength=disorder_strength,
        seed=disorder_seed,
        periodic=periodic,
    )

    terms: list[PauliTerm] = []
    for edge_index, (left, right) in enumerate(_nearest_edges(num_qubits, periodic)):
        if jzz != 0.0:
            terms.append(
                PauliTerm(
                    coefficient=-float(jzz * zz_mod[edge_index]),
                    pauli_string=((left, "Z"), (right, "Z")),
                )
            )
        if jxx != 0.0:
            terms.append(
                PauliTerm(
                    coefficient=-float(jxx * xx_mod[edge_index]),
                    pauli_string=((left, "X"), (right, "X")),
                )
            )

    for qubit in range(num_qubits):
        if hx != 0.0:
            terms.append(PauliTerm(coefficient=-float(hx), pauli_string=((qubit, "X"),)))
        hz_i = hz * z_mod[qubit]
        if hz_i != 0.0:
            terms.append(PauliTerm(coefficient=-float(hz_i), pauli_string=((qubit, "Z"),)))

    for edge_index, (left, right) in enumerate(_next_nearest_edges(num_qubits, periodic)):
        if jzz2 != 0.0:
            terms.append(
                PauliTerm(
                    coefficient=-float(jzz2 * zz2_mod[edge_index]),
                    pauli_string=((left, "Z"), (right, "Z")),
                )
            )

    if not terms:
        raise ValueError("at least one Hamiltonian coefficient must be nonzero")

    metadata = (
        ("spin_chain", (0.0, 0.0, 0.0)),
        ("mixed_field_ising", (float(jzz), float(hx), float(hz))),
    )
    return PauliHamiltonian(
        num_qubits=num_qubits,
        terms=tuple(terms),
        nuclear_repulsion=0.0,
        hf_energy=None,
        fci_energy=None,
        molecule=f"mixed_field_ising_q{num_qubits}",
        geometry=metadata,
        basis="spin-chain",
        mapping="pauli_spin",
    )


def build_random_xyz_grid_hamiltonian(
    *,
    grid_rows: int,
    grid_cols: int,
    z_field: float = 1.0,
    alpha_mean: float = 1.0,
    alpha_std: float = 0.25,
    beta_mean: float = 3.0,
    beta_std: float = 0.25,
    yy_anisotropy: float = 0.66,
    coupling_seed: int = 0,
) -> PauliHamiltonian:
    """Build the random 2D XYZ-grid Hamiltonian used as a harder VQE target.

    The model follows

    H = sum_i z_field Z_i
        + sum_<i,j> alpha_ij Z_i Z_j
        + sum_<i,j> beta_ij (X_i X_j + yy_anisotropy Y_i Y_j),

    where <i,j> are nearest-neighbor edges on an open rectangular grid and
    alpha_ij, beta_ij are deterministic normal samples for a given seed.
    """
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_rows and grid_cols must be positive")
    if alpha_std < 0.0 or beta_std < 0.0:
        raise ValueError("coupling standard deviations must be non-negative")

    num_qubits = int(grid_rows * grid_cols)
    edges, alphas, betas = sample_xyz_grid_couplings(
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        alpha_mean=alpha_mean,
        alpha_std=alpha_std,
        beta_mean=beta_mean,
        beta_std=beta_std,
        coupling_seed=coupling_seed,
    )

    terms: list[PauliTerm] = []
    for qubit in range(num_qubits):
        if z_field != 0.0:
            terms.append(PauliTerm(coefficient=float(z_field), pauli_string=((qubit, "Z"),)))
    for (left, right), alpha, beta in zip(edges, alphas, betas, strict=True):
        terms.append(
            PauliTerm(
                coefficient=float(alpha),
                pauli_string=((left, "Z"), (right, "Z")),
            )
        )
        terms.append(
            PauliTerm(
                coefficient=float(beta),
                pauli_string=((left, "X"), (right, "X")),
            )
        )
        terms.append(
            PauliTerm(
                coefficient=float(yy_anisotropy * beta),
                pauli_string=((left, "Y"), (right, "Y")),
            )
        )

    metadata = (
        ("xyz_grid", (float(grid_rows), float(grid_cols), float(yy_anisotropy))),
        ("alpha_normal", (float(alpha_mean), float(alpha_std), float(coupling_seed))),
        ("beta_normal", (float(beta_mean), float(beta_std), float(coupling_seed))),
    )
    return PauliHamiltonian(
        num_qubits=num_qubits,
        terms=tuple(terms),
        nuclear_repulsion=0.0,
        hf_energy=None,
        fci_energy=None,
        molecule=f"xyz_grid_{grid_rows}x{grid_cols}",
        geometry=metadata,
        basis="spin-grid",
        mapping="pauli_spin",
    )


def sample_xyz_grid_couplings(
    *,
    grid_rows: int,
    grid_cols: int,
    alpha_mean: float = 1.0,
    alpha_std: float = 0.25,
    beta_mean: float = 3.0,
    beta_std: float = 0.25,
    coupling_seed: int = 0,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray, np.ndarray]:
    edges = _grid_edges(grid_rows, grid_cols)
    rng = np.random.default_rng(int(coupling_seed))
    alphas = rng.normal(float(alpha_mean), float(alpha_std), size=len(edges))
    betas = rng.normal(float(beta_mean), float(beta_std), size=len(edges))
    return edges, alphas.astype(np.float64), betas.astype(np.float64)


def build_xyz_grid_control_features(
    *,
    depth: int,
    grid_rows: int,
    grid_cols: int,
    target_z_field: float = 1.0,
    target_alpha_mean: float = 1.0,
    target_alpha_std: float = 0.25,
    target_beta_mean: float = 3.0,
    target_beta_std: float = 0.25,
    yy_anisotropy: float = 0.66,
    initial_z_field: float = 1.0,
    initial_alpha_mean: float = 0.0,
    initial_alpha_std: float = 0.0,
    initial_beta_mean: float = 0.0,
    initial_beta_std: float = 0.0,
) -> jnp.ndarray:
    """Build per-layer physical descriptors for the XYZ-grid VQE task."""
    if depth <= 0:
        raise ValueError("depth must be positive")
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_rows and grid_cols must be positive")

    num_qubits = grid_rows * grid_cols
    edge_count = len(_grid_edges(grid_rows, grid_cols))
    denom = max(depth - 1, 1)
    layer_fraction = jnp.arange(depth, dtype=jnp.float64) / float(denom)
    smooth = layer_fraction * layer_fraction * (3.0 - 2.0 * layer_fraction)

    z_field = _interpolate(initial_z_field, target_z_field, smooth)
    alpha_mean = _interpolate(initial_alpha_mean, target_alpha_mean, smooth)
    alpha_std = _interpolate(initial_alpha_std, target_alpha_std, smooth)
    beta_mean = _interpolate(initial_beta_mean, target_beta_mean, smooth)
    beta_std = _interpolate(initial_beta_std, target_beta_std, smooth)
    scale = max(
        1.0,
        abs(float(initial_z_field)),
        abs(float(target_z_field)),
        abs(float(initial_alpha_mean)),
        abs(float(target_alpha_mean)),
        abs(float(initial_alpha_std)),
        abs(float(target_alpha_std)),
        abs(float(initial_beta_mean)),
        abs(float(target_beta_mean)),
        abs(float(initial_beta_std)),
        abs(float(target_beta_std)),
    )

    static = jnp.stack(
        [
            jnp.full_like(layer_fraction, grid_rows / 8.0),
            jnp.full_like(layer_fraction, grid_cols / 8.0),
            jnp.full_like(layer_fraction, num_qubits / 32.0),
            jnp.full_like(layer_fraction, edge_count / max(num_qubits, 1)),
            jnp.full_like(layer_fraction, depth / 128.0),
        ],
        axis=-1,
    )
    dynamic = jnp.stack(
        [
            layer_fraction,
            smooth,
            jnp.sin(jnp.pi * layer_fraction),
            jnp.cos(jnp.pi * layer_fraction),
            z_field / scale,
            alpha_mean / scale,
            alpha_std / scale,
            beta_mean / scale,
            beta_std / scale,
            jnp.full_like(layer_fraction, float(yy_anisotropy)),
        ],
        axis=-1,
    )
    return jnp.concatenate([dynamic, static], axis=-1)


def build_spin_control_features(
    *,
    depth: int,
    num_qubits: int,
    target_jzz: float = 1.0,
    target_hx: float = 0.8,
    target_hz: float = 0.2,
    target_jxx: float = 0.0,
    target_jzz2: float = 0.0,
    target_disorder_strength: float = 0.0,
    initial_jzz: float = 0.0,
    initial_hx: float = 2.0,
    initial_hz: float = 0.0,
    initial_jxx: float = 0.0,
    initial_jzz2: float = 0.0,
    initial_disorder_strength: float = 0.0,
    periodic: bool = False,
) -> jnp.ndarray:
    """Build per-layer physical control features for recurrent VQE.

    Each row describes the instantaneous point on a simple annealing path from
    an easy transverse-field Hamiltonian to the target interacting Hamiltonian.
    These are the inputs consumed by the classical head in the spin-chain VQE
    benchmark.
    """
    if depth <= 0:
        raise ValueError("depth must be positive")
    if num_qubits <= 1:
        raise ValueError("num_qubits must be greater than 1")

    denom = max(depth - 1, 1)
    layer_fraction = jnp.arange(depth, dtype=jnp.float32) / float(denom)
    smooth = layer_fraction * layer_fraction * (3.0 - 2.0 * layer_fraction)

    jzz = _interpolate(initial_jzz, target_jzz, smooth)
    jxx = _interpolate(initial_jxx, target_jxx, smooth)
    hx = _interpolate(initial_hx, target_hx, smooth)
    hz = _interpolate(initial_hz, target_hz, smooth)
    jzz2 = _interpolate(initial_jzz2, target_jzz2, smooth)
    disorder = _interpolate(initial_disorder_strength, target_disorder_strength, smooth)

    scale = max(
        1.0,
        abs(float(initial_jzz)),
        abs(float(target_jzz)),
        abs(float(initial_jxx)),
        abs(float(target_jxx)),
        abs(float(initial_hx)),
        abs(float(target_hx)),
        abs(float(initial_hz)),
        abs(float(target_hz)),
        abs(float(initial_jzz2)),
        abs(float(target_jzz2)),
        abs(float(initial_disorder_strength)),
        abs(float(target_disorder_strength)),
    )
    static = jnp.stack(
        [
            jnp.full_like(layer_fraction, num_qubits / 32.0),
            jnp.full_like(layer_fraction, depth / 128.0),
            jnp.full_like(layer_fraction, 1.0 if periodic else 0.0),
        ],
        axis=-1,
    )
    dynamic = jnp.stack(
        [
            layer_fraction,
            smooth,
            jnp.sin(jnp.pi * layer_fraction),
            jnp.cos(jnp.pi * layer_fraction),
            jzz / scale,
            jxx / scale,
            hx / scale,
            hz / scale,
            jzz2 / scale,
            disorder / scale,
        ],
        axis=-1,
    )
    return jnp.concatenate([dynamic, static], axis=-1)


def exact_ground_energy(
    hamiltonian: PauliHamiltonian,
    *,
    prefer_sparse: bool = True,
) -> float:
    """Return the exact ground-state energy for a Pauli Hamiltonian."""
    if hamiltonian.num_qubits <= 0:
        raise ValueError("hamiltonian.num_qubits must be positive")
    if prefer_sparse:
        try:
            return _sparse_exact_ground_energy(hamiltonian)
        except ImportError:
            pass
    if hamiltonian.num_qubits > 12:
        raise RuntimeError(
            "Dense exact diagonalization is limited to <=12 qubits when scipy is unavailable."
        )
    return _dense_exact_ground_energy(hamiltonian)


def cached_exact_ground_energy(
    hamiltonian: PauliHamiltonian,
    *,
    cache_dir: str | Path | None = "data/spin_chain_cache",
    max_qubits: int | None = 14,
    prefer_sparse: bool = True,
) -> float | None:
    """Compute or load an exact ground energy, optionally capped by qubit count."""
    if max_qubits is not None and hamiltonian.num_qubits > max_qubits:
        return None
    if cache_dir is None:
        return exact_ground_energy(hamiltonian, prefer_sparse=prefer_sparse)

    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / f"{spin_hamiltonian_cache_key(hamiltonian)}.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        return float(cached["exact_ground_energy"])

    energy = exact_ground_energy(hamiltonian, prefer_sparse=prefer_sparse)
    payload = {
        "num_qubits": hamiltonian.num_qubits,
        "num_pauli_terms": len(hamiltonian.terms),
        "exact_ground_energy": float(energy),
    }
    tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, cache_path)
    return float(energy)


def spin_hamiltonian_cache_key(hamiltonian: PauliHamiltonian) -> str:
    rows = [
        {
            "coefficient": round(float(term.coefficient), 12),
            "pauli_string": tuple((int(q), str(p).upper()) for q, p in term.pauli_string),
        }
        for term in hamiltonian.terms
    ]
    payload = {
        "num_qubits": hamiltonian.num_qubits,
        "terms": rows,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _nearest_edges(num_qubits: int, periodic: bool) -> tuple[tuple[int, int], ...]:
    edges = [(i, i + 1) for i in range(num_qubits - 1)]
    if periodic and num_qubits > 2:
        edges.append((num_qubits - 1, 0))
    return tuple(edges)


def _grid_edges(grid_rows: int, grid_cols: int) -> tuple[tuple[int, int], ...]:
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_rows and grid_cols must be positive")
    edges: list[tuple[int, int]] = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            index = row * grid_cols + col
            if col + 1 < grid_cols:
                edges.append((index, index + 1))
            if row + 1 < grid_rows:
                edges.append((index, index + grid_cols))
    return tuple(edges)


def _next_nearest_edges(num_qubits: int, periodic: bool) -> tuple[tuple[int, int], ...]:
    if num_qubits <= 2:
        return ()
    edges = [(i, i + 2) for i in range(num_qubits - 2)]
    if periodic and num_qubits > 3:
        edges.extend([(num_qubits - 2, 0), (num_qubits - 1, 1)])
    return tuple(edges)


def _deterministic_disorder(
    *,
    num_qubits: int,
    strength: float,
    seed: int,
    periodic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nearest_count = len(_nearest_edges(num_qubits, periodic))
    next_count = len(_next_nearest_edges(num_qubits, periodic))
    if strength == 0.0:
        return (
            np.ones(nearest_count, dtype=np.float64),
            np.ones(num_qubits, dtype=np.float64),
            np.ones(nearest_count, dtype=np.float64),
            np.ones(next_count, dtype=np.float64),
        )

    rng = np.random.default_rng(int(seed))
    low = 1.0 - float(strength)
    high = 1.0 + float(strength)
    return (
        rng.uniform(low, high, size=nearest_count),
        rng.uniform(low, high, size=num_qubits),
        rng.uniform(low, high, size=nearest_count),
        rng.uniform(low, high, size=next_count),
    )


def _interpolate(start: float, end: float, schedule: jnp.ndarray) -> jnp.ndarray:
    dtype = schedule.dtype
    return jnp.asarray(start, dtype=dtype) + (
        jnp.asarray(end, dtype=dtype) - jnp.asarray(start, dtype=dtype)
    ) * schedule


def _dense_exact_ground_energy(hamiltonian: PauliHamiltonian) -> float:
    matrix = np.zeros((2**hamiltonian.num_qubits, 2**hamiltonian.num_qubits), dtype=np.complex128)
    for term in hamiltonian.terms:
        matrix = matrix + float(term.coefficient) * _dense_pauli_term_matrix(
            hamiltonian.num_qubits,
            term.pauli_string,
        )
    return float(np.linalg.eigvalsh(matrix)[0].real)


def _dense_pauli_term_matrix(
    num_qubits: int,
    pauli_string: Iterable[tuple[int, str]],
) -> np.ndarray:
    term_ops = {int(qubit): str(pauli).upper() for qubit, pauli in pauli_string}
    matrix = np.asarray([[1.0]], dtype=np.complex128)
    for qubit in range(num_qubits):
        matrix = np.kron(matrix, _dense_pauli_matrix(term_ops.get(qubit, "I")))
    return matrix


def _dense_pauli_matrix(pauli: str) -> np.ndarray:
    if pauli == "I":
        return np.asarray([[1, 0], [0, 1]], dtype=np.complex128)
    if pauli == "X":
        return np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    if pauli == "Y":
        return np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
    if pauli == "Z":
        return np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
    raise ValueError(f"Unsupported Pauli operator {pauli!r}")


def _sparse_exact_ground_energy(hamiltonian: PauliHamiltonian) -> float:
    from scipy import sparse
    from scipy.sparse.linalg import eigsh

    dimension = 2**hamiltonian.num_qubits
    if dimension <= 16:
        return _dense_exact_ground_energy(hamiltonian)

    matrix = sparse.csr_matrix((dimension, dimension), dtype=np.complex128)
    for term in hamiltonian.terms:
        matrix = matrix + float(term.coefficient) * _sparse_pauli_term_matrix(
            hamiltonian.num_qubits,
            term.pauli_string,
        )
    eigenvalues = eigsh(matrix, k=1, which="SA", return_eigenvectors=False, tol=1e-9)
    return float(np.min(eigenvalues).real)


def _sparse_pauli_term_matrix(
    num_qubits: int,
    pauli_string: Iterable[tuple[int, str]],
):
    from scipy import sparse

    term_ops = {int(qubit): str(pauli).upper() for qubit, pauli in pauli_string}
    matrix = sparse.csr_matrix([[1.0]], dtype=np.complex128)
    for qubit in range(num_qubits):
        matrix = sparse.kron(matrix, _sparse_pauli_matrix(term_ops.get(qubit, "I")), format="csr")
    return matrix


def _sparse_pauli_matrix(pauli: str):
    from scipy import sparse

    return sparse.csr_matrix(_dense_pauli_matrix(pauli))
