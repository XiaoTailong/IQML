from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import jax.numpy as jnp


@dataclass(frozen=True)
class PauliTerm:
    coefficient: float
    pauli_string: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class PauliHamiltonian:
    num_qubits: int
    terms: tuple[PauliTerm, ...]
    nuclear_repulsion: float
    hf_energy: float | None
    fci_energy: float | None
    molecule: str
    geometry: tuple[tuple[str, tuple[float, float, float]], ...]
    basis: str
    mapping: str = "jordan_wigner"

    @property
    def coefficient_array(self) -> jnp.ndarray:
        return jnp.asarray([term.coefficient for term in self.terms], dtype=jnp.float32)


def linear_hydrogen_chain_geometry(
    num_atoms: int,
    spacing: float,
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    if num_atoms <= 0:
        raise ValueError("num_atoms must be positive")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    return tuple(("H", (0.0, 0.0, float(i * spacing))) for i in range(num_atoms))


@lru_cache(maxsize=32)
def build_hydrogen_chain_hamiltonian(
    num_atoms: int,
    spacing: float,
    basis: str = "sto-3g",
    multiplicity: int = 1,
    charge: int = 0,
    cache_dir: str | Path | None = "data/vqe_cache",
) -> PauliHamiltonian:
    """Build a Jordan-Wigner molecular Hamiltonian for a linear H chain.

    The first target benchmark is ``num_atoms=4``. In STO-3G this gives 8 spin
    orbitals and therefore an 8-qubit Hamiltonian.
    """
    try:
        from openfermion import MolecularData, get_fermion_operator, jordan_wigner
        from openfermionpyscf import run_pyscf
    except ImportError as exc:
        raise ImportError(
            "VQE chemistry generation requires openfermion, openfermionpyscf, and pyscf. "
            "Install them with: "
            "python -m pip install openfermion openfermionpyscf pyscf"
        ) from exc

    geometry = linear_hydrogen_chain_geometry(num_atoms=num_atoms, spacing=spacing)
    storage = None
    if cache_dir is not None:
        storage = Path(cache_dir)
        storage.mkdir(parents=True, exist_ok=True)

    molecule = MolecularData(
        geometry=list(geometry),
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
        filename=str(_molecule_cache_path(storage, num_atoms, spacing, basis)) if storage else "",
    )
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    qubit_operator = jordan_wigner(get_fermion_operator(molecule.get_molecular_hamiltonian()))
    qubit_operator.compress()

    terms = _pauli_terms_from_openfermion(qubit_operator.terms.items())
    return PauliHamiltonian(
        num_qubits=int(molecule.n_qubits),
        terms=terms,
        nuclear_repulsion=float(molecule.nuclear_repulsion),
        hf_energy=_optional_float(getattr(molecule, "hf_energy", None)),
        fci_energy=_optional_float(getattr(molecule, "fci_energy", None)),
        molecule=f"H{num_atoms}",
        geometry=geometry,
        basis=basis,
        mapping="jordan_wigner",
    )


def _pauli_terms_from_openfermion(
    raw_terms: Iterable[tuple[tuple[tuple[int, str], ...], complex]],
) -> tuple[PauliTerm, ...]:
    terms: list[PauliTerm] = []
    for pauli_string, coefficient in raw_terms:
        coeff = complex(coefficient)
        if abs(coeff.imag) > 1e-8:
            raise ValueError(f"Expected real molecular Hamiltonian coefficient, got {coeff}")
        terms.append(
            PauliTerm(
                coefficient=float(coeff.real),
                pauli_string=tuple((int(index), str(pauli)) for index, pauli in pauli_string),
            )
        )
    terms.sort(key=lambda term: (len(term.pauli_string), term.pauli_string))
    return tuple(terms)


def _molecule_cache_path(
    cache_dir: Path | None,
    num_atoms: int,
    spacing: float,
    basis: str,
) -> Path:
    if cache_dir is None:
        return Path("")
    safe_basis = basis.replace("/", "_").replace(" ", "_")
    safe_spacing = f"{spacing:.6f}".replace(".", "p")
    return cache_dir / f"h{num_atoms}_{safe_basis}_r{safe_spacing}"


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
