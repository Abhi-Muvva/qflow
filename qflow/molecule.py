"""
molecule.py
===========
Infrastructure layer: molecule construction, RHF, MO integrals,
FCI bitstring basis, and the H sigma-vector.

Nothing in this file is QFlow-specific. To use a different molecule
geometry or basis set, only this file needs to change.

Extension notes
---------------
- New molecule types (rings, 3D geometries): add a builder here,
  parallel to make_h_chain.
- Larger basis sets: pass basis= to make_h_chain; everything
  downstream adapts automatically.
- Active-space integral screening: modify run_rhf_and_integrals
  to return a truncated eri_packed.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1


def make_h_chain(nH: int, R_bohr: float, basis: str = "sto-3g") -> gto.Mole:
    """
    Build a PySCF Mole for a linear hydrogen chain.

    Atoms placed along x-axis at 0, R, 2R, ... (Bohr).
    symmetry is disabled so orbital indices are stable across geometries.
    """
    coords = [(i * R_bohr, 0.0, 0.0) for i in range(nH)]
    return gto.M(
        atom=[("H", c) for c in coords],
        unit="Bohr",
        basis=basis,
        charge=0,
        spin=0,
        symmetry=False,
    )


def run_rhf_and_integrals(mol: gto.Mole):
    """
    Run RHF and extract everything the QFlow machinery needs.

    Returns
    -------
    mf         : converged RHF object
    eps        : MO energies, shape (nmo,)
    occ        : MO occupations, shape (nmo,) — 2.0 or 0.0
    h1_mo      : one-electron integrals in MO basis, shape (nmo, nmo)
    eri_packed : two-electron integrals in MO basis, packed (compact=True)
    E_nuc      : nuclear repulsion energy (scalar)
    """
    mf = scf.RHF(mol)
    mf.max_cycle = 200
    mf.kernel()
    C          = np.asarray(mf.mo_coeff)
    eps        = np.asarray(mf.mo_energy)
    occ        = np.asarray(mf.mo_occ)
    h1_mo      = C.T @ mf.get_hcore() @ C
    eri_packed = ao2mo.kernel(mol, C, compact=True)
    E_nuc      = float(mol.energy_nuc())
    return mf, eps, occ, h1_mo, eri_packed, E_nuc


def build_fci_string_basis(nmo: int, nelec: int):
    """
    Enumerate all alpha/beta occupation strings for the full FCI space.

    Returns
    -------
    nalpha, nbeta         : electrons per spin
    alpha_strs, beta_strs : bitstring integer arrays
    a_index, b_index      : bitstring -> position dicts
    na, nb                : Hilbert-space dimensions per spin

    CI vectors are flat arrays of length na*nb indexed as vec[ia*nb + ib].
    """
    nalpha = nbeta = nelec // 2
    alpha_strs = cistring.gen_strings4orblist(range(nmo), nalpha)
    beta_strs  = cistring.gen_strings4orblist(range(nmo), nbeta)
    a_index    = {int(b): i for i, b in enumerate(alpha_strs)}
    b_index    = {int(b): i for i, b in enumerate(beta_strs)}
    na, nb     = len(alpha_strs), len(beta_strs)
    return nalpha, nbeta, alpha_strs, beta_strs, a_index, b_index, na, nb


def apply_H_fci(vec_flat, h1_mo, eri_packed, nmo, nelec, na, nb):
    """
    Sigma-vector: return H|vec> for a CI vector in the full FCI space.

    Uses PySCF absorb_h1e + contract_2e for efficient one- and
    two-electron contraction.
    """
    ci    = vec_flat.reshape(na, nb)
    h2eff = direct_spin1.absorb_h1e(h1_mo, eri_packed, nmo, nelec, fac=0.5)
    Hv    = direct_spin1.contract_2e(h2eff, ci, nmo, nelec)
    return Hv.reshape(na * nb)
