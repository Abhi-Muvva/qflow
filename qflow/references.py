"""
references.py
=============
Classical reference energies for Table I benchmarking.

Covers: HF, CAS-ED, CCSD (PySCF), CCSDT, CCSDTQ (CCpy), FCI/ED.

This file is entirely independent of the QFlow machinery.
It can be imported and run standalone to reproduce the non-QFlow
rows of Table I.

Extension notes
---------------
New methods (e.g. MRCISD, DMRG via Block2):
  - Add a new compute_* function following the same Refstate pattern.
  - Call it from Flow.ipynb and add the key to the method_order list
    in the Table I print cell.

CCpy version changes:
  - If CCpy's Driver API changes, only _ccpy_total_energy needs updating.
  - The public compute_ccsdt_energy_ccpy / compute_ccsdtq_energy_ccpy
    signatures stay the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from pyscf import gto, scf, fci, mcscf, cc

try:
    from ccpy.drivers.driver import Driver as _CCpyDriver
    _CCPY_AVAILABLE = True
except ImportError:
    _CCPY_AVAILABLE = False


@dataclass(frozen=True)
class Refstate:
    """Converged RHF state needed by all reference-energy functions."""
    mol:  gto.Mole
    mf:   Any       # scf.RHF instance
    e_hf: float


def build_refstate(
    mol: gto.Mole,
    symmetry_subgroup: Optional[str] = "D2h",
) -> Refstate:
    """
    Run RHF on mol and return a Refstate.

    symmetry_subgroup forces a finite point group so CCpy does not
    encounter 'Dooh'. Pass None to skip symmetry entirely.
    """
    if symmetry_subgroup is not None:
        mol = mol.copy()
        mol.symmetry = True
        mol.symmetry_subgroup = symmetry_subgroup
        mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = 200
    mf.kernel()
    return Refstate(mol=mol, mf=mf, e_hf=float(mf.e_tot))


def compute_hf_energy(refstate: Refstate) -> float:
    """RHF total energy."""
    return float(refstate.e_hf)


def compute_fci_energy(refstate: Refstate) -> float:
    """Full-space exact diagonalisation (ED) energy."""
    cisolver = fci.FCI(refstate.mf)
    e_fci, _ = cisolver.kernel()
    return float(e_fci)


def compute_casci_energy(
    refstate: Refstate,
    *,
    ncas: int,
    nelecas: int,
    use_natorb: bool = True,
) -> float:
    """
    CAS-ED energy: exact diagonalisation in the primary active space.

    For QFlow(4e,4o): ncas=4, nelecas=4.
    Uses the RHF MO ordering; verify that PySCF's default ordering
    matches the primary SES (index 0 from enumerate_ses).
    """
    solver          = mcscf.CASCI(refstate.mf, ncas=ncas, nelecas=nelecas)
    solver.natorb   = use_natorb
    solver.mo_coeff = refstate.mf.mo_coeff
    out             = solver.kernel()
    return float(out[0])


def compute_ccsd_energy_pyscf(refstate: Refstate) -> float:
    """CCSD total energy via PySCF."""
    solver = cc.CCSD(refstate.mf)
    e_corr = solver.kernel()[0]
    return float(refstate.e_hf + e_corr)


def _ccpy_total_energy(
    refstate: Refstate,
    *,
    method: str,
    nfrozen: int = 0,
    force_point_group_string: bool = True,
    ccpy_options: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Run a CCpy method and return the total energy.

    Handles CCpy's Driver attribute naming variability by probing
    several possible energy attribute names.
    """
    if not _CCPY_AVAILABLE:
        raise ImportError(f"CCpy is not installed; cannot compute {method.upper()}.")

    mol = refstate.mf.mol
    if force_point_group_string and getattr(mol, "symmetry", False) is True:
        mol.symmetry = mol.groupname

    pg = str(getattr(mol, "symmetry", "C1")).upper()
    if pg in {"DOOH", "COOV"}:
        raise ValueError(
            f"CCpy does not recognise point group '{pg}'. "
            "Build with mol.symmetry_subgroup='D2h' before RHF."
        )

    driver = _CCpyDriver.from_pyscf(refstate.mf, nfrozen=nfrozen)
    driver.options["energy_convergence"] = 1.0e-7
    driver.options["amp_convergence"]    = 1.0e-7
    driver.options["maximum_iterations"] = 80
    if ccpy_options:
        driver.options.update(ccpy_options)

    driver.run_cc(method=method.lower())

    for attr in ("total_energy", "e_tot", "cc_total_energy"):
        if hasattr(driver, attr):
            return float(getattr(driver, attr))

    e_ref = next(
        (float(getattr(driver, a))
         for a in ("reference_energy", "e_ref", "hf_energy")
         if hasattr(driver, a)), None
    )
    e_corr = next(
        (float(getattr(driver, a))
         for a in ("correlation_energy", "e_corr")
         if hasattr(driver, a)), None
    )

    if e_ref is not None and e_corr is not None:
        return e_ref + e_corr
    if e_corr is not None:
        return float(refstate.e_hf + e_corr)

    raise RuntimeError(
        f"Cannot locate energy on CCpy Driver after method='{method}'. "
        "Inspect dir(driver) to find the correct attribute."
    )


def compute_ccsdt_energy_ccpy(
    refstate: Refstate,
    *,
    nfrozen: int = 0,
    force_point_group_string: bool = True,
    ccpy_options: Optional[Dict[str, Any]] = None,
) -> float:
    """CCSDT total energy via CCpy."""
    return _ccpy_total_energy(
        refstate, method="ccsdt", nfrozen=nfrozen,
        force_point_group_string=force_point_group_string,
        ccpy_options=ccpy_options,
    )


def compute_ccsdtq_energy_ccpy(
    refstate: Refstate,
    *,
    nfrozen: int = 0,
    force_point_group_string: bool = True,
    ccpy_options: Optional[Dict[str, Any]] = None,
) -> float:
    """CCSDTQ total energy via CCpy."""
    return _ccpy_total_energy(
        refstate, method="ccsdtq", nfrozen=nfrozen,
        force_point_group_string=force_point_group_string,
        ccpy_options=ccpy_options,
    )
