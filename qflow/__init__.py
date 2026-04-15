"""
qflow
=====
QFlow(4e,4o) replication package — Kowalski & Bauman, arXiv:2305.05168v2.

Public API
----------
from qflow import run_qflow                  # main entry point
from qflow import make_h_chain               # molecule builder
from qflow import build_refstate, compute_*  # reference energies
"""

from .molecule import (
    make_h_chain,
    run_rhf_and_integrals,
    build_fci_string_basis,
    apply_H_fci,
)

from .ses import (
    enumerate_ses,
    build_ses_basis_vectors,
    sigma_int_keys_for_ses,
    build_sigma_sparse,
    init_sigma_pool_from_ses,
    precompute_ses_data,
    SigmaKey,
)

from .heff import (
    build_Heff,
    compute_ses_energy_stringmb,
)

from .optimizer import run_qflow

from .references import (
    Refstate,
    build_refstate,
    compute_hf_energy,
    compute_fci_energy,
    compute_casci_energy,
    compute_ccsd_energy_pyscf,
    compute_ccsdt_energy_ccpy,
    compute_ccsdtq_energy_ccpy,
)
