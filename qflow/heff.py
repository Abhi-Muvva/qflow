"""
heff.py
=======
Effective Hamiltonian layer.

Covers:
  - Building H_eff in the CAS basis (Eq. 15, N=1)
  - CAS determinant bit extraction
  - Anti-Hermitian generator τ_k in the CAS basis
  - UCC-style internal state |Ψ_int>
  - Energy expectation value
  - Commutator gradient (Eq. 19)
  - Exact-diagonalisation fallback energy

Extension notes
---------------
Quantum Natural Gradient (QNG) preconditioning:
  - Compute the quantum Fisher information matrix F_kl = <[τ_k, τ_l]>
    inside _gradients_commutator, then return F alongside raw gradients.
  - In optimizer.py, apply F^{-1} g before the SGD step.
  - Only _gradients_commutator and optimizer.py cycle function change.

Alternative ansatz (e.g. hardware-efficient, Givens rotations):
  - Replace _psi_int_cas with a new state-preparation function.
  - The gradient formula in _gradients_commutator stays the same as long
    as the ansatz is differentiable w.r.t. the same τ_k generators.

Larger CAS (QFlow(6e,6o)):
  - build_Heff is dimension-agnostic; only V_ses changes (from ses.py).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import expm as dense_expm
from scipy.sparse.linalg import expm_multiply

from .molecule import apply_H_fci
from .ses import build_sigma_sparse, _annihilate_orb, _create_orb, SigmaKey


def build_Heff(
    V_ses: np.ndarray,
    h1_mo,
    eri_packed,
    nmo: int,
    nelec: int,
    na: int,
    nb: int,
    alpha_strs,
    beta_strs,
    a_index: Dict[int, int],
    b_index: Dict[int, int],
    sigma_pool: Dict[SigmaKey, float],
    sigma_ext_keys: set,
) -> np.ndarray:
    """
    Build H_eff(hi) for one SES in its CAS basis (Eq. 15, N=1).

        H_eff = V^T  e^{-σ_ext}  H  e^{+σ_ext}  V

    Symmetrised at the end to suppress floating-point skew.
    """
    S_ext = build_sigma_sparse(
        nmo, alpha_strs, beta_strs, a_index, b_index,
        sigma_pool, sigma_ext_keys,
    )

    ncas        = V_ses.shape[1]
    H_eff       = np.zeros((ncas, ncas))
    any_nonzero = S_ext.nnz > 0

    for j in range(ncas):
        v_j = V_ses[:, j]
        if any_nonzero:
            u  = expm_multiply( S_ext,  v_j)
            Hu = apply_H_fci(u, h1_mo, eri_packed, nmo, nelec, na, nb)
            y  = expm_multiply(-S_ext,  Hu)
        else:
            y  = apply_H_fci(v_j, h1_mo, eri_packed, nmo, nelec, na, nb)
        H_eff[:, j] = V_ses.T @ y

    return 0.5 * (H_eff + H_eff.T)


def compute_ses_energy_stringmb(H_eff: np.ndarray, E_nuc: float) -> float:
    """
    Exact-diagonalisation energy for one SES: lowest eigenvalue of H_eff + E_nuc.
    Used as a diagnostic reference; the optimised energy uses _energy_expectation.
    """
    return float(np.linalg.eigvalsh(H_eff)[0]) + float(E_nuc)


def _extract_det_bits_cas(
    V_ses: np.ndarray,
    alpha_strs,
    beta_strs,
    nb: int,
) -> List[Tuple[int, int]]:
    """
    Return (alpha_bit, beta_bit) for each column of V_ses.
    Each column has exactly one nonzero entry at row idx = ia*nb + ib.
    """
    det_bits: List[Tuple[int, int]] = []
    for j in range(V_ses.shape[1]):
        idx = int(np.argmax(np.abs(V_ses[:, j])))
        ia  = idx // nb
        ib  = idx % nb
        det_bits.append((int(alpha_strs[ia]), int(beta_strs[ib])))
    return det_bits


def _tau_cas(
    det_bits_cas: List[Tuple[int, int]],
    key: SigmaKey,
) -> np.ndarray:
    """
    Anti-Hermitian generator τ = E - E† in the CAS basis for key ("x", holes, particles).

    Builds the excitation operator E by applying annihilation (holes) then
    creation (particles) operators to each CAS determinant, records the result
    and fermionic phase, then anti-Hermitises.
    """
    tag, holes, particles = key
    if tag != "x":
        raise ValueError(f"Unsupported key tag: {tag!r}")

    ncas      = len(det_bits_cas)
    det_index = {det_bits_cas[j]: j for j in range(ncas)}

    alpha_holes = sorted([h // 2 for h in holes if h % 2 == 0])
    beta_holes  = sorted([h // 2 for h in holes if h % 2 == 1])
    alpha_parts = sorted([p // 2 for p in particles if p % 2 == 0])
    beta_parts  = sorted([p // 2 for p in particles if p % 2 == 1])

    E_mat = np.zeros((ncas, ncas))

    for j, (ab, bb) in enumerate(det_bits_cas):
        new_a   = ab
        phase_a = 1
        valid   = True
        for h in alpha_holes:
            new_a, p = _annihilate_orb(new_a, h)
            if new_a is None:
                valid = False; break
            phase_a *= p
        if valid:
            for p_orb in alpha_parts:
                new_a, p = _create_orb(new_a, p_orb)
                if new_a is None:
                    valid = False; break
                phase_a *= p
        if not valid:
            continue

        new_b   = bb
        phase_b = 1
        for h in beta_holes:
            new_b, p = _annihilate_orb(new_b, h)
            if new_b is None:
                valid = False; break
            phase_b *= p
        if valid:
            for p_orb in beta_parts:
                new_b, p = _create_orb(new_b, p_orb)
                if new_b is None:
                    valid = False; break
                phase_b *= p
        if not valid:
            continue

        tgt = (int(new_a), int(new_b))
        if tgt in det_index:
            E_mat[det_index[tgt], j] += phase_a * phase_b

    return E_mat - E_mat.T


def _psi_int_cas(
    V_ses: np.ndarray,
    sigma_pool: Dict[SigmaKey, float],
    sigma_int_keys: set,
    *,
    alpha_strs,
    beta_strs,
    nb: int,
) -> np.ndarray:
    """
    Build |Ψ_int> in the CAS determinant basis via UCC ansatz:

        |Ψ_int> = exp( Σ_k θ_k τ_k ) |Φ_ref>

    where |Φ_ref> = e_0 (column-0 of V_ses, the SES reference determinant).
    """
    det_bits = _extract_det_bits_cas(V_ses, alpha_strs, beta_strs, nb)
    ncas     = V_ses.shape[1]
    e0       = np.zeros(ncas, dtype=np.float64)
    e0[0]    = 1.0

    sigma = np.zeros((ncas, ncas), dtype=np.float64)
    for k in sigma_int_keys:
        theta = float(sigma_pool.get(k, 0.0))
        if abs(theta) > 1e-14:
            sigma += theta * _tau_cas(det_bits, k)

    c = dense_expm(sigma) @ e0 if np.any(np.abs(sigma) > 1e-14) else e0.copy()
    nrm = float(np.linalg.norm(c))
    if nrm > 1e-14:
        c /= nrm
    return c


def _energy_expectation(H_eff: np.ndarray, c: np.ndarray, E_nuc: float) -> float:
    """VQE energy expectation: <c|H_eff|c> + E_nuc."""
    return float(c @ (H_eff @ c)) + float(E_nuc)


def _gradients_commutator(
    V_ses: np.ndarray,
    H_eff: np.ndarray,
    sigma_pool: Dict[SigmaKey, float],
    sigma_int_keys: set,
    *,
    alpha_strs,
    beta_strs,
    nb: int,
) -> Dict[SigmaKey, float]:
    """
    Commutator gradient (Eq. 19):

        grad_k = <Ψ_int| [H_eff, τ_k] |Ψ_int>

    where |Ψ_int> is built from the current pool amplitudes.
    """
    c        = _psi_int_cas(V_ses, sigma_pool, sigma_int_keys,
                            alpha_strs=alpha_strs, beta_strs=beta_strs, nb=nb)
    det_bits = _extract_det_bits_cas(V_ses, alpha_strs, beta_strs, nb)

    grads: Dict[SigmaKey, float] = {}
    for k in sigma_int_keys:
        T    = _tau_cas(det_bits, k)
        comm = H_eff @ T - T @ H_eff
        grads[k] = float(c @ (comm @ c))
    return grads
