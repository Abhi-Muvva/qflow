"""
ses.py
======
Sub-system embedding sub-algebra (SES) layer.

Covers:
  - Type aliases for sigma keys
  - SES enumeration (which pairs of occ/vir orbitals form each block)
  - CAS(4e,4o) basis vector construction (36 determinants per SES)
  - Internal key enumeration (35 spin-orbital excitations per SES)
  - Fermionic sign helpers used by sigma matrix and tau_cas
  - Sparse sigma matrix construction
  - Global pool initialisation and precomputation

Extension notes
---------------
Gradient-norm active-space pruning (arXiv:2410.11992 style):
  - Modify precompute_ses_data to accept a pruning mask or score vector.
  - After the first cycle, score each SES by max gradient norm on its
    owned keys; drop SES below threshold from ses_list before the next
    precompute_ses_data call.
  - No other file needs to change.

ADMM consensus coupling (overlapping SES):
  - The owned_keys split in precompute_ses_data enforces disjoint
    ownership. For ADMM you would relax this: allow multiple SES to
    own the same key and add a consensus penalty term. Modify
    precompute_ses_data and init_sigma_pool_from_ses accordingly.

Larger active spaces (e.g. QFlow(6e,6o)):
  - Change the combinations(occ_orbs, 2) calls in enumerate_ses and
    build_ses_basis_vectors to combinations(occ_orbs, 3), etc.
  - Update sigma_int_keys_for_ses rank range if needed.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse


ExcitationPairs = Tuple[Tuple[int, int], ...]
SigmaKey = tuple  # ("x", holes_spinorb_tuple, particles_spinorb_tuple)


# ── Fermionic sign helpers ───────────────────────────────────────────────────

def _popcount(x: int) -> int:
    """Number of set bits in integer x."""
    return bin(int(x)).count("1")


def _annihilate_orb(det: int, i: int):
    """
    Annihilate orbital i from bitstring det.
    Returns (new_det, phase) or (None, 0) if orbital is unoccupied.
    """
    if ((det >> i) & 1) == 0:
        return None, 0
    occ_below = _popcount(det & ((1 << i) - 1))
    return det & ~(1 << i), (-1) ** (occ_below % 2)


def _create_orb(det: int, a: int):
    """
    Create orbital a in bitstring det.
    Returns (new_det, phase) or (None, 0) if orbital is already occupied.
    """
    if ((det >> a) & 1) == 1:
        return None, 0
    occ_below = _popcount(det & ((1 << a) - 1))
    return det | (1 << a), (-1) ** (occ_below % 2)


def _phase_single(det: int, i: int, a: int) -> int:
    """Fermionic phase for a single excitation i->a on determinant det."""
    occ_below_i = _popcount(det & ((1 << i) - 1))
    sign1       = (-1) ** (occ_below_i % 2)
    det_removed = det & ~(1 << i)
    occ_below_a = _popcount(det_removed & ((1 << a) - 1))
    sign2       = (-1) ** (occ_below_a % 2)
    return int(sign1 * sign2)


def _apply_single(det: int, i: int, a: int):
    """Apply single excitation i->a to determinant det."""
    if ((det >> i) & 1) == 0 or ((det >> a) & 1) == 1:
        return None, 0
    new_det = (det & ~(1 << i)) | (1 << a)
    return int(new_det), _phase_single(det, i, a)


def _apply_pairs_to_det(det: int, pairs: ExcitationPairs) -> tuple:
    """
    Apply a sequence of (i->a) single excitations to one spin-string.
    Returns (new_det, sign) or (None, 0) if any step is invalid.
    """
    new_det    = int(det)
    sign_total = 1
    for (i, a) in pairs:
        out_det, sgn = _apply_single(new_det, int(i), int(a))
        if out_det is None:
            return None, 0
        new_det     = int(out_det)
        sign_total *= int(sgn)
    return int(new_det), int(sign_total)


# ── SES enumeration ──────────────────────────────────────────────────────────

def enumerate_ses(eps, occ):
    """
    Enumerate all (4e,4o) SES blocks for QFlow(4e,4o).

    Each SES = 2 occupied + 2 virtual orbitals.
      H6: C(3,2) x C(3,2) = 9  SES
      H8: C(4,2) x C(4,2) = 36 SES

    Sorted so index 0 is the PRIMARY active space:
      two highest-energy occupied + two lowest-energy virtual.
    """
    occ_orbs = np.where(occ > 0.0)[0].tolist()
    vir_orbs = np.where(occ == 0.0)[0].tolist()
    ses_list = []
    for occ_pair in combinations(occ_orbs, 2):
        for vir_pair in combinations(vir_orbs, 2):
            active = tuple(sorted(occ_pair + vir_pair))
            ses_list.append((active, occ_pair, vir_pair))
    ses_list.sort(key=lambda t: (
        -float(np.sum(eps[list(t[1])])),
         float(np.sum(eps[list(t[2])])),
    ))
    return occ_orbs, vir_orbs, ses_list


# ── CAS basis vectors ────────────────────────────────────────────────────────

def bit_from_occ(occ_list) -> int:
    """Convert a list/tuple of occupied orbital indices to a bitstring integer."""
    out = 0
    for i in occ_list:
        out |= (1 << int(i))
    return int(out)


def build_ses_basis_vectors(
    occ_ref: List[int],
    occ_pair,
    vir_pair,
    a_index: Dict[int, int],
    b_index: Dict[int, int],
    na: int,
    nb: int,
) -> np.ndarray:
    """
    Build V_ses — the (na*nb, 36) matrix spanning CAS(4e,4o) for one SES.

    Frozen orbitals outside the active set remain doubly occupied.
    Active orbitals distribute 2 alpha and 2 beta electrons among 4 spatial orbitals.
    Column 0 is the SES reference determinant (occ_pair in both spins);
    columns 1-35 are the remaining determinants in lexicographic order.
    """
    active_set = tuple(sorted(set(map(int, occ_pair)) | set(map(int, vir_pair))))
    if len(active_set) != 4:
        raise ValueError("SES active_set must have exactly 4 spatial orbitals.")

    occ_ref_set = set(int(x) for x in occ_ref)
    frozen_occ  = tuple(sorted(occ_ref_set - set(active_set)))

    det_pairs: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    for a_act in combinations(active_set, 2):
        alpha_occ = tuple(sorted(frozen_occ + tuple(sorted(a_act))))
        for b_act in combinations(active_set, 2):
            beta_occ = tuple(sorted(frozen_occ + tuple(sorted(b_act))))
            det_pairs.append((alpha_occ, beta_occ))

    occ_pair_sorted = tuple(sorted(map(int, occ_pair)))
    alpha_ref = tuple(sorted(frozen_occ + occ_pair_sorted))
    beta_ref  = tuple(sorted(frozen_occ + occ_pair_sorted))

    det_pairs_sorted = sorted(det_pairs, key=lambda p: (p[0], p[1]))
    if (alpha_ref, beta_ref) not in det_pairs_sorted:
        raise RuntimeError("Reference determinant not found in CAS list.")

    det_pairs_sorted.remove((alpha_ref, beta_ref))
    det_pairs_sorted = [(alpha_ref, beta_ref)] + det_pairs_sorted

    if len(det_pairs_sorted) != 36:
        raise RuntimeError(f"CAS determinant count != 36 (got {len(det_pairs_sorted)}).")

    V_cols: List[np.ndarray] = []
    for alpha_occ, beta_occ in det_pairs_sorted:
        ia = a_index[bit_from_occ(alpha_occ)]
        ib = b_index[bit_from_occ(beta_occ)]
        v  = np.zeros(na * nb)
        v[ia * nb + ib] = 1.0
        V_cols.append(v)

    return np.stack(V_cols, axis=1)


# ── Internal key enumeration ─────────────────────────────────────────────────

def sigma_int_keys_for_ses(
    occ_ref: list,
    occ_pair,
    vir_pair,
    a_index: dict,
    b_index: dict,
    na: int,
    nb: int,
    *,
    alpha_strs,
    beta_strs,
) -> set:
    """
    Enumerate the 35 spin-orbital excitation keys internal to one SES.

    Active spin-orbitals: i1α, i1β, i2α, i2β (occ) and a1α, a1β, a2α, a2β (vir).
    Excitation ranks 1-4 with spin conservation (Δnα = 0, Δnβ = 0):
      8 singles + 18 doubles + 8 triples + 1 quadruple = 35.

    Key format: ("x", holes_tuple, particles_tuple)
    where indices are spin-orbital indices: 2*spatial + spin (α=0, β=1).
    """
    occ_so = []
    for p in sorted(occ_pair):
        occ_so.append(2 * int(p))
        occ_so.append(2 * int(p) + 1)

    vir_so = []
    for p in sorted(vir_pair):
        vir_so.append(2 * int(p))
        vir_so.append(2 * int(p) + 1)

    keys: set = set()
    for rank in range(1, 5):
        for holes in combinations(occ_so, rank):
            for particles in combinations(vir_so, rank):
                h_alpha = sum(1 for h in holes if h % 2 == 0)
                p_alpha = sum(1 for p in particles if p % 2 == 0)
                if h_alpha == p_alpha:
                    keys.add(("x", tuple(sorted(holes)), tuple(sorted(particles))))

    if len(keys) != 35:
        raise RuntimeError(f"CAS internal key count != 35 (got {len(keys)}).")

    return keys


# ── Sparse sigma matrix ──────────────────────────────────────────────────────

def build_sigma_sparse(
    nmo,
    alpha_strs,
    beta_strs,
    a_index,
    b_index,
    sigma_pool,
    sigma_keys,
):
    """
    Build sparse anti-Hermitian σ in the full FCI basis.

    For each key ("x", holes, particles) with amplitude θ, accumulates
    θ*(E - E†) where E is the corresponding excitation operator.
    """
    na, nb = len(alpha_strs), len(beta_strs)
    dim    = na * nb
    rows, cols, data = [], [], []

    def add_entry(r: int, c: int, v: float):
        if r == c or abs(v) < 1e-14:
            return
        rows.append(r); cols.append(c); data.append(v)

    for key in sigma_keys:
        amp = float(sigma_pool.get(key, 0.0))
        if abs(amp) < 1e-12:
            continue

        tag, holes, particles = key
        if tag != "x":
            continue

        alpha_holes = sorted([h // 2 for h in holes if h % 2 == 0])
        beta_holes  = sorted([h // 2 for h in holes if h % 2 == 1])
        alpha_parts = sorted([p // 2 for p in particles if p % 2 == 0])
        beta_parts  = sorted([p // 2 for p in particles if p % 2 == 1])

        for ia_idx in range(na):
            ab = int(alpha_strs[ia_idx])
            for ib_idx in range(nb):
                bb      = int(beta_strs[ib_idx])
                col_idx = ia_idx * nb + ib_idx

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

                if new_a in a_index and new_b in b_index:
                    row_idx = a_index[new_a] * nb + b_index[new_b]
                    phase   = phase_a * phase_b
                    add_entry(row_idx, col_idx,  amp * phase)
                    add_entry(col_idx, row_idx, -amp * phase)

    if len(data) == 0:
        return sparse.csr_matrix((dim, dim))
    return sparse.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()


# ── Global pool and precompute ───────────────────────────────────────────────

def build_global_sigma_pool_keys_from_ses(
    ses_list,
    occ_ref: list,
    a_index: dict,
    b_index: dict,
    na: int,
    nb: int,
    *,
    alpha_strs,
    beta_strs,
) -> set:
    """
    Global pool = union of all per-SES internal keys (35 per SES).
    """
    all_keys: set = set()
    for (_, occ_pair, vir_pair) in ses_list:
        all_keys |= sigma_int_keys_for_ses(
            occ_ref=occ_ref,
            occ_pair=occ_pair,
            vir_pair=vir_pair,
            a_index=a_index,
            b_index=b_index,
            na=na,
            nb=nb,
            alpha_strs=alpha_strs,
            beta_strs=beta_strs,
        )
    return all_keys


def init_sigma_pool_from_ses(
    ses_list,
    occ_ref: list,
    a_index: dict,
    b_index: dict,
    na: int,
    nb: int,
    *,
    alpha_strs,
    beta_strs,
) -> tuple:
    """
    Initialise the Global Pool of Amplitudes (GPA) to zero.

    Returns (sigma_pool dict, all_keys set).
    """
    all_keys: set = build_global_sigma_pool_keys_from_ses(
        ses_list=ses_list,
        occ_ref=occ_ref,
        a_index=a_index,
        b_index=b_index,
        na=na,
        nb=nb,
        alpha_strs=alpha_strs,
        beta_strs=beta_strs,
    )
    sigma_pool = {k: 0.0 for k in all_keys}
    return sigma_pool, all_keys


def precompute_ses_data(
    ses_list,
    occ_ref,
    all_keys,
    a_index,
    b_index,
    na,
    nb,
    *,
    alpha_strs,
    beta_strs,
):
    """
    Precompute per-SES data needed every cycle: basis vectors, key splits,
    and first-claimant ownership.

    Returns
    -------
    V_ses_all      : list of (na*nb, 36) basis matrices
    int_keys_all   : list of per-SES internal key sets (35 each)
    ext_keys_all   : list of per-SES external key sets (all_keys - int_keys)
    owned_keys_all : list of frozensets — keys uniquely owned by each SES
                     (first-claimant rule; owned keys get gradient updates)
    """
    V_ses_all      = []
    int_keys_all   = []
    ext_keys_all   = []
    owned_keys_all = []
    seen_keys: set = set()

    for (active, occ_pair, vir_pair) in ses_list:
        V = build_ses_basis_vectors(
            occ_ref=occ_ref,
            occ_pair=occ_pair,
            vir_pair=vir_pair,
            a_index=a_index,
            b_index=b_index,
            na=na,
            nb=nb,
        )
        int_keys = sigma_int_keys_for_ses(
            occ_ref=occ_ref,
            occ_pair=occ_pair,
            vir_pair=vir_pair,
            a_index=a_index,
            b_index=b_index,
            na=na,
            nb=nb,
            alpha_strs=alpha_strs,
            beta_strs=beta_strs,
        )
        ext_keys = all_keys - int_keys
        owned    = frozenset(k for k in int_keys if k not in seen_keys)
        seen_keys |= owned

        V_ses_all.append(V)
        int_keys_all.append(int_keys)
        ext_keys_all.append(ext_keys)
        owned_keys_all.append(owned)

    return V_ses_all, int_keys_all, ext_keys_all, owned_keys_all
