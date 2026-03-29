"""
optimizer.py
============
QFlow optimization loop and printing utilities.

Covers:
  - One QFlow cycle: snapshot energies, SGD update sweep, end-of-cycle report
  - Outer loop: setup, convergence check, Table I comparison
  - Printing helpers: SES parameter table, ownership map

Extension notes
---------------
ADMM consensus coupling:
  - Add an ADMM penalty term to the gradient in run_qflow_cycle before
    the SGD step: grad_k += rho * (theta_k - z_k + u_k) for each owned key,
    then update z and u after the sweep.
  - The z and u vectors live in optimizer.py as outer-loop state.

Alternative optimizers (Adam, L-BFGS):
  - Replace the SGD line `sigma_pool[k] -= lr * grad` with a call to
    a stateful optimizer object. Add the optimizer state to the cycle
    function signature or close over it in run_qflow.

Convergence criteria:
  - The current criterion is: spread < conv_spread AND |ΔE| < conv_e
    AND max_grad < conv_g.
  - To switch to spread-only (paper Equivalence Theorem), remove the
    ΔE and max_grad checks in run_qflow.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .heff import (
    build_Heff,
    _psi_int_cas,
    _energy_expectation,
    _gradients_commutator,
)
from .ses import SigmaKey


# ── Key ordering and ownership ───────────────────────────────────────────────

def _sorted_sigma_keys(keys: set) -> list:
    """Deterministic ordering for sigma keys — used for stable printouts."""
    def key_sort(k):
        tag, holes, particles = k
        return (tag, len(holes), len(particles), tuple(holes), tuple(particles))
    return sorted(list(keys), key=key_sort)


def _build_ownership(int_keys_all: List[set]) -> Dict[tuple, int]:
    """First SES that contains a key becomes its owner (Eq. 18)."""
    ownership: Dict[tuple, int] = {}
    for i, int_keys in enumerate(int_keys_all):
        for k in int_keys:
            if k not in ownership:
                ownership[k] = i
    return ownership


# ── Printing helpers ─────────────────────────────────────────────────────────

def _ses_label(ses_list, i: int) -> str:
    """Short string label for SES i."""
    _, occ_pair, vir_pair = ses_list[i]
    return f"SES-{i:02d} occ{occ_pair} vir{vir_pair}"


def print_ses_params(ses_info: List[dict], ses_list, cycle: int, max_rows: int = 8):
    """Print a parameter snapshot table for all SES at a given cycle."""
    print(f"\n  [ Parameter snapshot — cycle {cycle} ]")
    for info in ses_info:
        i     = info["ses_idx"]
        label = _ses_label(ses_list, i)
        print(f"\n    {label}   E={info['E']:.8f} Ha")
        keys       = info["ordered_keys"]
        owned_mask = info["owned_mask"]
        tb         = info["theta_before"]
        gg         = info["grads"]
        ta         = info["theta_after"]
        print(f"    {'Key':<40}  {'θ_before':>10}  {'grad':>12}  {'θ_after':>10}  role")
        print(f"    {'-'*40}  {'-'*10}  {'-'*12}  {'-'*10}  ----")
        for idx, (k, b, g, a, owned) in enumerate(zip(keys, tb, gg, ta, owned_mask)):
            if idx >= max_rows:
                print(f"    ... ({len(keys)-max_rows} more keys)")
                break
            role = "θ^X " if owned else "θ^CP"
            flag = " ← ZERO GRAD on owned!" if (owned and abs(g) < 1e-12) else ""
            print(f"    {str(k):<40}  {b:>10.6f}  {g:>12.6e}  {a:>10.6f}  {role}{flag}")


def print_ownership(ses_list, ownership: Dict[tuple, int], limit: int = 40):
    """Print the first-claimant ownership map for the global pool."""
    print("\n[ Amplitude Ownership — first-claimant rule (Eq. 18) ]")
    items = sorted(ownership.items(), key=lambda kv: (kv[1], str(kv[0])))
    for idx, (k, owner) in enumerate(items):
        if idx >= limit:
            print(f"  ... ({len(items)-limit} more)")
            break
        print(f"  {str(k):<45}  ->  {_ses_label(ses_list, owner)}")


# ── One QFlow cycle ──────────────────────────────────────────────────────────

def run_qflow_cycle(
    cycle_idx: int,
    sigma_pool: Dict[tuple, float],
    *,
    V_ses_all: List[np.ndarray],
    int_keys_all: List[set],
    ext_keys_all: List[set],
    owned_keys_all: List[frozenset],
    h1_mo,
    eri_packed,
    nmo: int,
    nelec: int,
    na: int,
    nb: int,
    alpha_strs,
    beta_strs,
    a_index,
    b_index,
    E_nuc: float,
    lr: float,
) -> Tuple[List[float], float, Optional[float], Dict[tuple, int], List[dict], List[float]]:
    """
    Run one QFlow cycle.

    Cycle 0 is a snapshot-only pass (no parameter updates), mirroring the
    paper's convention that updates start from the second cycle.

    Returns
    -------
    E_end      : end-of-cycle energies for all SES
    spread_start : spread (max-min) at start of cycle
    max_grad   : max |gradient| over owned keys (None for cycle 0)
    ownership  : key -> owner SES index dict
    ses_info   : per-SES diagnostic dicts
    heff_gaps  : VQE vs exact-diag gap per SES at end of cycle
    """
    M         = len(V_ses_all)
    ownership = _build_ownership(int_keys_all)

    shared = dict(
        h1_mo=h1_mo, eri_packed=eri_packed, nmo=nmo, nelec=nelec,
        na=na, nb=nb, alpha_strs=alpha_strs, beta_strs=beta_strs,
        a_index=a_index, b_index=b_index,
    )

    E_snap: List[float] = []
    for i in range(M):
        H_eff = build_Heff(V_ses_all[i], sigma_pool=sigma_pool,
                           sigma_ext_keys=ext_keys_all[i], **shared)
        c0    = _psi_int_cas(V_ses_all[i], sigma_pool, int_keys_all[i],
                             alpha_strs=alpha_strs, beta_strs=beta_strs, nb=nb)
        E_snap.append(_energy_expectation(H_eff, c0, E_nuc))

    spread_start = float(max(E_snap) - min(E_snap))

    if cycle_idx == 0:
        ses_info = []
        for i in range(M):
            ordered = _sorted_sigma_keys(int_keys_all[i])
            ses_info.append({
                "ses_idx":      i,
                "ordered_keys": ordered,
                "theta_before": [float(sigma_pool.get(k, 0.0)) for k in ordered],
                "grads":        [0.0] * len(ordered),
                "theta_after":  [float(sigma_pool.get(k, 0.0)) for k in ordered],
                "owned_mask":   [k in owned_keys_all[i] for k in ordered],
                "E":            E_snap[i],
            })
        return E_snap, spread_start, None, ownership, ses_info, [0.0] * M

    max_grad = 0.0
    ses_info = []

    for i in range(M):
        ordered      = _sorted_sigma_keys(int_keys_all[i])
        theta_before = [float(sigma_pool.get(k, 0.0)) for k in ordered]

        H_eff = build_Heff(V_ses_all[i], sigma_pool=sigma_pool,
                           sigma_ext_keys=ext_keys_all[i], **shared)
        grads = _gradients_commutator(V_ses_all[i], H_eff, sigma_pool,
                                      int_keys_all[i], alpha_strs=alpha_strs,
                                      beta_strs=beta_strs, nb=nb)

        owned = owned_keys_all[i]
        for k in owned:
            g = abs(float(grads.get(k, 0.0)))
            if g > max_grad:
                max_grad = g

        for k in owned:
            sigma_pool[k] = float(sigma_pool.get(k, 0.0)) - lr * float(grads.get(k, 0.0))

        theta_after = [float(sigma_pool.get(k, 0.0)) for k in ordered]
        grad_list   = [float(grads.get(k, 0.0)) for k in ordered]
        owned_mask  = [k in owned for k in ordered]

        c   = _psi_int_cas(V_ses_all[i], sigma_pool, int_keys_all[i],
                           alpha_strs=alpha_strs, beta_strs=beta_strs, nb=nb)
        E_i = _energy_expectation(H_eff, c, E_nuc)

        ses_info.append({
            "ses_idx":      i,
            "ordered_keys": ordered,
            "theta_before": theta_before,
            "grads":        grad_list,
            "theta_after":  theta_after,
            "owned_mask":   owned_mask,
            "E":            E_i,
        })

    E_end: List[float]    = []
    heff_gaps: List[float] = []
    for i in range(M):
        H_eff_end = build_Heff(V_ses_all[i], sigma_pool=sigma_pool,
                               sigma_ext_keys=ext_keys_all[i], **shared)
        c_end     = _psi_int_cas(V_ses_all[i], sigma_pool, int_keys_all[i],
                                 alpha_strs=alpha_strs, beta_strs=beta_strs, nb=nb)
        E_vqe      = _energy_expectation(H_eff_end, c_end, E_nuc)
        E_exact    = float(np.linalg.eigvalsh(H_eff_end)[0]) + float(E_nuc)
        E_end.append(E_vqe)
        heff_gaps.append(E_vqe - E_exact)

    return E_end, spread_start, float(max_grad), ownership, ses_info, heff_gaps


# ── Outer loop ───────────────────────────────────────────────────────────────

def run_qflow(
    nH: int,
    R_bohr: float,
    *,
    lr: float = 0.03,
    max_cycles: int = 200,
    conv_spread: float = 2e-3,
    conv_e: float = 1e-6,
    conv_g: float = 1e-6,
    debug_cycles: int = 2,
) -> Tuple[List[List[float]], List[float], List[Optional[float]]]:
    """
    Full QFlow(4e,4o) outer loop for a linear H_n chain.

    Parameters
    ----------
    nH          : number of hydrogen atoms
    R_bohr      : H-H distance in Bohr
    lr          : SGD learning rate
    max_cycles  : hard cycle limit
    conv_spread : convergence threshold on SES energy spread (Ha)
    conv_e      : convergence threshold on |ΔE_primary| (Ha)
    conv_g      : convergence threshold on max owned gradient
    debug_cycles: print full parameter tables for cycles 0..debug_cycles

    Returns
    -------
    E_all_history : list of per-cycle SES energy lists
    spread_list   : list of per-cycle spread values
    max_grad_list : list of per-cycle max gradient values
    """
    from .molecule import make_h_chain, run_rhf_and_integrals, build_fci_string_basis
    from .ses import enumerate_ses, init_sigma_pool_from_ses, precompute_ses_data

    print(f"\n{'='*78}")
    print(f"  QFlow(4e,4o)  —  H{nH}  R = {R_bohr} bohr")
    print(f"  lr={lr}  max_cycles={max_cycles}  conv_spread={conv_spread*1e3:.2f} mHa")
    print(f"{'='*78}\n")

    print("[ Setup ] Building molecule and running RHF...")
    mol = make_h_chain(nH, R_bohr)
    mf, eps, occ, h1_mo, eri_packed, E_nuc = run_rhf_and_integrals(mol)
    nmo   = int(h1_mo.shape[0])
    nelec = int(mol.nelectron)
    print(f"          nmo={nmo}  nelec={nelec}  E_nuc={E_nuc:.6f}  E_RHF={mf.e_tot:.6f}")

    print("[ Setup ] Building FCI bitstring basis...")
    _, _, alpha_strs, beta_strs, a_index, b_index, na, nb = build_fci_string_basis(nmo, nelec)
    print(f"          FCI dim = {na}x{nb} = {na*nb}")

    print("[ Setup ] Enumerating SES...")
    occ_orbs, vir_orbs, ses_list = enumerate_ses(eps, occ)
    M = len(ses_list)
    print(f"          M={M} SES   Primary: occ={ses_list[0][1]} vir={ses_list[0][2]}")

    occ_ref = [p for p in range(nmo) if occ[p] > 0.5]

    print("[ Setup ] Initialising global pool...")
    sigma_pool, all_keys = init_sigma_pool_from_ses(
        ses_list, occ_ref=occ_ref, a_index=a_index, b_index=b_index,
        na=na, nb=nb, alpha_strs=alpha_strs, beta_strs=beta_strs,
    )
    print(f"          Pool size: {len(all_keys)} keys")

    print("[ Setup ] Precomputing SES data...")
    V_ses_all, int_keys_all, ext_keys_all, owned_keys_all = precompute_ses_data(
        ses_list, occ_ref, all_keys, a_index, b_index, na, nb,
        alpha_strs=alpha_strs, beta_strs=beta_strs,
    )
    print("          Done.\n")

    cycle_kwargs = dict(
        V_ses_all=V_ses_all, int_keys_all=int_keys_all,
        ext_keys_all=ext_keys_all, owned_keys_all=owned_keys_all,
        h1_mo=h1_mo, eri_packed=eri_packed, nmo=nmo, nelec=nelec,
        na=na, nb=nb, alpha_strs=alpha_strs, beta_strs=beta_strs,
        a_index=a_index, b_index=b_index, E_nuc=E_nuc, lr=lr,
    )

    E_all_history: List[List[float]]     = []
    spread_list:   List[float]           = []
    max_grad_list: List[Optional[float]] = []
    ownership_printed = False
    spread_now = float("inf")

    for cycle in range(max_cycles):
        E_all, spread_start, max_grad, ownership, ses_info, heff_gaps = run_qflow_cycle(
            cycle, sigma_pool, **cycle_kwargs
        )

        if not ownership_printed:
            print_ownership(ses_list, ownership, limit=40)
            ownership_printed = True

        E_all_history.append(list(E_all))
        spread_list.append(float(spread_start))
        max_grad_list.append(max_grad)

        E_primary  = float(E_all[0])
        spread_now = float(max(E_all) - min(E_all))

        if cycle <= debug_cycles:
            print_ses_params(ses_info, ses_list, cycle, max_rows=8)

        if cycle == 0:
            tag = "← snapshot (no updates)"
        else:
            dE  = (E_primary - E_all_history[-2][0]) * 1e3
            tag = (f"spread={spread_start*1e3:.3f} mHa  "
                   f"max_grad={max_grad:.2e}  ΔE={dE:+.3f} mHa")

        print(f"\n{'-'*78}")
        print(f"  CYCLE {cycle:>4}   {tag}")
        print(f"{'-'*78}")
        print(f"  {'SES':<26}  {'E (Ha)':>14}  {'ΔE_from_prim (mHa)':>20}  {'|θ_int|':>10}")
        print(f"  {'-'*26}  {'-'*14}  {'-'*20}  {'-'*10}")

        for i in range(M):
            label   = _ses_label(ses_list, i)
            e_i     = float(E_all[i])
            delta_p = (e_i - E_primary) * 1e3
            theta   = np.array([float(sigma_pool.get(k, 0.0)) for k in int_keys_all[i]])
            marker  = " ◄ PRIMARY" if i == 0 else ""
            print(f"  {label:<26}  {e_i:>14.7f}  {delta_p:>+20.4f}  "
                  f"{float(np.linalg.norm(theta)):>10.4f}{marker}")

        print(f"\n  E_primary = {E_primary:.7f} Ha")
        print(f"  Spread    = {spread_now*1e3:.4f} mHa")

        if cycle >= 1:
            prev_primary = float(E_all_history[-2][0])
            if (spread_now < conv_spread
                    and abs(E_primary - prev_primary) < conv_e
                    and (max_grad is not None and max_grad < conv_g)):
                print(f"\n  ✓ Converged at cycle {cycle}  spread={spread_now*1e3:.4f} mHa")
                break
    else:
        print(f"\n  ⚠ Reached max_cycles={max_cycles}. "
              f"final spread={spread_now*1e3:.3f} mHa")

    max_gap  = max(heff_gaps) if heff_gaps else 0.0
    mean_gap = sum(heff_gaps) / len(heff_gaps) if heff_gaps else 0.0
    print(f"\n  VQE vs H_eff exact: max_gap={max_gap*1e3:.4f} mHa  "
          f"mean_gap={mean_gap*1e3:.4f} mHa")
    print(f"  Primary SES gap   = {heff_gaps[0]*1e3:.4f} mHa")

    return E_all_history, spread_list, max_grad_list
