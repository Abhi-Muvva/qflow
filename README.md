# QFlow(4e,4o) — Replication & Extension

Replication of **Kowalski & Bauman, arXiv:2305.05168v2** for linear
hydrogen chains (H6, H8) in STO-3G, targeting Table I ground-state energies.

---

## File map

```
qflow/
├── __init__.py      Public API re-exports. Import everything from here.
├── molecule.py      Molecule construction, RHF, MO integrals, FCI basis, H sigma-vector.
├── ses.py           SES enumeration, CAS basis vectors, sigma keys, pool, precompute.
├── heff.py          H_eff construction, tau_cas, psi_int, gradients.
├── optimizer.py     One QFlow cycle, outer loop, convergence, printing.
└── references.py    All classical reference methods (HF → FCI, CC via CCpy).

Flow.ipynb           Main notebook. Runs all 4 benchmark cases and prints Table I.
README.md            This file.
```

---

## Data flow

```
molecule.py         make_h_chain → run_rhf_and_integrals → build_fci_string_basis
                                                                    |
ses.py              enumerate_ses → build_ses_basis_vectors         |
                    sigma_int_keys_for_ses → init_sigma_pool        |
                    precompute_ses_data                              |
                           |                                        |
heff.py             build_Heff (uses sigma_pool + apply_H_fci) ←───┘
                    _psi_int_cas → _gradients_commutator
                           |
optimizer.py        run_qflow_cycle → run_qflow (outer loop)
                           |
Flow.ipynb          calls run_qflow per case, then references.py for Table I
```

---

## File-by-file detail

### `molecule.py`
Infrastructure only — no QFlow logic.

| Function | Purpose |
|---|---|
| `make_h_chain(nH, R_bohr)` | Build PySCF Mole for linear H chain |
| `run_rhf_and_integrals(mol)` | RHF + MO integrals (h1_mo, eri_packed, E_nuc) |
| `build_fci_string_basis(nmo, nelec)` | Enumerate FCI bitstrings, return index dicts |
| `apply_H_fci(vec, ...)` | H sigma-vector via PySCF direct_spin1 |

**To extend:** add new molecule builders here (rings, 3D, non-H atoms).
Everything downstream adapts automatically as long as the same RHF
interface is returned.

---

### `ses.py`
The active-space heart of QFlow. All SES-specific logic lives here.

| Function | Purpose |
|---|---|
| `enumerate_ses(eps, occ)` | List all (2occ, 2vir) SES pairs, sorted by energy |
| `build_ses_basis_vectors(...)` | CAS(4e,4o) basis: (na\*nb, 36) matrix |
| `sigma_int_keys_for_ses(...)` | 35 spin-orbital excitation keys per SES |
| `build_sigma_sparse(...)` | Sparse anti-Hermitian σ in full FCI basis |
| `init_sigma_pool_from_ses(...)` | Initialise GPA to zero |
| `precompute_ses_data(...)` | Per-SES V, int_keys, ext_keys, owned_keys |
| `_annihilate_orb`, `_create_orb`, `_popcount`, ... | Fermionic sign helpers |

**Extension — gradient-norm active-space pruning (arXiv:2410.11992):**
After the first cycle, score each SES by `max(|grad|)` over its owned keys.
Drop low-scoring SES from `ses_list`, then call `precompute_ses_data` again.
Only this file and the outer loop in `optimizer.py` change.

**Extension — larger active spaces (QFlow(6e,6o)):**
Change `combinations(occ_orbs, 2)` in `enumerate_ses` and
`build_ses_basis_vectors` to `combinations(occ_orbs, 3)`, and update
the rank range in `sigma_int_keys_for_ses` accordingly.
The CAS dimension becomes C(6,3)^2 = 400 determinants.

---

### `heff.py`
Effective Hamiltonian construction and gradient computation.

| Function | Purpose |
|---|---|
| `build_Heff(V_ses, ..., sigma_pool, sigma_ext_keys)` | Eq. 15 (N=1): V^T e^{-σ_ext} H e^{σ_ext} V |
| `_extract_det_bits_cas(V_ses, ...)` | (alpha_bit, beta_bit) per CAS column |
| `_tau_cas(det_bits, key)` | Anti-Hermitian generator τ_k in CAS basis |
| `_psi_int_cas(V_ses, sigma_pool, ...)` | UCC state: exp(Σ θ_k τ_k) \|Φ_ref> |
| `_energy_expectation(H_eff, c, E_nuc)` | <c\|H_eff\|c> + E_nuc |
| `_gradients_commutator(...)` | Eq. 19: <Ψ_int\| [H_eff, τ_k] \|Ψ_int> |
| `compute_ses_energy_stringmb(H_eff, E_nuc)` | Exact-diag fallback (diagnostic) |

**Extension — Quantum Natural Gradient (QNG):**
Inside `_gradients_commutator`, compute the quantum Fisher information
matrix `F[k,l] = <[τ_k, τ_l]>`. Return it alongside raw gradients.
In `optimizer.py`, apply `F^{-1} @ g` before the SGD step.
This directly addresses the learning-rate sensitivity seen at R=3.0.

**Extension — alternative ansatz:**
Replace `_psi_int_cas` with a new state-preparation function (hardware-
efficient layers, Givens rotations, etc.). The gradient formula in
`_gradients_commutator` stays the same as long as the ansatz is
parameterised by the same τ_k generators.

---

### `optimizer.py`
The optimization loop. Touches all other modules but owns no physics.

| Function | Purpose |
|---|---|
| `run_qflow_cycle(cycle_idx, sigma_pool, ...)` | One full cycle: snapshot → SGD sweep → end energies |
| `run_qflow(nH, R_bohr, ...)` | Outer loop: setup, cycle iteration, convergence check |
| `print_ses_params(...)` | Per-SES parameter/gradient table (debug output) |
| `print_ownership(...)` | First-claimant ownership map |

**Extension — ADMM consensus coupling:**
In `run_qflow_cycle`, after computing gradients, add the ADMM penalty:
`grad_k += rho * (theta_k - z_k + u_k)` for each owned key.
After the full sweep, update the consensus variables z and u.
Store z, u as outer-loop state in `run_qflow`.

**Extension — Adam / L-BFGS:**
Replace the SGD line `sigma_pool[k] -= lr * grad` with a call to a
stateful optimizer object initialised once in `run_qflow` and passed
into each cycle call.

**Convergence criterion:**
Currently: spread < conv_spread AND |ΔE| < conv_e AND max_grad < conv_g.
To match the paper's Equivalence Theorem (spread-only), remove the
ΔE and max_grad checks in the convergence block of `run_qflow`.

---

### `references.py`
Classical reference energies. Fully independent of all QFlow files.

| Function | Purpose |
|---|---|
| `build_refstate(mol)` | Run RHF, return frozen Refstate dataclass |
| `compute_hf_energy(ref)` | HF total energy |
| `compute_casci_energy(ref, ncas, nelecas)` | CAS-ED (primary active space) |
| `compute_ccsd_energy_pyscf(ref)` | CCSD via PySCF |
| `compute_ccsdt_energy_ccpy(ref)` | CCSDT via CCpy |
| `compute_ccsdtq_energy_ccpy(ref)` | CCSDTQ via CCpy |
| `compute_fci_energy(ref)` | Full-space ED |

**Extension — new methods:**
Add a new `compute_*` function following the same Refstate pattern.
Add the key to `method_order` in Flow.ipynb cell 5.

---

### `Flow.ipynb`
The only entry point. Six cells:

| Cell | Content |
|---|---|
| 0 | Imports + shared `qflow_results` dict |
| 1 | H6, R=2.0 — run QFlow, plot, store result |
| 2 | H6, R=3.0 — run QFlow, plot, store result |
| 3 | H8, R=2.0 — run QFlow, plot, store result |
| 4 | H8, R=3.0 — run QFlow, plot, store result |
| 5 | Compute all reference energies + print full Table I |

Cells 1-4 can be run independently in any order.
Cell 5 reads from `qflow_results` populated by cells 1-4.

---

## Quick-start

```python
# Run a single case
from qflow import run_qflow
history, spreads, grads = run_qflow(6, 2.0, lr=0.25, max_cycles=40)
print(f"Converged energy: {history[-1][0]:.8f} Ha")

# Reference energies only
from qflow import make_h_chain, build_refstate, compute_fci_energy
mol = make_h_chain(6, 2.0)
ref = build_refstate(mol)
print(f"FCI: {compute_fci_energy(ref):.8f} Ha")
```

---

## Dependencies

| Package | Version tested | Purpose |
|---|---|---|
| PySCF | ≥ 2.3 | RHF, integrals, FCI sigma-vector, CASCI, CCSD |
| NumPy | ≥ 1.24 | All linear algebra |
| SciPy | ≥ 1.10 | `expm_multiply` for H_eff, `expm` for psi_int |
| CCpy | any | CCSDT, CCSDTQ (optional — only references.py) |
| Matplotlib | ≥ 3.7 | Convergence plots in notebook |

CCpy is optional. If not installed, `compute_ccsdt_energy_ccpy` and
`compute_ccsdtq_energy_ccpy` will raise `ImportError` when called, but
all QFlow functionality remains available.
