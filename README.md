# TrimCI-Flow: Classical QFlow with TrimCI Sub-Solver

## Motivation

QFlow (Kowalski & Bauman, arXiv:2305.05168v2) fragments a molecular electronic
structure problem into overlapping active spaces (SES) and solves them
self-consistently through a shared parameter pool. The original implementation
uses a VQE-style solver for each fragment's effective Hamiltonian.

Dr. Otten's challenge: every QFlow benchmark to date uses active spaces small
enough to solve exactly on a classical computer. The quantum advantage claim is
unsubstantiated until the fragment sizes reach the regime where classical exact
diagonalization fails. His group's TrimCI solver (Zhang & Otten,
arXiv:2511.14734) can handle active spaces of 36+ orbitals — far beyond exact
diag — making it the natural classical sub-solver to stress-test whether QFlow's
fragmentation actually helps.

**The question:** Does QFlow's fragmentation, with TrimCI as the sub-solver,
converge faster (fewer determinants, fewer iterations) than running TrimCI
directly on the full problem?

**Target system:** [Fe4S4] active space — 36 orbitals, 54 electrons (27α+27β).
Dr. Otten's group has near-exact TrimCI reference energies (~10K determinants,
E ≈ −327.1920 Ha) and DMRG results for validation.

---

## The Fe4S4 system

**Source:** `fcidump_cycle_6` from TrimCI orbital optimization (cycle 6).

| Property | Value |
|---|---|
| Orbitals | 36 |
| Electrons | 54 (27α + 27β) |
| Virtual orbitals per spin | 9 |
| Nuclear repulsion | 0.0 (absorbed into effective Hamiltonian) |
| TrimCI reference energy | −327.1920 Ha |
| TrimCI determinants | 10,095 (core set: 9,177) |
| Leading CI coefficient | −0.763 (strong multireference character) |
| Orbital energy range | −13.58 to −8.86 Ha (no clear HOMO-LUMO gap) |

Key features that affect fragmentation design:

1. **Dense occupation:** 27 out of 36 orbitals occupied per spin. Only 9 virtual.
   Standard QFlow(4e,4o) enumeration gives C(27,2) × C(9,2) = 12,636 SES —
   combinatorial explosion from the occupied side.

2. **No clean orbital energy gap:** All orbital energies are between −8.9 and
   −13.6 Ha. There is no HOMO-LUMO gap to define a clear occupied/virtual
   boundary. The "occupied" and "virtual" labels come from the reference
   determinant, which only has coefficient −0.763 (meaning the system is
   far from single-reference).

3. **Strong correlation:** The leading coefficient of −0.763 means the
   reference determinant accounts for only ~58% of the wavefunction weight.
   Many determinants contribute significantly. This is exactly the regime
   where fragmentation could help — or could fail badly.

---

## The scaling wall: why current QFlow cannot work for Fe4S4

The current QFlow implementation builds the dressed effective Hamiltonian via:

```
H_eff = V^T  e^{-σ_ext}  H  e^{+σ_ext}  V
```

where σ_ext is a sparse matrix and the exponentials act on vectors in the
**full FCI space**. This is the coupling mechanism — σ_ext contains other
fragments' amplitudes, so when they update, H_eff changes.

The problem: full FCI dimension for Fe4S4 is C(36,27)² ≈ 7 × 10^13
determinants. We cannot store a single vector in this space, let alone
evaluate matrix exponentials on it.

This means the current `build_Heff` implementation (which uses
`scipy.sparse.linalg.expm_multiply` on full-FCI-length vectors) is
computationally impossible for Fe4S4, regardless of which sub-solver
we use. The bottleneck is H_eff construction, not the sub-solver.

### Three paths to tractable coupling

**Path A — Integral-level BCH dressing (preserves QFlow amplitude pool)**

Evaluate the similarity transform as a Baker-Campbell-Hausdorff series
on the integral tensors:

```
H_dressed = H + [H, σ_ext] + (1/2)[[H, σ_ext], σ_ext] + ...
```

where H = (h1, eri) and σ_ext = (σ1, σ2) are all one-body and two-body
tensors in the 36-orbital basis. The commutators at each order produce
new (h1', eri') tensors of the same dimensions. Tensor sizes: (36,36)
for one-body, (36,36,36,36) ≈ 1.7M entries for two-body. Trivially
manageable on a laptop.

After computing dressed integrals, slice to each fragment's orbital
subset and feed to TrimCI. The amplitude pool stores σ_ext as tensor
elements, and the coupling is through integral modification.

This is the same commutator algebra as DUCC Phase 3B, applied for
intra-active-space fragmentation rather than basis-set reduction.
Requires the normal-ordered commutator engine (comm_11, comm_12,
comm_21, comm_22) or automated tools (OpenFermion, Qcombo, WICK&D).

**Path B — Mean-field embedding (DMET-style, simpler)**

No σ_ext. Each fragment's integrals are dressed with a mean-field
correction from other fragments' one-particle reduced density matrices
(1-RDMs):

```
h1_eff[p,q] = h1[p,q] + Σ_{r ∈ external} (2*eri[p,q,r,r] - eri[p,r,r,q]) * γ_r
```

where γ_r is the occupation of orbital r from neighboring fragment
solutions. Iterate until fragment 1-RDMs are self-consistent.

Advantages: no commutator algebra, well-established theory, simple
to implement. Disadvantage: loses the UCC amplitude pool — the coupling
is through density matrices, not through excitation amplitudes. This
is closer to DMET than to QFlow.

**Path C — No coupling (fragmentation baseline)**

Simplest possible approach. Slice integrals per fragment, solve each
with TrimCI independently, sum fragment energies with double-counting
corrections. No iteration, no self-consistency.

This doesn't give accurate total energies, but it answers the core
question: can you cover the important determinant space of Fe4S4 by
solving several smaller problems, using fewer total determinants than
brute-force TrimCI on the full 36-orbital space?

### Recommended implementation order

```
Path C (baseline)  →  Path B (coupling)  →  Path A (full QFlow)
   days                  ~1 week               ~2 weeks
```

Start with Path C — it's implementable immediately and gives us a
baseline. Path B adds coupling without commutator algebra. Path A
is the "true QFlow" but requires either fixing the Phase 3B commutator
engine or using automated tools.

---

## Fragment design for Fe4S4

### Why QFlow(4e,4o) enumeration doesn't work here

The paper's approach: enumerate ALL C(n_occ, 2) × C(n_vir, 2) SES.
For H8 STO-3G (4 occ, 4 vir): 36 SES. Manageable.
For Fe4S4 (27 occ, 9 vir): 12,636 SES. The occupied-side combinatorics
explode. This isn't inherently fatal (each (4e,4o) fragment is trivially
small for TrimCI), but 12,636 fragments means 12,636 TrimCI calls per
cycle, and the global pool has hundreds of thousands of amplitude keys.

### Better: physically motivated fragments

Fe4S4 has 4 iron and 4 sulfur atoms. The 36 active orbitals are
primarily Fe 3d and S 3p in character. Natural fragment groupings:

1. **By atom pair:** Fe-S bond-centered fragments, each containing the
   d-orbitals of one Fe and p-orbitals of its bonded S atoms.
   ~8-12 orbitals per fragment, ~4-8 fragments. Overlap at shared S atoms.

2. **By orbital correlation:** Use TrimCI's `compute_orbital_mutual_information`
   on the existing 10K-determinant wavefunction to identify strongly
   correlated orbital clusters. Group orbitals with high mutual information
   into fragments.

3. **By energy windows:** Group orbitals by orbital energy ranges.
   The energies cluster into bands: ~(−13.6 to −12.5), ~(−12.5 to −11.0),
   ~(−11.0 to −8.9). Each band could define a fragment.

4. **Sliding windows:** Order orbitals (by energy, mutual information, or
   atom), then take overlapping windows of size W with stride S.
   E.g., W=15, S=10 on 36 orbitals → 3 fragments:
   orbitals [0-14], [10-24], [20-35]. Each pair overlaps by 5 orbitals.

### Recommendation for starting

Use option 4 (sliding windows) for initial implementation — it's
simple, systematic, and the overlap provides natural coupling regions.
Tune W and S as hyperparameters. Then compare against option 2
(correlation-guided) once we have baseline results.

For the orbital ordering, sort by h1 diagonal (orbital energy). This
puts similar-energy orbitals together, which tends to group correlated
orbitals since correlation is strongest between near-degenerate orbitals.

### Fragment size vs TrimCI cost tradeoff

| Fragment size (orbs) | Electrons | CAS dim | TrimCI feasible? | Fragments needed |
|---|---|---|---|---|
| 10 | ~15 | C(10,8)²=2025 | Exact diag works | ~5-6 |
| 15 | ~22 | C(15,11)²≈10^6 | TrimCI needed | ~3-4 |
| 20 | ~30 | C(20,15)²≈10^9 | TrimCI needed | ~2-3 |
| 25 | ~38 | C(25,19)²≈10^11 | TrimCI works | ~2 |
| 36 (full) | 54 | C(36,27)²≈10^13 | TrimCI works (10K dets) | 1 |

Sweet spot is probably 12-18 orbitals per fragment: big enough to
capture meaningful correlation, small enough that TrimCI converges
quickly with few determinants. This is what Dr. Otten was getting at.

---

## Implementation plan

### Phase 0: Standalone validation

**Goal:** Verify both codebases independently on a shared test case.

**Tasks:**
1. Run QFlow on H6 R=2.0 STO-3G → confirm convergence to known energies
2. Run TrimCI standalone on H6 STO-3G → confirm TrimCI energy matches FCI
3. Run TrimCI on Fe4S4 FCIDUMP → confirm it reproduces −327.1920 Ha
4. Compute orbital mutual information on Fe4S4 wavefunction → understand
   which orbitals are strongly correlated → inform fragment design

**Integral format bridge:**
- QFlow: `eri_packed` (1D, compact=True from ao2mo)
- TrimCI: `eri` as 4D array or flattened 1D
- FCIDUMP: `read_fcidump` returns (h1, eri_4D, n_elec, n_orb, E_nuc, ...)
- Conversion: `ao2mo.restore(1, eri_packed, nmo)` → 4D
- Both use chemist notation (pq|rs), no index permutation needed

### Phase 1: Fragmentation engine

**Goal:** Fragment the Fe4S4 orbital space into overlapping subsets.

**New file:** `fragment.py`

```python
def fragment_by_sliding_window(n_orb, orbital_order, window_size, stride):
    """
    Create overlapping orbital fragments using a sliding window.

    Parameters
    ----------
    n_orb        : total number of orbitals (36 for Fe4S4)
    orbital_order: permutation array — orbital indices sorted by desired
                   criterion (e.g., energy, atom, mutual information)
    window_size  : number of orbitals per fragment (e.g., 15)
    stride       : step between windows (e.g., 10)

    Returns
    -------
    fragments    : list of lists, each containing orbital indices for one fragment
    """

def fragment_by_mutual_information(n_orb, mi_matrix, n_fragments, min_overlap):
    """
    Create fragments by clustering orbitals with high mutual information.
    Uses spectral clustering or agglomerative clustering on the MI matrix.
    """

def extract_fragment_integrals(h1_full, eri_full, fragment_orbs):
    """
    Slice h1 and eri to a fragment's orbital subset.

    h1_frag[i,j] = h1_full[frag[i], frag[j]]
    eri_frag[i,j,k,l] = eri_full[frag[i], frag[j], frag[k], frag[l]]

    Also returns the electron count for the fragment (from the reference
    determinant's occupation of those orbitals).
    """

def fragment_electron_count(ref_alpha_bits, ref_beta_bits, fragment_orbs):
    """
    Count how many electrons from the reference determinant fall in this fragment.
    """
```

### Phase 2: TrimCI adapter

**Goal:** Clean interface for calling TrimCI on fragment integrals.

**New file:** `trimci_adapter.py`

```python
def solve_fragment_trimci(h1_frag, eri_frag, n_alpha_frag, n_beta_frag,
                          n_orb_frag, config=None):
    """
    Run TrimCI on a single fragment.

    Parameters
    ----------
    h1_frag, eri_frag : fragment integrals (from extract_fragment_integrals)
    n_alpha_frag, n_beta_frag : electrons in this fragment
    n_orb_frag        : orbitals in this fragment
    config            : TrimCI configuration dict (max_final_dets, threshold, etc.)

    Returns
    -------
    energy   : float — fragment ground-state energy
    dets     : list of TrimCI Determinant objects
    coeffs   : list of float — CI coefficients
    n_dets   : int — number of determinants used
    """

def solve_fragment_exact(h1_frag, eri_frag, n_alpha_frag, n_beta_frag,
                         n_orb_frag):
    """
    Exact diagonalization fallback for small fragments (n_orb <= ~14).
    Uses PySCF FCI solver.
    """
```

### Phase 3: Uncoupled fragmentation baseline (Path C)

**Goal:** Fragment Fe4S4 into overlapping subsets, solve each independently
with TrimCI, analyze total determinant usage vs brute-force.

**New file:** `trimci_flow.py`

```python
def run_fragmented_trimci(fcidump_path, window_size, stride,
                          trimci_config=None):
    """
    Fragment-and-solve without inter-fragment coupling.

    1. Read FCIDUMP
    2. Create fragments via sliding window
    3. For each fragment: extract integrals, run TrimCI
    4. Report: per-fragment energy, determinant count, total dets
    5. Compare total dets against brute-force (10,095 for Fe4S4)
    """
```

**Key metrics to report:**
- Total determinants across all fragments vs brute-force (10,095)
- Per-fragment energy and determinant count
- Which fragments need the most determinants (identifies the hardest subproblems)

### Phase 4: Mean-field coupling (Path B)

**Goal:** Add inter-fragment coupling via 1-RDM embedding.

**Additions to `trimci_flow.py`:**

```python
def compute_fragment_rdm1(dets, coeffs, fragment_orbs, n_orb_frag):
    """
    Compute 1-RDM from TrimCI wavefunction for a fragment.
    γ[p,q] = Σ_{I,J} c_I c_J <I| a†_p a_q |J>
    """

def dress_integrals_meanfield(h1_frag, eri_full, fragment_orbs,
                               external_rdm1):
    """
    Add mean-field correction from external orbitals.

    h1_eff[p,q] = h1_frag[p,q]
                  + Σ_{r ∈ external} γ_r * (2*eri[p,q,r,r] - eri[p,r,r,q])

    where γ_r is the diagonal of external_rdm1 (occupation numbers).
    """

def run_selfconsistent_fragments(fcidump_path, window_size, stride,
                                  max_iterations=20, convergence=1e-6,
                                  trimci_config=None):
    """
    Self-consistent fragment solve with mean-field coupling.

    1. Read FCIDUMP, create fragments
    2. Initial solve: each fragment with bare integrals
    3. Compute 1-RDMs for each fragment
    4. Dress each fragment's integrals with other fragments' RDMs
    5. Re-solve each fragment
    6. Check convergence: |ΔE| < threshold for all fragments
    7. Repeat 3-6 until converged
    """
```

### Phase 5: Full QFlow coupling (Path A) — future

**Goal:** Replace mean-field coupling with integral-level σ_ext dressing
via BCH commutators. This restores the full QFlow amplitude-pool structure.

Requires: normal-ordered commutator engine for 1-body + 2-body operators.
Options:
- Fix the comm_11/comm_12/comm_21/comm_22 from Phase 3B
- Use OpenFermion for symbolic commutator evaluation
- Use Qcombo or WICK&D for automated code generation

This phase is deferred until Phases 0-4 produce results that inform
whether the additional complexity of amplitude-level coupling is worth
the implementation cost over mean-field coupling.

---

## File map

```
TrimCI_Flow/
├── README.md              This file
├── fragment.py             Orbital fragmentation: sliding window, MI-based, integral slicing
├── trimci_adapter.py       TrimCI wrapper: takes fragment integrals, returns energy + wavefunction
├── trimci_flow.py          Main driver: fragmented solve, mean-field coupling, convergence loop
├── analysis.py             Post-processing: determinant counts, energy comparisons, MI analysis
├── Flow_TrimCI.ipynb       Main notebook: runs all phases, produces comparison tables
│
├── qflow/                  Original QFlow codebase (reference implementation)
│   ├── molecule.py         Molecule setup, RHF, MO integrals
│   ├── ses.py              SES enumeration, basis vectors, pool init
│   ├── heff.py             H_eff construction, UCC ansatz, gradients
│   ├── optimizer.py        QFlow cycle and outer loop
│   └── references.py       Classical reference methods
│
├── trimci/                 TrimCI codebase
│   └── (full TrimCI package)
│
└── data/
    └── Fe4S4_251230orbital_-327.1920_10kdets/
        ├── fcidump_cycle_6       36-orbital FCIDUMP (orbital-optimized)
        └── dets.npz              TrimCI reference wavefunction (10,095 dets)
```

---

## Key things to keep in mind

### 1. The reference determinant is NOT a good single-reference

The leading CI coefficient is −0.763 (58% weight). This means any method
that assumes a dominant reference (HF, CCSD, perturbation theory) will
struggle. TrimCI's strength is precisely that it doesn't need a good
reference — it builds the wavefunction from random determinants.

QFlow's SES enumeration, which is built on occupied/virtual partitioning
from a reference determinant, may need adaptation. The occupied/virtual
distinction is weak for this system.

### 2. The FCIDUMP has nuclear repulsion = 0

This is an effective Hamiltonian from orbital optimization. The energy
−327.1920 Ha is the total electronic energy in this orbital basis.
When comparing energies, do NOT add a nuclear repulsion term.

### 3. Orbital energies don't define a natural fragmentation

The h1 diagonal values (orbital energies) range from −8.9 to −13.6 Ha
with no clear gaps. Energy-based orbital ordering for sliding windows
will group similar-energy orbitals, which is a reasonable starting point
but not chemically motivated.

Orbital mutual information (from the existing TrimCI wavefunction) is a
better guide — it directly measures which orbital pairs are correlated.

### 4. Fragment electron count needs care

With 27 occupied orbitals per spin in 36 total, a 15-orbital fragment
will typically contain ~11 occupied and ~4 virtual (per spin). The
electron count per fragment depends on which orbitals are included
and which reference determinant we use.

For the FCIDUMP reference (from dets.npz row 0): alpha occupies
orbitals {0-20, 22-23, 30, 33-35}, beta occupies {0-1, 4-5, 10,
12, 14-27, 29-35}. These are NOT the same sets — the alpha and beta
occupations differ significantly, reflecting the open-shell character
of the Fe4S4 cluster.

### 5. Total energy from fragments is not straightforward

Summing fragment energies double-counts electron-electron interactions
for orbitals that appear in multiple fragments. For Path C (no coupling),
we cannot get a meaningful total energy — we can only compare
per-fragment determinant usage.

For Path B (mean-field coupling), the converged energy is well-defined
but not variational in general.

### 6. TrimCI configuration matters

TrimCI has many tunable parameters (max_final_dets, threshold,
core_set_ratio, etc.). For the full 36-orbital problem, Dr. Otten's
group used specific settings to get 10K determinants. For fragments,
the optimal settings will be different. Start with TrimCI's auto mode
and tune from there.

### 7. The real comparison metric is determinant count

Dr. Otten explicitly said timing comparisons aren't meaningful on
different hardware. The metric that matters: how many total determinants
(summed across all fragment TrimCI solves) does the fragmented approach
need to reach the same energy accuracy as brute-force TrimCI on the
full space (10,095 determinants for −327.1920 Ha)?

If fragmented TrimCI uses fewer total determinants for comparable
accuracy, QFlow-style fragmentation adds value. If not, fragmentation
is just overhead.

---

## Dependencies

| Package | Purpose |
|---|---|
| PySCF ≥ 2.3 | RHF, integrals, FCI (exact diag for small fragments) |
| NumPy ≥ 1.24 | Linear algebra |
| SciPy ≥ 1.10 | Matrix functions (expm, logm) |
| TrimCI | Sub-solver for fragment Hamiltonians |
| Matplotlib ≥ 3.7 | Convergence plots, MI heatmaps |
| scikit-learn | Spectral clustering for MI-based fragmentation (optional) |

---

## References

1. Kowalski & Bauman, arXiv:2305.05168v2 (2023) — QFlow algorithm
2. Kowalski, arXiv:2410.11992 (2024) — sub-QFlow with DUCC
3. Zhang & Otten, arXiv:2511.14734 (2025) — TrimCI algorithm
4. Bauman et al., JCP 151, 014107 (2019) — DUCC implementation
5. Wouters et al., JCTC 12, 2706 (2016) — DMET with CI solvers (methodological parallel)