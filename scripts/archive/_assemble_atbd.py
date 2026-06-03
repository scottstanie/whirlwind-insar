#!/usr/bin/env python3
"""One-off: rebuild ATBD-whirlwind.md §3–§9 from the code-verified audit drafts.

Reads the seven per-subsystem drafts produced by the ATBD-currency audit
(stored under the audit dir) and splices them into the existing ATBD, replacing
the stale algorithm sections (§3–§8) and rewriting §9 (implementation details +
a new status/benchmark subsection). §1, §2 (with a §2.4 fix), §10, and the
appendices (Appendix C fixed) are preserved. Idempotent-ish: operates on the
current ATBD by top-level "## " section boundaries.
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path("/Users/staniewi/repos/whirlwind-insar")
AUDIT = Path("/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/atbd_audit")
ATBD = ROOT / "ATBD-whirlwind.md"

def draft(name: str) -> str:
    return (AUDIT / f"{name}_draft.md").read_text().rstrip("\n")

# §7 = solver draft §7.1–§7.5 + ssp draft §7.6–§7.7 (ssp's are more detailed).
solver = draft("solver")
ssp = draft("ssp")
solver_head = solver.split("### 7.6", 1)[0].rstrip("\n")   # keep 7.1–7.5 (+ the "## 7." heading)
section7 = solver_head + "\n\n" + ssp.strip()

DRAFTS = {
    3: draft("api"),
    4: draft("residue"),
    5: draft("cost"),
    6: draft("network"),
    7: section7,
    8: draft("integrate"),
}

# --- §9 full rewrite (corrected structures + new 9.6 status/benchmark) ---
SECTION9 = r"""## 9. Implementation Details

### 9.1 Implementation Architecture

Whirlwind is implemented in **Rust**, with a small Python binding layer:

- **`crates/whirlwind-core`** (Rust): all algorithms — residue computation,
  cost build, MCF solver, integration, tiling/robustness, connected components,
  synthetic-ifg simulator. Parallelism via `rayon`.
- **`crates/whirlwind-cli`** (Rust): `whirlwind` binary (`simulate`, `unwrap`).
- **`crates/whirlwind-py`** (`pyo3`/`maturin`): Python bindings, importable as
  `whirlwind`. The top-level `unwrap` returns `(phase, conncomp)`.

### 9.2 Key Data Structures

#### 9.2.1 Rectangular Grid Graph (`grid.rs`)

```rust
pub struct RectangularGridGraph {
    pub m: usize,          // node rows  (= pixel rows + 1)
    pub n: usize,          // node cols  (= pixel cols + 1)
    pub n_v: usize,        // vertical pairs   = (m-1)*n
    pub n_h: usize,        // horizontal pairs = m*(n-1)
    pub num_forward: usize,// = 2*n_v + 2*n_h
}
// Forward arc IDs partitioned [DOWN(n_v), UP(n_v), RIGHT(n_h), LEFT(n_h)];
// reverse of forward arc a is a + num_forward (O(1) transpose).
// node_id(i,j) = i*n + j. Per-node outdegree <= 8 (SmallVec8).
```

#### 9.2.2 Network (`network.rs`)

```rust
pub struct Network<'a> {
    graph: &'a RectangularGridGraph,
    pub excess: Vec<i32>,        // b_i (supply/demand = residues)
    pub potential: Vec<i64>,     // pi_i (dual variables; i64 to avoid overflow)
    pub cost_fwd: Vec<i32>,      // forward arc cost (reverse = -fwd)
    pub is_saturated: BitVec,    // (fwd,rev) bit pair per arc; (true,true)=FORBIDDEN
    // multi-unit / convex extras:
    flow_count: Vec<i32>,        // signed flow (reuse + convex modes)
    reuse_mode: bool, convex_mode: bool,
    offsets: Vec<i32>, weights: Vec<i32>,  // convex parabola params
    // optional ground sub-layout for new_with_mask_and_ground
}
```

Three construction modes select the capacity/cost model (see §6.2): unit-capacity
(`new`/`new_with_mask`), flow-reuse (`new_reuse_with_mask`, the production
default), and convex (`new_convex_with_mask`). `from_topology` builds the
non-raster network for the sparse triangulated path; `warm_start` is
`#[doc(hidden)]` and unsafe to call (breaks Dial's reduced-cost invariant).

#### 9.2.3 Dijkstra Backends

Multi-source Dijkstra has three backends, selected once per process via
`WHIRLWIND_DIJKSTRA` (`OnceLock`-cached):

- **Dial serial** (default): bucket queue over bounded integer reduced costs;
  `O(V + E + max_reduced_cost)` per solve. Has a full-completion variant
  (`dial::run_full`) used by `run_full_dijkstra`.
- **Dial parallel** (`dial-par`/`parallel`, experimental): not actually faster
  than serial on measured workloads; owns its own buffers (not allocation-free).
- **Binary heap** (`heap`): `O((V+E) log V)`. **Convex mode forces the heap
  unconditionally** (Dial's bucket count would explode on ~1e6 marginal costs);
  there is *no* full-completion heap, so convex "full" dispatch silently uses the
  early-exit heap.

`ShortestPaths` (`dist:i64`, `pred_arc/pred_node/source:i32`, `popped:bool`) is
allocated once and reused across PD/SSP iterations via `reset()` + the `*_into`
variants; `popped`/`was_reached` distinguish a finalized distance from a merely
relaxed one.

### 9.3 Numerical Considerations

#### 9.3.1 Cost Scaling

Float LLR costs are converted to `i32` to drive Dial's bucket-queue Dijkstra.
The **production** Carballo path scales by `CARBALLO_COST_SCALE = 6.0` with a
50-nat LLR cap, so the maximum integer cost is `6 × 50 = 300`. The diagnostic
parity path (`compute_carballo_costs_parity`) scales by `100` to match Python
`ww-orig`. (A separate `COST_SCALE = 100.0` constant is used by the CRLB and
convex cost builders, **not** by the production Carballo path.)

#### 9.3.2 Masked Regions

Masks (`true` = valid) are handled differently per stage and per entry point:

- **Residue compute** (`compute_with_mask`): zeros any interior residue whose
  2×2 pixel loop touches a masked pixel, and skips boundary-edge deposits with a
  masked endpoint (§4.2). Without this, `0+0j` masked pixels generate a wall of
  spurious residues at every mask boundary.
- **Network construction**: *two* mechanisms (§6.3). Arc-forbidding
  (`forbid_masked_arcs`) pre-saturates both directions of masked-edge arcs —
  used by the CRLB, convex, conncomp, ground and tiled paths. The **default
  coherence solver `unwrap_reuse` and the parity `unwrap_linear` deliberately do
  NOT forbid** masked arcs; they pass `mask = None` to construction, rely on the
  cost stage to zero masked-arc costs, route freely, and NaN masked pixels after
  integration. Forbidding masked arcs *isolates* residues and drops NISAR
  matching from ~99 % to ~42 %, hence the cost-zeroing default.
- **Integration** (`integrate_with_mask`): independent BFS per valid component;
  masked pixels left as `NaN` (§8.2).

### 9.4 Performance Characteristics

- **Bottleneck**: the Dijkstra shortest-path computations in the MCF solve. On a
  single-tile whole-image solve of a NISAR-scale frame the exact MCF dominates
  end-to-end runtime. The SSP fallback is the sharpest cost if it uses the
  multi-source search; the full-Dijkstra single-tile path therefore uses
  single-source SSP (see §9.6).
- **Memory**: `O(pixels)`; a whole-image solve of a 4176×4257 NISAR frame peaks
  at ≈6.4 GB RSS (the ≈72 M-arc residual network). Tiling bounds peak memory to
  tile scale.

### 9.5 Limitations and Assumptions

1. **2D only**: this ATBD covers the 2D unwrap; the 3D/time-series pipeline is in
   `ATBD-3d.md`.
2. **Statistical model**: assumes the Carballo/Lee cost model fits the data, and
   an accurate effective number of looks.
3. **Filter size**: 7×7 smoothing (Carballo's original used 5×5).
4. **Tiled robustness layer is heuristic**: the default large-frame path
   (`unwrap_tiled_robust`) — seam reconciliation, coarse anchor + multi-scale
   cascade, sliver healing, gated multi-shift re-solve — is empirically tuned
   against benchmark scenes, **not proven optimal**, and can produce invalid
   (fast-but-wrong) results on fragmented NISAR scenes. It carries environment
   escape hatches (`WHIRLWIND_NO_ANCHOR`, `WHIRLWIND_NO_HEAL`). Only the
   single-tile kernel (§3.1, §9.6) is verified.

### 9.6 Implementation Status, Verified Paths & Benchmarks

This subsection records what is *validated* versus *evolving*, and the
benchmark numbers behind the claims, so changes can be measured against a known
baseline.

**Entry point → solver / cost / mask map**

| Public fn | Network | Cost | Dijkstra | Mask | Status |
|---|---|---|---|---|---|
| `unwrap` (default, tiled >512 px) | reuse (per tile) | `compute_carballo_costs` | early-exit, 50 it + multi-source SSP | forbid (tiled) | **WIP heuristic** |
| `unwrap_reuse` (whole-image default) | reuse | `compute_carballo_costs` | early-exit, 50 it | cost-zero + NaN | reaches its cost optimum |
| `unwrap_linear` (single-tile) | unit-capacity | `compute_carballo_costs_parity` | full-completion, 8 it + single-source SSP | cost-zero + NaN | **verified (Python parity)** |
| `unwrap_convex` | convex | `compute_snaphu_smooth_costs` | heap | forbid | research prototype (#65) |
| `components_only` | unit-capacity | `compute_carballo_costs` | forbid | no MCF solve | — |

**Single-tile benchmark (D_077, 4176×4257, vs production GUNW = SNAPHU).** The
single-tile kernel is both faster and more accurate than single-tile SNAPHU:

| Unwrapper (single tile) | Runtime | per-component match vs production |
|---|---|---|
| **whirlwind `unwrap_linear`** | **≈160 s** | **99.49 %** (matches Python `ww-orig`) |
| SNAPHU (`cost=smooth, init=mcf`) | ≈588 s | 99.30 % |
| PHASS | ≈19.6 s | 94.7 % |

Peak RSS ≈6.4 GB (no swap). Reference SNAPHU/PHASS timings are in
`snaphu_ref/D_077.log` / `phass_ref.log`. Benchmark the verified path with
`scripts/bench_nisar_gunw_whirlwind.py --solver linear --nlooks 16`.

**SSP-fallback cost (a known sharp edge).** `unwrap_linear` runs 8 full-Dijkstra
PD iterations *then falls through to SSP* — and on D_077 it does reach SSP (the
PD iterations alone reach only ≈11 %; the SSP fallback routes the bulk). The SSP
fallback's runtime therefore dominates, and it depends critically on the SSP
*algorithm*:

- The multi-source `ssp::run` seeds every excess node, runs to
  all-deficits-popped, and augments **one** path per iteration —
  i.e. effectively a near-whole-image Dijkstra *per single unit of flow*. On the
  D_077 whole-image graph this costs ≈1472 s.
- A **single-source** SSP (early-exit per source) routes the same flow in ≈160 s.

The fast figure above is with the single-source SSP. **Dual-SSP fix
(implemented):** the multi-source `ssp::run` is kept for the early-exit/tiled
path (where it is fast — it is catastrophic only on large *whole-image* graphs),
and `ssp::run_single_source` is used only by `run_full_dijkstra` (single-tile),
restoring D_077 from ≈1472 s back to **≈160 s / 99.49 %** (verified post-fix).
The single-source potential update keeps reduced costs non-negative after every
early-exit Dijkstra — popped nodes get their exact distance; unpopped nodes keep
a zero shift, which is exactly "cap at the sink distance" since any unpopped node
has `dist ≥ d_sink` by Dijkstra pop order — so `debug_assert!(rc >= 0)` holds
with **no clamp**. The invariant is guarded by the debug test
`single_source_ssp_keeps_nonnegative_reduced_costs` (a steep noisy ramp that
reaches the SSP fallback); the tiled/default path is byte-unchanged (only
`run_full_dijkstra`'s fall-through branches to the single-source variant).

> **Tiling is not yet validated** on fragmented NISAR scenes (see §9.5 item 4).
> The single-tile kernel is the trustworthy reference to measure tiling against.
"""

# --- targeted small fixes to preserved sections ---
S24_OLD = """**Properties:**

- Residues are integers, typically in $\\{-1, 0, +1\\}$
- The sum of all residues over the image is zero (conservation)
- Positive residues act as flow **sources**, negative as **sinks** in the network formulation"""
S24_NEW = """**Properties:**

- Residues are integers, typically in $\\{-1, 0, +1\\}$
- The sum over the **entire augmented grid** (interior nodes *plus* the signed
  boundary frame, §4.2–4.3) is exactly zero by Stokes' theorem — the boundary
  deposits balance the interior winding. (For a smooth non-wrapping image every
  residue is zero.)
- Positive residues act as flow **sources**, negative as **sinks** in the network formulation"""

# Appendix C example 2 ("negative cost") is wrong: costs are clamped >= 0.
APPC_OLD = """Cost:
$$
c = -\\ln\\left(\\frac{0.90}{0.10}\\right) = -\\ln(9.0) \\approx -2.20
$$

This **negative cost** encourages adding a $2\\pi$ cycle, which is correct for a large phase gradient near a wrapping discontinuity."""
APPC_NEW = """Cost (before clamping):
$$
-\\ln\\left(\\frac{0.90}{0.10}\\right) = -\\ln(9.0) \\approx -2.20
$$

The raw LLR is negative (a $2\\pi$ correction is favored near a wrapping
discontinuity), but **the implementation clamps every forward-arc cost to
$\\ge 0$** — the \"encourage a cut\" signal is expressed as a *near-zero* cost on
the cut direction together with the asymmetric $\\alpha\\le 0\\Rightarrow c_{\\max}$
rule on the opposite direction (§5.4), not as a literal negative number."""

OBJECTIVE_OLD = """For the **linear** cost (Carballo / CRLB), the objective minimizes total signed cost subject to flow conservation $b_i$ at every node:

$$
\\min_{f}\\ \\sum_{e\\in\\text{forward arcs}} c_e\\, f_e,
\\qquad \\text{(unit-capacity: } 0 \\le f_e \\le 1\\text{; reuse: } f_e \\in \\mathbb{Z}\\text{).}
$$"""
OBJECTIVE_NEW = """For the **unit-capacity linear** cost (Carballo / CRLB), the objective minimizes total signed cost subject to flow conservation $b_i$ at every node:

$$
\\min_{f}\\ \\sum_{e\\in\\text{forward arcs}} c_e\\, f_e,
\\qquad 0 \\le f_e \\le 1.
$$

The **flow-reuse** mode is intentionally not this fixed linear objective. It is a PHASS-style shared-cut heuristic: the first unit placed on an unused arc pays the Carballo cost, then any arc with existing nonzero `flow_count` is relaxed at reduced cost 0 so later demands can reuse the same branch cut for free. This preserves the residue-neutralizing flow constraint while avoiding the capacity-1 boundary-stacking artifact on steep coherent ramps."""

FALLBACK_LINE_OLD = """3. **Fallback**: after `max_iter` iterations, or on no-progress, route any remaining excess with successive shortest paths. (A separate `run_no_ssp` variant omits this fallback for NISAR-scale problems where SSP on residual residues is prohibitively slow.)"""
FALLBACK_LINE_NEW = """3. **Fallback**: after `max_iter` iterations, or on no-progress, route any remaining excess with successive shortest paths. The fallback variant depends on the PD mode: early-exit PD uses multi-source SSP; full-completion PD (`unwrap_linear`) uses single-source SSP. A separate `run_no_ssp` variant exists only as a diagnostic to measure how much routing the fallback contributes."""

SSP_SECTION_OLD = """After the primal-dual loop terminates — either on hitting `max_iter` (default 8) or on a no-progress stall — any remaining excess is drained by a **successive shortest paths (SSP)** fallback (`ssp::run`). This fall-through is unconditional inside the shared primal-dual driver, so it is reached from *both* the early-exit (`run`) and full-completion (`run_full_dijkstra`) entry points. (The `run_no_ssp` variant exists precisely to *skip* this fallback when matching Python ww-orig behavior; see §7.7.)

Each SSP iteration:

1. Runs a **full multi-source Dijkstra** over the residual graph — every positive-excess node is seeded at distance 0 (the same routine the primal-dual phase uses), relaxing the entire reachable graph.
2. Selects the single reached deficit node with the smallest distance, and traces predecessor arcs back to its source seed.
3. Augments **one unit** of flow along that **one** source→deficit path, adjusting the two endpoints' excess.
4. Applies the same potential update as the primal-dual phase, $\\pi_i \\gets \\pi_i - d_i$ (capped at $d_{\\max}$ for nodes not finalized by an early-exit Dijkstra), keeping reduced costs non-negative.

A safety counter caps the loop at $4|V|$ iterations and asserts convergence (panicking with `\"SSP did not converge\"` otherwise).

Because a full graph-wide Dijkstra is re-run for *every single unit* of augmentation, SSP is correct but expensive: its cost scales with the residual flow $F$ left after the primal-dual phase. On whole-image, NISAR-scale graphs (tens of millions of arcs) this is prohibitively slow, which is why production paths rely on the primal-dual phase to route essentially all flow and treat SSP only as a small-residue safety net."""
SSP_SECTION_NEW = """After the primal-dual loop terminates — either on hitting `max_iter` or on a no-progress stall — any remaining excess is drained by a **successive shortest paths (SSP)** fallback. The shared driver chooses the fallback by PD mode: early-exit `run` falls through to multi-source `ssp::run`, while full-completion `run_full_dijkstra` falls through to single-source `ssp::run_single_source`.

Each **multi-source SSP** iteration:

1. Runs an early-exit multi-source Dijkstra over the residual graph — every positive-excess node is seeded at distance 0, and the search stops after all currently reachable deficits are finalized.
2. Selects the single reached deficit node with the smallest distance, and traces predecessor arcs back to its source seed.
3. Augments **one unit** of flow along that **one** source→deficit path, adjusting the two endpoints' excess.
4. Applies the same potential update as the primal-dual phase, $\\pi_i \\gets \\pi_i - d_i$ (capped at $d_{\\max}$ for nodes not finalized by an early-exit Dijkstra), keeping reduced costs non-negative.

A safety counter caps the loop at $4|V|$ iterations and asserts convergence (panicking with `\"SSP did not converge\"` otherwise).

The **single-source SSP** variant instead snapshots the current source list, skips any source already drained by an earlier augmentation, runs Dijkstra from that one source only, stops when the first deficit is popped, and augments that one path. Its potential update is written in the equivalent capped form `π += d_sink - dist[v]` for popped nodes while unpopped nodes keep a zero shift. This is valid only when the entry potentials already have non-negative reduced costs, which the full-completion PD path provides.

Because one Dijkstra search is re-run for *every single unit* of augmentation, SSP is correct but expensive: its cost scales with the residual flow $F$ left after the primal-dual phase. The multi-source variant is appropriate for tiled/small graphs but can approach a near-whole-image search per unit on NISAR-scale single-tile graphs. The single-source variant avoids that case by searching from one source only and stopping at the first popped deficit; it is used only after full-completion PD."""

COMPLEXITY_OLD = """- **SSP fallback**: $O(F \\cdot (|E| + |V| \\log |V|))$, where $F$ is the residual flow remaining after the primal-dual phase — one full multi-source Dijkstra per unit augmented. This dominates if much flow reaches SSP.
- **Space**: $O(|V| + |E|)$

In practice the primal-dual phase routes essentially all flow, leaving SSP a small residue; on very large graphs the SSP fallback is bypassed entirely (`run_no_ssp`) because a per-unit Dijkstra is catastrophically slow at NISAR scale."""
COMPLEXITY_NEW = """- **SSP fallback**: $O(F \\cdot \\text{Dijkstra search})$, where $F$ is the residual flow remaining after the primal-dual phase. Multi-source SSP can approach one near-global search per unit on large graphs; single-source SSP usually explores only the neighborhood from one source to its nearest deficit.
- **Space**: $O(|V| + |E|)$

In practice the runtime depends on how much flow reaches SSP. On D_077, SSP does the bulk of the final routing, so the single-source fallback is the runtime lever for the verified single-tile path. `run_no_ssp` is useful for diagnostics, not for parity or production quality."""

VERIFIED_OLD = """The single-tile kernel of §3.1 is the validated core: `unwrap_linear` is
bit-checked against the Python reference, and the whole-image `unwrap_reuse`
solve reaches its cost optimum (no negative cycles remain)."""
VERIFIED_NEW = """The single-tile kernel of §3.1 is the validated core: `unwrap_linear` is
checked against the Python reference, and the whole-image `unwrap_reuse`
solve reaches its cost optimum (no negative cycles remain)."""

def replace_section(text: str, num: int, new_body: str) -> str:
    # Match "## {num}. ..." up to the next top-level "## " (or end).
    pat = re.compile(rf"^## {num}\. .*?(?=^## |\Z)", re.S | re.M)
    new = new_body.rstrip("\n") + "\n\n---\n\n"
    # Use a function replacement so backslashes (LaTeX) aren't treated as escapes.
    out, k = pat.subn(lambda _m: new, text)
    assert k == 1, f"section {num}: expected 1 match, got {k}"
    return out

def replace_once(text: str, old: str, new: str, label: str) -> str:
    n_old = text.count(old)
    if n_old == 0 and text.count(new) == 1:
        return text
    assert n_old == 1, f"{label}: expected exactly 1 occurrence, got {n_old}"
    return text.replace(old, new)

t = ATBD.read_text()
for num in (3, 4, 5, 6, 7, 8):
    t = replace_section(t, num, DRAFTS[num])
t = replace_section(t, 9, SECTION9)
t = replace_once(t, S24_OLD, S24_NEW, "§2.4")
t = replace_once(t, VERIFIED_OLD, VERIFIED_NEW, "§3.3 verified wording")
t = replace_once(t, OBJECTIVE_OLD, OBJECTIVE_NEW, "§6.4 reuse objective")
t = replace_once(t, FALLBACK_LINE_OLD, FALLBACK_LINE_NEW, "§7.1 fallback")
t = replace_once(t, SSP_SECTION_OLD, SSP_SECTION_NEW, "§7.6 SSP")
t = replace_once(t, COMPLEXITY_OLD, COMPLEXITY_NEW, "§7.7 complexity")
t = replace_once(t, APPC_OLD, APPC_NEW, "AppendixC")
version_text = (
    "*Document Version: 3.0 — algorithm sections audited against the code "
    "(see §9.6 for verified-vs-WIP status and benchmarks).*  \n"
    "*Last Updated: 2026-06-03*"
)
t, k = re.subn(
    r"\*Document Version: .*?\*\s*\n\*Last Updated: .*?\*",
    version_text,
    t,
    count=1,
    flags=re.S,
)
assert k == 1, f"version: expected exactly 1 replacement, got {k}"
ATBD.write_text(t)
print(f"ATBD rewritten: {len(t.splitlines())} lines")
