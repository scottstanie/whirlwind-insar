# Whirlwind Algorithm Theoretical Basis Document (ATBD)

## Executive Summary

Whirlwind is a Bayesian minimum-cost network flow algorithm for 2D phase unwrapping of interferometric synthetic aperture radar (InSAR) data. The algorithm formulates phase unwrapping as a minimum-cost flow problem on a rectangular grid graph, where edge costs are derived from Bayesian probability densities that account for both coherence and local phase gradient statistics. The network flow problem is solved using a primal-dual algorithm, and the final unwrapped phase is obtained by integrating the unwrapped phase gradients.

> **Default architecture note (updated 2026-06-03).** The MCF + primal-dual core described in this ATBD (Sections 2–8) is correct and unchanged. The public `unwrap` entry point now **defaults to the verified single-tile linear solver** (`unwrap_linear`: ww-orig-parity Carballo cost, capacity-1 MCF, adaptive PD/SSP fallback for masked frames - §7.6.1), which matches Python `ww-orig` across the validated NISAR frame set. A **tiled** pipeline still exists (per-tile MCF → global coarse anchor → multi-scale cascade → feathered seam composite) and is selected opt-in via `multilook>1`, an explicit `tile_size`, or `WHIRLWIND_UNWRAP_SOLVER=tiled`, but **it is NOT a validated path** - see the status note below before using it. See [`paper/report_anchor_cascade.md`](paper/report_anchor_cascade.md) and [`paper/tiling.md`](paper/tiling.md).
>
> **Verified-vs-WIP (see §9.6).** The single-tile linear kernel is the *validated* core and the default: `unwrap_linear` is checked against the Python `ww-orig` reference (99.49 % on D_077; matches ww-orig across the masked-frame set, §9.6) and beats single-tile SNAPHU on **both** speed and accuracy (≈37 s / 99.49 % vs ≈588 s / 99.30 %). **The tiled + anchor pipeline is NOT working and NOT validated.** It produced good numbers on *select* scenes but **failed on most**, and there is **no working version at present** - it was not salvageable until ww-original quality was re-matched on the single-tile path (now done); any tiled benchmark elsewhere (e.g. "99.84 % NISAR") is from those select scenes and must not be read as a product claim. Treat §9.6 as the canonical status/benchmark table.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Mathematical Background](#2-mathematical-background)
3. [Algorithm Overview](#3-algorithm-overview)
4. [Residue Computation](#4-residue-computation)
5. [Bayesian Cost Function](#5-bayesian-cost-function)
6. [Network Flow Formulation](#6-network-flow-formulation)
7. [Primal-Dual Solution](#7-primal-dual-solution)
8. [Phase Integration](#8-phase-integration)
9. [Implementation Details](#9-implementation-details)
10. [References](#10-references)

---

## 1. Introduction

### 1.1 Problem Statement

InSAR phase measurements are inherently wrapped to the interval $[-\pi, \pi)$. The phase unwrapping problem seeks to recover the true (unwrapped) phase $\psi$ from the observed wrapped phase $\phi$:

$$
\psi = \phi + 2\pi k
$$

where $k \in \mathbb{Z}$ is an unknown integer cycle count that varies spatially across the interferogram.

### 1.2 Challenges

Phase unwrapping is ill-posed due to:

- **Phase noise** from decorrelation, which can cause the wrapped gradient to differ from the true gradient by integer multiples of $2\pi$
- **Residues** (phase inconsistencies) that indicate locations where no continuous unwrapping can reconcile all local gradients
- **Ambiguity** in determining the correct integer $k$ for each phase gradient

### 1.3 Whirlwind Approach

Whirlwind addresses these challenges through:

1. A **Bayesian framework** that uses coherence to weight the reliability of phase gradients
2. **Network flow optimization** to find the minimum-cost unwrapping that neutralizes all residues
3. **Statistical cost functions** (Carballo PDFs) that incorporate both phase and coherence information

---

## 2. Mathematical Background

### 2.1 Phase Model

The observed wrapped phase $\hat{\psi}$ can be decomposed as:

$$
\hat{\psi} = \phi_s + \phi_N + 2\pi k
$$

where:

- $\phi_s$ is the true signal phase (the quantity we want to recover, modulo $2\pi$)
- $\phi_N$ is phase noise due to decorrelation
- $k$ is the integer ambiguity (unknown)
- $\hat{\psi}$ denotes the measured (observed) wrapped phase

Note: The hat notation $\hat{\psi}$ follows the convention in Carballo's work, denoting the observed quantity rather than an estimate.

### 2.2 Phase Gradients

For adjacent pixels, the phase gradient difference is:

$$
\Delta \hat{\psi} = \Delta \phi_s + \Delta \phi_N + 2\pi \Delta k
$$

The key insight is that **if we can determine $\Delta k$ for each edge, we can recover the unwrapped phase gradients**, and from those, the unwrapped phase field (up to a global constant).

### 2.3 Bayesian Formulation

The maximum likelihood estimate for $\Delta k$ is:

$$
\Delta k^* = \arg\max_{\Delta k} f(\Delta \hat{\psi} \mid \Delta k)
$$

This requires a probability model for the phase gradient conditioned on the integer ambiguity. Carballo's approach integrates over the unknown true coherence $\gamma$ using the sample coherence $\hat{\gamma}$ and the phase noise PDF from Lee et al.

**Important principle from Geoff's notes**: "Don't rewrap!" - the algorithm works directly with wrapped gradients and integer cycle corrections, never rewrapping intermediate results.

### 2.4 Residues

A **residue** at a grid node is the sum of wrapped phase gradients around a 2x2 pixel loop, normalized by $2\pi$:

$$
r = \mathrm{round}\left(\frac{1}{2\pi} \oint \nabla \phi \cdot d\ell\right)
$$

Residues are topological defects indicating phase inconsistency. For any continuous phase field, the sum around a closed loop must be zero. Non-zero residues indicate that the wrapped gradients are inconsistent with any continuous unwrapping.

**Properties:**

- Residues are integers, typically in $\{-1, 0, +1\}$
- The sum over the **entire augmented grid** (interior nodes *plus* the signed
  boundary frame, §4.2–4.3) is exactly zero by Stokes' theorem - the boundary
  deposits balance the interior winding. (For a smooth non-wrapping image every
  residue is zero.)
- Positive residues act as flow **sources**, negative as **sinks** in the network formulation

---

## 3. Algorithm Overview

Whirlwind exposes a layered API. The inner kernel - and the public default - is
a single whole-image minimum-cost-flow (MCF) solve (`unwrap_linear`); the public
entry point also produces connected-component labels and an opt-in tiling
robustness layer. This section describes both, and which paths are verified
versus still evolving.

**Input (all entry points):** complex interferogram `igram`, coherence
magnitude `corr`, number of effective looks `nlooks`, and (optionally) a
boolean validity `mask`.

### 3.1 Inner single-tile kernel (the verified path)

For a frame that fits in one tile, the algorithm is the classic five stages on
the whole image:

```
[1] Wrapped phase:        φ = angle(igram)
[2] Residues:             r = residue(φ)
[3] Bayesian arc costs:   c = carballo_costs(igram, corr, nlooks, mask)
[4] Min-cost flow:        primal_dual(network(r, c))   # 50 iterations
[5] Integrate gradients:  ψ = integrate(φ, flow)       # NaN masked pixels
```

The default whole-image solver is `unwrap_linear` (`lib.rs`), the verified
single-tile capacity-1 MCF kernel: a unit-capacity network with the
ww-orig-parity Carballo cost and `run_full_dijkstra` (8 primal-dual iterations,
then a single-source SSP and an adaptive PD-resume fallback that re-balances
masked frames - §7.6.1). It matches the Python `whirlwind_orig` reference across
the validated NISAR frame set. Masked arcs are **not** forbidden - they are
given cost 0 so MCF can route through masked regions, and masked pixels are set
to NaN after integration.

Two other single-tile variants exist (opt-in / non-default):

* `unwrap_reuse` - a PHASS-style whole-image *flow-reuse* solver
  (`Network::new_reuse_with_mask`): arcs are multi-unit (no capacity-1
  saturation) and the Dial bucket-queue Dijkstra overrides the reduced cost to 0
  on any arc that already carries flow, so once one wrap-line is laid down
  subsequent demands route along it for free. It reaches its cost optimum (no
  negative cycles remain) but is **experimental/research, not validated** as the
  default; selected only by `WHIRLWIND_UNWRAP_SOLVER=reuse`.
* `unwrap_convex` - a **research prototype** of a SNAPHU-style convex
  (quadratic-in-flow) cost (issue #65). Selected at the whole-image level only
  by `WHIRLWIND_UNWRAP_SOLVER=convex`; solve backend chosen by
  `WHIRLWIND_CONVEX_SOLVE ∈ {pd (default), ssp, cancel}`.

### 3.2 Public entry point and opt-in tiling layer

The public unwrap (`unwrap_coherence_with_components`, exposed to Python as
`_unwrap_native` and consumed by dolphin) returns **both** unwrapped phase and
connected-component labels:

```
phase  = unwrap_coherence(igram, corr, nlooks, mask,
                          tile_size, tile_overlap, multilook)
comps  = components_only(igram, corr, nlooks, mask, params)
return (phase, comps)
```

`components_only` grows SNAPHU-style components directly from the global Carballo
cost grid **without running an MCF solve** (labels depend only on mask-forbidden
arcs and raw arc costs, both fixed at network construction), so it is
independent of how - or whether - phase was solved, and costs one global cost
grid in memory (`O(pixels)`).

**Default = single-tile, no auto-tiling** (`unwrap_coherence`): with
`tile_size == 0` (the default) and `multilook ≤ 1`, the whole image is solved
single-tile by the verified `unwrap_linear` kernel (§3.1) - there is **no**
automatic switch to tiling at any frame size. The **tiled** pipeline is opt-in
and is selected *only* by an explicit `tile_size ≥ 4`, any `multilook > 1`, or
`WHIRLWIND_UNWRAP_SOLVER=tiled`. It is **NOT a validated path**: it fails on most
scenes (≈65–89 % per-component match vs ≈99–100 % single-tile) and must never be
described as the default/shipped/production path.

* **Single-tile** frames go through `unwrap_linear` (§3.1), or `unwrap_reuse` /
  `unwrap_convex` when `WHIRLWIND_UNWRAP_SOLVER=reuse|convex`.
* **Tiled** frames (opt-in only) go through `unwrap_tiled_robust`, which adds,
  on top of the per-tile reuse solves:
  1. parallel per-tile MCF solve (each tile uses the §3.1 reuse kernel);
  2. global reconciliation of per-tile integer-2π offsets via an MCF on the
     tile grid;
  3. gated feathered compositing of the overlaps;
  4. a global coarse anchor plus a multi-scale cascade (`f = 16, 8, 4`) to pin
     each region's integer cycle level;
  5. bounded coherence-gated sliver healing;
  6. a **gated multi-shift re-solve**: if the result tears coherent terrain
     (coherent-cut rate above a fixed floor - the signature of a tile-seam
     artifact or a wrong global winding on a fragmented scene), the tile grid
     is re-run shifted by fractions of the tile step and the result with the
     fewest coherent cuts is kept, followed by a localized `seam_repair`.
  Stages 4–6 can be toggled for diagnostics via `WHIRLWIND_NO_ANCHOR` and
  `WHIRLWIND_NO_HEAL`.

A parallel CRLB-cost family (`unwrap_crlb_*`, variance-driven rather than
coherence-driven) mirrors this structure; anchor/cascade parity with the
coherence path is still pending (issue #35).

### 3.3 Verified vs. work-in-progress

The single-tile `unwrap_linear` kernel of §3.1 is both the validated core **and
the public default**: it is checked against the Python `ww-orig` reference and
matches it across the validated NISAR frame set. (The opt-in whole-image
`unwrap_reuse` solver reaches its cost optimum - no negative cycles remain - but
is experimental, not the default.) The **tiled robustness layer of §3.2 is
opt-in and NOT validated** - an empirically tuned, still-evolving heuristic
whose seam reconciliation, anchor/cascade, multi-shift gate, and sliver healing
are calibrated against benchmark scenes rather than proven optimal (hence the
environment escape hatches), and it fails on most scenes. Section 9.5 lists its
known limitations. The detailed mathematics of stages [2]–[5] follow in
Sections 4–8; the tiling and stitching machinery is detailed in Section 9.

---

---

## 4. Residue Computation

### 4.1 Definition

Residues are computed on a grid of **nodes** with dimensions $(m+1) \times (n+1)$, where the input wrapped-phase array has dimensions $m \times n$. An **interior** node $(r,c)$ sits at the center of the 2x2 pixel block $\{(r{-}1,c{-}1),(r{-}1,c),(r,c{-}1),(r,c)\}$, and its residue is the integer winding number of the wrapped gradients around that loop. The outer **frame** of the grid (row $0$, row $m$, column $0$, column $n$) holds the wrap counts along the four image edges (see §4.2).

Implementation: `residue::compute_with_mask` in `crates/whirlwind-core/src/residue.rs`; `residue::compute(φ)` is the unmasked convenience wrapper that calls `compute_with_mask(φ, None)`.

### 4.2 Algorithm

The integer cycle difference of two wrapped phases is

$$
\texttt{cycle\_diff}(a,b) = \mathrm{round}\!\left(\frac{a-b}{2\pi}\right) \in \mathbb{Z}.
$$

**Interior residues.** For each interior node $(r,c)$ (with $1 \le r \le m-1$, $1 \le c \le n-1$), let $i=r-1$, $j=c-1$ and take the four surrounding pixels $\phi_{00}=\phi[i,j]$, $\phi_{01}=\phi[i,j{+}1]$, $\phi_{10}=\phi[i{+}1,j]$, $\phi_{11}=\phi[i{+}1,j{+}1]$. The residue is the counter-clockwise curl of the integer-rounded gradients around the loop, written to the single node $(r,c)$:

```
residue(r, c) =   cycle_diff(φ_10, φ_00)
                + cycle_diff(φ_11, φ_10)
                + cycle_diff(φ_01, φ_11)
                + cycle_diff(φ_00, φ_01)
```

Each residue row depends only on pixel rows $r{-}1$ and $r$, so rows are computed in parallel (rayon).

**Boundary frame.** The outer frame is **not** zero. Each image-edge pixel gradient deposits its wrap count on a unique frame node, with signs chosen so the full grid balances to zero (see §4.3):

```
top edge    φ[0,   j]→φ[0,   j+1]:  residue(0,   j+1) += cycle_diff(φ[0,   j+1], φ[0,   j])
bottom edge φ[m-1, j]→φ[m-1, j+1]:  residue(m,   j+1) -= cycle_diff(φ[m-1, j+1], φ[m-1, j])
left edge   φ[i,   0]→φ[i+1, 0]:    residue(i+1, 0)   -= cycle_diff(φ[i+1, 0],   φ[i,   0])
right edge  φ[i, n-1]→φ[i+1, n-1]:  residue(i+1, n)   += cycle_diff(φ[i+1, n-1], φ[i,   n-1])
```

The four corner nodes $(0,0),(0,n),(m,0),(m,n)$ are never written and stay zero. These frame charges let the MCF drain wrap lines that exit through an image edge (a "wrap line ends here") instead of forcing each to pair with a distant interior residue, which would otherwise produce long flow paths and large integer-surface variations after integration.

**Masking.** When a pixel-grid mask is supplied (`true` = valid, `false` = masked/invalid), `compute_with_mask`:

- leaves an interior residue at $0$ if **any** of the four pixels in its 2x2 loop is masked;
- skips a boundary-edge deposit if **either** pixel of that edge segment is masked.

NaN/invalid pixels are replaced by zeros upstream and would otherwise generate a wall of spurious large residues along the mask boundary that leak charge into the valid region's MCF problem; masking keeps the flow problem confined to where the phase is meaningful.

### 4.3 Properties

- The residue grid has one more row and one more column than the phase array: $(m+1)\times(n+1)$.
- Interior residues are integers, typically in $\{-1,0,+1\}$; positive residues act as flow **sources**, negative as **sinks**.
- The sum over the **entire** grid - interior nodes *and* the signed boundary frame - is exactly zero: $\sum_{r,c} r_{r,c} = 0$. By Stokes' theorem the counter-clockwise boundary contour integral of the wrap rates equals the total interior winding, so the boundary deposits carry the opposite sign and the augmented total balances. This source/sink balance is what makes the MCF problem solvable.
- For a smooth, non-wrapping image (range within $[-\pi,\pi]$) every `cycle_diff` rounds to $0$, so both interior and boundary residues vanish.

> **Note (diagnostic parity path).** The standard/production unwrap paths (`unwrap_reuse`, convex) keep the boundary frame populated and rely on a single *ground* node connected to every boundary residue for edge drainage. The diagnostic `unwrap_linear` (`crates/whirlwind-core/src/lib.rs`) instead explicitly **zeros** the residue frame (`row 0`, `row -1`, `col 0`, `col -1`) to bit-match the original Python `ww-orig` solver and is not part of the production residue stage.

---

## 5. Bayesian Cost Function

Whirlwind ships **two** Carballo-style edge-cost implementations. They share the same statistical motivation (Lee 1994 multilook phase noise + smoothed local gradient) and the same per-arc layout, but differ in how the per-arc log-likelihood is obtained and in their default scale:

| Function (`crates/whirlwind-core/src/cost/mod.rs`) | Used by                                                                     | Probability source                                                                 | Int scale                               | Masking rule                                  |
| -------------------------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------- | --------------------------------------------- |
| `compute_carballo_costs` (§5.1–5.5)                | **Production**: `unwrap`, `unwrap_reuse`, the tiled solver, conncomp regrow | Analytical Lee-1994 CDF LUT built at runtime (`cost/lut.rs`); `p_0 = 1-p_1`        | `CARBALLO_COST_SCALE = 6` (max int 300) | cost = 0 where **either** endpoint masked     |
| `compute_carballo_costs_parity` (§5.6)             | **Diagnostic only**: `unwrap_linear` (single-tile Rust/Python parity)       | Embedded ww-orig pre-sampled spline tables (`cost/spline_lut.rs`); `p_0 + p_1 ≠ 1` | `100` (matches Python)                  | cost = 0 only where **both** endpoints masked |

### 5.1 Carballo Probability Model

The cost of pushing one unit of flow on an arc is a **log-likelihood ratio** for "this gradient should receive a +1 cycle correction" vs. "no correction":

$$
c = -\ln\left(\frac{p_1}{p_0}\right),\qquad p_1 = P(\Delta k = +1 \mid \hat\alpha,\hat\gamma,L),\quad p_0 = P(\Delta k = 0 \mid \hat\alpha,\hat\gamma,L)
$$

where $\hat\alpha$ is the smoothed local phase gradient, $\hat\gamma$ the per-edge coherence, and $L$ the number of looks. In both implementations the float cost is clamped to be **non-negative**; forward arcs never carry a negative cost.

### 5.2 Smooth Phase Gradient Estimation

Wrapped per-edge gradients are formed from complex conjugate products (`phase_dy = arg(igram[i+1,j]·conj(igram[i,j]))`, `phase_dx` analogously), then smoothed with a **separable 7x7 box filter** (`box_filter_2d`, nearest-edge replication; Carballo's original paper used 5x5). The smoothed gradient $\hat\alpha$ estimates the underlying signal gradient by averaging over local phase noise.

**Mask handling (important):** both cost paths deliberately use the *biased* `smooth_phase_gradients` (which averages in the `0+0j` of masked pixels) rather than the mask-aware variant. The bias is a feature: it drags $\hat\alpha\to0$ within ~3 px of a mask boundary, raising the Carballo cost there and acting as an implicit fence that discourages MCF from routing 2π errors across the boundary. A mask-aware filter (`smooth_phase_gradients_with_mask`) exists but is intentionally **not** the default - using it worsens 2π block errors on real NISAR data.

### 5.3 Coherence-Based Quality

The coherence for each edge is the **minimum** of the two adjacent pixels, $\gamma_{\text{edge}} = \min(\gamma_1,\gamma_2)$, so a low-quality pixel always weakens its incident edges.

### 5.4 Probability Lookup (production path: analytical Lee-1994 CDF)

The production `compute_carballo_costs` does **not** use B-splines or stored p0/p1 tables. It builds a cost LUT at runtime from the Lee 1994 PDF (`cost/lee_pdf.rs`, `pdf(α,γ,L)`, evaluated in log-space with the Euler-transformed ₂F₁). For each γ it numerically integrates the PDF into a normalized CDF (2001-node trapezoid over $[-\pi,\pi]$), then for $\alpha>0$:

$$
p_1 = \text{CDF}(\alpha-\pi),\qquad p_0 = 1-p_1,\qquad c(\alpha,\gamma)=\min\!\big(\max(-\ln(p_1/p_0),0),\,c_{\max}\big)
$$

with $c_{\max} = $ `MAX_CARBALLO_COST = 50` nats. For $\alpha \le 0$ the cost is forced to $c_{\max}$. Limits: at $\alpha=\pi$ (a wrap line) $\text{CDF}(0)=0.5\Rightarrow c=0$ (free to cut); as $\alpha\to0^+$ (smooth interior) $c\to c_{\max}$ (never cut). The $\alpha\le0$ → $c_{\max}$ rule makes the cost **strongly asymmetric in the sign of $\alpha$**, which is load-bearing for the per-direction split in §5.5. The LUT is a 101(γ)x501(α) bilinear table, built once per `nlooks` (rounded to 0.1) and leaked to `'static`; γ is clamped to $[0,0.999]$, α to $[-\pi,\pi]$.

The diagnostic parity path uses pre-computed tables instead - see §5.6.

### 5.5 Cost Computation for Four Directions and Integer Scaling

Costs are computed for all four arc directions and packed into one `Vec<i32>` indexed by arc id, in slab order `[DOWN, UP, RIGHT, LEFT]`. The sign of $\hat\alpha$ fed to the cost depends on direction (vertical pixel edges → RIGHT/LEFT, horizontal pixel edges → DOWN/UP):

```
cost_rt = c(-phase_dy_smooth, corr_dy)   cost_lt = c(+phase_dy_smooth, corr_dy)
cost_dn = c(+phase_dx_smooth, corr_dx)   cost_up = c(-phase_dx_smooth, corr_dx)
```

The float LLR is converted to `i32` via `round(c · CARBALLO_COST_SCALE)` with `CARBALLO_COST_SCALE = 6.0`, so the maximum integer cost is `6 x 50 = 300` (chosen to keep Dial's bucket-queue Dijkstra fast while using the correct Lee 1994 shape). Reverse-arc costs are the negation of the forward cost and are reconstructed by `Network` on demand. Where **either** endpoint pixel is masked, the arc cost is set to `0`.

### 5.6 Parity / single-tile cost (`compute_carballo_costs_parity`)

`unwrap_linear` (the Rust/Python parity replica - *not* a production entry point) uses `compute_carballo_costs_parity`, which reproduces the original ww-orig spline model exactly:

- **Probabilities** come from embedded, pre-sampled tables read via **trilinear** interpolation (`cost/spline_lut.rs`) - there is no tri-cubic B-spline evaluator in Rust; the Python `.npz`/`.pkl` splines were sampled onto a dense grid that the Rust reads directly. The grid is α: 31 uniform pts in $[-\pi,\pi]$; γ: 11 pts $[0,0.1,\dots,1.0]$; $L$: 11 log-spaced pts $[1,\dots,80]$ (clamped at lookup). Tables ship as five little-endian `f32` blobs embedded in the binary: `carballo_grid_phase.bin` (31), `carballo_grid_corr.bin` (11), `carballo_grid_nlooks.bin` (11), `carballo_p0.bin` and `carballo_p1.bin` (each 31·11·11 = 3751). Here $p_0 = P(\Delta k=0)$ and $p_1 = P(\Delta k=\pm1)$, and in general **$p_0 + p_1 \neq 1$**.
- **Cost** = `round(100 · max(-ln(p_1/p_0), 0))`, with both probabilities floored at `1e-30`. The scale is `100` (matching Python's `100·-log(p1/p0)`), *not* the production path's 6.
- **Masking** zeros the cost only where **both** endpoint pixels are invalid (matching Python's `mask = ~valid`, zero where `mask[a] && mask[b]`); a boundary arc with one valid pixel keeps a nonzero cost.
- **Override:** setting `WHIRLWIND_CARBALLO_LUT_DIR` to a directory containing the same five `.bin` files replaces the embedded tables at first use (for experiments).

Smoothing (biased 7x7 box), the min-of-endpoints edge coherence, and the four-direction sign convention are identical to the production path (§5.2–5.5).

### 5.7 Cost Interpretation

| Coherence | Phase gradient $\hat\alpha$ | Cost behavior        | Interpretation                                         |
| --------- | --------------------------- | -------------------- | ------------------------------------------------------ |
| High      | $\approx 0$ (smooth)        | Large (→ $c_{\max}$) | Confident $\Delta k = 0$; strongly penalize a cut here |
| High      | near $+\pi$ (wrap line)     | $\approx 0$          | Confident $\Delta k = +1$; cheap to cut                |
| Any       | $\le 0$ (production path)   | $c_{\max}$           | Wrong-sign correction for this direction; never cut    |
| Low       | any                         | small                | Uncertain; the edge barely influences the solution     |

Note this differs from the older description: forward-arc costs are clamped non-negative (never negative), and the production path is **asymmetric** in the sign of $\hat\alpha$ rather than "symmetric near zero" at low coherence.

---

## 6. Network Flow Formulation

### 6.1 Graph Construction

Phase unwrapping is posed as a **minimum-cost flow (MCF) problem** on a rectangular residue grid (`RectangularGridGraph`, `grid.rs`):

- **Nodes**: an $m \times n$ grid with $m = m_{\text{phase}}+1$, $n = n_{\text{phase}}+1$ (one node per 2x2 pixel loop). `node_id(i,j) = i·n + j`.
- **Arcs**: 4-connected. Each grid edge carries **two independent forward arcs** (one per direction - the two directions are separate Carballo cost decisions), each with its own residual reverse partner, i.e. 4 arc slots per interior pair. Forward arc IDs are partitioned for $O(1)$ transpose:

  | Range                 | Direction | Tail → Head         |
  | --------------------- | --------- | ------------------- |
  | $[0,\ n_v)$           | DOWN      | $(i,j)\to(i{+}1,j)$ |
  | $[n_v,\ 2n_v)$        | UP        | $(i{+}1,j)\to(i,j)$ |
  | $[2n_v,\ 2n_v{+}n_h)$ | RIGHT     | $(i,j)\to(i,j{+}1)$ |
  | $[2n_v{+}n_h,\ N_f)$  | LEFT      | $(i,j{+}1)\to(i,j)$ |

  with $n_v=(m{-}1)n$, $n_h=m(n{-}1)$, $N_f = 2n_v+2n_h$ forward arcs. The reverse partner of forward arc $a$ is $a+N_f$. (Per-arc cost vectors must follow this same `[DOWN, UP, RIGHT, LEFT]` order.)
- **Supply/Demand**: node $i$ has supply $b_i = r_i$ (its residue / winding count); the problem is balanced, $\sum_i b_i = 0$.
- **Cost**: a forward arc has cost $c_{ij}$; its reverse has cost $-c_{ij}$.

### 6.2 Residual Graph and Capacity Modes

The solver operates on the **residual graph** (forward + reverse arc per direction), letting it "undo" flow on reverse arcs. The `Network` (`network.rs`) supports three capacity/cost modes, selected at construction; the public default (`unwrap_linear`) is **unit-capacity**:

- **Unit-capacity MCF** (`Network::new` / `new_with_mask`): each forward arc has capacity 1, tracked by a per-arc saturation bit pair. Pushing a unit saturates the forward arc and opens its reverse. Used by the verified public default `unwrap_linear`, by connected-component growth, and by `components_only`.
- **Flow-reuse mode** (`new_reuse_with_mask`, the network used by the opt-in `unwrap_reuse` solver and the per-tile reuse solves): arcs are **multi-unit** (signed integer `flow_count`, no saturation on push). Once an arc carries any flow, Dial overrides its reduced cost to 0 so later demands reuse the same wrap-line for free (PHASS-style). This removes the capacity-1 boundary-stacking failure on steep clean ramps.
- **Convex mode** (`new_convex_with_mask`, SNAPHU-style): arcs are multi-unit with a **parabolic per-arc cost** (§6.4). Dial uses the *marginal* cost of one more unit rather than `cost_fwd`. Used by `unwrap_convex` and the `WHIRLWIND_UNWRAP_SOLVER=convex` path.

Masked edges are encoded as a **forbidden** state (both directions saturated, never carrying flow); see §6.3.

### 6.3 Mask Handling

Two distinct mechanisms exist, and which one applies depends on the entry point:

- **Arc forbidding** (`forbid_masked_arcs`): when a pixel-grid mask is passed to construction, every arc crossing a pixel-edge with ≥1 invalid endpoint is pre-saturated in **both** directions (the *forbidden* state), removing it from the residual graph. Used by the CRLB-coherence, convex, conncomp, ground, and tiled paths.
- **Cost-zeroing + post-NaN**: the default coherence solver (`unwrap_linear`) and the opt-in `unwrap_reuse` deliberately pass **no mask** to construction (no arcs forbidden), rely on the cost stage to zero masked-arc costs so MCF routes through masked regions freely, then mark masked pixels `NaN` after integration. Empirically, forbidding masked arcs *isolates* residues inside masked regions and drops NISAR matching from ~99 % to ~42 %, hence the cost-zeroing default on the coherence path.

### 6.4 Minimum-Cost Flow Objective

For the **unit-capacity linear** cost (Carballo / CRLB), the objective minimizes total signed cost subject to flow conservation $b_i$ at every node:

$$
\min_{f}\ \sum_{e\in\text{forward arcs}} c_e\, f_e,
\qquad 0 \le f_e \le 1.
$$

The **flow-reuse** mode is intentionally not this fixed linear objective. It is a PHASS-style shared-cut heuristic: the first unit placed on an unused arc pays the Carballo cost, then any arc with existing nonzero `flow_count` is relaxed at reduced cost 0 so later demands can reuse the same branch cut for free. This preserves the residue-neutralizing flow constraint while avoiding the capacity-1 boundary-stacking artifact on steep coherent ramps.

For the **convex** cost the per-arc term is parabolic in the integer signed flow $k_e$:

$$
\min_{k}\ \sum_{e} w_e\,\bigl(k_e\cdot 100 - O_e\bigr)^2,
$$

with $w_e$ an inverse-variance weight, $O_e$ a preferred-flow offset, and $100 = $ `NSHORTCYCLE`. Dijkstra uses the marginal cost $\Delta c_e = w_e\,(\pm 2\cdot100\cdot(k_e\cdot100-O_e) + 100^2)$. Because that marginal is negative at $k_e=0$ whenever $|O_e|>50$, the convex network is first pre-loaded so each arc sits at its parabola minimum $k^\* = \mathrm{round}(O_e/100)$ (`preload_convex_min`, with node excess adjusted to keep conservation); thereafter every residual marginal is $\ge 0$, zero initial potentials are valid, and successive-shortest-paths stays sound (the ordered-parallel-arc reduction of convex-cost MCF).

An optional **virtual ground node** (`new_with_mask_and_ground`) connects every boundary residue to a sink with two unit-capacity arcs of cost `ground_cost`, letting boundary wrap-line terminations drain at the image edge. It is used only by the `*_grounded` diagnostics - it corrupts dense interior-residue real data and is not on the default path.

### 6.5 Why This Works

- Positive residues (sources) export flow; negative residues (sinks) import it; the net flow on each directed edge is the integer $2\pi$ correction applied to that phase gradient (§8.4).
- The minimum-cost flow pairs sources with sinks along the statistically most probable (lowest-cost) correction paths.
- Neutralizing all residues guarantees the integrated unwrapped phase is **path-independent**.
- The flow-reuse and convex modes give the same residue-neutralizing guarantee while removing the capacity-1 stacking artifact on steep coherent ramps.

---

## 7. Primal-Dual Solution

### 7.1 Algorithm Overview

The primal-dual algorithm solves the min-cost flow problem through repeated multi-source shortest-path computations. A single shared loop (`primal_dual::run_impl`) implements two completion modes:

- **Early-exit mode** (`primal_dual::run`, `max_iter = 50`) - used by the opt-in tiled solve, the opt-in `unwrap_reuse`/`unwrap_convex` solvers, conncomp, and integration. Dijkstra stops as soon as all sinks are finalized.
- **Full-completion mode** (`primal_dual::run_full_dijkstra`, `max_iter = 8`) - used by the verified public default `unwrap_linear` (and `unwrap_linear_ext_costs`). Dijkstra runs until the queue is empty, matching Python ww-orig's `dijkstra_pd` / `primal_dual(maxiter=8)`.

Each iteration:

1. **Initialization**: all flows zero, all potentials zero, excess set from the residue charges.
2. **Iterate** until no progress is possible:
   - Terminate if total positive excess or total negative deficit reaches 0.
   - Break to the SSP fallback if total excess did not decrease this iteration ("no progress").
   - Run **multi-source Dijkstra** from all excess nodes using reduced costs.
   - **Augment** unit flow along shortest paths from sources to sinks.
   - **Update potentials** to keep reduced costs non-negative.
3. **Fallback**: after `max_iter` iterations, or on no-progress, route any remaining excess with successive shortest paths. The fallback variant depends on the PD mode: early-exit PD uses multi-source SSP; full-completion PD (`unwrap_linear`) uses single-source SSP. A separate `run_no_ssp` variant exists only as a diagnostic to measure how much routing the fallback contributes.

### 7.2 Reduced Costs

The **reduced cost** of an arc from tail $i$ to head $j$ is

$$\bar{c}_{ij} = c_{ij} - \pi_i + \pi_j$$

the standard convention (Ahuja et al., 1993). After each potential update, reduced costs on residual arcs with positive capacity stay non-negative, enabling Dijkstra. Three cost cases are handled inline at relaxation: a **used** arc has reduced cost 0 (PHASS-style reuse); in **convex** mode the cost term is the arc's marginal cost; otherwise it is the linear arc cost.

### 7.3 Multi-Source Dijkstra

Dijkstra is seeded with every excess node at distance 0 and finds shortest paths from any source to all reachable nodes. Two completion behaviors exist:

- **Early-exit** (`dijkstra_multi_source_into`): stops once every deficit (sink) has been *popped* (finalized). Further relaxation cannot change a finalized distance, so the late tail of each PD iteration is trimmed.
- **Full-completion** (`dijkstra_multi_source_full_into`): runs until the queue empties, so every reachable node is popped with its exact shortest distance.

A node's distance is trustworthy only once it is **popped**; the `ShortestPaths::popped` flag (queried via `was_reached`) distinguishes a finalized node from one that has merely been relaxed (early-exit can leave relaxed-but-unpopped nodes with a non-final finite distance). Scratch buffers (`ShortestPaths` plus the augment/cycle-detection scratch) are allocated once and reused across all PD iterations via `ShortestPaths::reset` and the `*_into` Dijkstra variants.

Backend selection is described in §9.2.3.

### 7.4 Flow Augmentation

After Dijkstra, one unit of flow is routed to each reachable deficit node:

- For each sink that was popped, walk the predecessor chain (`pred_node` / `pred_arc`) back to its seed source (the node whose predecessor arc is $-1$). The source is read from the **end of the walk**, not from the cached `source` field, which relaxation can leave stale.
- Cyclic predecessor chains are detected with a per-walk epoch stamp and discarded.
- Candidate paths are sorted by `(source, hop count)` and applied so each source contributes **at most one unit per iteration**. (Sorting by Dijkstra distance instead would break the non-negativity invariant relied on by the convex solver, so hop count is used.)
- For an applied path: push a unit of flow on every arc, then increment the sink's excess and decrement the source's excess.

### 7.5 Potential Update

After augmentation, potentials are updated by subtracting the shortest-path distance:

$$\pi_v \gets \pi_v - d_v \quad\text{(popped nodes)}$$

For nodes **not** popped this iteration (unreached, or relaxed-but-not-finalized under early-exit) the effective distance is **capped at** $d_{\max}$, the largest distance among popped nodes:

$$\pi_v \gets \pi_v - d_{\max} \quad\text{(non-popped nodes)}$$

This cap keeps the potentials valid (Ahuja, Magnanti & Orlin §9): without it, residual arcs crossing the Dijkstra search frontier would acquire negative reduced cost on the next iteration, producing cyclic predecessor chains. In full-completion mode every reachable node is popped, so $d_{\max}$ is never applied and every node receives its exact distance - matching Python's `update_potential_pd` and giving the tight reduced costs that let each iteration route more flow (closing a ~5.5% quality gap on masked single-tile scenes).

### 7.6 Fallback to Successive Shortest Paths

After the primal-dual loop terminates - either on hitting `max_iter` or on a no-progress stall - any remaining excess is drained by a **successive shortest paths (SSP)** fallback. The shared driver chooses the fallback by PD mode: early-exit `run` falls through to multi-source `ssp::run`, while full-completion `run_full_dijkstra` falls through to single-source `ssp::run_single_source`.

Each **multi-source SSP** iteration:

1. Runs an early-exit multi-source Dijkstra over the residual graph - every positive-excess node is seeded at distance 0, and the search stops after all currently reachable deficits are finalized.
2. Selects the single reached deficit node with the smallest distance, and traces predecessor arcs back to its source seed.
3. Augments **one unit** of flow along that **one** source→deficit path, adjusting the two endpoints' excess.
4. Applies the same potential update as the primal-dual phase, $\pi_i \gets \pi_i - d_i$ (capped at $d_{\max}$ for nodes not finalized by an early-exit Dijkstra), keeping reduced costs non-negative.

A safety counter caps the loop at $4|V|$ iterations and asserts convergence (panicking with `"SSP did not converge"` otherwise).

The **single-source SSP** variant instead snapshots the current source list, skips any source already drained by an earlier augmentation, runs Dijkstra from that one source only, stops when the first deficit is popped, and augments that one path. Its potential update is written in the equivalent capped form `π += d_sink - dist[v]` for popped nodes while unpopped nodes keep a zero shift. This is valid only when the entry potentials already have non-negative reduced costs, which the full-completion PD path provides.

Because one Dijkstra search is re-run for *every single unit* of augmentation, SSP is correct but expensive: its cost scales with the residual flow $F$ left after the primal-dual phase. The multi-source variant is appropriate for tiled/small graphs but can approach a near-whole-image search per unit on NISAR-scale single-tile graphs. The single-source variant avoids that case by searching from one source only and stopping at the first popped deficit; it is used only after full-completion PD.

#### 7.6.1 Masked-frame stranding and the guarded adaptive fallback

On **heavily-masked** frames (e.g. NISAR D_074 at ~6 % valid), the masked "sea" is a vast cost-0 region (both-invalid arcs cost 0 and are not forbidden, matching ww-orig). The single-source SSP processes its source list once and augments greedily one source at a time; on such frames this can **fragment the residual graph** so that a few remaining excess nodes end up trapped in tiny residual components that contain no deficit. Those sources are then **stranded** - the network never reaches balance, and the leftover ±2π discontinuities corrupt large regions of the integrated phase. (The single-tile parity path bisects cleanly to this: residues and costs are byte-identical to ww-orig - feeding ww-orig's exact costs through the Rust solver reproduces the Rust output exactly - so the divergence is purely in the flow the solver builds.) Python `ww-orig` does **not** hit this: it runs `primal_dual(maxiter=8)` and its SSP completes; the Rust single-source SSP's greedy one-pass does not, on these frames.

The fix is a **guarded adaptive fallback** in `run_full_dijkstra` (used only by `unwrap_linear`): run the usual PD(8) + SSP; if any excess remains, **resume the multi-source primal-dual** (which does not fragment the residual graph the way the single-source SSP does) in chunks, retrying the SSP each round, up to a cap (`WHIRLWIND_LINEAR_PD_CAP`, default 512 iterations). Because the first SSP already drains the easy residues, the resume typically finishes in ~16 more PD iterations. The final imbalance is always reported. Crucially the order matters: SSP-first-then-resume-PD converges far faster than running many PD iterations up front (which on D_075 left the network unbalanced after 300 iterations / 415 s, whereas PD(8)+SSP+resume balances in ~90 s). This is a **robustness lever, not literal ww-orig parity** - ww-orig fixes the same frames in a single PD(8)+SSP pass; the remaining trajectory difference (why the Rust single-source SSP strands where ww-orig's completes) is documented but not yet closed. The change is confined to the parity path; the production reuse/tiled solvers are untouched.

### 7.7 Complexity

- **Primal-dual phase**: $O(k \cdot (|E| + |V| \log |V|))$ where $k$ is the number of iterations
- **SSP fallback**: $O(F \cdot \text{Dijkstra search})$, where $F$ is the residual flow remaining after the primal-dual phase. Multi-source SSP can approach one near-global search per unit on large graphs; single-source SSP usually explores only the neighborhood from one source to its nearest deficit.
- **Space**: $O(|V| + |E|)$

In practice the runtime depends on how much flow reaches SSP. On D_077, SSP does the bulk of the final routing, so the single-source fallback is the runtime lever for the verified single-tile path. `run_no_ssp` is useful for diagnostics, not for parity or production quality.

---

## 8. Phase Integration

### 8.1 Overview

After the min-cost-flow solve, every pixel-grid edge carries an **integer cycle correction** (the net residue-arc flow across that edge). Integration turns those per-edge corrections back into an absolute unwrapped phase field. Rather than accumulate a running floating-point phase, whirlwind tracks an **integer cycle count** `K[p]` per pixel and emits

$$
\phi_{\text{unwrapped}}[p] = \psi[p] + 2\pi \cdot K[p],
$$

where $\psi[p]$ is the wrapped input. Carrying `K` as an integer is what keeps the output exactly congruent to the wrapped input (see §8.6). Implemented in `crates/whirlwind-core/src/integrate.rs`.

### 8.2 Integration Strategy

There are two paths, selected by whether a validity mask is supplied:

- **Unmasked fast path** (`integrate`, `integrate_with_flow`). Seeds `K = 0` at $(0,0)$ and sweeps the whole pixel grid in fixed order - down column 0, then left-to-right across each row - so every pixel is reached exactly once. `K` is propagated as a single running integer (the column-0 head count plus a per-row interior count).
- **Masked path** (`integrate_with_mask`, `integrate_with_flow_masked`). 4-connected BFS over valid pixels. The grid is swept in raster order; at the first still-unvisited valid pixel of **each** connected component a fresh BFS is started with `K = 0`. Every disconnected valid component is therefore integrated independently, and the absolute $2\pi$ offset between components is left unconstrained (it is unobservable from the wrapped data). Masked pixels are returned as `NaN`; the `is_nan()` state doubles as the BFS visited-marker.

`integrate_with_mask` with a `None` mask delegates to the unmasked `integrate`.

### 8.3 Integer Cycle Offset

Integration never accumulates a floating-point wrapped difference. Instead it adds, per edge, the integer number of $2\pi$ cycles needed to bring the raw phase difference into the principal interval:

$$
N(a, b) = -\,\mathrm{round}\!\left(\frac{a - b}{2\pi}\right), \qquad \mathrm{wrap}(a-b) = (a-b) + 2\pi\,N(a,b).
$$

`N(a,b) \in \{-1, 0, +1\}` for $a, b \in [-\pi, \pi]$ (function `wrap_n_cycle`). With Rust's round-half-away-from-zero, the implied wrap interval is $[-\pi, \pi)$; this `round` convention is shared with the cost pipeline's `wrap`.

### 8.4 Flow Extraction

Each pixel step adds `N` (§8.3) plus the **net integer flow** across the residue-graph edge that the step crosses. The crossing edge for a *horizontal* pixel step is the column of DOWN/UP residue arcs; for a *vertical* pixel step it is the row of RIGHT/LEFT residue arcs:

```text
// Horizontal pixel step (i, j-1) -> (i, j):
fwd = g.down_arc(i, j)       // forward (down) arc index
rev = g.up_arc(i + 1, j)     // reverse (up) arc index
net_flow = arc_flow(fwd) - arc_flow(rev)

// Vertical pixel step (i-1, 0) -> (i, 0):   [column-0 head in the fast path]
fwd = g.right_arc(i, 0)
rev = g.left_arc(i, 1)
net_flow = arc_flow(rev) - arc_flow(fwd)
```

`down_arc/up_arc/right_arc/left_arc` return `Option<usize>` forward-arc indices into the residual graph; `Network::arc_flow` reads the unit-capacity flow on an arc. In the masked BFS the same edge is used for both directions of traversal with the sign flipped (e.g. a RIGHT-neighbor step adds `arc_flow(fwd) - arc_flow(rev)`, a LEFT-neighbor step adds `arc_flow(rev) - arc_flow(fwd)`).

For the PHASS-style reuse solver, `integrate_with_flow` / `integrate_with_flow_masked` read **multi-unit signed flow** from an `&[i32]` slab indexed by forward-arc id (`flow[fwd]`) instead of `Network::arc_flow`, since that solver can route $|f| > 1$ on a single arc.

### 8.5 Integration Algorithm

```text
// Unmasked fast path (integrate)
col0_cycles = 0
for i in 0..m:
    if i > 0:                                   // vertical step down column 0
        col0_cycles += wrap_n_cycle(psi[i,0], psi[i-1,0])
                     + ( arc_flow(left_arc(i,1)) - arc_flow(right_arc(i,0)) )
    cycles = col0_cycles
    for j in 0..n:
        if j > 0:                               // horizontal step across the row
            cycles += wrap_n_cycle(psi[i,j], psi[i,j-1])
                    + ( arc_flow(down_arc(i,j)) - arc_flow(up_arc(i+1,j)) )
        unw[i,j] = psi[i,j] + 2*pi * (cycles as f32)
```

`cycles` (= `K`) is an `i32`; the unwrapped value is materialized once per pixel from the **integer** count. The masked path is the same recurrence carried along a BFS frontier, with a per-pixel `i32` cycle array and a fresh `K = 0` seed per connected component.

### 8.6 Numerical Precision

The output is exactly congruent to the wrapped input modulo $2\pi$, independent of image size. Because `K` is an integer, the only floating-point operations per pixel are the single multiply-and-add `psi[p] + 2*pi * (K as f32)`; error does **not** accumulate along the integration path. This is single precision (`f32`, `std::f32::consts::TAU`) - double-precision accumulation is unnecessary and is not used. The earlier float-accumulator integrator (`phi_accum += d_phi`, as in SNAPHU's original `IntegratePhase`) had error growing with path length and was replaced by this integer formulation (cf. the isce3/SNAPHU fix, commit fe6cba72). A regression test (`unwrap_is_congruent_to_wrapped_input`) asserts $|\mathrm{wrap}(\phi_{\text{unwrapped}} - \psi)| < 10^{-4}$ rad.

---

## 9. Implementation Details

### 9.1 Implementation Architecture

Whirlwind is implemented in **Rust**, with a small Python binding layer:

- **`crates/whirlwind-core`** (Rust): all algorithms - residue computation,
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
50-nat LLR cap, so the maximum integer cost is `6 x 50 = 300`. The diagnostic
parity path (`compute_carballo_costs_parity`) scales by `100` to match Python
`ww-orig`. (A separate `COST_SCALE = 100.0` constant is used by the CRLB and
convex cost builders, **not** by the production Carballo path.)

#### 9.3.2 Masked Regions

Masks (`true` = valid) are handled differently per stage and per entry point:

- **Residue compute** (`compute_with_mask`): zeros any interior residue whose
  2x2 pixel loop touches a masked pixel, and skips boundary-edge deposits with a
  masked endpoint (§4.2). Without this, `0+0j` masked pixels generate a wall of
  spurious residues at every mask boundary.
- **Network construction**: *two* mechanisms (§6.3). Arc-forbidding
  (`forbid_masked_arcs`) pre-saturates both directions of masked-edge arcs -
  used by the CRLB, convex, conncomp, ground and tiled paths. The **default
  solver `unwrap_linear` and the opt-in `unwrap_reuse` deliberately do
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
- **Memory**: `O(pixels)`; a whole-image solve of a 4176x4257 NISAR frame peaks
  at ≈6.4 GB RSS (the ≈72 M-arc residual network). Tiling bounds peak memory to
  tile scale.

### 9.5 Limitations and Assumptions

1. **2D only**: this ATBD covers the 2D unwrap; the 3D/time-series pipeline is in
   `ATBD-3d.md`.
2. **Statistical model**: assumes the Carballo/Lee cost model fits the data, and
   an accurate effective number of looks.
3. **Filter size**: 7x7 smoothing (Carballo's original used 5x5).
4. **Tiled robustness layer is heuristic (opt-in, not validated)**: the opt-in
   tiled path (`unwrap_tiled_robust`) - seam reconciliation, coarse anchor +
   multi-scale cascade, sliver healing, gated multi-shift re-solve - is
   empirically tuned against benchmark scenes, **not proven optimal**, and can
   produce invalid (fast-but-wrong) results on fragmented NISAR scenes. It
   carries environment
   escape hatches (`WHIRLWIND_NO_ANCHOR`, `WHIRLWIND_NO_HEAL`). Only the
   single-tile kernel (§3.1, §9.6) is verified.

### 9.6 Implementation Status, Verified Paths & Benchmarks

This subsection records what is *validated* versus *evolving*, and the
benchmark numbers behind the claims, so changes can be measured against a known
baseline.

**Entry point → solver / cost / mask map**

| Public fn                                                                                               | Network          | Cost                            | Dijkstra                                                                | Mask            | Status                                                                          |
| ------------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------- | ----------------------------------------------------------------------- | --------------- | ------------------------------------------------------------------------------- |
| `unwrap` **(public default → `unwrap_linear`)**                                                         | unit-capacity    | `compute_carballo_costs_parity` | full-completion, 8 it + single-source SSP + adaptive PD-resume (§7.6.1) | cost-zero + NaN | **verified (Python parity)** - default since 2026-06-03                         |
| `unwrap` *tiled path* (opt-in: `multilook>1`, explicit `tile_size`, or `WHIRLWIND_UNWRAP_SOLVER=tiled`) | reuse (per tile) | `compute_carballo_costs`        | early-exit, 50 it + multi-source SSP                                    | forbid (tiled)  | **NOT validated** - worked on select scenes, failed on most; no working version |
| `unwrap_reuse` (whole-image reuse)                                                                      | reuse            | `compute_carballo_costs`        | early-exit, 50 it                                                       | cost-zero + NaN | reaches its cost optimum; **not validated** as default (PHASS path, opt-in)     |
| `unwrap_linear` (single-tile; the default kernel)                                                       | unit-capacity    | `compute_carballo_costs_parity` | full-completion, 8 it + single-source SSP + adaptive PD-resume (§7.6.1) | cost-zero + NaN | **verified (Python parity)**; masked frames balanced via adaptive fallback      |
| `unwrap_convex`                                                                                         | convex           | `compute_snaphu_smooth_costs`   | heap                                                                    | forbid          | research prototype (#65)                                                        |
| `components_only`                                                                                       | unit-capacity    | `compute_carballo_costs`        | forbid                                                                  | no MCF solve    | -                                                                               |

**Single-tile benchmark (D_077, 4176x4257, vs production GUNW = SNAPHU).** The
single-tile kernel is both faster and more accurate than single-tile SNAPHU:

| Unwrapper (single tile)          | Runtime   | per-component match vs production      |
| -------------------------------- | --------- | -------------------------------------- |
| **whirlwind `unwrap_linear`**    | **≈37 s** | **99.49 %** (matches Python `ww-orig`) |
| SNAPHU (`cost=smooth, init=mcf`) | ≈588 s    | 99.30 %                                |
| PHASS                            | ≈19.6 s   | 94.7 %                                 |

Peak RSS ≈6.4 GB (no swap). Reference SNAPHU/PHASS timings are in
`snaphu_ref/D_077.log` / `phass_ref.log`. Benchmark the verified path with
`scripts/bench_nisar_gunw_whirlwind.py --solver linear --nlooks 16`.

**Full 13-frame NISAR GUNW sweep - whirlwind vs ww-orig vs PHASS vs ICU (2026-06-03).**
Per-component match vs the production GUNW unwrap (= snaphu), runtime, and peak
RSS, single-tile, one heavy unwrap at a time. `whirlwind` = the public `unwrap`
default (single-tile linear + the adaptive PD/SSP fallback of §7.6.1).
Reproduce with `scripts/sweep_all_unwrappers.sh` (→ `results.csv` + `SUMMARY.md`);
it drives `scripts/run_native_one.py` (whirlwind / ww-orig, in the whirlwind env)
and `scripts/tophu_compare.py` (PHASS / ICU / snaphu, in an isce3+tophu env).

| frame | ww %  | ww s | ww GB | ww-orig % | orig s | PHASS % | ph s | ICU %/s     |
| ----- | ----- | ---- | ----- | --------- | ------ | ------- | ---- | ----------- |
| A_013 | 100.0 | 13.5 | 3.2   | 100.0     | 35.5   | 99.3    | 6.3  | 100.0 / 525 |
| A_016 | 100.0 | 19.8 | 3.6   | 100.0     | 44.0   | 99.6    | 12.5 | -           |
| A_018 | 100.0 | 18.0 | 3.4   | 100.0     | 39.6   | 85.7    | 4.7  | -           |
| A_020 | 99.8  | 27.6 | 3.7   | 99.8      | 52.0   | 99.4    | 7.3  | -           |
| A_022 | 100.0 | 25.3 | 3.7   | 100.0     | 48.3   | 99.4    | 6.4  | -           |
| A_025 | 58.0  | 30.2 | 3.8   | **70.3**  | 54.5   | 67.0    | 5.6  | -           |
| A_028 | 100.0 | 33.4 | 3.6   | 100.0     | 55.3   | 92.9    | 11.5 | -           |
| A_030 | 100.0 | 35.5 | 4.1   | 100.0     | 63.1   | 75.4    | 6.1  | -           |
| D_074 | 98.8  | 18.5 | 3.5   | 98.8      | 37.4   | 91.2    | 5.7  | 86.3 / 0.6  |
| D_075 | 88.2  | 82.4 | 3.9   | 88.2      | 106.3  | 48.4    | 15.4 | -           |
| D_077 | 99.5  | 61.9 | 3.5   | 99.5      | 85.7   | 94.7    | 16.8 | -           |
| D_078 | 99.8  | 37.9 | 3.5   | 99.9      | 59.1   | 96.9    | 10.5 | -           |
| A_035 | 100.0 | 22.5 | 3.2   | 100.0     | 46.7   | 94.6    | 8.5  | -           |

Findings:
- **Parity: whirlwind ≡ ww-orig on 12 of 13 frames** (identical to ±0.1 %). The
  table's lone exception, **A_025** (58.0 here, bridge *off*), is now fixed to
  **99.99 %** by the default-on bridging pass (`unwrap(bridge=True)`) - so with the
  default settings whirlwind is at-or-above ww-orig on **all 13** (see the A_025
  bridging section below).
- whirlwind is **~1.5–2x faster than Python ww-orig** (Rust) at ~3–4 GB vs 4–5 GB,
  and single-tile snaphu is ~588 s - so whirlwind is **~7–45x faster than snaphu**
  while matching/beating its per-comp.
- whirlwind **beats PHASS on quality** on most frames, sometimes by a lot
  (D_075 88.2 vs 48.4; A_030 100 vs 75.4; A_028 100 vs 92.9; A_018 100 vs 85.7),
  though PHASS is ~2–4x faster (5–17 s) at ~2 GB.
- **The *isce3* ICU (via tophu) is impractical** - 525 s on the *easy* frame
  (the 0.6 s on D_074 is an artefact of its 94 %-masked tiny valid region);
  sampled, not swept. The **original *isce2* mroipac ICU** (a different engine)
  is a very different story - fast and competitive (see below).

**The original isce2 ICU (Giangi's `mroipac` C extension).** A separate, much
older implementation than the isce3/tophu ICU above - and the one most veteran
InSAR users will recognize. Run single-patch via `scripts/icu_isce2_run.py` in an
isce2 env, scored with the *same* `percomp_match`. ICU estimates coherence
internally (phase-sigma), so masked pixels are filled with **random** phase (not
zeroed - a constant region looks perfectly coherent and corrupts ICU's seeding).

| frame | isce2-ICU %/s  | coverage | whirlwind % | note                                               |
| ----- | -------------- | -------- | ----------- | -------------------------------------------------- |
| A_016 | 100.0 / 120    | 99.6 %   | 100.0       | matches                                            |
| D_074 | 100.0 / 157    | 99.9 %   | 98.8        | matches/beats                                      |
| D_077 | 96.2 / 133     | 80.7 %   | 99.5        | ICU drops low-coh land below its growing threshold |
| A_025 | **73.2** / 122 | 99.9 %   | **58.0**    | **ICU beats whirlwind on the river**               |

So the classic ICU is *not* "just bad": it is fast (~2–3 min), hits 100 % with
full coverage on clean frames, and on the A_025 river its tree/bootstrap
referencing recovers the cross-bank integer gauge better than our MCF (73 vs 58,
≈ ww-orig 70 / PHASS 67). That makes A_025 a *referencing* weakness, not a
fundamental one - and 73 % is the concrete target for the bridging work below to
beat. ICU's one weak spot is coverage on lower-coherence land (D_077 at 81 %),
which the MCF fills.

**A_025 - the residual bridging gap, now SOLVED (`unwrap(bridge=True)`, default on).**
A low-coherence/masked river split A_025 into disconnected slabs; each slab is
internally correct but the MCF integrator seeds every disconnected valid region at
its own arbitrary 2π level, so the *relative* offset across the river was
under-determined (whirlwind 0.580, ww-orig 0.703, PHASS ≈ 0.67). The fix is a fast
post-integration pass over the right partition:

- **The free gauge is between INTEGRATION components** - the 4-connected components
  of the valid mask, which is the partition `integrate_with_mask` seeds - *not* the
  (strictly finer) conncomp labels. (A_030 has 230 conncomps yet scores 100 %
  because the integrator already bridged them across low-cost cuts.) Putting a shift
  variable on conncomps was the old post-hoc dead-end; on integration components a
  single-region (or coherently connected) frame is a structural no-op.
- Each region is re-levelled to a coherent x8 coarse anchor, with the shift taken
  **relative to the largest region**, gated to regions the coarse scale connects
  (data-supported) and vetoed unless the offset is cleanly integer.
- Connected-components labelling is a native binding (`whirlwind.label_components`,
  the same BFS as `integrate_with_mask`) - **no scipy/scikit-image dependency**.

13-frame validation (`scripts/bench_bridge_all.py`, per-comp before/after + pixels
moved; bridge cost +0.5–1.0 s/frame):

|              | A_025                    | the other 12                                  |
| ------------ | ------------------------ | --------------------------------------------- |
| per-comp     | **58.0 → 99.99 %**       | unchanged (0 regressions)                     |
| pixels moved | 5.25 M (the offset slab) | 0 on 11/13; D_075 moved 0.1 % (score-neutral) |

So **whirlwind now matches/at-or-above ww-orig on all 13 frames.** Prototype +
diagnostics: `scripts/proto_bridge_a025.py`, `scripts/diag_bridge_partition.py`.
Future work: the wider, fully-decorrelated-gap case where x8 does *not* bridge (the
offset is then a labelled *convention*, not a measurement) - A_025's river was narrow
enough to be genuinely data-supported.

**SSP-fallback cost (a known sharp edge).** `unwrap_linear` runs 8 full-Dijkstra
PD iterations *then falls through to SSP* - and on D_077 it does reach SSP (the
PD iterations alone reach only ≈11 %; the SSP fallback routes the bulk). The SSP
fallback's runtime therefore dominates, and it depends critically on the SSP
*algorithm*:

- The multi-source `ssp::run` seeds every excess node, runs to
  all-deficits-popped, and augments **one** path per iteration -
  i.e. effectively a near-whole-image Dijkstra *per single unit of flow*. On the
  D_077 whole-image graph this costs ≈1472 s.
- A **single-source** SSP (early-exit per source) routes the same flow far
  faster (≈61 s before the rescan fix below; ≈37 s after it - see the next
  block).

The fast figure above is with the single-source SSP. **Dual-SSP fix
(implemented):** the multi-source `ssp::run` is kept for the early-exit/tiled
path (where it is fast - it is catastrophic only on large *whole-image* graphs),
and `ssp::run_single_source` is used only by `run_full_dijkstra` (single-tile),
restoring D_077 from ≈1472 s back to **≈61 s / 99.49 %**, then to **≈37 s** after
the per-source rescan elimination below (verified post-fix).
The single-source potential update keeps reduced costs non-negative after every
early-exit Dijkstra - popped nodes get their exact distance; unpopped nodes keep
a zero shift, which is exactly "cap at the sink distance" since any unpopped node
has `dist ≥ d_sink` by Dijkstra pop order - so `debug_assert!(rc >= 0)` holds
with **no clamp**. The invariant is guarded by the debug test
`single_source_ssp_keeps_nonnegative_reduced_costs` (a steep noisy ramp that
reaches the SSP fallback); the tiled/default path is byte-unchanged (only
`run_full_dijkstra`'s fall-through branches to the single-source variant).

**Per-source Dial-`k` rescan eliminated (2026-06-04, ~1.4–2.4x on residue-heavy
frames).** `WHIRLWIND_DEBUG` timing revealed the single-source SSP's real cost was
*not* the Dijkstra traversals but the **`max_reduced_cost_par` rescan it ran once
per source** to size the Dial buckets - an O(E) scan over ~38 M arcs x ~1k sources
= **34 s of D_077's 61 s (~52 %)**. A naive hoist is unsafe (the capped potential
update *grows* potentials, so the max reduced cost rises and a stale `k` would
alias). Fix: maintain `max_rc` across sources (one tight scan), and if any
relaxation sees `rc ≥ k`, discard that source's partial Dijkstra, recompute
`max_rc`, and retry - an under-sized `k` can never commit. **D_077 61→37 s, D_075
2.4x, others 1.4–1.7x; optimal cost byte-identical, per-comp unchanged on all 13
frames, 79/79 core tests green.** Profiler: `scripts/prof_pdssp.py`,
`WHIRLWIND_DEBUG=1` (`max_reduced_cost_scan=…ms`).

> **Tiling is not yet validated** on fragmented NISAR scenes (see §9.5 item 4).
> The single-tile kernel is the trustworthy reference to measure tiling against.

---

## 10. References

### Primary References

1. **Carballo, G. F.**, & Fieguth, P. W. (2002). "Probabilistic cost functions for network flow phase unwrapping." *IEEE Transactions on Geoscience and Remote Sensing*, 40(11), 2192-2203.

2. **Lee, J. S.**, Hoppel, K. W., Mango, S. A., & Miller, A. R. (1994). "Intensity and phase statistics of multilook polarimetric and interferometric SAR imagery." *IEEE Transactions on Geoscience and Remote Sensing*, 32(5), 1017-1028.

3. **Touzi, R.**, Lopes, A., Bruniquel, J., & Vachon, P. W. (1999). "Coherence estimation for SAR imagery." *IEEE Transactions on Geoscience and Remote Sensing*, 37(1), 135-149.

### Network Flow Algorithms

4. **Ahuja, R. K.**, Magnanti, T. L., & Orlin, J. B. (1993). *Network Flows: Theory, Algorithms, and Applications*. Prentice Hall.

5. **Goldberg, A. V.**, & Tarjan, R. E. (1990). "Finding minimum-cost circulations by successive approximation." *Mathematics of Operations Research*, 15(3), 430-466.

### Phase Unwrapping Background

6. **Ghiglia, D. C.**, & Pritt, M. D. (1998). *Two-Dimensional Phase Unwrapping: Theory, Algorithms, and Software*. Wiley.

7. **Chen, C. W.**, & Zebker, H. A. (2001). "Two-dimensional phase unwrapping with use of statistical models for cost functions in nonlinear optimization." *Journal of the Optical Society of America A*, 18(2), 338-351.

---

## Appendix A: Mathematical Notation

| Symbol                   | Description                                          |
| ------------------------ | ---------------------------------------------------- |
| $\psi$                   | Unwrapped phase                                      |
| $\phi$                   | Wrapped phase (measured)                             |
| $\hat{\psi}$             | Observed wrapped phase (following Carballo notation) |
| $\phi_s$                 | True signal phase                                    |
| $\phi_N$                 | Phase noise                                          |
| $k$, $\Delta k$          | Integer cycle ambiguity / correction                 |
| $\gamma$, $\hat{\gamma}$ | True coherence / sample coherence                    |
| $L$                      | Number of looks                                      |
| $r_{i,j}$                | Residue at node $(i,j)$                              |
| $c_{ij}$                 | Cost on arc $(i,j)$                                  |
| $\bar{c}_{ij}$           | Reduced cost: $c_{ij} - \pi_i + \pi_j$               |
| $f_{ij}$                 | Flow on arc $(i,j)$                                  |
| $\pi_i$                  | Potential (dual variable) at node $i$                |
| $b_i$                    | Supply/demand at node $i$ (equals residue)           |

---

## Appendix B: Algorithm Pseudocode

```python
def unwrap_linear_parity(igram, corr, nlooks, mask=None):
    """
    Verified single-tile parity path (`unwrap_linear`).

    Parameters
    ----------
    igram : array_like, complex
        Complex interferogram (m x n)
    corr : array_like, float
        Coherence values [0, 1] (m x n)
    nlooks : float
        Effective number of looks (≥ 1)
    mask : array_like, bool, optional
        Valid pixel mask (True = valid) (m x n)

    Returns
    -------
    unwrapped_phase : ndarray, float
        Unwrapped phase in radians (m x n)
    """
    # Stage 1: Extract wrapped phase
    phase = angle(igram)  # [-π, π]

    # Stage 2: Compute residues, then match Python ww-orig boundary semantics
    residue = compute_residues_unmasked(phase)  # (m+1) x (n+1)
    zero_boundary_frame(residue)

    # Stage 3: Compute ww-orig parity costs
    cost = compute_carballo_costs_parity(igram, corr, nlooks, mask)

    # Stage 4: Formulate and solve network flow
    graph = RectangularGridGraph(residue.shape)
    network = Network(graph, residue.flatten(), cost, capacity=1)
    run_full_dijkstra(network, maxiter=8)  # includes single-source SSP fallback

    # Stage 5: Integrate unwrapped gradients
    unwrapped_phase = integrate_unwrapped_gradients(phase, network)
    if mask is not None:
        unwrapped_phase[~mask] = NaN

    return unwrapped_phase
```

The public production `unwrap` adds the tiled/anchor/cascade layer described in
§3.2 around the per-tile reuse solver; the pseudocode above is the smaller
validated whole-image kernel used for parity and benchmarking.

---

## Appendix C: Cost Interpretation Examples

### Example 1: High coherence, small gradient

- Coherence: $\gamma = 0.9$
- Smooth phase gradient: $\hat{\alpha} = 0.1$ rad
- Number of looks: $L = 10$

The Carballo PDFs give:
- $P(\Delta k = 0) \approx 0.95$
- $P(\Delta k = +1) \approx 0.05$

Cost:
$$
c = -\ln\left(\frac{0.05}{0.95}\right) = -\ln(0.053) \approx 2.94
$$

This **high positive cost** penalizes adding a $2\pi$ cycle, which is correct since we're confident no correction is needed.

### Example 2: High coherence, large gradient

- Coherence: $\gamma = 0.9$
- Smooth phase gradient: $\hat{\alpha} = 3.0$ rad (near $\pi$)
- Number of looks: $L = 10$

The Carballo PDFs give:
- $P(\Delta k = 0) \approx 0.10$
- $P(\Delta k = +1) \approx 0.90$

Cost (before clamping):
$$
-\ln\left(\frac{0.90}{0.10}\right) = -\ln(9.0) \approx -2.20
$$

The raw LLR is negative (a $2\pi$ correction is favored near a wrapping
discontinuity), but **the implementation clamps every forward-arc cost to
$\ge 0$** - the "encourage a cut" signal is expressed as a *near-zero* cost on
the cut direction together with the asymmetric $\alpha\le 0\Rightarrow c_{\max}$
rule on the opposite direction (§5.4), not as a literal negative number.

### Example 3: Low coherence

- Coherence: $\gamma = 0.3$
- Any phase gradient
- Number of looks: $L = 10$

With low coherence, the PDFs become nearly equal:
- $P(\Delta k = 0) \approx P(\Delta k = +1)$

Cost:
$$
c \approx -\ln(1) = 0
$$

This **near-zero cost** means the edge doesn't strongly influence the solution, which is appropriate when we have low confidence in the measurement.

---

*Document Version: 3.0 - algorithm sections audited against the code (see §9.6 for verified-vs-WIP status and benchmarks).*  
*Last Updated: 2026-06-03*
