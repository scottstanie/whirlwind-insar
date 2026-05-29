# Whirlwind Algorithm Theoretical Basis Document (ATBD)

## Executive Summary

Whirlwind is a Bayesian minimum-cost network flow algorithm for 2D phase unwrapping of interferometric synthetic aperture radar (InSAR) data. The algorithm formulates phase unwrapping as a minimum-cost flow problem on a rectangular grid graph, where edge costs are derived from Bayesian probability densities that account for both coherence and local phase gradient statistics. The network flow problem is solved using a primal-dual algorithm, and the final unwrapped phase is obtained by integrating the unwrapped phase gradients.

> **Production architecture note.** The MCF + primal-dual core described in this ATBD (Sections 2–8) is correct and unchanged, but it is the **per-tile** engine, not the top-level method. The shipped default wraps it in a **tiled** pipeline: per-tile MCF → global coarse anchor (multilook the complex igram ×8, solve that tiny image whole — seam-free and runaway-free — upsample, and snap each region's integer 2π level to it by coherence-weighted mode) → multi-scale cascade (`coarse_refine` at f=16,8,4) → feathered seam composite. A whole-image MCF *runs away* on real noisy scenes (NISAR: 80 % K-match, 18 % multi-cycle); the tiled path reaches **99.79 % K-match, 0 % multi-cycle, 3.9 s** vs SNAPHU 9×9's ~17 min — no Goldstein filtering required. Noisy / moderate-coherence scenes (e.g. Sentinel-1) use a `multilook=L` down-look first. See [`paper/report_anchor_cascade.md`](paper/report_anchor_cascade.md) and [`TILING_DESIGN.md`](TILING_DESIGN.md). (committed e24e0ed / 8aa7a1d)

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

**Important principle from Geoff's notes**: "Don't rewrap!" — the algorithm works directly with wrapped gradients and integer cycle corrections, never rewrapping intermediate results.

### 2.4 Residues

A **residue** at a grid node is the sum of wrapped phase gradients around a 2×2 pixel loop, normalized by $2\pi$:

$$
r = \mathrm{round}\left(\frac{1}{2\pi} \oint \nabla \phi \cdot d\ell\right)
$$

Residues are topological defects indicating phase inconsistency. For any continuous phase field, the sum around a closed loop must be zero. Non-zero residues indicate that the wrapped gradients are inconsistent with any continuous unwrapping.

**Properties:**

- Residues are integers, typically in $\{-1, 0, +1\}$
- The sum of all residues over the image is zero (conservation)
- Positive residues act as flow **sources**, negative as **sinks** in the network formulation

---

## 3. Algorithm Overview

The Whirlwind algorithm consists of five main stages:

**Input:** Complex interferogram, coherence magnitude, number of effective looks, and (optionally) mask of invalid pixels

```
[1] Compute wrapped phase: φ = angle(igram)
[2] Detect residues: r = residue(φ)
[3] Compute Bayesian costs: c = carballo_costs(igram, corr, nlooks)
[4] Solve min-cost flow: primal_dual(network(r, c))
[5] Integrate unwrapped gradients: ψ = integrate(φ, flow)
```

**Output:** $\psi$ (unwrapped phase)

Each stage is described in detail in the following sections.

---

## 4. Residue Computation

### 4.1 Definition

Residues are computed on a grid of **nodes** with dimensions $(m+1) \times (n+1)$, where the input phase array has dimensions $m \times n$. Each interior node $(i,j)$ is surrounded by four pixels, and the residue measures the inconsistency of the phase around the 2×2 block.

### 4.2 Algorithm

The residue computation processes each 2×2 block of pixels and distributes the contribution to the surrounding nodes:

```cpp
// Initialize output array to zeros: (m+1) × (n+1)
residue = zeros(m+1, n+1)

// Helper: compute integer cycle difference
cycle_diff_residual(a, b) = round((a - b) / (2π))

// Process interior 2×2 blocks
for i = 0 to m-2:
    for j = 0 to n-2:
        φ_00 = wrapped_phase(i, j)
        φ_10 = wrapped_phase(i+1, j)
        φ_01 = wrapped_phase(i, j+1)

        // Upward difference (row i+1 to row i)
        di = cycle_diff_residual(φ_00, φ_10)
        // Leftward difference (col j+1 to col j)  
        dj = cycle_diff_residual(φ_01, φ_00)

        residue(i+1, j)   += di
        residue(i, j+1)   += dj
        residue(i+1, j+1) -= di + dj

// Process last column (j = n-1)
for i = 0 to m-2:
    d = cycle_diff_residual(φ(i, n-1), φ(i+1, n-1))
    residue(i+1, n-1) += d
    residue(i+1, n)   -= d

// Process last row (i = m-1)
for j = 0 to n-2:
    d = cycle_diff_residual(φ(m-1, j+1), φ(m-1, j))
    residue(m-1, j+1) += d
    residue(m, j+1)   -= d
```

### 4.3 Properties

- The sum of all residues is exactly zero: $\sum_{i,j} r_{i,j} = 0$
- Boundary nodes (edges of the grid) accumulate residues from incomplete loops
- The residue grid has one more row and column than the phase array

---

## 5. Bayesian Cost Function

### 5.1 Carballo Probability Model

The cost function is based on the work by Carballo & Fieguth (2002), which models the probability of different integer unwrapping corrections given the local phase gradient and coherence.

For each edge between adjacent pixels, we compute the **log-likelihood ratio**:

$$
c = -\ln\left(\frac{P(\Delta k = +1 \mid \hat{\alpha}, \hat{\gamma}, L)}{P(\Delta k = 0 \mid \hat{\alpha}, \hat{\gamma}, L)}\right)
$$

where:

- $\hat{\alpha}$ is the **smoothed local phase gradient** (estimated signal gradient)
- $\hat{\gamma}$ is the **sample coherence** (quality metric)
- $L$ is the **number of looks** (multi-looking parameter)

This cost represents how much more likely it is that no cycle correction is needed ($\Delta k = 0$) versus a positive correction ($\Delta k = +1$).

### 5.2 Smooth Phase Gradient Estimation

The local phase gradient is estimated using a **7×7 uniform filter** (note: Carballo's original paper used 5×5):

```python
# Compute complex phase differences
dy_igram = igram[1:, :] * igram[:-1, :].conj()  # Row differences
dx_igram = igram[:, 1:] * igram[:, :-1].conj()  # Column differences

# Extract phase of differences
phase_dy = angle(dy_igram)
phase_dx = angle(dx_igram)

# Smooth with 7×7 uniform filter
phase_dy_smooth = uniform_filter(phase_dy, size=(7, 7), mode='nearest')
phase_dx_smooth = uniform_filter(phase_dx, size=(7, 7), mode='nearest')
```

This provides an estimate of the underlying signal gradient $\Delta \phi_s$ by averaging over local phase noise.

**Key insight from Geoff's notes**: The algorithm uses the *smoothed* phase gradient, not the measured phase gradient of individual edges. This is the "average local phase gradient" that estimates $\Delta \phi_s$.

### 5.3 Coherence-Based Quality

The coherence for each edge is the **minimum** of the two adjacent pixels:

$$
\gamma_{\text{edge}} = \min(\gamma_{\text{pixel}_1}, \gamma_{\text{pixel}_2})
$$

This conservative choice ensures that low-quality pixels appropriately reduce confidence in the edge.

### 5.4 PDF Lookup via B-Splines

The probabilities $P(\Delta k = 0)$ and $P(\Delta k = +1)$ are pre-computed from the Carballo PDF model and stored as **tri-cubic B-spline** interpolants. The splines are functions of three variables:

- **Phase difference**: $\hat{\alpha} \in [-\pi, \pi]$
- **Coherence**: $\hat{\gamma} \in [0, 1]$
- **Number of looks**: $L > 0$

The B-splines (`carballo-pdf-0-spline.pkl` and `carballo-pdf-1-spline.pkl`) provide fast, smooth interpolation of these probability densities.

### 5.5 Cost Computation for Four Directions

For the rectangular grid, costs are computed for arcs in four directions. The sign of the phase gradient input to the PDF depends on the arc direction:

```python
# Load pre-computed Carballo PDF splines
spline_pdf0, spline_pdf1 = load_carballo_pdf_splines()

def compute_cost(phase_diff, min_corr):
    p1 = spline_pdf1((phase_diff, min_corr, nlooks))
    p0 = spline_pdf0((phase_diff, min_corr, nlooks))
    return -log(p1 / p0)

# Costs for four arc directions
cost_up = compute_cost(-phase_dx_smooth, corr_dx)  # Upward arcs
cost_lt = compute_cost(phase_dy_smooth, corr_dy)   # Leftward arcs
cost_dn = compute_cost(phase_dx_smooth, corr_dx)   # Downward arcs
cost_rt = compute_cost(-phase_dy_smooth, corr_dy)  # Rightward arcs
```

The sign negations account for the fact that traversing an edge in the opposite direction corresponds to the negative of the phase gradient.

### 5.6 Cost Interpretation

| Coherence | Phase Gradient | Cost Behavior | Interpretation |
|-----------|---------------|---------------|----------------|
| High | Small | Large positive | Confident that $\Delta k = 0$; penalize corrections |
| High | Large (near $\pm\pi$) | Small or negative | Confident that $\Delta k = \pm 1$; encourage correction |
| Low | Any | Near zero (symmetric) | Uncertain; edge shouldn't strongly influence solution |

From Geoff's notes: When cost is zero, we are equally confident in $\Delta k \in \{-1, 0, +1\}$, so the arc doesn't contribute much to the total cost.

---

## 6. Network Flow Formulation

### 6.1 Graph Construction

The phase unwrapping problem is formulated as a **minimum-cost flow problem** on a rectangular grid graph:

- **Nodes**: $(m+1) \times (n+1)$ grid (residue locations)
- **Arcs**: 4-connected directed arcs (up, down, left, right) between adjacent nodes
- **Supply/Demand**: Node $i$ has supply $b_i = r_i$ (the residue value)
- **Capacity**: Each forward arc has capacity 1 (unit capacity)
- **Cost**: Forward arc has cost $c_{ij} \geq 0$; reverse arc has cost $-c_{ij}$

### 6.2 Residual Graph

The algorithm operates on a **residual graph** where:

- Each edge in the original graph becomes two directed arcs (forward and reverse)
- Forward arcs have non-negative costs from the Carballo model
- Reverse arcs have the negation of the forward arc cost

This allows the algorithm to "undo" flow decisions by pushing flow on reverse arcs.

### 6.3 Flow Interpretation

The net flow $f$ on an edge represents the integer correction to the phase gradient:

$$
\Delta \psi_{\text{unwrapped}} = \Delta \phi_{\text{wrapped}} + 2\pi \cdot f
$$

where $f$ can be positive (add cycles), negative (subtract cycles), or zero.

### 6.4 Minimum-Cost Flow Problem

The objective is to find flows that satisfy supply/demand constraints at minimum total cost:

$$
\min_{f} \sum_{(i,j) \in \text{forward arcs}} c_{ij} \cdot f_{ij}
$$

subject to:

- **Flow conservation**: Net flow into each node equals its demand (negative of residue)
- **Capacity constraints**: $0 \leq f_{ij} \leq 1$ for forward arcs

### 6.5 Why This Works

- Positive residues (sources) must export flow; negative residues (sinks) must import flow
- The minimum-cost solution finds paths that pair sources with sinks using the statistically most probable corrections
- Flow on an edge indicates that a $2\pi$ correction is applied to that phase gradient
- Neutralizing all residues ensures the unwrapped phase is **path-independent**

---

## 7. Primal-Dual Solution

### 7.1 Algorithm Overview

The primal-dual algorithm solves the min-cost flow problem through iterative shortest-path computations:

1. **Initialization**: All flows zero, all potentials zero
2. **Iteration** (repeat until no excess nodes remain):
   - Run Dijkstra from **all excess nodes simultaneously** using reduced costs
   - **Augment flow** along shortest paths from sources to sinks
   - **Update potentials** to maintain non-negative reduced costs
3. **Fallback**: After `maxiter` iterations (default 8), switch to successive shortest paths if excess remains

### 7.2 Reduced Costs

The **reduced cost** of an arc from node $i$ (tail) to node $j$ (head) is:

$$
\bar{c}_{ij} = c_{ij} - \pi_i + \pi_j
$$

where $\pi_i$ is the potential at node $i$. This is the standard convention from network flow theory (Ahuja et al., 1993).

Reduced costs have the key property that after potential updates, all reduced costs on residual arcs with positive capacity remain non-negative, enabling the use of Dijkstra's algorithm.

### 7.3 Multi-Source Dijkstra

The primal-dual variant runs Dijkstra from multiple sources simultaneously:

```cpp
dijkstra_pd(dijkstra, network):
    // Add all excess nodes as sources
    for each source in network.excess_nodes():
        dijkstra.add_source(source)  // distance = 0

    // Standard Dijkstra relaxation
    while not dijkstra.done():
        (tail, distance) = dijkstra.pop_next_unvisited_vertex()
        dijkstra.visit_vertex(tail, distance)

        for each (arc, head) in network.outgoing_arcs(tail):
            if not network.is_arc_saturated(arc):
                arc_length = network.arc_reduced_cost(arc, tail, head)
                dijkstra.relax_edge(arc, tail, head, distance + arc_length)
```

This finds shortest paths from *any* source to *all* reachable nodes.

### 7.4 Flow Augmentation

After Dijkstra, flow is augmented from sources to deficit nodes:

```cpp
augment_flow_pd(network, dijkstra):
    // For each deficit node, find its nearest source and augment
    for each sink in deficit_nodes (one per source):
        network.increase_node_excess(sink, 1)  // Satisfy demand

        // Push flow along shortest path from source to sink
        for each (tail, arc) in dijkstra.predecessors(sink):
            network.increase_arc_flow(arc, 1)

        network.decrease_node_excess(source, 1)  // Consume supply
```

Each iteration can satisfy multiple source-sink pairs simultaneously.

### 7.5 Potential Update

After augmentation, potentials are updated to maintain optimality conditions:

$$
\pi_i \gets \pi_i - d_i
$$

where $d_i$ is the shortest path distance to node $i$ from any source. This ensures all reduced costs remain non-negative for the next iteration.

### 7.6 Fallback to Successive Shortest Paths

After `maxiter` primal-dual iterations (default 8), if excess nodes remain, the algorithm switches to the **successive shortest paths** algorithm, which processes one source at a time until all flow is routed.

### 7.7 Complexity

- **Primal-dual phase**: $O(k \cdot |V| \log |V|)$ where $k$ is the number of iterations
- **SSP fallback**: $O(|E| \cdot |V| \log |V|)$ worst case
- **Space**: $O(|V| + |E|)$

In practice, the primal-dual phase handles most of the flow, making the algorithm efficient.

---

## 8. Phase Integration

### 8.1 Overview

After solving the network flow problem, we have the **integer corrections** for each phase gradient. The final step integrates these corrected gradients to obtain the unwrapped phase field.

### 8.2 Integration Strategy

Starting from a **seed point** at $(0,0)$, the algorithm:

1. Integrates down the first column
2. Integrates across each row from left to right

This ensures every pixel is reached exactly once.

### 8.3 Wrapped Difference Function

The wrapped difference between two phase values is:

$$
\Delta \phi_{\text{wrapped}} = a - b - 2\pi \cdot \mathrm{round}\left(\frac{a - b}{2\pi}\right)
$$

This maps the difference to the interval $[-\pi, \pi)$.

### 8.4 Flow Extraction

For each edge between adjacent pixels, the integer correction is the **net flow** between the two bordering residue nodes:

```cpp
// For vertical edges (between rows i-1 and i, column j):
// The bordering nodes are (i, j) and (i, j+1)
node0 = Vertex(i, j)
node1 = Vertex(i, j+1)
// Net leftward flow determines the correction
arc_right = get_right_edge(node0)  // node0 → node1
arc_left = get_left_edge(node1)    // node1 → node0
net_flow = arc_flow(arc_left) - arc_flow(arc_right)

// For horizontal edges (row i, between columns j-1 and j):
// The bordering nodes are (i, j) and (i+1, j)
node0 = Vertex(i, j)
node1 = Vertex(i+1, j)
// Net downward flow determines the correction
arc_down = get_down_edge(node0)  // node0 → node1
arc_up = get_up_edge(node1)      // node1 → node0
net_flow = arc_flow(arc_down) - arc_flow(arc_up)
```

### 8.5 Integration Algorithm

```cpp
// Seed point
unwrapped_phase(0, 0) = wrapped_phase(0, 0)

// Integrate down first column
φ_accum = unwrapped_phase(0, 0)
for i = 1 to m-1:
    Δφ_wrapped = wrapped_diff(wrapped_phase(i, 0), wrapped_phase(i-1, 0))
    
    // Get nodes bordering this vertical edge
    node0 = Vertex(i, 0)
    node1 = Vertex(i, 1)
    net_flow = arc_flow(get_left_edge(node1)) - arc_flow(get_right_edge(node0))
    
    Δφ_unwrapped = Δφ_wrapped + 2π * net_flow
    φ_accum += Δφ_unwrapped
    unwrapped_phase(i, 0) = φ_accum

// Integrate across each row
for i = 0 to m-1:
    φ_accum = unwrapped_phase(i, 0)
    
    for j = 1 to n-1:
        Δφ_wrapped = wrapped_diff(wrapped_phase(i, j), wrapped_phase(i, j-1))
        
        // Get nodes bordering this horizontal edge
        node0 = Vertex(i, j)
        node1 = Vertex(i+1, j)
        net_flow = arc_flow(get_down_edge(node0)) - arc_flow(get_up_edge(node1))
        
        Δφ_unwrapped = Δφ_wrapped + 2π * net_flow
        φ_accum += Δφ_unwrapped
        unwrapped_phase(i, j) = φ_accum
```

### 8.6 Numerical Precision

The integration uses **double precision accumulation** to minimize numerical error, even when the output array is single precision.

---

## 9. Implementation Details

### 9.1 Implementation Architecture

Whirlwind is implemented in **Rust**, with a small Python binding layer:

- **`crates/whirlwind-core`** (Rust): All algorithms — residue computation,
  cost build, min-cost flow solver, integration, synthetic-ifg simulator.
  Parallelism via `rayon`.
- **`crates/whirlwind-cli`** (Rust): `whirlwind` binary, `simulate` and
  `unwrap` subcommands.
- **`crates/whirlwind-py`** (`pyo3`/`maturin`): Python bindings, importable
  as `whirlwind`. Exposes `unwrap`, `compute_residues`, `simulate_ifg`,
  `wrap_phase`, `diagonal_ramp`.

### 9.2 Key Data Structures

#### 9.2.1 Rectangular Grid Graph

```rust
pub struct RectangularGridGraph {
    pub num_rows: usize,  // number of node rows  (= pixel rows + 1)
    pub num_cols: usize,  // number of node cols  (= pixel cols + 1)
    // 4-connected: each node has up to 4 neighbors.
    // Residual graph has 2 parallel arcs per edge (forward + reverse).
}
```

#### 9.2.2 Network

```rust
pub struct Network<'a> {
    graph: &'a RectangularGridGraph,
    pub excess: Vec<i32>,       // b_i (supply/demand = residues)
    pub potential: Vec<i64>,    // π_i (dual variables, i64 to avoid overflow)
    pub cost_fwd: Vec<i32>,     // c_ij for forward arcs (reverse = -fwd)
    pub is_saturated: BitVec,   // unit-capacity flow stored as a bitvec
}
```

#### 9.2.3 Dijkstra Variants

The algorithm supports two Dijkstra implementations:

- **Dial's algorithm**: For integer costs, uses bucket queue for O(1) decrease-key
- **Binary heap Dijkstra**: For real-valued costs

### 9.3 Numerical Considerations

#### 9.3.1 Cost Scaling

Costs are scaled by `COST_SCALE = 100.0` and stored as `i32`, which lets Dial's
bucket-queue Dijkstra run in `O(1)` per relax. The scale factor balances
quantization error (≤ 0.005 per arc) against the largest bucket index Dial
needs to allocate.

#### 9.3.2 Masked Regions

Pixel-grid masks (`true` = valid) are propagated into both the residue and
network stages:

- **Residue compute** zeros any residue whose 2×2 pixel loop touches a masked
  pixel — without this, the arbitrary phase values in masked regions (typically
  `igram = 0 + 0j` from upstream `nan_to_num`) generate a wall of spurious
  residues at every mask boundary that dominate the MCF problem.
- **Network construction** pre-saturates every arc that crosses an invalid
  pixel-edge, so Dijkstra skips those arcs entirely. This is strictly stronger
  than just setting edge costs high — saturated arcs are removed from the
  residual graph rather than merely penalised.
- **Integration** BFS-walks the valid region from the first valid pixel and
  leaves masked pixels as `NaN` in the output.

### 9.4 Performance Characteristics

- **Typical runtime**: $O(mn \log(mn))$ for $m \times n$ image
- **Memory**: $O(mn)$ for network storage
- **Bottleneck**: Dijkstra shortest path computations in primal-dual iterations

### 9.5 Limitations and Assumptions

1. **2D only**: Current implementation is limited to 2D phase unwrapping
2. **Rectangular grid**: Assumes regular grid structure
3. **Unit capacity**: Each edge can carry at most 1 unit of flow per direction
4. **Statistical model**: Assumes Carballo PDF model is appropriate for the data
5. **Number of looks**: Requires accurate estimate of effective number of looks
6. **Filter size**: Uses 7×7 smoothing window (differs from Carballo's original 5×5)

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

| Symbol | Description |
|--------|-------------|
| $\psi$ | Unwrapped phase |
| $\phi$ | Wrapped phase (measured) |
| $\hat{\psi}$ | Observed wrapped phase (following Carballo notation) |
| $\phi_s$ | True signal phase |
| $\phi_N$ | Phase noise |
| $k$, $\Delta k$ | Integer cycle ambiguity / correction |
| $\gamma$, $\hat{\gamma}$ | True coherence / sample coherence |
| $L$ | Number of looks |
| $r_{i,j}$ | Residue at node $(i,j)$ |
| $c_{ij}$ | Cost on arc $(i,j)$ |
| $\bar{c}_{ij}$ | Reduced cost: $c_{ij} - \pi_i + \pi_j$ |
| $f_{ij}$ | Flow on arc $(i,j)$ |
| $\pi_i$ | Potential (dual variable) at node $i$ |
| $b_i$ | Supply/demand at node $i$ (equals residue) |

---

## Appendix B: Algorithm Pseudocode

```python
def unwrap(igram, corr, nlooks, mask=None):
    """
    Whirlwind phase unwrapping algorithm.

    Parameters
    ----------
    igram : array_like, complex
        Complex interferogram (m × n)
    corr : array_like, float
        Coherence values [0, 1] (m × n)
    nlooks : float
        Effective number of looks (≥ 1)
    mask : array_like, bool, optional
        Valid pixel mask (True = valid) (m × n)

    Returns
    -------
    unwrapped_phase : ndarray, float
        Unwrapped phase in radians (m × n)
    """
    # Stage 1: Extract wrapped phase
    phase = angle(igram)  # [-π, π]

    # Stage 2: Compute residues
    residue = compute_residues(phase)  # (m+1) × (n+1)

    # Stage 3: Compute Bayesian costs
    cost = compute_carballo_costs(igram, corr, nlooks, mask)

    # Stage 4: Formulate and solve network flow
    graph = RectangularGridGraph(residue.shape)
    network = Network(graph, residue.flatten(), cost, capacity=1)
    primal_dual(network, maxiter=8)

    # Stage 5: Integrate unwrapped gradients
    unwrapped_phase = integrate_unwrapped_gradients(phase, network)

    return unwrapped_phase
```

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

Cost:
$$
c = -\ln\left(\frac{0.90}{0.10}\right) = -\ln(9.0) \approx -2.20
$$

This **negative cost** encourages adding a $2\pi$ cycle, which is correct for a large phase gradient near a wrapping discontinuity.

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

*Document Version: 2.0*  
*Last Updated: 2026-05-24*
