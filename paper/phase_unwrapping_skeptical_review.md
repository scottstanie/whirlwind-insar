# Phase Unwrapping Literature: Skeptical Internal Review

Last checked: 2026-05-27.

Purpose: give us enough command of the phase-unwrapping literature to decide
what to cite, what to group aggressively, and what not to overclaim in the
GRSL-length paper. This is not intended to become a paper section.

## Executive Takeaway

The defensible literature positioning is narrow:

1. Do not claim a new phase-unwrapping formulation in the broad sense. MCF,
   statistical network-flow costs, graph cuts, and 3D/time-series unwrapping
   are all old and well-cited.
2. Do claim that `whirlwind-rs` combines a production-quality 2D MCF solve
   with phase-linking-specific information that the standard tools do not use:
   CRLB-derived per-pixel variance, exact temporal-closure diagnostics, and
   boundary-aware residue handling.
3. Treat SNAPHU as the serious baseline. It is still the open, operational
   reference for SAR interferograms.
4. Treat spurt/EMCF as the serious time-series comparator. It attacks a
   related 3D problem, but via the classic separable temporal-then-spatial EMCF
   decomposition rather than our per-IG CRLB MCF plus closure diagnostic.
5. Treat PUMA/Kamui, scikit-image, Tophu, PHASS/ICU, and newer ML work as
   relevant context, not as main comparators unless a reviewer asks.

Compressed paper sentence:

> Phase unwrapping methods range from residue branch-cut and quality-guided
> path-following approaches, through least-squares and graph-cut formulations,
> to network-flow methods for SAR interferograms. The dominant operational
> SAR baseline remains SNAPHU, which uses statistical-cost nonlinear
> network-flow optimization. Time-series extensions such as 3D unwrapping and
> EMCF exploit temporal redundancy, but are generally separable from the
> phase-linking uncertainty model. Our contribution is therefore not a new
> global objective, but a CRLB-weighted, boundary-aware implementation of
> residue-grid MCF together with exact temporal-closure diagnostics for
> phase-linked stacks.

## Citation Set For A Five-Page Paper

Minimum citation cluster:

- Goldstein, Zebker, Werner 1988: residue branch cuts and the InSAR origin
  story. DOI: https://doi.org/10.1029/RS023i004p00713
- Ghiglia and Romero 1994, or Ghiglia and Pritt 1998: least-squares /
  minimum-norm background. DOI: https://doi.org/10.1364/JOSAA.11.000107
- Costantini 1998: network-programming / MCF phase unwrapping. DOI:
  https://doi.org/10.1109/36.673674
- Chen and Zebker 2000: network view, intractability of exact L0, two
  algorithms. DOI: https://doi.org/10.1364/JOSAA.17.000401
- Chen and Zebker 2001/2002: SNAPHU statistical-cost nonlinear network-flow
  model and large-interferogram implementation. DOIs:
  https://doi.org/10.1364/JOSAA.18.000338 and
  https://doi.org/10.1109/TGRS.2002.802453
- Carballo and Fieguth 2000: probabilistic MCF edge costs. DOI:
  https://doi.org/10.1109/36.868876
- Bioucas-Dias and Valadao 2007: PUMA graph-cut formulation. DOI:
  https://doi.org/10.1109/TIP.2006.888351
- Hooper and Zebker 2007: 3D phase unwrapping for InSAR time series. DOI:
  https://doi.org/10.1364/JOSAA.24.002737
- Pepe and Lanari 2006: EMCF for multitemporal DInSAR. DOI:
  https://doi.org/10.1109/TGRS.2006.873207
- Shanker and Zebker 2010: edgelist multidimensional/time-series formulation.
  DOI: https://doi.org/10.1364/JOSAA.27.000605
- Ansari, De Zan, Bamler 2018 or Ansari, De Zan, Parizzi 2021: phase
  linking / modern stack phase estimation. DOIs:
  https://doi.org/10.1109/TGRS.2018.2826045 and
  https://doi.org/10.1109/TGRS.2020.3003421

If references are tight, cite Ghiglia/Pritt as the broad review/book, then
SNAPHU, Costantini, PUMA, Hooper/Zebker, Pepe/Lanari, and phase linking.

## Taxonomy

### 1. Branch cuts and residue pairing

Canonical example: Goldstein, Zebker, Werner 1988.

What they do: compute residues, place branch cuts between opposite-sign
residues or to boundaries, then integrate while avoiding the cuts.

Why reviewers care: this is the historical InSAR phase-unwrapping baseline.

Skeptical view: fast and intuitive, but local/cut-placement choices dominate
failure modes. Not a credible competitor for modern noisy wide-area stacks
unless heavily engineered.

Paper treatment: one clause only. "Early branch-cut methods..." is enough.

### 2. Least-squares / minimum-norm / Poisson-family methods

Canonical examples: Ghiglia and Romero 1994; Ghiglia and Pritt 1998.

What they do: solve for a phase field whose gradients best match wrapped
differences, often with weights and fast transforms or iterative solvers.

Strengths: fast, smooth, mathematically clean, useful outside InSAR.

Weaknesses for us: tends to spread residue errors into smooth bias rather than
placing explicit integer discontinuities. In SAR decorrelation, the integer
ambiguity structure matters.

Paper treatment: cite as part of the broad family, then move on.

### 3. Network programming / MCF

Canonical example: Costantini 1998.

What it does: formulates phase unwrapping as an integer network-programming
problem. Residues become supplies/demands, and flow encodes 2*pi gradient
corrections.

Why it matters: this is the lineage for our 2D solver. We should be explicit
that the residue-grid MCF formulation is not novel.

Skeptical view: strong global formulation, but cost generation and solver
engineering determine practical behavior. "MCF" alone is not a differentiator.

Paper treatment: cite directly when introducing the residue-grid flow model.

### 4. Statistical-cost nonlinear network flow: SNAPHU

Canonical examples: Chen and Zebker 2000, 2001, 2002. Public implementation:
https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/

What it claims:

- SNAPHU frames SAR phase unwrapping as MAP estimation.
- It uses topography, deformation, and smooth statistical models.
- It solves the posed nonlinear optimization approximately with network-flow
  techniques.
- Its public page states version 2.0.7 is the latest distribution as of
  February 2024, and memory is roughly 100 MB per 1 million pixels in
  single-tile mode.

Why it is the serious baseline:

- It is open enough for operational use.
- It is integrated into many SAR pipelines.
- It produces connected-component labels and is trusted by users.
- Dolphin, SNAP/ASF workflows, ISCE-style workflows, and other tools still
  commonly route through SNAPHU or wrappers.

What our code does differently:

- We solve a simpler linear unit-capacity MCF model, not SNAPHU's nonlinear
  statistical-cost tree-pivot optimizer.
- We batch residue pairing with multi-source primal-dual shortest-path passes.
- We use CRLB variance for phase-linked interferograms rather than sample
  coherence.
- We compute boundary residues explicitly and/or expose a ground-node variant.
- We emit exact temporal closure diagnostics for phase-linked stacks.

Skeptical view: reviewers can fairly ask why a simpler objective should beat
or replace SNAPHU. The answer is not "better statistical model in general."
The answer is: for phase-linked stacks, the input noise information is
different; our implementation uses that information and is much faster on the
tested cases while matching SNAPHU modulo 2*pi. SNAPHU remains the baseline,
not a straw man.

Paper treatment: cite SNAPHU in the introduction and benchmark section. Avoid
claiming SNAPHU does one shortest-path search per residue pair. Its source
shows a more nuanced modified network-simplex-style nonlinear optimizer with
MST/MCF initialization, source-specific tree solves, candidate negative
reduced-cost arcs, sorting, and pivots.

### 5. Probabilistic cost generators

Canonical example: Carballo and Fieguth 2000. Related phase-noise statistics:
Lee et al. 1994.

What they do: derive MCF edge weights from SAR phase statistics, coherence,
and local slope rather than ad hoc reliability weights.

Why it matters: this is the direct ancestry of Whirlwind's coherence-cost
path. The current `unwrap_crlb` keeps the same intuition, but changes the
weight source from sample coherence to phase-linking variance.

Skeptical view: this is not our novelty. It is the correct prior art for the
cost-shape idea. We should cite it and say we replace the uncertainty weight.

Paper treatment: one method paragraph.

### 6. Graph cuts / PUMA

Canonical example: Bioucas-Dias and Valadao 2007. Current Python implementation
to be aware of: Kamui, https://github.com/yoyolicoris/kamui

What it claims:

- PUMA formulates phase unwrapping as first-order MRF energy minimization.
- For convex clique potentials, including classical Lp with p >= 1, it gives
  an exact graph-cut solution.
- For nonconvex/discontinuity-preserving potentials, it becomes NP-hard and
  uses approximations.

Why reviewers may mention it: it is a strong global optimizer with good
general image-processing credentials, and Kamui exposes both ILP and PUMA-like
methods.

Skeptical view:

- It is not the operational SAR standard.
- It optimizes a different objective from SAR statistical-cost SNAPHU.
- It does not automatically solve phase-linked-stack uncertainty, connected
  components, or SAR masks.
- In Kamui's own README, the ILP path is described as computationally heavy,
  with subgraph division still TODO. The PUMA extra depends on a GPL maxflow
  implementation.

Paper treatment: cite PUMA in a "other global optimization approaches" clause.
Do not spend paragraph budget unless reviewers specifically ask.

### 7. Quality-guided path following

Canonical example: Herraez et al. 2002. Open implementation:
`skimage.restoration.unwrap_phase`.

What it does: sort local phase connections by reliability and unwrap along a
non-continuous high-reliability path.

Why it matters: common in Python/scientific imaging; supports 1D/2D/3D arrays
and masks.

Skeptical view:

- scikit-image docs cite Herraez directly and do not expose SAR coherence or
  CRLB as a statistical cost.
- There is no connected-component or SAR residue-flow output.
- Useful for smooth or generic phase fields; not a serious noisy-InSAR
  production baseline.

Paper treatment: probably omit in a five-page GRSL unless the intro includes
a broad method taxonomy.

### 8. 3D/time-series unwrapping

Canonical examples:

- Hooper and Zebker 2007: theoretical 3D framework and algorithms for InSAR
  time series.
- Pepe and Lanari 2006: EMCF for multitemporal DInSAR, using spatial and
  temporal relationships.
- Shanker and Zebker 2010: edgelist integer-programming formulation for
  multidimensional and time-series InSAR.
- spurt docs: https://spurt.readthedocs.io/en/latest/tutorials/emcf-3d/

What spurt says:

- It implements the MCF component of EMCF.
- The 3D problem is approximated by two-stage 2D unwrapping.
- First, double-difference phases are temporally unwrapped in time/Bperp space
  to generate unwrapped spatial gradients.
- Then those gradients are spatially unwrapped to form interferograms.
- Its docs say the impact of cost functions is less well understood for EMCF
  and 3D unwrappers than for 2D unwrappers like SNAPHU.

Why this is closest to our paper's "time-series" angle:

- It is open.
- It is explicitly InSAR time-series phase unwrapping.
- It uses temporal graph structure rather than independent per-IG unwraps.

Why it does not subsume us:

- It is not using phase-linking CRLB rasters as the same per-pixel noise model
  through 2D cost, tree priority, and reference selection.
- It performs a separable temporal-then-spatial EMCF workflow; our current
  default performs global 2D MCF per IG and then uses temporal closure mainly
  as an exact reliability diagnostic.
- It does not appear to provide the same exact integer closure-quality map for
  phase-linked interferograms.

Paper treatment: cite Pepe/Lanari and spurt, then state the distinction
plainly. Do not frame spurt as bad; frame it as a different decomposition.

### 9. Multiscale wrappers and operational toolchains

Examples:

- Tophu: https://tophu.readthedocs.io/
- Dolphin unwrap methods:
  https://dolphin-insar.readthedocs.io/en/latest/reference/dolphin/unwrap/
- ISCE3 / ISCE2 unwrappers, PHASS, ICU, SNAPHU wrappers.
- Moraine: https://kanglcn.github.io/moraine/

What they do:

- Tophu is a multiscale 2D InSAR unwrapping framework; its example wraps
  SNAPHU as the actual unwrap callback.
- Dolphin exposes `snaphu`, `icu`, `phass`, `spurt`, and `whirlwind` style
  choices in its ecosystem.
- Moraine includes point-cloud phase unwrapping APIs, but the docs shown are
  wrappers around GAMMA's `mcf_pt` and require CUDA-heavy infrastructure.

Skeptical view:

- These are important ecosystem references, but most are wrappers,
  orchestration, or multiscale strategies rather than a new core competitor.
- Do not cite all of them in a five-page letter. Keep them in the internal
  response cache for reviewer questions.

### 10. Machine-learning and diffusion claims

Recent examples:

- UnwrapDiff 2025: conditional diffusion guided by SNAPHU, claims average
  10.11 percent NRMSE reduction over SNAPHU on synthetic/noisy cases.
  https://arxiv.org/abs/2512.04749
- Song et al. 2026: diffusion framework for large-scale and complex
  earthquake-related deformation, claiming large image scaling and handling of
  fault discontinuities. https://arxiv.org/abs/2603.21378
- "When Less Is More" 2026: U-Net benchmarking for operational early-warning
  style InSAR phase unwrapping, with code and speed claims.
  https://arxiv.org/abs/2605.00896

Skeptical view:

- Mostly preprint/recent-work territory, not yet the operational standard.
- They often optimize for specific deformation scenes, synthetic supervision,
  or image-to-image reconstruction metrics.
- They generally do not address phase-linked CRLB propagation, exact closure,
  residue-grid MCF diagnostics, or connected components in the same way.
- Some use SNAPHU/MCF as input guidance, which makes them post-processors or
  denoisers relative to the classical unwrapping step.

Paper treatment: ignore unless reviewers explicitly ask for ML context. In a
GRSL letter, citing these distracts from the core method.

## Open-Source Competitor Matrix

| Tool | Core idea | Open status | Why it matters | Skeptical assessment |
|---|---|---:|---|---|
| SNAPHU | Statistical-cost nonlinear network-flow optimizer | Public C source, latest page says 2.0.7 in Feb 2024 | Operational SAR baseline | Main benchmark. Do not dismiss. |
| spurt | EMCF for time-series InSAR | PyPI/GitHub, pre-alpha metadata | Closest open 3D/time-series comparator | Different separable formulation; cost-function behavior still noted as open in docs. |
| Tophu | Multiscale 2D unwrapping framework | Open docs/package | Operational tiling/multiscale wrapper | Usually wraps SNAPHU/PHASS; not a new core statistical solver. |
| Kamui | ILP / graph cuts / PUMA-like methods | MIT core, PUMA extra uses GPL maxflow | General global optimization competitor | Computationally heavy; not InSAR production baseline; no CRLB/closure story. |
| scikit-image `unwrap_phase` | Herraez reliability-sorted path following | Open | Common Python generic phase unwrap | No SAR statistical cost, no coherence/CRLB weighting, no connected components. |
| ISCE ICU / PHASS | ISCE-family unwrappers | Open within ISCE ecosystem | Common in pipelines | Need benchmark only if reviewer asks; less central than SNAPHU. |
| Moraine | InSAR post-processing with point-cloud MCF wrappers | GPL, CUDA-heavy, under active development | Emerging open ecosystem | Docs show wrappers around GAMMA MCF for PU; not directly comparable to our open core. |
| ML/diffusion methods | Learned phase reconstruction or postprocessing | Mixed; mostly recent preprints | Reviewer may ask due current hype | Not a core deterministic SAR MCF competitor; should not be central. |

## Claims We Should Avoid

- Avoid: "first MCF phase unwrapper."
  Correct: "builds on network-programming/MCF phase unwrapping."

- Avoid: "first statistical cost for InSAR unwrapping."
  Correct: "replaces coherence-derived uncertainty with CRLB-derived variance
  for phase-linked interferograms."

- Avoid: "first 3D/time-series unwrapping."
  Correct: "uses exact phase-linking closure to diagnose/correct integer
  ambiguities in a phase-linked stack."

- Avoid: "SNAPHU is slow because it does one graph search per residue pair."
  Correct: "SNAPHU solves a more general nonlinear statistical-cost problem
  with tree pivots and iterative flow increments; our simpler linear MCF
  admits batched multi-source shortest-path augmentation."

- Avoid: "nothing open source is close."
  Correct: "we are not aware of an open InSAR unwrapper that combines
  CRLB-weighted 2D MCF, explicit boundary residues or ground-node handling,
  and exact temporal-closure diagnostics for phase-linked stacks."

- Avoid: "closure correction always improves results."
  Correct: "closure residuals are an exact diagnostic; the current tree
  correction is retained but off by default because it can propagate outliers."

## What Reviewers May Ask

**Why not just use SNAPHU?**

Because SNAPHU is coherence/statistical-model driven for ordinary
interferograms. Phase-linked products already contain a better per-pixel
uncertainty estimate: the CRLB phase variance. Our method plugs that into the
MCF cost and then exploits exact temporal closure as a stack diagnostic.
SNAPHU remains the benchmark, not the thing being replaced by argument.

**Why not spurt/EMCF?**

EMCF is the right prior art for time-series unwrapping. The distinction is the
decomposition and uncertainty model: spurt approximates 3D unwrapping by
temporal and spatial 2D MCF steps; our current default keeps per-IG global MCF
and uses the phase-linked temporal graph for closure diagnostics and anchoring.

**Why not PUMA?**

PUMA is a strong graph-cut formulation for Lp/MRF objectives. It is not the
SAR statistical-cost baseline, not a phase-linked CRLB method, and not the
operational tool most InSAR pipelines use. It deserves a citation, not a full
comparison in a five-page letter.

**Why trust a simpler linear MCF over SNAPHU's nonlinear model?**

We should not argue abstract superiority. We should show empirical agreement
modulo 2*pi, speed, and the phase-linking-specific features. Simpler is a
tradeoff: it enables fast batched primal-dual solves and exact diagnostics, but
does not subsume SNAPHU's full nonlinear cost machinery.

**What is genuinely new?**

Most likely publishable novelty:

- CRLB-derived uncertainty used directly as the MCF cost weight for
  phase-linked interferograms.
- Boundary-residue handling / ground-node discussion with real failure-mode
  evidence.
- Exact integer closure residual as a per-pixel reliability diagnostic for
  phase-linked stacks.
- Fast, reproducible open implementation with validation against Dolphin
  SNAPHU on a real stack.

## Suggested Related-Work Paragraph

Use this as a starting point for the actual paper:

> Phase unwrapping has a long history in InSAR, including branch-cut residue
> methods, least-squares and minimum-norm approaches, network-programming and
> MCF formulations, statistical-cost network-flow optimization, and graph-cut
> objectives. SNAPHU remains the dominant open operational baseline for large
> SAR interferograms, using statistical cost models and nonlinear network-flow
> optimization. Multitemporal methods such as 3D phase unwrapping and EMCF
> exploit temporal redundancy, and open implementations such as spurt provide
> separable temporal-then-spatial MCF workflows. Our work is complementary:
> for phase-linked stacks we retain a residue-grid MCF per interferogram, but
> replace coherence-derived uncertainty with the CRLB variance emitted by the
> phase linker, repair boundary residue handling, and use exact temporal
> closure to form integer-valued reliability diagnostics.

## Source Notes

Primary literature and tool docs used while compiling this note:

- SNAPHU project page:
  https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/
- Chen and Zebker 2000:
  https://opg.optica.org/abstract.cfm?uri=josaa-17-3-401
- Chen and Zebker 2001:
  https://pubmed.ncbi.nlm.nih.gov/11205980/
- Chen and Zebker 2002:
  https://doi.org/10.1109/TGRS.2002.802453
- Costantini 1998:
  https://doi.org/10.1109/36.673674
- Carballo and Fieguth 2000:
  https://doi.org/10.1109/36.868876
- Bioucas-Dias and Valadao 2007:
  https://pubmed.ncbi.nlm.nih.gov/17357730/
- Goldstein, Zebker, Werner 1988:
  https://doi.org/10.1029/RS023i004p00713
- Ghiglia and Romero 1994:
  https://doi.org/10.1364/JOSAA.11.000107
- Herraez et al. 2002:
  https://opg.optica.org/abstract.cfm?uri=ao-41-35-7437
- scikit-image unwrap docs:
  https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html
- Hooper and Zebker 2007:
  https://opg.optica.org/abstract.cfm?uri=josaa-24-9-2737
- Pepe and Lanari 2006:
  https://doi.org/10.1109/TGRS.2006.873207
- Shanker and Zebker 2010:
  https://opg.optica.org/abstract.cfm?uri=josaa-27-3-605
- spurt EMCF docs:
  https://spurt.readthedocs.io/en/latest/tutorials/emcf-3d/
- Tophu docs:
  https://tophu.readthedocs.io/
- Kamui README:
  https://github.com/yoyolicoris/kamui
- Dolphin unwrap docs:
  https://dolphin-insar.readthedocs.io/en/latest/reference/dolphin/unwrap/
- Moraine docs:
  https://kanglcn.github.io/moraine/
- Recent ML preprints:
  https://arxiv.org/abs/2512.04749,
  https://arxiv.org/abs/2603.21378,
  https://arxiv.org/abs/2605.00896
