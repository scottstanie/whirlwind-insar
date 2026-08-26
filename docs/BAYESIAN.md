# Why whirlwind is Bayesian

Whirlwind's original subtitle was "a Bayesian phase unwrapper." This page
explains what that means precisely: what probability the solver actually
maximizes, why the Carballo formulation earns the word *Bayesian* in a way
that SNAPHU's self-described MAP estimator does not, and on which kinds of
interferograms the two formulations should be expected to diverge.

This is a theory companion to the [algorithm notes](ALGORITHM.md) and the
[full ATBD](ATBD-whirlwind.md) (especially ATBD §2.3 and §5). It cites the
implementation where the theory is load-bearing, and it flags honestly where
the implementation approximates the clean theory.

## 1. What is actually being estimated

Phase unwrapping never estimates a continuous phase surface directly. The
output is constrained to be *congruent* to the input: every unwrapped pixel is
the wrapped value plus an integer number of cycles. Equivalently, the unknowns
are one integer per pixel-grid edge,

$$
\Delta k_e \in \mathbb{Z},
$$

the cycle correction applied to that edge's wrapped gradient. Once every
$\Delta k_e$ is chosen, the unwrapped surface follows by integration (up to a
global constant). Not every assignment is admissible: the corrected gradients
must be curl-free, which is exactly the residue-balancing flow-conservation
constraint of the network problem (ATBD §4, §6). So a network-flow unwrapper
is a **discrete estimator**: it selects one configuration
$\{\Delta k_e\}$ from the feasible set $\mathcal{K}$ of integer fields
consistent with *some* surface.

Both whirlwind and SNAPHU search this same feasible set with the same network
machinery. The entire statistical difference is in how each candidate
$\Delta k_e$ is scored.

## 2. The hierarchical model behind the Carballo cost

The Carballo & Fieguth (2002) cost is built from a layered ("hierarchical")
probability model. Reading from the bottom up:

1. **Phase noise.** A single multilooked pixel's phase error follows the Lee
   et al. (1994) PDF $f(\phi_N \mid \gamma, L)$, parameterized by true
   coherence $\gamma$ and looks $L$.
2. **Gradient noise.** An edge's gradient noise $n$ is the difference of two
   pixel noises, so its PDF is the (circular) self-correlation of the Lee PDF,
   supported on $[-2\pi, 2\pi]$
   (`scripts/generate_carballo_tables.py::gradient_noise_cdf`).
3. **Observation model.** The true unwrapped gradient on the edge is
   $\alpha + n$, where $\alpha$ is the true signal slope. The wrap count is
   determined by which $2\pi$ interval that lands in:

    $$
    P(\Delta k = j \mid \alpha, \gamma, L)
    \;=\;
    \Pr\bigl\{\, \alpha + n \in [\,(2j{-}1)\pi,\ (2j{+}1)\pi\,) \,\bigr\}.
    $$

4. **Slope uncertainty (prior on $\alpha$).** The true slope is not observed;
   whirlwind estimates it by local smoothing (a 7×7 box over the complex
   gradients, ATBD §5.2), giving $\hat\alpha$. Carballo's eq. (15) models the
   estimation error as Gaussian, $\alpha \sim \mathcal{N}(\hat\alpha,
   \sigma^2(\hat\gamma, N_{\text{win}}))$, with $\sigma \to \pi/\sqrt{3}$
   (uniform) as coherence vanishes. The slope is then **marginalized**:

    $$
    P(\Delta k = j \mid \hat\alpha, \hat\gamma, L)
    \;=\;
    \int P(\Delta k = j \mid \alpha, \gamma, L)\;
    p(\alpha \mid \hat\alpha, \hat\gamma)\, d\alpha .
    $$

    (`generate_carballo_tables.py::marginalized_probability`.)

5. **Coherence uncertainty (prior on $\gamma$).** The sample coherence
   $\hat\gamma$ is a biased, noisy estimate of the true $\gamma$ — strongly
   biased high at low coherence and few looks. The original whirlwind tables
   additionally integrated over the true coherence given the sample value,
   using the sampling distribution of $\hat\gamma \mid \gamma, L$ (Touzi et
   al., 1999) inverted through Bayes' rule:

    $$
    P(\Delta k = j \mid \hat\alpha, \hat\gamma, L)
    \;=\;
    \iint P(\Delta k = j \mid \alpha, \gamma, L)\;
    p(\alpha \mid \hat\alpha, \hat\gamma)\;
    p(\gamma \mid \hat\gamma, L)\; d\alpha\, d\gamma .
    $$

The result is two tabulated fields, $p_0 = P(\Delta k = 0 \mid \cdot)$ and
$p_1 = P(\Delta k = +1 \mid \cdot)$, indexed by $(\hat\alpha, \hat\gamma, L)$.
Two structural features are worth noticing:

- **These are probability *masses*, not density values.** Each one is the
  integral of a noise density over a full $2\pi$ interval, further averaged
  over the nuisance parameters. In Bayesian terms, $P(\Delta k = j \mid
  \hat\alpha, \hat\gamma, L)$ is a **posterior predictive probability** of the
  discrete unknown.
- **$p_0 + p_1 \neq 1$.** The remaining mass sits on $\Delta k = -1$ (and
  beyond). The distribution over $\Delta k$ is a genuine normalized
  distribution over the integers; the cost only ever needs the two hypotheses
  that a unit of flow in a given direction discriminates between. The
  opposite-direction arc on the same pixel edge carries the $\Delta k = -1$
  hypothesis, via the sign flip $p_{-1}(\hat\alpha) = p_{+1}(-\hat\alpha)$ —
  that is exactly the four-direction sign convention of ATBD §5.5.

!!! note "Implementation variants"
    The default single-tile path (`compute_carballo_costs_parity`, ATBD §5.6)
    reads the pre-sampled $p_0/p_1$ tables described above, which marginalize
    the slope (and, in the original generator, the coherence). The opt-in
    analytical path (`compute_carballo_costs`, ATBD §5.4) rebuilds a cost LUT
    from the Lee CDF at runtime and collapses to a two-hypothesis split
    $p_0 = 1 - p_1$ with plug-in $\hat\gamma$ — it keeps the interval-mass
    structure but drops the nuisance marginalization. The bundled tables are a
    reconstruction of the original whirlwind generator
    (`scripts/generate_carballo_tables.py`), not a bit-exact recovery.

## 3. The posterior the flow solve maximizes

Assume (as Carballo does, and as SNAPHU does) that edges are conditionally
independent given the local data summaries. The posterior over a correction
field $\mathbf{k} = \{\Delta k_e\}$ is then

$$
P(\mathbf{k} \mid \text{data})
\;\propto\;
\mathbb{1}[\mathbf{k} \in \mathcal{K}]
\prod_{e} P(\Delta k_e \mid \hat\alpha_e, \hat\gamma_e, L),
$$

where the indicator is the curl/flow-conservation constraint — the only
"prior" over configurations, and it is a hard physical one: corrections that
do not balance the residues correspond to no surface at all.

Take the negative log and subtract the constant baseline
$-\sum_e \ln p_0$ (the score of the all-zero correction). What remains is a
sum over the edges that carry flow:

$$
\hat{\mathbf{k}}
= \arg\max_{\mathbf{k} \in \mathcal{K}} P(\mathbf{k} \mid \text{data})
= \arg\min_{\mathbf{k} \in \mathcal{K}}
  \sum_{e:\ \Delta k_e \neq 0} -\ln\frac{p_{\Delta k_e}}{p_0}
  \;\approx\;
  \arg\min_{\text{flow } f}\ \sum_e c_e\, f_e,
\qquad
c_e = -\ln\frac{p_1}{p_0}.
$$

That last expression is exactly the minimum-cost-flow objective of ATBD §6.4.
**So yes: the quantity being maximized is a posterior probability** — the
(independence-factorized) posterior of the integer correction field given the
interferogram, coherence, and looks — or equivalently the posterior odds of
the correction field against the all-zero field. The MCF solver finds the
exact optimum of that objective over the feasible set.

The "$\approx$" hides three implementation departures from the pure statement,
worth naming so the claim stays honest:

1. **Non-negative clamp.** Forward-arc costs are clamped at zero (ATBD §5.1).
   Where the data *favor* a cut ($p_1 > p_0$, raw LLR negative), the model says
   the solver should be paid to cut; the implementation only makes the cut
   free. This biases the solution slightly toward fewer corrections than the
   unclamped posterior mode.
2. **Unit capacity / linearization.** A $|\Delta k| = 2$ correction costs two
   unit-flow traversals, i.e. $2 \cdot (-\ln p_1/p_0)$ rather than
   $-\ln(p_2/p_0)$. On the default unit-capacity network, multi-cycle
   corrections on a single edge are routed around rather than stacked
   (ATBD §6.2).
3. **Independence.** The product over edges is a pseudo-posterior: adjacent
   gradients share pixels and share the smoothed $\hat\alpha$, so they are not
   independent. SNAPHU makes the same approximation; it is what makes the
   problem a tractable network flow at all.

## 4. Why this is "actually Bayesian"

Three properties together justify the label, and each one is a specific
modeling decision rather than a slogan:

1. **The discrete unknown is scored by posterior probability mass.** The
   cost compares $P(\Delta k = 1 \mid \text{data})$ against $P(\Delta k = 0
   \mid \text{data})$, where each side is an *integral* of the noise density
   over its cycle interval. The competing hypotheses are weighed by how much
   probability the model actually assigns them — which matters precisely when
   the noise density is broad, skewed, or heavy-tailed, i.e. at low coherence
   and low looks, where unwrappers earn their keep.
2. **Nuisance parameters are marginalized, not plugged in.** The true slope
   and (in the original tables) the true coherence are integrated out under
   their posteriors given the local estimates. A plug-in estimator inherits
   the full bias of $\hat\gamma$ and pretends $\hat\alpha$ is exact; the
   marginalized cost automatically deflates its own confidence where the
   estimates are unreliable.
3. **The prior over solutions is the physical constraint set.** Conditional
   on the local data, the noise model induces a proper normalized distribution
   over each integer $\Delta k$ (this is why $p_0 + p_1 \neq 1$ is a feature,
   not a bug), and the only additional prior imposed is the hard congruence/
   curl constraint. Nothing about the scene (slope statistics, deformation
   models) is smuggled in — the data model carries all the information.

The estimate is still a *MAP point estimate* of that posterior — whirlwind
does not report posterior uncertainty over surfaces. "Bayesian" here describes
how the objective is constructed, not the form of the output.

## 5. How SNAPHU's formulation differs

SNAPHU (Chen & Zebker, 2001) describes itself as an approximate
maximum-a-posteriori estimator: it maximizes
$\prod_e f(\Delta\phi_e \mid I_1, I_2, \hat\gamma)$ over congruent solutions,
where $f$ is a statistical model for the *unwrapped* gradient given the
observed intensities and coherence, and each candidate correction is scored by
evaluating that density at the point $\Delta\hat\psi_e + 2\pi \Delta k_e$.
The structure looks similar — same feasible set, same independence
approximation, same network solve — but it differs from the Carballo
construction in three specific places:

1. **Density at a point vs. probability of the integer.** SNAPHU's score for
   $\Delta k$ is a density *value*, $f(\Delta\hat\psi + 2\pi\Delta k \mid
   \cdot)$, not the probability that the wrap count *is* $\Delta k$. The two
   coincide only in the limit of sharply peaked noise. At low coherence the
   Lee density is broad and its interval masses can rank hypotheses
   differently than its pointwise heights — exactly the regime where the
   choices are hard. Carballo's interval integration is the difference between
   asking "how tall is the density here?" and "how much belief does the model
   actually place on this cycle?"
2. **Plug-in nuisance parameters.** SNAPHU conditions on the sample coherence
   (and intensity-derived quantities) directly. No integration over the true
   coherence or the slope-estimate error is performed, so the costs inherit
   the well-known high bias of $\hat\gamma$ at low coherence/low looks and
   treat the local gradient estimate as exact. In estimation terms this is a
   profile/empirical-Bayes shortcut rather than a posterior.
3. **The prior depends on the cost mode — and the production mode has none.**
   In `topo` and `defo` modes SNAPHU's $f$ genuinely folds in scene priors
   (terrain-slope statistics through the imaging geometry; a deformation-
   gradient model), so "SNAPHU has no prior" is too strong as a blanket
   statement. But in **`smooth` mode** — the configuration the NISAR GUNW
   production pipeline actually runs (`cost=smooth`, `init=mcf`; see the
   [SNAPHU/PHASS notes](SNAPHU_PHASS_SPEED.md)) — the cost is a
   coherence-weighted quadratic in the corrected gradient's deviation from its
   local mean:

    $$
    c_e(k) = w_e \,\bigl(k \cdot 100 - O_e\bigr)^2,
    \qquad w_e \propto 1/\sigma^2(\hat\gamma_e, L),
    $$

    (mirrored by whirlwind's experimental `compute_snaphu_smooth_costs`,
    ATBD §6.4). That is a Gaussian likelihood with a flat prior over the
    congruent solutions: **maximum likelihood**, i.e. coherence-weighted least
    squares on the cycle corrections. For the smooth mode, the
    "MAP-in-name, ML-in-practice" characterization is exactly right; for
    topo/defo the fairer statement is "a plug-in density maximization with a
    scene prior, but still not a posterior over the discrete wrap counts."

In one line: **SNAPHU maximizes a plug-in conditional density evaluated at
discrete points; whirlwind maximizes a marginalized posterior probability of
the discrete unknowns themselves.** Both then commit to the mode of their
respective objective over the same feasible set.

## 6. Where the difference should show up in real interferograms

Because both methods agree on the feasible set and both are exact network
solvers of their objective, they can only diverge where their *costs* rank
candidate cuts differently. The model differences above predict three regimes:

**Low coherence, few looks.** This is where marginalization bites. The sample
coherence is biased high there, so a plug-in weight $1/\sigma^2(\hat\gamma)$
systematically *overtrusts* noisy regions: SNAPHU-smooth assigns them more
cost than the data justify, and discontinuities get pushed around them along
detours. The marginalized Carballo cost collapses toward zero in genuinely
uninformative regions ($p_1 \approx p_0$), letting the flow pass through noise
freely and concentrating the solution's structure where the data have
authority. Expect the largest placement differences for cuts through
decorrelated farmland, water margins, dense vegetation, and anywhere
$L$ is small.

**Near-Nyquist fringe rates.** Where the true gradient approaches $\pm\pi$
per sample (steep terrain, near-fault coseismic deformation), the slope
marginalization spreads posterior mass smoothly between $\Delta k = 0$ and
$\pm 1$, so costs fall toward zero *along* the true wrap lines while the
directional asymmetry ($\hat\alpha \le 0 \Rightarrow$ effectively forbidden in
the analytical path, ATBD §5.4) keeps wrong-sign cuts expensive. SNAPHU-smooth
scores the same edges with a parabola that is symmetric about its local-mean
offset; it has no equivalent hard directional prohibition, and its
deviation-from-local-mean offset can under-respond to *coherent, spatially
sustained* dense fringes (where raw gradient ≈ local mean, so the offset stays
small). Expect different failure modes when fringes alias: whirlwind errs by
following the directional model, SNAPHU-smooth errs by over-smoothing.

**Quadratic vs. linear flow costs.** SNAPHU-smooth's convex cost makes the
second unit of flow on an arc more expensive than the first, so it prefers to
*spread* a multi-cycle discontinuity across parallel paths. Whirlwind's linear
unit-capacity cost makes cuts pay the same price wherever they go, so it
*concentrates* discontinuities along the single cheapest corridor. On wide
low-coherence corridors crossed by several fringes, the two will draw visibly
different cut geometries — usually the same integer surface on the coherent
sides, but with the transition placed differently inside the noise.

**Where it will not matter.** On well-correlated scenes with modest fringe
rates, the posterior over each $\Delta k_e$ is sharply concentrated, every
reasonable cost model ranks the hypotheses identically, and both solvers
recover the same surface. This is exactly what the
[13-frame NISAR benchmark](NISAR_SUMMARY.md) shows: whirlwind's Carballo-cost
MCF and production single-tile SNAPHU agree on the overwhelming majority of
per-component $2\pi$ levels. The Bayesian construction is insurance for the
hard minority of edges, not a different answer on the easy ones.

## 7. Honest caveats

- The per-edge independence factorization means neither method maximizes a
  true joint posterior; both maximize a pseudo-posterior. "Actually Bayesian"
  refers to the construction of the per-edge probabilities, not to a full
  joint inference.
- The non-negative cost clamp and the unit-capacity linearization (§3) shave
  the objective away from the exact posterior mode in data-favored-cut and
  multi-cycle situations.
- The shipped probability tables are a documented *reconstruction* of the
  original whirlwind generator; the original's exact true-coherence
  integration survives in notes rather than code
  (`scripts/generate_carballo_tables.py`, "Important limitations").
- The output is a point estimate. No calibrated per-pixel uncertainty is
  produced, even though the ingredients (per-edge posteriors) would in
  principle support one.

## References

1. Carballo, G. F., & Fieguth, P. W. (2002). "Probabilistic cost functions
   for network flow phase unwrapping." *IEEE TGRS*, 40(11), 2192–2203.
2. Chen, C. W., & Zebker, H. A. (2001). "Two-dimensional phase unwrapping
   with use of statistical models for cost functions in nonlinear
   optimization." *JOSA A*, 18(2), 338–351.
3. Lee, J. S., Hoppel, K. W., Mango, S. A., & Miller, A. R. (1994).
   "Intensity and phase statistics of multilook polarimetric and
   interferometric SAR imagery." *IEEE TGRS*, 32(5), 1017–1028.
4. Touzi, R., Lopes, A., Bruniquel, J., & Vachon, P. W. (1999). "Coherence
   estimation for SAR imagery." *IEEE TGRS*, 37(1), 135–149.
