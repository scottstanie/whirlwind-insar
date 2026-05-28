# Recent Phase Unwrapping Literature: 2010-2026 Skeptical Review

Last checked: 2026-05-27.

This note is specifically about recent work we could get called out for
missing, especially deep-learning phase unwrapping after roughly 2010. It
complements `phase_unwrapping_skeptical_review.md`, which covers the older
algorithmic lineage.

## Bottom Line

Reviewers can fairly object if the paper cites no recent phase-unwrapping work.
There has been substantial activity since 2010, and since about 2019 there has
been a real wave of deep-learning phase unwrapping papers.

But the recent literature does not obviously invalidate our positioning. Most
deep-learning work is:

- single-interferogram 2D PU, not phase-linked stack PU;
- supervised on simulated or semi-synthetic data;
- evaluated against SNAPHU/MCF, not replacing SNAPHU as operational baseline;
- focused on wrap-count segmentation, direct phase regression, denoising plus
  unwrapping, or deep assistance to a classical solver;
- not built around CRLB variance from phase linking;
- not emitting connected components, residue-flow diagnostics, or exact
  temporal-closure reliability.

The one recent paper that is closest to a "you should cite this" comparator is
Jiang, Xu, Hooper, and Xie 2026, a SegFormer wrap-count method for large-scale
low-coherence interferograms. It is TGRS, directly InSAR, and explicitly claims
large real-data validation and reduced unwrapping errors versus classical MCF.
It still solves a different problem than ours: low-gradient tectonic
single-interferogram unwrapping across decorrelated regions, not CRLB-weighted
phase-linked stacks.

## What To Add To The 5-Page Paper

Add one compact "recent learning methods" sentence in the introduction or
related-work paragraph, not a separate section:

> Recent learning-based phase unwrappers have recast 2D PU as wrap-count
> segmentation, phase-discontinuity prediction, direct phase regression, or
> model-based deep unrolling, with applications to noisy InSAR and low-coherence
> interferograms. These methods are complementary to our setting: we retain a
> residue-grid MCF solver, but use the CRLB variance emitted by phase linking
> and the exact temporal closure relation of phase-linked stacks.

If we can only cite 3 recent items:

1. Baek and Jung 2022 review of DL-based InSAR phase unwrapping.
2. Sica et al. 2020/2022 CNN coherence-driven InSAR PU, because it is a clean
   early InSAR-specific DL baseline.
3. Jiang et al. 2026 wrap-count SegFormer low-coherence InSAR PU, because it is
   recent, TGRS, large-scale, and likely reviewer-visible.

If we can cite 5-6 recent items:

- Add PU-GAN 2022 for direct one-step InSAR regression/GAN.
- Add MoDL-PU 2025 for model-based + DL hybrid InSAR PU.
- Add Chen et al. CVPR 2024 for unsupervised deep unrolling, if we want a
  general-computer-vision recent citation outside SAR.

## Recent Literature Map

### 1. 2010-2016: multidimensional/time-series and robust optimization

This is the "post-classical, pre-DL" period. It matters because it shows that
the field did not stop after SNAPHU.

Representative work:

- Shanker and Zebker 2010, edgelist phase unwrapping for time-series InSAR.
  It replaces closed loops by reliable edges as the basic construct for
  multidimensional/time-series PU.
- Costantini, Malvarosa, and Minati 2011/2012, redundant integration / sparse
  multidimensional formulation.
- Multibaseline cluster-analysis methods, e.g. Yu/Liu/Xing/Bao line of work,
  including cluster-analysis-based multibaseline PU and noise-robust variants.
- Kamilov et al. 2015, isotropic inverse-problem PU with sparse gradient-error
  fidelity and higher-order TV regularization.
- Multiscale and wavelet/modulo-wavelet approaches around 2016.

Skeptical relevance:

- These are algorithmically interesting, but not direct competitors to our
  current paper unless we make very broad "new 3D PU" claims.
- For GRSL, cite Hooper/Zebker 2007, Shanker/Zebker 2010, and Pepe/Lanari
  2006/spurt if we need time-series context; do not expand the whole tree.

### 2. 2019-2022: PhaseNet-style wrap-count segmentation

This is the start of the modern DL PU wave.

Representative work:

- PhaseNet, Spoorthi et al. 2019: frames 2D PU as dense wrap-count
  classification/semantic segmentation.
- PhaseNet 2.0, Spoorthi et al. 2020: DenseNet-style network, improved loss,
  more noise robustness, still synthetic/general phase-image oriented.
- Sica et al. 2020/2022: CNN-based coherence-driven InSAR PU. This is directly
  relevant because it adds interferometric coherence as a network input and
  tests on TanDEM-X style data.
- Li et al. 2022: InSAR PU using gradient information fusion / improved
  PhaseNet, motivated by class imbalance and use of gradient information.
- PUnet and related U-Net/attention variants: use U-Net, attention, positional
  encoding, or related image-segmentation architectures for InSAR PU.

What these methods usually predict:

- absolute wrap count `k(x, y)`;
- phase-jump / branch-cut / discontinuity masks;
- wrap-count gradients;
- or a direct unwrapped phase field.

Strength:

- Fast inference after training.
- Can learn patterns that violate simple Itoh continuity assumptions.
- Promising in severe noise, dense fringes, and discontinuity settings.

Skeptical read:

- Wrap-count segmentation has a finite class range and class-imbalance problem.
- Generalization depends strongly on training data realism.
- Many evaluations are synthetic or semi-synthetic; real ground truth is hard.
- Direct regression may not guarantee congruence: `wrap(unwrapped)` can differ
  from the input wrapped phase unless the loss explicitly enforces it.
- These methods are usually not conservative scientific estimators: uncertainty,
  topology, connected components, and path-independent integer consistency are
  often secondary.

Paper stance:

- Cite one review plus one InSAR-specific method.
- Do not benchmark unless reviewers demand it.

### 3. 2021-2024: discontinuity prediction, joint denoise+unwrap, and GANs

Representative work:

- Wu et al. 2021 TGRS, deep-learning phase-discontinuity prediction for 2D SAR
  interferograms. DENet uses interferogram, range/azimuth phase gradients, and
  residue maps to predict discontinuities.
- Zhou et al. 2022 PU-GAN, conditional GAN one-step 2D InSAR PU. It explicitly
  addresses the problem that pure L2 direct regression can blur phase and fail
  congruence.
- PUnet 2023: U-Net plus attention/positional encoding for robust InSAR PU,
  framed as simultaneously denoising and unwrapping.
- U-Net approaches for InSAR denoising plus unwrapping, 2023 Remote Sensing and
  related work.
- Chen et al. 2024 CVPR unsupervised deep unrolling for PU. Not InSAR-specific,
  but important as a modern computer-vision/optimization hybrid: no paired GT
  phases are needed for training, but known noise statistics are assumed.

Skeptical read:

- These are real research directions, not noise.
- Most are still not "drop-in operational InSAR unwrappers."
- Many improve image metrics versus classical methods but do not expose the
  integer ambiguity, topology, or failure regions in a way downstream InSAR
  pipelines expect.
- Diffusion/GAN/regression methods can produce visually good fields while being
  less transparent about exact modulo consistency and uncertainty.

Paper stance:

- Mention as a family. Cite PU-GAN or Wu et al. if we want an InSAR DL example
  beyond Sica.
- For our contribution, emphasize "we stay inside the integer MCF framework"
  rather than arguing DL is bad.

### 4. 2024-2026: datasets, transformers, model-based DL, and operational scale

This is the recent cluster most likely to come up in review.

Representative work:

- InSAR-DLPU dataset, Zhou/Yu et al. 2024: public benchmark dataset for
  DL-based InSAR PU, with 31,100 paired wrapped/absolute phase patches, mostly
  SRTM-derived simulation plus real TanDEM-X patches.
- Unwrap-Net 2024: encoder-decoder InSAR PU assisted by airborne LiDAR data;
  claims improved SSIM/RMSE versus SNAPHU and PhaseNet, with code/dataset
  promised.
- MMPhU-Net 2024: multi-model fusion network for large-gradient subsidence
  deformation.
- MoDL-PU 2025: model-based deep learning for InSAR PU, explicitly responding
  to generalization limits of pure DL by combining model-based PU knowledge and
  Res-UNet-Inception training. It covers single-baseline and multibaseline
  tasks.
- PIPNet 2025: deep network for multibaseline InSAR PU based on pure integer
  programming.
- Jiang, Xu, Hooper, Xie 2026: wrap-count-based SegFormer for large-scale
  low-coherence interferograms. Trains on >20k simulated samples plus >10k
  real-world samples from COMET-LiCSAR, corrects/reunwraps patches via a
  reliability metric, mosaics overlapping patches, and reports 28-83 percent
  reduction in real-interferogram unwrapping errors versus classical MCF.
- 2026 transformer/U-Net variants: ResUCTransNet, FPUNet, DMP-PUNet, etc.
  These show the architecture race has begun: channel transformers, Fourier
  features, dilated multipath designs, and public benchmark use.

Skeptical read:

- The field is moving quickly. A paper with zero recent references looks stale.
- The Jiang/Hooper paper is especially relevant because it is not merely a
  generic DL demo; it is large-scale InSAR and has real-data validation.
- However, these papers still mostly target single interferograms, DEM/topo PU,
  subsidence funnels, low-coherence tectonic scenes, or multibaseline DEM PU.
  None is obviously the same as "phase-linked stack, CRLB cost, exact closure
  diagnostic."
- Many results optimize per-image MAE/RMSE/SSIM or detected-error fraction,
  not stack closure, uncertainty propagation, or downstream time-series
  consistency.

Paper stance:

- Cite Jiang et al. 2026 if space allows. It inoculates us against "what about
  recent DL?" more effectively than citing ten U-Net variants.
- Phrase it as complementary: recent DL methods learn wrap counts or unwrapped
  phase from data; our method uses the phase-linked covariance/CRLB information
  in a deterministic MCF framework.

### 5. 2025-2026: diffusion and foundation-model flavored work

Representative work:

- UnwrapDiff 2025 arXiv: conditional diffusion for robust InSAR PU, using
  SNAPHU/MCF output as conditional guidance, claiming average 10.11 percent
  NRMSE reduction versus SNAPHU on synthetic/noisy cases and difficult dyke
  intrusion examples.
- Song et al. 2026 arXiv: diffusion-based framework for large-scale complex
  events and discontinuous deformation.
- Other U-Net/transformer/foundation-model-ish papers are appearing quickly.

Skeptical read:

- These are worth knowing, but preprint diffusion work should not dominate a
  short GRSL methods paper unless reviewers specifically ask.
- If a model conditions on SNAPHU, it is closer to post-processing/refinement
  than a replacement for a core deterministic unwrapper.
- The hard questions remain: generalization, physical consistency, exact
  modulo congruence, uncertainty, and operational failure masks.

Paper stance:

- Do not cite diffusion unless we need to show awareness of the absolute latest
  preprints.
- Keep the main cited DL set to peer-reviewed InSAR papers.

## What Recent DL Methods Claim, And How To Answer

### General limitations worth stating

The strongest generic limitations are:

- Training data realism and domain shift. Most methods train on simulated,
  semi-synthetic, DEM-derived, or curated "good" real examples. Generating
  fully realistic raw-SLC-level training data is expensive, but more importantly,
  many papers never attempt that; they simulate at the interferogram/phase/noise
  level. The right critique is not just "synthetic is slow"; it is that the
  learned model is only as good as the deformation, atmosphere, decorrelation,
  filtering, masking, and sensor distributions represented in training.
- Hardware/deployment dependency. Training requires GPUs, and competitive
  large-scale inference generally assumes GPU infrastructure, patch batching,
  and model/runtime management. CPU inference may be possible but is not the
  operating point these methods optimize for.
- Model-quality dependency. The result is a property of the architecture,
  labels, training set, class range, loss, preprocessing, post-processing, and
  domain match. A good network architecture can still fail if the training
  distribution misses the target failure mode.
- Reproducibility gap. Several papers release partial code or datasets, but
  fewer provide a maintained, permissively licensed, pretrained,
  command-line-ready InSAR unwrapper that can be applied to arbitrary GeoTIFF
  interferograms without retraining.

### Claim: DL breaks the Itoh limitation

Reasonable. A supervised model can learn priors over dense fringes, aliasing,
or decorrelated regions where local gradient integration fails.

Our answer: we are not relying on plain Itoh integration. We solve an integer
residue-flow problem with a statistical cost and then evaluate exact temporal
closure. Different axis.

### Claim: DL is faster at inference

Often true, especially per tile on GPU.

Our answer: inference speed alone is not the operational metric. We need
congruence, uncertainty, masks/components, closure, and reproducibility. Also,
our deterministic Rust MCF is already fast enough for the reported pipeline and
does not require a trained model for the target domain.

### Claim: DL beats SNAPHU/MCF on noisy or low-coherence cases

Sometimes plausible, especially in large low-coherence masks or synthetic
training/test regimes.

Our answer: cite it, do not deny it. But our paper is about phase-linked stacks
where CRLB rasters and exact temporal closure are available. We should avoid
claiming best possible single-IG unwrap on every noisy scene.

### Claim: DL provides an operational path for automated large-scale products

Increasingly plausible. Jiang et al. 2026 is exactly this direction.

Our answer: our operational path is deterministic, uncertainty-aware, and
integrates with phase linking. DL may be complementary as a denoiser, mask
generator, or prior for pathological low-coherence regions.

## Relevance Ranking For Our Paper

High relevance:

- Baek and Jung 2022 review: establishes that DL InSAR PU is an active area and
  gives categories.
- Sica et al. 2020/2022: InSAR-specific CNN with coherence input.
- Jiang et al. 2026: large-scale low-coherence SegFormer wrap-count method.
- spurt/EMCF and Dolphin docs/JOSS: recent open InSAR time-series ecosystem.

Medium relevance:

- PU-GAN 2022: important one-step InSAR DL regression/GAN.
- Wu et al. 2021: phase-discontinuity prediction for SAR interferograms.
- MoDL-PU 2025: model-based DL hybrid.
- InSAR-DLPU 2024 dataset: useful if discussing benchmarks.

Low relevance for GRSL, but useful for reviewer response:

- Chen et al. CVPR 2024 unsupervised deep unrolling.
- PUnet, ResUCTransNet, FPUNet, DMP-PUNet, MMPhU-Net, PIPNet.
- General optical/metrology DL phase unwrapping reviews.
- Diffusion preprints.

## Open DL Comparator Candidates

After a quick open-source pass, these are the most plausible candidates. The
practical conclusion is that "open code exists" is not the same as "reasonable
baseline for our paper."

| Repo / method | What is open | Practical comparison status |
|---|---|---|
| `Wu-Patrick/Deformation-Monitoring` | TensorFlow 1.13 implementation with checkpoints for mining-induced deformation; README says it targets localized rapid deformation and suggests 180x180-ish patches. | Most runnable InSAR DL candidate found. Niche target, old TF stack, personal-academic-use statement, not phase-linked/CRLB. |
| `zhoulifan/InSAR-DLPU` | Dataset of 31,100 wrapped/absolute phase pairs, mostly SRTM-derived simulation plus 100 TanDEM-X examples. | Useful benchmark data, not a model implementation. |
| `kqwang/Phase_unwrapping_by_U-Net` | MIT U-Net demo code and dataset generation/test scripts. | General phase-unwrapping demo, not InSAR production; appears to require training weights. |
| `yangwangyangzi48/UNWRAPNETV1` | Minimal Unwrap-Net code; README says datasets/other files will be uploaded later. | Not ready for fair comparison without missing data/weights. |
| `Heyuxiao123/FPUNet` | Minimal FPUNet training code, MIT. | Very thin repo; no obvious pretrained weights or full benchmark package. |
| CVPR 2024 unsupervised deep unrolling | Paper/supplement open; code not found in quick search. | Interesting scientifically, but not an InSAR operational comparator. |
| Jiang et al. 2026 SWIPHU | Accepted manuscript is open; no public code found in quick search. | Strong recent citation, but not currently runnable as an external baseline. |

So the fair paper language is:

> We did not compare to learning-based unwrappers because the most relevant
> recent methods either do not provide a maintained, pretrained open
> implementation suitable for arbitrary InSAR interferograms, or target a
> different setting such as isolated rapid deformation, DEM/topographic
> unwrapping, low-coherence interseismic single interferograms, or general
> optical phase images. We cite them as complementary recent work.

## Suggested Actual Related-Work Paragraph

This is more current than the paragraph in the older note:

> Recent phase-unwrapping work includes multidimensional and time-series
> formulations, robust inverse-problem solvers, and a rapidly growing body of
> learning-based methods. Deep networks have been used to predict wrap counts,
> phase discontinuities, gradient ambiguities, or unwrapped phase directly, with
> InSAR-specific variants using coherence, residues, attention/transformer
> modules, and public or semi-synthetic training datasets. These methods show
> promise for noisy or low-coherence single interferograms, but they generally
> do not use the CRLB uncertainty rasters emitted by phase linking or the exact
> temporal closure relation of phase-linked interferogram stacks. We therefore
> treat them as complementary to the deterministic residue-grid MCF framework
> used here.

## Suggested Bib Items To Add

Use BibTeX later; for now, these are the references I would add by hand:

- Baek, W. K. and Jung, H. S., "A Review on Deep-learning-based Phase
  Unwrapping Technique for Synthetic Aperture Radar Interferometry," Korean
  Journal of Remote Sensing, 2022. DOI: 10.7780/kjrs.2022.38.6.2.2.
- Sica, F., Calvanese, F., Scarpa, G., and Rizzoli, P., "A CNN-Based
  Coherence-Driven Approach for InSAR Phase Unwrapping," IEEE GRSL, 2022
  (published online 2020). DOI: 10.1109/LGRS.2020.3029565.
- Zhou, L., Yu, H., Pascazio, V., and Xing, M., "PU-GAN: A One-Step 2-D InSAR
  Phase Unwrapping Based on Conditional Generative Adversarial Network," IEEE
  TGRS, 2022.
- Chen, Z., Quan, Y., and Ji, H., "Unsupervised Deep Unrolling Networks for
  Phase Unwrapping," CVPR, 2024.
- Zhou, L. and Yu, H., "MoDL-PU: Model-Based Deep Learning for InSAR Phase
  Unwrapping," IEEE TGRS, 2025. DOI: 10.1109/TGRS.2025.3549607.
- Jiang, K., Xu, W., Hooper, A. J., and Xie, L., "A Wrap-Count-Based Phase
  Unwrapping Method for Large-Scale, Low-Coherence Interferograms Using Deep
  Learning," IEEE TGRS, 2026. DOI: 10.1109/TGRS.2026.3660028.

## Source Notes

Most useful sources checked:

- Baek and Jung 2022 DL-InSAR PU review:
  https://pure.uos.ac.kr/en/publications/a-review-on-deep-learning-based-phase-unwrapping-technique-for-sy
- Wang et al. 2022 comparative review of deep learning spatial PU:
  https://www.researching.cn/Articles/OJc3ee5c3ec89a4a1a
- Sica et al. CNN coherence-driven InSAR PU:
  https://elib.dlr.de/140351/
- PUnet 2023:
  https://www.frontiersin.org/articles/10.3389/fenvs.2023.1138399/full
- PU-GAN 2022:
  https://ricerca.uniparthenope.it/handle/11367/105158
- Unwrap-Net 2024:
  https://doi.org/10.1016/j.isprsjprs.2024.11.009
- MoDL-PU 2025:
  https://colab.ws/articles/10.1109%2Ftgrs.2025.3549607
- Jiang/Xu/Hooper/Xie 2026:
  https://eprints.whiterose.ac.uk/id/eprint/240021/
- InSAR-DLPU dataset:
  https://github.com/zhoulifan/InSAR-DLPU
- CVPR 2024 unsupervised deep unrolling:
  https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Unsupervised_Deep_Unrolling_Networks_for_Phase_Unwrapping_CVPR_2024_paper.pdf
- UnwrapDiff preprint:
  https://arxiv.org/abs/2512.04749
