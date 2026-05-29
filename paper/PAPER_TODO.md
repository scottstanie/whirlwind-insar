# Paper (whirlwind3d.tex) — TODO once the code/examples settle

Deferred on purpose: the algorithm is still moving (anchor, cascade, feather,
multilook=, and an Atlanta fine-scale fix in progress), so lock the code +
example figures FIRST, then write the paper details. This is just the running
list of what will need changing, from the 2026-05-28 docs audit.

## whirlwind3d.tex
- **Abstract / framing**: currently presents three orthogonal contributions
  (CRLB-weighted cost, boundary-residue fix, tiled mode + virtual-ground). It
  does NOT describe the production method (tiled + global coarse anchor +
  multi-scale cascade + feathered composite; multilook-first for noisy scenes).
  Decide scope: keep as "precursor architecture" paper, or expand to claim the
  NISAR/Atlanta no-Goldstein results.
- **Results**: no NISAR / anchor+cascade result or figure. If we want to claim
  it (99.x% K-match, 3.9 s vs SNAPHU 17 min; Atlanta 97.7% via multilook),
  a NEW figure is needed (generate from the NISAR/Atlanta runs — heavy compute,
  do it deliberately). The 3 existing Palos Verdes figures stay valid.
- **Metric definition**: define "K-match" precisely (per-pixel integer-cycle
  agreement on SNAPHU cc==1 mainland) and distinguish it from the paper's
  existing "100% mod-2π / 2.31 rad RMS" Capella numbers (a different, weaker
  metric). Also note (per S.S.) that K-match on an easy scene is a soft target —
  the real bar is *visual* correctness (no seam lines / blocks), not the %.
- **Discussion / closure**: align with the "tiling beats whole-image because
  the linear cost's global optimum contains the runaway" finding.

## handoff.md
- tl;dr table is stale (pre-anchor/cascade). Either refresh or point to
  paper/report_anchor_cascade.md as the current source of truth.

## Source of truth meanwhile
paper/report_anchor_cascade.md (NISAR + Atlanta, method + figures + numbers).
