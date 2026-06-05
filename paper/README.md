# IEEE GRSL paper draft

Five-page letter draft based on the
[IEEE Geoscience and Remote Sensing Letters template](https://www.overleaf.com/latex/templates/ieee-geoscience-and-remote-sensing-letters-official-ieee-latex-template/zckwcmrxbvhg).

## Build

```
cd paper
latexmk -pdf whirlwind3d.tex
```

Tested with TeX Live 2024+. `IEEEtran.cls` is bundled with all standard
TeX Live / MacTeX installs.

## Source of truth for content

The repo-root [`ATBD-3d.md`](../ATBD-3d.md) is the working long-form
document; this letter is a compressed view focused on the publishable
claims:

1. CRLB-weighted arc cost replacing Lee coherence (§II-A here, §4 of the ATBD).
2. Residue-grid boundary fix (§II-B here, §10.1 of the ATBD).
3. Per-pixel quality from closure residuals (§II-C here, §10.5 of the ATBD).
4. Tiled mode + virtual ground-node MCF (§II-D here, §10.3 and §10.6 of the ATBD).

Keep the two in sync when claims change.

## Figures

Copies of the three figures live in `paper/figures/`, mirrored from
`docs/figures/` so the LaTeX build is self-contained. Regenerate the
sources with:

- `scripts/reproduce.sh --full` - `fig_palos_verdes_full_wrapped_vs_unwrapped.png`
- `scripts/reproduce.sh` (1024² tile) - `fig_palos_verdes_1024_wrapped_vs_unwrapped.png`
- `scripts/bench_tile_memory.py` - `fig_tile_memory.png` (memory-profile plot)

After regenerating, copy the new versions into `paper/figures/` before
rebuilding the PDF.
