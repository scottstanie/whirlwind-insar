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

The `ATBD-3d.md` in the repo root is the working long-form document.
This letter is a compressed view focused on the publishable claims:

1. CRLB-weighted arc cost replacing Lee coherence (§II-A in the letter,
   §4 in the ATBD).
2. Residue-grid boundary fix (§II-B in the letter, §10.1 in the ATBD).
3. Per-pixel quality from closure residuals (§II-C, ATBD §10.5 backlog).
4. Tiled mode + virtual ground-node MCF (§II-D, ATBD §10.3, §10.6).

Keep the two in sync when claims change.

## Figures

Symlinked from `docs/figures/` at commit time. Regenerate via
`scripts/reproduce.sh` and `scripts/bench_tile_memory.py`. Placeholder if
needed — the letter as written embeds three figures that all exist:

- `fig_palos_verdes_full_wrapped_vs_unwrapped.png`
- `fig_palos_verdes_1024_wrapped_vs_unwrapped.png`
- `fig_tile_memory.png`

## Still TODO before submission

- Numbers for the SBAS-inversion comparison vs SNAPHU+SBAS (not in the
  current results table).
- Author affiliations / acknowledgements section.
- The reference list is a placeholder set; need to be filled in with
  the actual journal-style citations for SNAPHU, EMI, EVD/SqueeSAR,
  spurt, LAMBDA, Carballo costs, Whirlwind, and dolphin.
- A possible §III on tile-stitching robustness (the 256+64 anomaly is
  an interesting limitation worth a paragraph; covered in ATBD §10.6).
- Decide whether to also discuss the ground-node grounded variant's
  failure on real data; currently a paragraph in §II-D.
