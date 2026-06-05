"""File the pre-release audit findings as GitHub issues (idempotent).

Run from the whirlwind-insar repo root (gh infers the repo from the remote):
    python scripts/release_prep/file_issues.py

Re-running is safe: issues whose exact title already exists are skipped.
Created 2026-06-01 from the multi-agent pre-release audit.
"""

from __future__ import annotations

import json
import subprocess
import sys

LABELS = {
    "bug": ("d73a4a", "Incorrect behavior"),
    "release": ("0e8a16", "Blocks / part of the PyPI release"),
    "cleanup": ("c5def5", "Remove dead/vestigial code"),
    "api": ("1d76db", "Public Python/CLI surface"),
    "docs": ("0075ca", "Documentation"),
    "testing": ("fbca04", "Test coverage / validation"),
    "3d": ("5319e7", "3D / temporal-closure path"),
    "hygiene": ("bfdadc", "Repo hygiene / first impressions"),
    "ci": ("ededed", "CI / build"),
    "dolphin": ("d4c5f9", "Downstream dolphin wrapper"),
}

ISSUES: list[dict] = [
    {
        "title": "Bug: min_size_px silently dropped on the Goldstein path of unwrap_with_conncomp",
        "labels": ["bug", "release"],
        "body": """The default Goldstein branch ignores `min_size_px`.

`python/whirlwind/__init__.py:171-175` (the branch taken whenever `goldstein_alpha > 0`, i.e. the default 0.7) calls `_unwrap_with_conncomp_native(...)` passing `min_size_frac` and `max_ncomps` but **not** `min_size_px`. The no-Goldstein branch (lines 161-165) passes all three.

**Effect:** a caller setting e.g. `min_size_px=500` silently gets the native default (100) on the common path - a silent wrong result.

**Fix:** add `min_size_px=min_size_px` to the call at line 171. One line. Add a regression test that the value round-trips on the Goldstein path.

Found by the 2026-06-01 pre-release audit; verified by direct read.""",
    },
    {
        "title": "3D CRLB tiling path lacks the 2D robustness fixes (anchor / cascade / multi-shift / seam-repair)",
        "labels": ["3d"],
        "body": """`unwrap_crlb` (`crates/whirlwind-py/src/lib.rs:182`) → `whirlwind_core::tile::unwrap_crlb_tiled` (`crates/whirlwind-core/src/tile.rs:146`), which builds its **own** `TileGrid` and does plain median-based stitching. It does **not** call `unwrap_tiled_robust`, the global coarse anchor, the multi-scale cascade, the gated multi-shift re-solve, or seam-repair - i.e. it is missing all of #27 (anchor/cascade/multilook), #30 (multi-shift + seam-repair), and #31 (conncomp floor). It got only the earlier NaN-variance / cost-inversion fixes.

Since closure defaults off, this **is** the production-default 3D path.

**Decision needed:** either route CRLB tiling through `unwrap_tiled_robust`, or explicitly document the 3D per-IG tiler as the simpler/less-robust median-stitch path. Also consider exposing a `multilook=` kwarg on `unwrap_crlb` (the 2D `unwrap` has one).

Remember to update the dolphin wrapper if the 3D API changes (see the dolphin-sync issue).""",
    },
    {
        "title": "Remove vestigial cost env-var experiments (LLR / DEVIATION / HARD_CUT / CONVEX_OFFSET_*)",
        "labels": ["cleanup", "release"],
        "body": """`crates/whirlwind-core/src/cost/mod.rs` carries several dead or proven-worse env toggles. For a clean first release, delete the branches (no default-path behavior change):

- `WHIRLWIND_LLR_COST` - mathematically **identically zero** (Lee PDF is 2π-periodic; lines 425-428). Removing it collapses `cost_dir` from 3-way to 2-way and drops an unconditional LUT allocation.
- `WHIRLWIND_DEVIATION_COST` + the `need_raw` plumbing - documented negative (NISAR 92.5%→86.5%, halves coverage).
- `WHIRLWIND_HARD_CUT_THRESH` - pathological at 1.0, −1.4pp at 2.0.
- `WHIRLWIND_CONVEX_OFFSET_FLIP` - polarity ruled out (commit 9347acf).
- `WHIRLWIND_CONVEX_OFFSET_RAW` - the correct deviation offset is already the default; the toggle only reverts to the old incorrect behavior. Delete toggle + now-misleading comment.

Net: collapses `cost_dir` to a clean 2-way (PHASS / Carballo). Handle `WHIRLWIND_PHASS_COST` and `WHIRLWIND_COH_BIAS_CORRECT` as a separate decision (keep-and-document vs remove).""",
    },
    {
        "title": "Remove superseded pyramid module (pyramid.rs ~692 lines) + unwrap_pyramid export",
        "labels": ["cleanup"],
        "body": """`crates/whirlwind-core/src/pyramid.rs` (~692 lines) and the `unwrap_pyramid` export (`lib.rs:393-424`, `python/whirlwind/__init__.py:26`) are fully superseded by the #30 gated multi-shift re-solve (for fragmented A_016) and the #29 REUSE default (for corner cases). Not on any default path, no env toggle, not in the CLI.

**Keep** `paper/pyramid_aliasing.md` as the internal reference (it documents the real `L·g > π` aliasing finding). Note `auto_base_factor` (`pyramid.rs:349-375`) has a documented catastrophic blind spot and must never ship enabled.""",
    },
    {
        "title": "Python type stubs incomplete; prototype warnings invisible in .pyi",
        "labels": ["api", "release"],
        "body": """A first-time user running a type checker sees a half-typed package.

- `python/whirlwind/_native.pyi` is missing stubs for four **exported** functions: `unwrap_convex`, `unwrap_grounded`, `unwrap_pyramid`, `unwrap_reuse` (exported at `lib.rs:817-826`).
- `python/whirlwind/__init__.pyi` re-imports only 11 of 23 `__all__` entries; omits the wrappers `unwrap_crlb_stack` and `unwrap_with_conncomp`.
- "Prototype" / "Specialized - not a general substitute" warnings in the Rust docstrings (`unwrap_convex` lib.rs:341, `unwrap_reuse` 426, `unwrap_grounded` 448) don't reach the stubs.
- `unwrap_reuse` is now the **default tile solver** (`tile.rs:1512`) yet still docstring-labeled "Prototype" - stale; reword.

Decide which prototypes stay public; sync the stubs to the actual pyo3 signatures.""",
    },
    {
        "title": "Stale conncomp docs + CLI/Python naming divergence",
        "labels": ["docs", "api"],
        "body": """- `lib.rs:599` and `lib.rs:615` say components below `min_size_frac` are dropped, but the actual control is the absolute `min_size_px=100` floor (`min_size_frac` is a vestigial cap that only *raises* the floor on huge frames). Update the prose.
- CLI exposes `--min-component-frac` (`main.rs:103`) and hardcodes `min_size_px=100` (`main.rs:257`); Python splits `min_size_px` + `min_size_frac`. Pick one vocabulary, and consider hiding `min_size_frac` from the public Python signature.""",
    },
    {
        "title": "Consolidate overlapping tiling docs (TILING_DESIGN.md vs paper/tiling.md)",
        "labels": ["docs"],
        "body": """`TILING_DESIGN.md` (root + docs, ~226 lines, describes stages partly superseded) overlaps with `paper/tiling.md` (~247 lines, the authoritative account). `paper/handoff.md:21` even tells readers to "read paper/tiling.md first." Consolidate to one source of truth before release.""",
    },
    {
        "title": "Validate scene-tuned constants on diverse fragmented scenes (COH_CUT_FLOOR, seam_repair)",
        "labels": ["testing"],
        "body": """Several load-bearing constants are tuned on one fragmented scene (A_016). Risk = generalization to unseen decorrelated scenes.

- `COH_CUT_FLOOR=1.5e-3` (`tile.rs:450`) gates whether the **entire** multi-shift re-solve fires (the 55%→97% fix). A_016 ≈ 6.7e-3, clean ≤ 5.6e-4. A new scene landing in the gap silently won't trigger it.
- `seam_repair` constants `MIN_CLUSTER=500 / MIN_BLOCK=4000 / MARGIN=220 / MAX_WIN=1400` (`tile.rs:711-714`), A_016-tuned, no ablation. Clean GUNW frames are exact no-ops (proves *safety*, not *optimality*).

Test on 3-5 geographically diverse fragmented scenes; if margins don't hold, promote `COH_CUT_FLOOR` to an env var / document.""",
    },
    {
        "title": "Sparse unwrapper: no Python-level test + no max_edge_length heuristic",
        "labels": ["testing"],
        "body": """`unwrap_sparse` is public API but has zero Python-level test coverage (all 5 sparse tests are Rust-internal). The mandatory `max_edge_length` cutoff (`sparse.rs:42-47`) has no heuristic - a wrong guess produces "garbage" per the docstring. Integration seed is hardcoded to pixel 0 (`sparse.rs:187,233`); the 50-iteration `primal_dual` cap (`sparse.rs:81`) is unjustified.

Add a ~30-50 line Python smoke test before exposing on PyPI; document the deliberate single-pass design (it shares none of the architectural 2D fixes, by design).""",
    },
    {
        "title": "conncomp test coverage gaps",
        "labels": ["testing"],
        "body": """`crates/whirlwind-core/src/conncomp.rs:187-277` has no test that `max_ncomps` caps output, no test of the `cost_threshold` effect, and no test of descending-size renumbering. Add cheap synthetic tests to catch silent regressions.""",
    },
    {
        "title": "3D closure off-by-default needs a high-visibility note; README time-series claim vs CI",
        "labels": ["docs", "3d"],
        "body": """Tree closure regresses on real data (`ATBD-3d.md:370-373`: 2D + reference anchor = 2.29 rad median; + tree closure = 5.61 rad), so `unwrap_stack.py` defaults `closure_mode="off"`. This is honest and correct, but needs a prominent README/docstring note: **3D is not a closed-loop unwrapper by default; closure is opt-in and known to regress on tight unwraps.**

Also: `README.md:3` headlines "full phase-linked time-series stacks," but CI validates only 2D - there is no 3D CLI subcommand and no 3D CI test. Soften the headline or add a lightweight 3D smoke test. (`closure::refine_mcf` is a properly-gated diagnostic - keep, leave flagged.)""",
    },
    {
        "title": "Repo hygiene: drop top-level doc symlinks (keep real files in docs/) + remove docs/requirements.txt",
        "labels": ["hygiene"],
        "body": """Per maintainer:
- The top-level symlinks (`ENV_VARS.md`, `PERFORMANCE.md`, `TILING_DESIGN.md`, `figures` → `docs/...`) were a mkdocs-build workaround. Keep the real files in `docs/` and remove the top-level symlinks. Update any references (README, handoff) that point at the root paths.
- Remove `docs/requirements.txt` in favor of the `pyproject.toml` docs dependency-group. **Check `.readthedocs.yaml`** first - if it installs `docs/requirements.txt`, switch it to the pyproject docs group / uv before deleting.

Leave the `scripts/` research record intact. The 294 KB session transcript at repo root is already `*.txt`-ignored - just don't commit it.""",
    },
    {
        "title": "Release prep: bump Node20 actions, verify py3.14 wheels, ship to PyPI",
        "labels": ["release", "ci"],
        "body": """- Bump `actions/checkout` + `actions/setup-python` to current majors before the GitHub Node20 deprecation (June 16).
- Verify py3.14 wheel / maturin support on the macOS + Windows runners (`.github/workflows/CI.yml:52`) or cap at 3.11.
- Confirm `release.yml` has the `PYPI_API_TOKEN`/trusted-publishing secret and the wheel build matrix is correct.
- Dist name `whirlwind-insar`, import module `whirlwind`.""",
    },
    {
        "title": "[dolphin] Sync downstream wrapper + fix always-skipped integration test",
        "labels": ["dolphin"],
        "body": """Downstream dolphin (`/Users/staniewi/repos/dolphin`) embeds whirlwind:
- `src/dolphin/unwrap/_whirlwind.py` - calls `ww.unwrap_with_conncomp(igram, corr, nlooks, mask=)` with all defaults.
- `WhirlwindOptions` (`src/dolphin/workflows/config/_unwrap_options.py:140`) - exposes only `num_threads`.

**(a) Bug:** `tests/test_unwrap.py:13` and `tests/test_show_versions.py:12` gate on `find_spec("whirlwind_rs")`, but the module is `whirlwind` → `TestWhirlwind` is always skipped. Fix the name.
**(b) Opportunity:** surface conncomp params (`min_size_px`), `multilook`, and `goldstein_alpha` in `WhirlwindOptions` so dolphin users get the new levers.
**(c) Standing rule:** update `_whirlwind.py` + `WhirlwindOptions` whenever whirlwind's API/defaults change. The current 2D cleanup (bug fix, env-var purge, pyramid removal, stub sync) does **not** change dolphin behavior (defaults + `WHIRLWIND_NUM_THREADS` only).""",
    },
]


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def main() -> int:
    # Ensure labels exist (idempotent; --force updates color/desc if present).
    for name, (color, desc) in LABELS.items():
        run(
            [
                "gh",
                "label",
                "create",
                name,
                "--color",
                color,
                "--description",
                desc,
                "--force",
            ],
            check=False,
        )

    existing = run(
        ["gh", "issue", "list", "--state", "all", "--limit", "300", "--json", "title"]
    )
    have = {i["title"] for i in json.loads(existing.stdout or "[]")}

    for spec in ISSUES:
        if spec["title"] in have:
            print(f"skip (exists): {spec['title']}")
            continue
        cmd = [
            "gh",
            "issue",
            "create",
            "--title",
            spec["title"],
            "--body",
            spec["body"],
        ]
        for lbl in spec["labels"]:
            cmd += ["--label", lbl]
        res = run(cmd, check=False)
        out = (res.stdout or res.stderr).strip()
        print(
            f"{'OK  ' if res.returncode == 0 else 'FAIL'}: {spec['title']}\n      {out}"
        )
        if res.returncode != 0:
            return res.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
