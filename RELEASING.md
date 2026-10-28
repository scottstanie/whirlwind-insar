# Releasing whirlwind

The version is defined in `Cargo.toml` under `[workspace.package].version`.
Maturin reads it when building the wheel (`pyproject.toml` uses
`dynamic = ["version"]`). To release:

1. Actions -> **Bump version** -> *Run workflow*. Pick `patch`, `minor`, or `major` (or type an explicit version). It runs `cargo set-version`, commits the bumped `Cargo.toml` and `Cargo.lock` to `main`, and pushes a `v*` tag.
2. The tag triggers the `Release` workflow. Watch it; on success the artifacts are on the run page and on PyPI.
3. Within a few hours, `regro-cf-autotick-bot` opens a PR against `conda-forge/whirlwind-insar-feedstock` to update [conda forge](https://anaconda.org/channels/conda-forge/packages/whirlwind-insar/files)

The `Release` workflow also attaches prebuilt CLI binaries (Linux x86_64/aarch64, macOS x86_64/arm64, Windows x64) to the GitHub release.
