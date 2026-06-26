# Releasing whirlwind

Run the **Bump version** workflow (Actions -> Bump version -> Run workflow) and pick `patch`, `minor`, or `major`. That is the whole release. It bumps the single version source (`Cargo.toml`), commits it, and pushes a `v*` tag, which triggers the `Release` workflow to build and publish the wheels.

The version is defined once, in `Cargo.toml` `[workspace.package].version`, and read by maturin into the wheel (pyproject uses `dynamic = ["version"]`). So a release is just a version bump plus a `v*` tag; the `Bump version` workflow does both.

1. Actions -> **Bump version** -> *Run workflow*. Pick `patch`, `minor`, or `major` (or type an explicit version). It runs `cargo set-version`, commits the bumped `Cargo.toml` and `Cargo.lock` to `main`, and pushes a `v*` tag.
2. The tag triggers the `Release` workflow. Watch it; on success the artifacts are on the run page and on PyPI.
3. Within a few hours, `regro-cf-autotick-bot` opens a PR against `conda-forge/whirlwind-insar-feedstock` to update [conda forge](https://anaconda.org/channels/conda-forge/packages/whirlwind-insar/files)

In addition to the wheels, the `Release` workflow attaches prebuilt `whirlwind` CLI binaries (linux x86_64/aarch64, macOS x86_64/arm64, Windows x64) to the GitHub Release for each tag, so non-Python users can download a single executable. This is fully automatic - no extra setup beyond the `v*` tag.
