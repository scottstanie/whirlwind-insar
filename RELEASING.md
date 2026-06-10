# Releasing whirlwind

End-to-end release flow for the Python distribution. The Rust crates (`whirlwind-core`, `whirlwind-cli`) are not currently published to crates.io; this doc covers PyPI and conda-forge.

In addition to the wheels, the `Release` workflow attaches prebuilt `whirlwind` CLI binaries (linux x86_64/aarch64, macOS x86_64/arm64, Windows x64) to the GitHub Release for each tag, so non-Python users can download a single executable. This is fully automatic - no extra setup beyond the `v*` tag.

## TL;DR

Run the **Bump version** workflow (Actions -> Bump version -> Run workflow) and pick `patch`, `minor`, or `major`. That is the whole release. It bumps the single version source (`Cargo.toml`), commits it, and pushes a `v*` tag, which triggers the `Release` workflow to build and publish the wheels.

The version lives in exactly one place, `Cargo.toml` `[workspace.package].version`.  `pyproject.toml` declares `dynamic = ["version"]`, so maturin reads the version from `Cargo.toml` at build time; there is no second copy to keep in sync.

The `Release` workflow (`.github/workflows/release.yml`) takes it from there: builds abi3 wheels for linux (manylinux + musllinux, x86_64 + aarch64), macOS (x86_64 + arm64), and Windows (x64), builds an sdist, publishes everything to PyPI via trusted publishing, and uploads the prebuilt CLI binaries to the GitHub Release.

Conda-forge then auto-updates from PyPI (see below).

## One-time setup

### PyPI trusted publishing

The release workflow authenticates to PyPI via OIDC - no API token in
GitHub secrets. To set it up:

1. Decide how to create the PyPI project. A manual first upload creates
   the project and claims the name immediately. A [pending publisher][pending]
   lets the first trusted-publishing upload create the project, but it does
   **not** reserve the name until that upload succeeds.
2. On PyPI → *Manage* → *Publishing* → *Add a new pending publisher*
   (or *Add a publisher* if the project already exists), enter:
   - **PyPI project name:** `whirlwind-insar`
   - **Owner:** `scottstanie`
   - **Repository name:** `whirlwind-insar`
   - **Workflow name:** `release.yml`
   - **Environment name:** `pypi`
3. In GitHub → *Settings* → *Environments* → *New environment*, create
   an environment called `pypi`. Optionally add required reviewers so
   publishes require manual approval.

[pending]: https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/

## Cutting a release

The version is defined once, in `Cargo.toml` `[workspace.package].version`, and read by maturin into the wheel (pyproject uses `dynamic = ["version"]`). So a release is just a version bump plus a `v*` tag; the `Bump version` workflow does both.

### Default path (CI)

1. Actions -> **Bump version** -> *Run workflow*. Pick `patch`, `minor`, or
   `major` (or type an explicit version). It runs `cargo set-version`, commits the bumped `Cargo.toml` and `Cargo.lock` to `main`, and pushes a `v*` tag.
2. The tag triggers the `Release` workflow. Watch it; on success the artifacts are on the run page and on PyPI.
3. Within a few hours, `regro-cf-autotick-bot` opens a PR against
   `conda-forge/whirlwind-insar-feedstock`. Review and merge.

### Local path (or if `main` is protected against direct pushes)

```bash
cargo install cargo-edit          # one-time; provides `cargo set-version`
git checkout main && git pull
cargo set-version --bump patch    # bumps Cargo.toml + Cargo.lock together
git commit -am "Release v$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"
# push to main (directly, or open a PR and merge), then tag the merged commit:
v=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
git tag -a "v$v" -m "v$v" && git push origin main "v$v"
```

## Re-running a failed release

`pypa/gh-action-pypi-publish` is invoked with `skip-existing: true`, so
re-running the workflow after a partial failure is safe - already
uploaded artifacts are skipped, missing ones are uploaded.

If the tag itself needs to be moved (don't do this once PyPI has
accepted a release - bump the version instead):

```bash
git tag -d v$(VERSION)
git push origin :refs/tags/v$(VERSION)
git tag -a v$(VERSION) -m "v$(VERSION)"
git push origin v$(VERSION)
```

## Manual / smoke-test builds

Run the `Release` workflow via *Actions* → *Release* → *Run workflow*.
On `workflow_dispatch` the publish job is skipped (`if:` guard on the
tag ref), but every wheel/sdist artifact is still produced and
downloadable from the run page - useful for validating wheel builds
before tagging.
