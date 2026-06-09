# Releasing whirlwind

End-to-end release flow for the Python distribution. The Rust crates
(`whirlwind-core`, `whirlwind-cli`) are not currently published to
crates.io; this doc covers PyPI and conda-forge.

## TL;DR

Run the **Bump version** workflow (Actions -> Bump version -> Run workflow) and
pick `patch`, `minor`, or `major`. That is the whole release. It bumps the
single version source (`Cargo.toml`), commits it, and pushes a `v*` tag, which
triggers the `Release` workflow to build and publish the wheels.

The version lives in exactly one place, `Cargo.toml` `[workspace.package].version`.
`pyproject.toml` declares `dynamic = ["version"]`, so maturin reads the version
from `Cargo.toml` at build time; there is no second copy to keep in sync.

The `Release` workflow (`.github/workflows/release.yml`) takes it from
there: builds abi3 wheels for linux (manylinux + musllinux, x86_64 +
aarch64), macOS (x86_64 + arm64), and Windows (x64), builds an sdist,
and publishes everything to PyPI via trusted publishing.

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

### Conda-forge feedstock (first release only)

conda-forge does not auto-create packages; you submit a recipe once,
and after acceptance the feedstock auto-updates on every PyPI release.

1. Wait for the first PyPI release to be live.
2. Fork [`conda-forge/staged-recipes`][staged] and add a recipe under
   `recipes/whirlwind-insar/meta.yaml`. Skeleton:

   ```yaml
   {% set name = "whirlwind-insar" %}
   {% set version = "0.1.0" %}

   package:
     name: {{ name|lower }}
     version: {{ version }}

   source:
     url: https://pypi.org/packages/source/{{ name[0] }}/{{ name }}/{{ name }}-{{ version }}.tar.gz
     sha256: <fill in from the PyPI sdist>

   build:
     number: 0
     script: {{ PYTHON }} -m pip install . -vv --no-deps --no-build-isolation
     skip: true  # [py<311]

   requirements:
     build:
       - {{ compiler('rust') }}
       - {{ compiler('c') }}
     host:
       - python
       - pip
       - maturin >=1.5,<2
     run:
       - python
       - numpy >=1.21

   test:
     imports:
       - whirlwind
     commands:
       - pip check
     requires:
       - pip

   about:
     home: https://github.com/scottstanie/whirlwind-insar
     license: MIT
     license_file: LICENSE
     summary: Fast Rust-backed 2D InSAR phase unwrapper
     dev_url: https://github.com/scottstanie/whirlwind-insar

   extra:
     recipe-maintainers:
       - scottstanie
   ```

3. Open a PR to `staged-recipes`. Reviewers will merge once linting
   passes; conda-forge bot then creates `conda-forge/whirlwind-insar-feedstock`
   and grants you maintainer rights.

After the feedstock exists, **no further action is needed in this
repo** - `regro-cf-autotick-bot` opens an automatic PR to the feedstock
on every new PyPI release. Merge it (or wait for the maintainer team to
merge) and conda-forge ships the new build.

[staged]: https://github.com/conda-forge/staged-recipes

## Cutting a release

The version is defined once, in `Cargo.toml` `[workspace.package].version`, and
read by maturin into the wheel (pyproject uses `dynamic = ["version"]`). So a
release is just a version bump plus a `v*` tag; the `Bump version` workflow does
both.

### Default path (CI)

1. Actions -> **Bump version** -> *Run workflow*. Pick `patch`, `minor`, or
   `major` (or type an explicit version). It runs `cargo set-version`, commits
   the bumped `Cargo.toml` and `Cargo.lock` to `main`, and pushes a `v*` tag.
2. The tag triggers the `Release` workflow. Watch it; on success the artifacts
   are on the run page and on PyPI.
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
