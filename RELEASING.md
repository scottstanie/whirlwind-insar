# Releasing whirlwind-rs

End-to-end release flow for the Python distribution. The Rust crates
(`whirlwind-core`, `whirlwind-cli`) are not currently published to
crates.io; this doc covers PyPI and conda-forge.

## TL;DR

```bash
# 1. Bump versions in:
#      Cargo.toml                                  (workspace.package.version)
#      crates/whirlwind-py/pyproject.toml          (project.version)
#    Commit, open PR, merge to main.

# 2. Tag the release commit on main and push the tag.
git checkout main && git pull
git tag -a v0.1.1 -m "v0.1.1"
git push origin v0.1.1
```

The `Release` workflow (`.github/workflows/release.yml`) takes it from
there: builds abi3 wheels for linux (manylinux + musllinux, x86_64 +
aarch64), macOS (x86_64 + arm64), and Windows (x64), builds an sdist,
and publishes everything to PyPI via trusted publishing.

Conda-forge then auto-updates from PyPI (see below).

## One-time setup

### PyPI trusted publishing

The release workflow authenticates to PyPI via OIDC — no API token in
GitHub secrets. To set it up:

1. Reserve the project name by uploading at least one release manually,
   or by [registering a pending publisher][pending] before the first
   release.
2. On PyPI → *Manage* → *Publishing* → *Add a new pending publisher*
   (or *Add a publisher* if the project already exists), enter:
   - **PyPI project name:** `whirlwind-rs`
   - **Owner:** `scottstanie`
   - **Repository name:** `whirlwind-rs`
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
   `recipes/whirlwind-rs/meta.yaml`. Skeleton:

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
     home: https://github.com/scottstanie/whirlwind
     license: MIT
     license_file: LICENSE   # add a LICENSE file to the repo if missing
     summary: Bayesian min-cost-flow phase unwrapper for InSAR
     dev_url: https://github.com/scottstanie/whirlwind

   extra:
     recipe-maintainers:
       - scottstanie
   ```

3. Open a PR to `staged-recipes`. Reviewers will merge once linting
   passes; conda-forge bot then creates `conda-forge/whirlwind-rs-feedstock`
   and grants you maintainer rights.

After the feedstock exists, **no further action is needed in this
repo** — `regro-cf-autotick-bot` opens an automatic PR to the feedstock
on every new PyPI release. Merge it (or wait for the maintainer team to
merge) and conda-forge ships the new build.

[staged]: https://github.com/conda-forge/staged-recipes

## Cutting a release

1. Open a release-bump PR:
   - Bump `version` in `Cargo.toml` (workspace.package).
   - Bump `version` in `crates/whirlwind-py/pyproject.toml`.
   - Run `cargo update -p whirlwind-core -p whirlwind-cli -p whirlwind-py`
     to refresh `Cargo.lock` with the new versions.
   - Update the changelog if there is one.
2. Merge to `main` after CI is green.
3. Tag the release commit and push:

   ```bash
   git checkout main && git pull
   git tag -a v$(VERSION) -m "v$(VERSION)"
   git push origin v$(VERSION)
   ```

4. Watch the `Release` workflow. On success, the artifacts are visible
   under the workflow run and on PyPI.
5. Within a few hours, `regro-cf-autotick-bot` opens a PR against
   `conda-forge/whirlwind-rs-feedstock`. Review and merge.

## Re-running a failed release

`pypa/gh-action-pypi-publish` is invoked with `skip-existing: true`, so
re-running the workflow after a partial failure is safe — already
uploaded artifacts are skipped, missing ones are uploaded.

If the tag itself needs to be moved (don't do this once PyPI has
accepted a release — bump the version instead):

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
downloadable from the run page — useful for validating wheel builds
before tagging.
