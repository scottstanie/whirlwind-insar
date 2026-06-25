# syntax=docker/dockerfile:1
#
# Container for the NISAR GUNW comparison (aws-batch/compare_gunw.py).
#
# Unlike the top-level ./Dockerfile (which ships only the pure-Rust `whirlwind`
# CLI), this image bundles the **Python** `whirlwind` package, because the GUNW
# comparison reads HDF5 (h5py), re-wraps the production phase, and calls the
# public `whirlwind.unwrap(...)` API. The wheel is built from this repo's source
# so reviewers test the latest code, not a PyPI release.
#
# Build (from the repo root, so the build context is the whole repo):
#   docker build -f aws-batch/Dockerfile -t whirlwind-gunw .
#
# Run locally on one product (mount a dir for inputs/outputs):
#   docker run --rm -e EARTHDATA_TOKEN=$EARTHDATA_TOKEN -v "$PWD/out:/work/out" \
#     whirlwind-gunw \
#     https://nisar.asf.earthdatacloud.nasa.gov/.../<ID>.h5 --out-dir /work/out

# ---- Stage 1: build the whirlwind wheel (needs Rust + maturin) -------------
FROM python:3.12-bookworm AS builder
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
# Pin matches rust-toolchain.toml (1.95.0); rustup also reads that file from the
# repo, so cargo/maturin use the same toolchain CI does.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain 1.95.0 --profile minimal
WORKDIR /src
COPY . .
RUN pip install --no-cache-dir "maturin>=1.7,<2.0" \
    && maturin build --release --out /wheels

# ---- Stage 2: slim runtime (no Rust, no compilers) -------------------------
FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="whirlwind-gunw" \
      org.opencontainers.image.description="whirlwind vs NISAR L2 GUNW comparison" \
      org.opencontainers.image.source="https://github.com/scottstanie/whirlwind-insar"

# Runtime deps for compare_gunw.py. All ship manylinux wheels, so the slim
# image needs no build toolchain. Split into two layers: earthaccess pulls
# aiobotocore, which pins botocore to a range; installing boto3 in a *second*
# pip call lets it resolve against the already-installed botocore instead of
# backtracking over dozens of boto3 versions.
RUN pip install --no-cache-dir \
    "numpy>=1.21" \
    "h5py>=3.8" \
    "matplotlib>=3.7" \
    "pandas>=2.0" \
    "earthaccess>=0.11" \
    "requests>=2.31" \
    "psutil>=5.9"
RUN pip install --no-cache-dir boto3

# Install the freshly-built whirlwind wheel.
COPY --from=builder /wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Headless matplotlib + unbuffered logs (so AWS CloudWatch shows progress live).
ENV MPLBACKEND=Agg PYTHONUNBUFFERED=1

COPY aws-batch/compare_gunw.py /opt/whirlwind/compare_gunw.py
COPY aws-batch/sample_granules.txt /opt/whirlwind/sample_granules.txt
WORKDIR /work
ENTRYPOINT ["python", "/opt/whirlwind/compare_gunw.py"]
CMD ["--help"]
