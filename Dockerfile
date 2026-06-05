# syntax=docker/dockerfile:1
#
# Containerized whirlwind CLI (the `whirlwind` binary from crates/whirlwind-cli).
# Pure Rust - the only raster I/O is the `tiff` crate (0.10), so there is NO GDAL
# / libtiff / system geo dependency: the runtime image is just glibc + the binary.
#
#   docker build -t whirlwind .
#   docker run --rm whirlwind --help
#   docker run --rm -v "$PWD:/data" whirlwind unwrap \
#       --phase /data/wrapped.tif --cor /data/cor.tif --nlooks 16 \
#       --out /data/unw.tif --conncomp /data/conncomp.tif
#
# Build stage: compile only the CLI (-p whirlwind-cli pulls in whirlwind-core but
# NOT whirlwind-py, so no PyO3 / Python toolchain is needed).
FROM rust:1-bookworm AS builder
WORKDIR /src
COPY . .
RUN cargo build --release -p whirlwind-cli

# Runtime stage: slim Debian (glibc) + the single static-ish binary (~a few MB).
FROM debian:bookworm-slim
LABEL org.opencontainers.image.title="whirlwind" \
      org.opencontainers.image.description="Rust InSAR 2D phase unwrapper (CLI)" \
      org.opencontainers.image.source="https://github.com/scottstanie/whirlwind-insar"
COPY --from=builder /src/target/release/whirlwind /usr/local/bin/whirlwind
ENTRYPOINT ["whirlwind"]
CMD ["--help"]
