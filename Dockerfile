# Reference image for rbitnet-server (Phase 6 — optional Docker).
# Build: docker build -t rbitnet:local .
# Run (mount GGUF + tokenizer): docker run --rm -e RBITNET_MODEL=/model/model.gguf -v /path/on/host:/model:ro -p 8080:8080 rbitnet:local

FROM rust:1-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build -p bitnet-server --bin rbitnet-server --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/rbitnet-server /usr/local/bin/rbitnet-server
ENV RBITNET_BIND=0.0.0.0:8080
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/rbitnet-server"]
