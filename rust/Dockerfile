FROM rust:1-slim-bookworm AS build
WORKDIR /app
# Cache dependencies separately from source
COPY Cargo.toml ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release && rm -rf src
COPY src ./src
RUN touch src/main.rs && RUSTC_VERSION="$(rustc --version)" cargo build --release

FROM debian:bookworm-slim
COPY --from=build /app/target/release/bench-rust /usr/local/bin/bench-rust
RUN useradd -m app
USER app
EXPOSE 8000
CMD ["bench-rust"]
