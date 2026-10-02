# Release process

Workspace version lives in the root `Cargo.toml` (`[workspace.package].version`). Keep it aligned with tags and notes.

Overall **feature and gap tracking** for the repo: [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md).

## Prebuilt binaries (GitHub Actions)

Pushing a tag matching `v*` (for example `v0.2.0`) runs [`.github/workflows/release.yml`](../.github/workflows/release.yml). It builds **`rbitnet-server`** and the **`rbitnet`** CLI (`cargo build -p bitnet-server -p rbitnet-cli --release --locked`) on **Linux (x86_64)**, **Windows (x86_64)**, and **macOS** (architecture matches the runner, e.g. `arm64` on Apple Silicon), then uploads archives to a **GitHub Release** for that tag. Each archive contains **both** executables:

- `rbitnet-server-vX.Y.Z-linux-<arch>.tar.gz` — includes `rbitnet-server` and `rbitnet`
- `rbitnet-server-vX.Y.Z-macos-<arch>.tar.gz` — includes `rbitnet-server` and `rbitnet`
- `rbitnet-server-vX.Y.Z-windows-<arch>.zip` — includes `rbitnet-server.exe` and `rbitnet.exe`

Requirements: `Cargo.lock` must be committed so `--locked` succeeds.

## Steps

1. Update [CHANGELOG.md](../CHANGELOG.md) and GitHub release notes: user-visible fixes, new env vars, breaking HTTP changes.
2. Bump `version` in `Cargo.toml` (semver).
3. Commit and tag: `git tag v0.x.y && git push origin v0.x.y` (or create the tag from the GitHub UI). This triggers the release workflow and attaches the binaries.
4. CI (see `.github/workflows/ci.yml`) should be green on the release branch before tagging — including the **`smoke-openai`** job (Akasha OpenAI contract via stub).
5. **Akasha contract checklist** (see [AKASHA_INFER.md](AKASHA_INFER.md)): SSE chat, `/v1/models`, frozen [AKASHA_METRICS.md](AKASHA_METRICS.md) series, recipe `bitnet-b158`, and `scripts/smoke_openai.sh` documented for operators.
6. **Benchmarks checklist (issue #23 / follow-up #41):** before calling a release “measured”, append at least one **real GGUF** row (not stub) to [BENCHMARKS_RESULTS.md](BENCHMARKS_RESULTS.md) for the frozen TinyLlama Q4_K_M reference (or document why skipped). Prefer also running `scripts/compare_llamacpp_rbitnet.sh` when `llama-bench` is available. Update [MODEL_MATRIX.md](MODEL_MATRIX.md) RSS/tok/s cells to match. Stub-only rows must stay labeled as API overhead. Keep the **#41 follow-up table** in `BENCHMARKS_RESULTS.md` honest: mark unfilled Llama-3.2 / BitNet / feature-flag deltas as **non mesuré** rather than omitting them. Record host + `rustc` + git SHA on every published row.
7. Release archives are the unified **server+CLI** artefacts (`rbitnet` + `rbitnet-server` in one zip/tarball). Homebrew / WinGet formulas under `packaging/` remain **experimental** until real release SHA-256 values replace the placeholders (see [RELEASE_PACKAGING.md](RELEASE_PACKAGING.md)).

## Semver guidance

- **MAJOR**: breaking HTTP API or `Engine` API changes.
- **MINOR**: new endpoints, new env vars with defaults, new optional behavior.
- **PATCH**: bug fixes, docs, dependency updates without behavior change.
