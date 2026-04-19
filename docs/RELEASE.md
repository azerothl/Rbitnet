# Release process

Workspace version lives in the root `Cargo.toml` (`[workspace.package].version`). Keep it aligned with tags and notes.

Overall **feature and gap tracking** for the repo: [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md).

## Prebuilt binaries (GitHub Actions)

Pushing a tag matching `v*` (for example `v0.2.0`) runs [`.github/workflows/release.yml`](../.github/workflows/release.yml). It builds **`rbitnet-server`** in release mode (`cargo build -p bitnet-server --release --locked`) on **Linux (x86_64)**, **Windows (x86_64)**, and **macOS** (architecture matches the runner, e.g. `arm64` on Apple Silicon), then uploads archives to a **GitHub Release** for that tag:

- `rbitnet-server-vX.Y.Z-linux-<arch>.tar.gz`
- `rbitnet-server-vX.Y.Z-macos-<arch>.tar.gz`
- `rbitnet-server-vX.Y.Z-windows-<arch>.zip`

Requirements: `Cargo.lock` must be committed so `--locked` succeeds.

## Steps

1. Update `CHANGELOG` or GitHub release notes: user-visible fixes, new env vars, breaking HTTP changes.
2. Bump `version` in `Cargo.toml` (semver).
3. Commit and tag: `git tag v0.x.y && git push origin v0.x.y` (or create the tag from the GitHub UI). This triggers the release workflow and attaches the binaries.
4. CI (see `.github/workflows/ci.yml`) should be green on the release branch before tagging.

## Semver guidance

- **MAJOR**: breaking HTTP API or `Engine` API changes.
- **MINOR**: new endpoints, new env vars with defaults, new optional behavior.
- **PATCH**: bug fixes, docs, dependency updates without behavior change.
