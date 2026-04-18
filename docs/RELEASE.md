# Release process

Workspace version lives in the root `Cargo.toml` (`[workspace.package].version`). Keep it aligned with tags and notes.

## Steps

1. Update `CHANGELOG` or GitHub release notes: user-visible fixes, new env vars, breaking HTTP changes.
2. Bump `version` in `Cargo.toml` (semver).
3. Commit and tag: `git tag v0.x.y && git push origin v0.x.y` (or create the tag from the GitHub UI).
4. CI (see `.github/workflows/ci.yml`) should be green on the release branch.

## Semver guidance

- **MAJOR**: breaking HTTP API or `Engine` API changes.
- **MINOR**: new endpoints, new env vars with defaults, new optional behavior.
- **PATCH**: bug fixes, docs, dependency updates without behavior change.
