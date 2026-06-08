# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- [CHANGELOG.md](CHANGELOG.md) (this file), [docs/ENV_REFERENCE.md](docs/ENV_REFERENCE.md), [docs/profiling/](docs/profiling/README.md), [docs/INFERENCE_STACK_V2.md](docs/INFERENCE_STACK_V2.md).
- `rbitnet serve` / `rbitnet-server`: optional `--api-key` and `--bind` when env vars are unset.
- `RBITNET_MAX_PROMPT_TOKENS` optional HTTP guard; `Engine::count_prompt_tokens` / `ModelExecutor::count_prompt_tokens`.
- Startup validation for zero-valued caps; bundle install validates GGUF/tokenizer files on disk.
- Optional `optional_engine_load_from_env_smoke` (`RBITNET_TEST_GGUF` + tokenizer beside GGUF).

### Changed

- Documentation: benchmarks baseline procedure, DEPLOYMENT rate-limit sketch, LIMITATIONS timeout semantics, STATUS/USAGE cross-links.
