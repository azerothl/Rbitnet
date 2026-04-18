# Limitations

This page sets expectations for performance, formats, and architectures. For compatibility rules when **exporting** checkpoints, see [TRAINING_AND_COMPATIBILITY.md](TRAINING_AND_COMPATIBILITY.md).

## Performance

- **CPU-first:** Throughput is limited by single-node CPU performance; there is no GPU backend in this repository.
- **No distributed inference:** One process loads one GGUF for real generation (stub/toy modes are separate smoke paths).

## GGUF / GGML

- **Unknown `ggml_type` values** fail with a clear error (`UnsupportedGgmlType`) once a tensor is dequantized; see `crates/bitnet-core/src/ggml/types.rs` for layout coverage.
- **Tensor names** must follow llama.cpp-style conventions; odd exports may need renaming or loader extensions.

## Tokenizer

- The bundled generation path loads **`tokenizer.json`** via the Hugging Face `tokenizers` crate. If your workflow only produced `tokenizer.model`, convert or obtain a compatible `tokenizer.json` for Rbitnet.

## HTTP server

- **Inference timeout:** Long generations are cut off with HTTP 504 after `RBITNET_INFERENCE_TIMEOUT_SECS` (the blocking task may still finish in the thread pool afterward).
- **Concurrency:** At most `RBITNET_MAX_CONCURRENT` generations at once; extra requests receive HTTP 503.
- **Auth:** When `RBITNET_API_KEY` is set, protect upstream with TLS and a reverse proxy for anything beyond localhost.
