---
title: Rbitnet
description: Rbitnet is a Rust inference runtime for GGUF models with an OpenAI-compatible HTTP server.
---

# Rbitnet

Rbitnet is a Rust inference runtime for GGUF models with an OpenAI-compatible HTTP server. Compatibility depends on the model architecture, tensor layout and execution backend.

## Choose a model and backend

Check the GGUF architecture matrix and known limitations before selecting a model. Use the deployment and environment references when configuring the server. Benchmark results are tied to the named model, runtime and hardware.

## Sources

- [GGUF architecture matrix](https://github.com/azerothl/Rbitnet/blob/main/docs/ARCHITECTURE_GGUF_MATRIX.md)
- [Known limits](https://github.com/azerothl/Rbitnet/blob/main/docs/LIMITATIONS.md)
- [Deployment](https://github.com/azerothl/Rbitnet/blob/main/website/docs/content/deployment.md)
