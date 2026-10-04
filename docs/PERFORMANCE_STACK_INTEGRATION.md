# Combined performance candidate

This branch stages the optional KV formats, F32 paged attention hoisting,
continuous Native Llama decoder, exact Qwen draft verifier, persistent F32
Llama/Qwen context store, ordered GPT normalization, structured-output refusals
and the leased MoE arena. Individual measurements belong to their source
branches. They do not validate this combined build.

The merge preserves both sides of added Native APIs, context fields and test
modules. Structured-output refusal runs before continuous or draft dispatch.
The following cross-feature protections were added here:

- Native continuous kernels and portable Llama transport refuse encoded KV
  before scratch allocation, copying or sequence mutation.
- Continuous Llama options refuse context tiers until the worker participates
  in the same capture/restore policy. The ordinary tiered runtime remains the
  supported path for that store.
- A validated Qwen portable import invalidates any outstanding speculative
  nonce. A refused malformed import leaves the existing nonce intact. A failed
  accepted transport poisons speculation until reset or successful replacement.

Actual-model preservation fixtures cover F16/Q8 dense/paged owners with graphs
off/on, unchanged page counters and continuation logits after refusal, plus
Qwen imported GDN/convolution state and stale/fresh nonce behavior. They are
prepared, not yet compiled or executed. An additional option test verifies
refusal before Native initialization. Optional features retain their defaults.

Fresh workspace, Native, model, HTTP/SSE and cross-engine CPU/GPU measurements
remain pending. This branch is not a release, a claim of engine parity or
completion of the performance epic. Learned prediction, KIVI, MoE continuous
serving, true tokenizer-aware grammar and non-NVIDIA validation are still
separate work.
