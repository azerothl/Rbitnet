# Structured output and tools

Structured generation is not implemented for the shipped tokenizers. OpenAI `json_object`/`json_schema` and tool-call requests return HTTP 501 with a JSON error, including when `stream: true`. The refusal occurs before acquiring an inference permit or opening SSE. Anthropic tool generation and incoming `tool_use`/`tool_result` blocks are also refused.

The current research mask treats token ID 123 as ASCII `{`. With Hugging Face BPE or SentencePiece, that ID can denote an unrelated subword. A standalone JSON/schema post-validator does not make generation constrained, and a streaming response cannot retract text already emitted. The high-level Engine, real runtime/executor entry points and API now reject these requests rather than applying that mask or ignoring tool fields.

- `RBITNET_STRUCTURED_OUTPUT=json`, `tool`, `tool-call`, or `tool_call` returns an Engine/CLI error or HTTP 501. Default `off` keeps ordinary generation available.
- OpenAI `response_format: {"type":"text"}` remains ordinary generation. `json_object` and `json_schema` return `structured_output_not_supported`.
- Nonempty `tools`, forced/named `tool_choice`, and deprecated `functions`/`function_call` requests return `tool_generation_not_supported`. An empty list or an explicit choice `none` allows ordinary text; `auto` without definitions requires no tool generation.
- Anthropic tool definitions or forced choices return the same capability error. Incoming tool blocks are refused even without definitions, preventing their content from being dropped silently.

The low-level ASCII fixture and the standalone schema validator remain available for research and validation of existing text. They are not evidence of JSON-constrained generation. A future implementation must map actual token pieces and special tokens, handle UTF-8 and EOS, enforce the grammar/schema before emission, and validate real JSON and tool responses via HTTP and SSE. This change closes one misleading capability in issue #24; it does not implement that future grammar or complete the other conversions in #24.
