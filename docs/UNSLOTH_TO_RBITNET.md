# Unsloth / TRL to Rbitnet

Fine-tune upstream, export one supported GGUF and its tokenizer, then serve with Rbitnet. This guide uses a Llama 3 Instruct model, Q4_K_M weights and the CPU backend. It includes a downloadable reference pack so you can check the serving steps before training your own model.

Run shell commands from the Rbitnet repository root. Build the serving binaries once:

```text
cargo build --release --locked -p rbitnet-cli -p bitnet-server -p bitnet-core --example inspect_gguf --bins
```

The commands below use Windows PowerShell. On Linux/macOS, use `./target/release/rbitnet` and `./target/release/examples/inspect_gguf` instead of the `.exe` paths. Python, Torch, Unsloth and converters are needed only on the training/export machine.

## 1. Obtain the GGUF and matching tokenizer

Choose your own fine-tune or the reference pack. Both end at this layout:

```text
models/exported-llama/model.gguf
models/exported-llama/tokenizer.json
```

### Export an Unsloth fine-tune

After training a supported Llama-family model in Unsloth, run this in the same Python session, where `model` and `tokenizer` are the objects used for training:

```python
model.save_pretrained_gguf("exported-llama", tokenizer, quantization_method="q4_k_m")
tokenizer.save_pretrained("exported-llama")
```

Unsloth chooses the generated GGUF filename. Copy that file to `models/exported-llama/model.gguf` and the generated `tokenizer.json` to the adjacent path above. Keep `tokenizer_config.json` with your export record: it describes the training chat template. The serving recipe below uses the built-in `llama3` format, so the fine-tune must use that format too.

Use the [official Unsloth export guide](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf) for exporter setup. Its quantization list covers more formats than Rbitnet supports; use the compatibility table below when choosing an export.

### Export a TRL / Hugging Face fine-tune

Save full weights and the training tokenizer in the Python training session. For a LoRA adapter on an unquantized base, merge with [PEFT](https://huggingface.co/docs/peft/developer_guides/lora) first:

```python
trained_model = trainer.model
if hasattr(trained_model, "merge_and_unload"):
    trained_model = trained_model.merge_and_unload()
trained_model.save_pretrained("merged-llama", safe_serialization=True)
tokenizer.save_pretrained("merged-llama")
```

For QLoRA, reload the original base weights in floating point and attach/merge the saved adapter before conversion; the snippet above assumes a floating-point base. Keep the tokenizer used in training, including any added tokens.

In an upstream [llama.cpp checkout](https://github.com/ggml-org/llama.cpp), install its converter dependencies and convert the merged directory. These commands use PowerShell and assume `merged-llama` is next to that checkout:

```powershell
git clone https://github.com/ggml-org/llama.cpp llama.cpp
New-Item -ItemType Directory -Force models/exported-llama | Out-Null
python -m pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py merged-llama --outfile model-f16.gguf --outtype f16
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF
cmake --build llama.cpp/build --config Release --target llama-quantize
& ./llama.cpp/build/bin/Release/llama-quantize.exe model-f16.gguf models/exported-llama/model.gguf Q4_K_M
Copy-Item merged-llama/tokenizer.json models/exported-llama/tokenizer.json
```

Skip `git clone` if you already have that checkout. With a single-configuration build, `llama-quantize` is under `build/bin/` rather than `build/bin/Release/`; use the binary your build produced. Record the converter commit with `git -C llama.cpp rev-parse HEAD`. See the upstream [quantizer instructions](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) for platform build details. Rbitnet serving uses the exported files after conversion.

### Download the reference export without training

This public Unsloth pack is already in Rbitnet's catalog. The commands pin both repositories to the revisions inspected for this walkthrough:

```powershell
New-Item -ItemType Directory -Force models/exported-llama | Out-Null
curl.exe -fL --retry 3 -o models/exported-llama/model.gguf "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/b69aef112e9f895e6f98d7ae0949f72ff09aa401/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
if ($LASTEXITCODE -ne 0) { throw "GGUF download failed" }
curl.exe -fL --retry 3 -o models/exported-llama/tokenizer.json "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/5a8abab4a5d6f164389b1079fb721cfab8d7126c/tokenizer.json"
if ($LASTEXITCODE -ne 0) { throw "Tokenizer download failed" }
```

On Linux/macOS, `mkdir -p models/exported-llama` and the same `curl -fL --retry 3 -o ...` commands download the files. The GGUF is 807,694,368 bytes. Its publisher SHA-256 is `3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1`; the tokenizer SHA-256 is `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b`.

## 2. Check format and provenance

```powershell
.\target\release\examples\inspect_gguf.exe models/exported-llama/model.gguf
if ($LASTEXITCODE -ne 0) { throw "GGUF inspection failed" }
Get-FileHash models/exported-llama/model.gguf, models/exported-llama/tokenizer.json -Algorithm SHA256
```

For the reference pack, compare both hashes to the values above. For your own export, record its SHA-256 at export time and compare it after transfer. A local hash identifies the bytes; a publisher hash or your export record supplies the expected value.

The inspector parses the file and reports tensor types. Actual generation in step 4 checks the loader and tokenizer together. GGUF metadata alone does not establish successful inference.

| Export / tensor types | Rbitnet path |
|-----------------------|--------------|
| F32, F16, BF16; Q4_0/Q4_1, Q5_0/Q5_1, Q8_0/Q8_1 | CPU dequantization available |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | CPU dequantization available; Q4_K_M is a mixture of tensor types |
| TQ1_0, TQ2_0 | Native BitNet scope; use [BITNET_NATIVE.md](BITNET_NATIVE.md) and its recipe |
| IQ variants, Q8_K, integer/F64 tensors, NVFP4, Q1_0 | The size parser recognizes some of these, but `tensor_to_f32` does not decode them; this walkthrough does not support them |
| MXFP4 | Dequantization exists; architecture restrictions still apply, outside this Llama walkthrough |
| Unknown or removed type IDs | Refused with `UnsupportedGgmlType` or an invalid-GGUF error |

The implemented type list is in [`ggml/dequant.rs`](../crates/bitnet-core/src/ggml/dequant.rs). The `llama` recipe assumes Llama-shaped tensors and metadata. Qwen3, Mixtral, experimental Qwen3.5 and roadmap MoE/MLA exports need their own [architecture checks](LIMITATIONS.md); changing a filename or forcing `architecture=llama` cannot convert their topology.

`RBITNET_TRUSTED_MODELS_ONLY` and `RBITNET_MODEL_SHA256` protect CLI download/install paths. Direct `recipe` / `serve` calls do not enforce that download check. Verify the local files before serving; the smoke helper below independently checks the expected GGUF hash. See [catalog provenance](CURATED_MODELS.md).

## 3. Start the native server

Inspect the [serve recipe](../recipes/exported-llama.recipe.json), then start it from the repository root:

```powershell
.\target\release\rbitnet.exe recipe recipes/exported-llama.recipe.json --print-only
.\target\release\rbitnet.exe recipe recipes/exported-llama.recipe.json
```

The second command starts the HTTP server in that terminal. The recipe sets CPU execution, Llama 3 chat formatting, one concurrent request, bind `127.0.0.1:18079` and disables stub/toy modes. It keeps supported weights quantized in the GGUF mmap. Relative model paths resolve against your working directory.

For other fine-tunes, copy the recipe and set the paths and the chat format used during training. The server supports `llama3`, `chatml` and `raw`, plus its documented [chat template handling](USAGE.md). A tokenizer file alone does not guarantee the correct conversation template or stop-token behavior.

## 4. Make a real `/v1` request

In a second PowerShell terminal, run the helper for the reference pack:

```powershell
.\scripts\smoke_gguf.ps1 -ModelPath models/exported-llama/model.gguf -ExpectedSha256 3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1
```

For your own export, pass its recorded SHA-256. The helper checks readiness, the served GGUF path, the file hash and a non-empty completion. It rejects stub/toy servers and a different loaded model, then writes the metadata, request and actual response to `target/gguf-smoke.json`. Set `RBITNET_API_KEY` if the server requires authentication; credentials are excluded from the record. This is a loading/API smoke, so review the generated text and run a golden comparison before claiming quality or parity.

On Linux/macOS, call the same endpoint with the model ID returned by `/v1/models`:

```bash
curl -fsS http://127.0.0.1:18079/ready
curl -fsS http://127.0.0.1:18079/v1/models
# Replace MODEL_ID with data[0].id from the previous response.
curl -fsS http://127.0.0.1:18079/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL_ID","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":16,"temperature":0}'
```

Review `/v1/models` for `loaded: true`, `ready: true`, `metadata.architecture: llama` and your GGUF path. Reject model IDs `rbitnet-stub` and `rbitnet-toy` in both the listing and the completion, even if a GGUF path is present. Require non-empty response text and positive `usage.prompt_tokens` / `usage.completion_tokens` before accepting loading/API evidence. This manual curl path does not perform the PowerShell helper's hash/path assertions.

## Recorded reference result

On 2026-10-03, after the inference repairs in PR #82, the same pinned Unsloth Llama-3.2-1B-Instruct Q4_K_M pack passed the Windows CPU recipe and hash/path checks. The capital-of-France request returned **`Paris.`**, with 22 prompt tokens and two completion tokens. See the [actual response and metadata](validation/2026-10-03-unsloth-llama32-1b.json). The [published CPU/CUDA comparison](benchmarks/2026-10-03-parity-round2/README.md) and its real-model sequence checks cover this same GGUF SHA-256.

The [2026-10-02 record](validation/2026-10-02-unsloth-llama32-1b.json) remains available: loading succeeded at that time, but the response was repetitive and factually incorrect. It describes the earlier runtime.

This run uses a prebuilt upstream GGUF. It does not validate a new Unsloth/TRL training run, local GGUF conversion, GPU performance or golden token parity. If your export has repeated or incorrect output, check the tokenizer, training template and special tokens, then compare against a reference engine on the same GGUF. Keep the failure visible in your validation record.

Runtime boundaries: [NATIVE_FIRST.md](NATIVE_FIRST.md). Akasha integration: [AKASHA_INFER.md](AKASHA_INFER.md). Curation and recipes: [CURATED_MODELS.md](CURATED_MODELS.md). This walkthrough implements the delivery requested by [#79](https://github.com/azerothl/Rbitnet/issues/79).
