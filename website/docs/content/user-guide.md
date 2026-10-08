# User guide

Rbitnet runs a language model on your machine and exposes it as a local OpenAI-compatible API. You do not need Python to chat. You need the `rbitnet` program, a GGUF model file, and its tokenizer.

The server listens on `http://127.0.0.1:8080` by default. A small web chat is at `http://127.0.0.1:8080/ui`.

## Install

**Linux or macOS**

```bash
curl -fsSL https://raw.githubusercontent.com/azerothl/Rbitnet/main/scripts/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

You can also download `rbitnet.exe` and `rbitnet-server.exe` from the [GitHub releases](https://github.com/azerothl/Rbitnet/releases) page (tag `v0.2.0` and later).

Check that the program is on your PATH:

```bash
rbitnet --version
```

## Download a small model

TinyLlama is the easiest first model. It is slow on CPU but small enough to fit on a laptop.

```bash
rbitnet models install tinyllama:q4 --dir ./models
rbitnet up tinyllama:q4 --dir ./models
```

`up` writes `rbitnet.toml` with the paths of the weights and the tokenizer. Keep `tokenizer.json` next to the `.gguf` file.

## Start the server

From the directory that contains `rbitnet.toml`:

```bash
rbitnet serve
```

Leave that terminal open. Then open `http://127.0.0.1:8080/ui` in a browser, or call the API:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "content-type: application/json" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}"
```

Streaming uses the same URL with `"stream": true`. The response is server-sent events and ends with `data: [DONE]`.

Stop the server with Ctrl+C in the terminal that is running `rbitnet serve`.

## Try the API without a model

If you only want to check that the HTTP server answers:

```bash
RBITNET_STUB=1 rbitnet serve
```

On Windows PowerShell:

```powershell
$env:RBITNET_STUB = "1"
rbitnet serve
```

Stub mode returns synthetic text. It does not load a GGUF.

## Everyday settings

| Goal | What to do |
|------|------------|
| Listen on another port | `rbitnet serve --bind 127.0.0.1:8081` |
| Require a key on `/v1` | `rbitnet serve --api-key your-secret` and send `Authorization: Bearer your-secret` |
| Point at a model file yourself | Set `RBITNET_MODEL` to the `.gguf` path and `RBITNET_TOKENIZER` if the tokenizer is not beside it |
| See whether the model is loaded | Open `http://127.0.0.1:8080/ready` |

Health and metrics (`/health`, `/ready`, `/metrics`) stay open even when an API key is set.

## If something fails

| What you see | What to do |
|--------------|------------|
| Tokenizer missing | Put `tokenizer.json` next to the `.gguf`, or set `RBITNET_TOKENIZER` |
| Model not loaded | Check `model` in `rbitnet.toml`, or set `RBITNET_MODEL` |
| Out of memory | Use a smaller quant, or close other programs. TinyLlama Q4 is the safe first choice |
| Port already in use | Pass another `--bind` address |
| Very slow replies | CPU inference on a large model is slow. Use a smaller model, or read the advanced guide for an NVIDIA GPU |

## What this program is not

Rbitnet does not train models. It does not convert Hugging Face checkpoints. If you already have a `.gguf` plus a tokenizer file, you can serve it. Training and export are covered in the advanced guide.

For CUDA, several concurrent chats, or saving a long prompt so the next turn starts faster, use the [advanced guide](advanced-guide.md).
