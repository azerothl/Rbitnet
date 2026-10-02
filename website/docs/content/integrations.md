# Integrations

Rbitnet exposes an OpenAI-compatible base URL. Start the server first:

```bash
export RBITNET_MODEL=/absolute/path/to/model.gguf
export RBITNET_TOKENIZER=/absolute/path/to/tokenizer.json
rbitnet serve
```

For a no-weight smoke test:

```bash
RBITNET_STUB=1 rbitnet serve --open-ui
```

The default base URL is `http://127.0.0.1:8080`. If `RBITNET_API_KEY` is set, pass it as `Authorization: Bearer <key>` or `x-api-key: <key>`.

## curl

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rbitnet-llama",
    "messages": [{"role": "user", "content": "Say hello from Rbitnet"}],
    "max_tokens": 64,
    "temperature": 0.7
  }'
```

## Python `openai`

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed-unless-RBITNET_API_KEY-is-set",
)

response = client.chat.completions.create(
    model="rbitnet-llama",
    messages=[{"role": "user", "content": "Write one sentence about local inference."}],
    max_tokens=64,
)
print(response.choices[0].message.content)
```

## Node `openai`

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8080/v1",
  apiKey: "not-needed-unless-RBITNET_API_KEY-is-set",
});

const response = await client.chat.completions.create({
  model: "rbitnet-llama",
  messages: [{ role: "user", content: "Write one sentence about local inference." }],
  max_tokens: 64,
});

console.log(response.choices[0].message.content);
```

## LiteLLM

Use an OpenAI-compatible route and point it at Rbitnet:

```yaml
model_list:
  - model_name: rbitnet-local
    litellm_params:
      model: openai/rbitnet-llama
      api_base: http://127.0.0.1:8080/v1
      api_key: not-needed-unless-RBITNET_API_KEY-is-set
```

Then call LiteLLM with `model="rbitnet-local"`.

## Akasha

Rbitnet is the local OpenAI-compatible backend used by Akasha's `BitNetProvider`. Configure Akasha with the Rbitnet base URL and a model id returned by `GET /v1/models`:

```yaml
providers:
  bitnet:
    base_url: "http://127.0.0.1:8080"

task_types:
  conversation:
    primary:
      provider: bitnet
      model: rbitnet-llama
```
