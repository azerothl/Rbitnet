# KV sidecar connector (optional)

External KV storage inspired by [PegaFlow](https://github.com/novitalabs/pegaflow): decouple KV lifecycle from `rbitnet-server` so warm prefixes survive process restarts and can be shared across `rbitnet-proxy` runners.

## Environment

| Variable | Description |
|----------|-------------|
| `RBITNET_KV_SIDECAR_URL` | Base URL (`http://127.0.0.1:9000`) — enables sidecar client |
| `RBITNET_KV_SIDECAR_TIMEOUT_SECS` | HTTP timeout (default `30`) |

## API sketch (future)

- `PUT /v1/kv/prefix` — upload block table + metadata
- `GET /v1/kv/prefix/{model_id}/{prefix_hash}` — fetch block ids for warm prefill

Implementation stub: [`crates/bitnet-core/src/kv_sidecar.rs`](../crates/bitnet-core/src/kv_sidecar.rs).

## Deployment

Use with sticky routing in [`DEPLOYMENT.md`](DEPLOYMENT.md) so repeated prompts hit the same worker or sidecar shard.
