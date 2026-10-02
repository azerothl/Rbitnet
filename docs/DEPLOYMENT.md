# Deployment notes

Rbitnet is typically run **beside Akasha** or behind a **reverse proxy** on a trusted network.

## systemd (user unit sketch)

Adjust paths and the Akasha user as needed.

```ini
[Unit]
Description=Rbitnet OpenAI-compatible server
After=network.target

[Service]
Type=simple
Environment=RBITNET_BIND=127.0.0.1:8080
Environment=RBITNET_MODEL=/var/lib/rbitnet/model.gguf
Environment=RBITNET_API_KEY=change-me
Environment=RBITNET_MAX_CONCURRENT=4
ExecStart=/usr/local/bin/rbitnet-server
Restart=on-failure

[Install]
WantedBy=default.target
```

Build `rbitnet-server` with `cargo install --path crates/bitnet-server --locked` or copy the release binary.

## Nginx TLS termination (sketch)

Terminate TLS on Nginx and proxy to `127.0.0.1:8080`. Do **not** expose `rbitnet-server` without authentication on `0.0.0.0` unless you have another security layer.

**Rate limiting and abuse:** Rbitnet does not ship an in-process rate limiter. For internet-facing edges, add **`limit_req`** (or equivalent) in Nginx, or use [Caddy](https://caddyserver.com/docs/caddyfile/directives/rate_limit) / your cloud load balancer, in addition to TLS and `RBITNET_API_KEY`.

```nginx
limit_req_zone $binary_remote_addr zone=rbitnet:10m rate=10r/s;

location /v1/ {
    limit_req zone=rbitnet burst=20 nodelay;
    proxy_pass http://127.0.0.1:8080;
    proxy_set_header Authorization $http_authorization;
    proxy_read_timeout 600s;
}
```

## Multiple replicas (prefix locality)

If you run **several** `rbitnet-server` processes (or multiple proxy hosts) behind a load balancer and enable **prefix KV** reuse or session-heavy workloads, **random** balancing prevents cache hits on a single worker. Options:

- **Sticky sessions (preferred for agents):** send `X-Rbitnet-Session: <id>` (or cookie `rbitnet_session=<id>`) on every request. `rbitnet-proxy` echoes the header and an `X-Rbitnet-Sticky-Bucket` hash when sticky is enabled.
- **Hash routing:** `hash $http_x_rbitnet_session consistent` (or `$http_authorization`) so the same session always hits the same instance.
- **Proxy sticky map:** set `RBITNET_PROXY_STICKY=1` so the proxy remembers `session → model` when the JSON body omits `model` (useful for multi-turn clients that only set the session id).

Example (same TLS block as above; adjust upstream names):

```nginx
upstream rbitnet_backends {
    hash $http_x_rbitnet_session consistent;
    server 127.0.0.1:8080;
    server 127.0.0.1:8081;
}

location /v1/ {
    proxy_pass http://rbitnet_backends;
    proxy_set_header Authorization $http_authorization;
    proxy_set_header X-Rbitnet-Session $http_x_rbitnet_session;
    proxy_read_timeout 600s;
}
```

`RBITNET_PROXY_REPLICAS` documents the planned replica count and drives the sticky hash-bucket plumbing (`sticky_hash_bucket`). Today the proxy still supervises **one native child per model id**; multi-replica same-model children are not spawned yet — use an external LB with the sticky header until that lands.

### Multi-session prefix hit-rate scenario

With **`RBITNET_PREFIX_KV=1`** on a single runner (or a sticky-pinned replica), ≥2 concurrent agent sessions that share a long system/tools prefix should raise `rbitnet_core_prefix_hit` on `/metrics` after warm-up:

1. Start one worker (or pin both clients via the same sticky session / replica).
2. Session A and session B send chat turns with the **same** system + tools preamble and different user tails.
3. After the first request warms the radix, subsequent turns from either session should increment `rbitnet_core_prefix_hit` (Akasha alias `prefix_hit`). Unit gate: `agent_style_prefix_hit_rate_after_warmup` in `bitnet-core` targets ≥70% after warm-up.

Without sticky co-location across replicas, each worker keeps a private radix and multi-session hit rate collapses to near zero.

See [LIMITATIONS.md — Multi-replica deployments](LIMITATIONS.md#multi-replica-deployments) and [AUTOTUNE_DESIGN.md](AUTOTUNE_DESIGN.md) (kernel autotune is deferred).

## Health checks

- **Liveness:** `GET /health` — process is up.
- **Readiness:** `GET /ready` — `503` if a GGUF is configured but `tokenizer.json` cannot be resolved (stub/toy modes are ready when enabled).
- **Metrics:** `GET /metrics` — Prometheus text format (no auth by default; restrict at the proxy if exposed).

## Docker

A **reference** multi-stage build lives at the repository root [`Dockerfile`](../Dockerfile). It compiles `rbitnet-server` and runs as `ENTRYPOINT`; mount your `.gguf` and tokenizer at runtime and set `RBITNET_MODEL` (and usually `RBITNET_BIND`, `RBITNET_API_KEY`).

```bash
docker build -t rbitnet:local .
docker run --rm -p 8080:8080 \
  -e RBITNET_MODEL=/model/model.gguf \
  -v /abs/path/on/host:/model:ro \
  rbitnet:local
```

The default `RBITNET_BIND` in the image is `0.0.0.0:8080` — use TLS and auth at the edge.
