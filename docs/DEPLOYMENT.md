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

If you run **several** `rbitnet-server` processes behind a load balancer and later enable **prefix KV** reuse or session-heavy workloads, **random** balancing prevents cache hits on a single worker. Options:

- **Sticky sessions:** route the same client (cookie or client IP) to the same upstream.
- **Hash routing:** `hash $http_authorization consistent` (or a custom header) so the same API key always hits the same instance.

Example (same TLS block as above; adjust upstream names):

```nginx
upstream rbitnet_backends {
    hash $http_authorization consistent;
    server 127.0.0.1:8080;
    server 127.0.0.1:8081;
}

location /v1/ {
    proxy_pass http://rbitnet_backends;
    proxy_set_header Authorization $http_authorization;
    proxy_read_timeout 600s;
}
```

See [LIMITATIONS.md — Multi-replica deployments](LIMITATIONS.md#multi-replica-deployments).

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
