//! Multi-process runner proxy for Rbitnet.
//!
//! The proxy keeps the OpenAI-compatible HTTP surface in one parent process and
//! supervises one native child runner process per configured model id. External
//! inference daemons are not part of the default runtime path.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct RegistryFile {
    #[serde(rename = "default")]
    default_model: Option<String>,
    models: HashMap<String, ProxyModelEntry>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ProxyModelEntry {
    pub gguf: PathBuf,
    #[serde(default)]
    pub tokenizer: Option<PathBuf>,
    #[serde(default)]
    pub architecture: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyRegistry {
    pub default_model: Option<String>,
    pub models: HashMap<String, ProxyModelEntry>,
}

impl ProxyRegistry {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let parsed: RegistryFile =
            serde_json::from_str(text).map_err(|e| format!("registry JSON: {e}"))?;
        if parsed.models.is_empty() {
            return Err("registry must contain at least one model".into());
        }
        if let Some(default) = parsed.default_model.as_ref() {
            if !parsed.models.contains_key(default) {
                return Err(format!(
                    "registry default model '{default}' is not in models"
                ));
            }
        }
        Ok(Self {
            default_model: parsed.default_model,
            models: parsed.models,
        })
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::from_json(&text)
    }

    pub fn default_or_first_model(&self) -> Option<String> {
        if let Some(default) = self.default_model.as_ref() {
            return Some(default.clone());
        }
        self.models.keys().min().cloned()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceBackend {
    LocalWorkers,
    Vllm { base_url: String },
}

#[derive(Debug, Clone)]
pub struct ProxyConfig {
    pub bind: String,
    pub registry_path: Option<PathBuf>,
    pub runner_bin: String,
    pub api_key: Option<String>,
    pub max_body_bytes: usize,
    pub ready_timeout: Duration,
    pub request_timeout: Duration,
    pub backend: InferenceBackend,
}

impl ProxyConfig {
    pub fn from_env() -> Result<Self, String> {
        let bind = std::env::var("RBITNET_PROXY_BIND")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| std::env::var("RBITNET_BIND").ok())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "127.0.0.1:8080".into());
        let registry_path = std::env::var("RBITNET_MODEL_REGISTRY")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        let runner_bin =
            std::env::var("RBITNET_RUNNER_BIN").unwrap_or_else(|_| default_runner_bin());
        let api_key = std::env::var("RBITNET_API_KEY")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let max_body_bytes = parse_usize_env("RBITNET_MAX_BODY_BYTES", 1024 * 1024)?;
        let ready_timeout =
            Duration::from_secs(parse_u64_env("RBITNET_RUNNER_READY_TIMEOUT_SECS", 60)?);
        let request_timeout =
            Duration::from_secs(parse_u64_env("RBITNET_PROXY_REQUEST_TIMEOUT_SECS", 600)?);
        let backend = parse_inference_backend()?;
        Ok(Self {
            bind,
            registry_path,
            runner_bin,
            api_key,
            max_body_bytes,
            ready_timeout: ready_timeout.max(Duration::from_secs(1)),
            request_timeout: request_timeout.max(Duration::from_secs(1)),
            backend,
        })
    }
}

#[derive(Debug, Default)]
struct WorkerRuntime {
    child: Option<Child>,
    base_url: Option<String>,
    failures: u32,
    backoff_until: Option<Instant>,
}

#[derive(Debug)]
struct Worker {
    id: String,
    entry: ProxyModelEntry,
    runtime: Mutex<WorkerRuntime>,
}

#[derive(Clone)]
pub struct ProxyState {
    config: Arc<ProxyConfig>,
    registry: Arc<ProxyRegistry>,
    client: Client,
    workers: Arc<HashMap<String, Arc<Worker>>>,
}

pub fn create_proxy_app(config: ProxyConfig, registry: ProxyRegistry) -> Result<Router, String> {
    let state = build_proxy_state(config, registry)?;
    Ok(router_with_state(
        state.clone(),
        state.config.max_body_bytes,
    ))
}

fn build_proxy_state(config: ProxyConfig, registry: ProxyRegistry) -> Result<ProxyState, String> {
    let client = Client::builder()
        .timeout(config.request_timeout)
        .build()
        .map_err(|e| format!("HTTP client: {e}"))?;
    let workers = registry
        .models
        .iter()
        .map(|(id, entry)| {
            (
                id.clone(),
                Arc::new(Worker {
                    id: id.clone(),
                    entry: entry.clone(),
                    runtime: Mutex::new(WorkerRuntime::default()),
                }),
            )
        })
        .collect::<HashMap<_, _>>();
    Ok(ProxyState {
        config: Arc::new(config),
        registry: Arc::new(registry),
        client,
        workers: Arc::new(workers),
    })
}

pub async fn run_proxy() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = ProxyConfig::from_env().map_err(|e| format!("invalid proxy config: {e}"))?;
    let registry_path = config
        .registry_path
        .clone()
        .ok_or("RBITNET_MODEL_REGISTRY is required for rbitnet-proxy")?;
    let registry = ProxyRegistry::load(&registry_path)?;
    let bind = config.bind.clone();
    let state = build_proxy_state(config, registry)?;
    let app = router_with_state(state.clone(), state.config.max_body_bytes);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .map_err(|e| format!("bind {bind}: {e}"))?;
    info!("rbitnet-proxy listening on http://{bind}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(state))
        .await
        .map_err(|e| format!("proxy server error: {e}"))?;
    Ok(())
}

async fn shutdown_signal(state: ProxyState) {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            error!(%e, "failed to install Ctrl-C handler");
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(e) => {
                error!(%e, "failed to install SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    state.shutdown_children().await;
}

fn router_with_state(state: ProxyState, max_body_bytes: usize) -> Router {
    let cors = if std::env::var("RBITNET_CORS_ANY").as_deref() == Ok("1") {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
    } else {
        CorsLayer::new()
            .allow_origin(AllowOrigin::list([
                HeaderValue::from_static("http://localhost:3000"),
                HeaderValue::from_static("http://127.0.0.1:3000"),
                HeaderValue::from_static("http://localhost:8080"),
                HeaderValue::from_static("http://127.0.0.1:8080"),
            ]))
            .allow_methods([Method::GET, Method::POST])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::AUTHORIZATION,
                HeaderName::from_static("x-api-key"),
                HeaderName::from_static("x-request-id"),
            ])
    };

    Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/metrics", get(metrics))
        .route("/", get(root_health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state)
        .layer(cors)
        .layer(
            TraceLayer::new_for_http().make_span_with(|req: &Request<axum::body::Body>| {
                let id = req
                    .headers()
                    .get("x-request-id")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("-");
                tracing::info_span!(
                    "proxy_http_request",
                    method = %req.method(),
                    path = %req.uri().path(),
                    request_id = %id,
                )
            }),
        )
        .layer(middleware::from_fn(add_request_id_if_missing))
}

async fn add_request_id_if_missing(mut req: Request<axum::body::Body>, next: Next) -> Response {
    if req.headers().get("x-request-id").is_none() {
        if let Ok(v) = HeaderValue::from_str(&Uuid::new_v4().to_string()) {
            req.headers_mut().insert("x-request-id", v);
        }
    }
    next.run(req).await
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok\n")
}

async fn ready(State(state): State<ProxyState>) -> Response {
    let ready = match &state.config.backend {
        InferenceBackend::Vllm { .. } => true,
        InferenceBackend::LocalWorkers => !state.registry.models.is_empty(),
    };
    if ready {
        (StatusCode::OK, "ready\n").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "not ready\n").into_response()
    }
}

async fn metrics(State(state): State<ProxyState>) -> impl IntoResponse {
    let mut body = String::from("rbitnet_proxy_up 1\n");
    for (id, worker) in &state.workers {
        let runtime = worker.runtime.lock().await;
        if let Some(base) = runtime.base_url.as_ref() {
            let url = format!("{base}/metrics");
            if let Ok(resp) = state.client.get(&url).send().await {
                if let Ok(text) = resp.text().await {
                    for line in text.lines() {
                        if line.starts_with('#') || line.trim().is_empty() {
                            continue;
                        }
                        body.push_str(&format!("worker_{id}_{line}\n"));
                    }
                }
            }
        }
    }
    (StatusCode::OK, body)
}

async fn root_health(State(state): State<ProxyState>, headers: HeaderMap) -> Response {
    if let Err(r) = check_auth(&state, &headers) {
        return *r;
    }
    (StatusCode::OK, "rbitnet proxy OK\n").into_response()
}

async fn list_models(State(state): State<ProxyState>, headers: HeaderMap) -> Response {
    if let Err(r) = check_auth(&state, &headers) {
        return *r;
    }
    if let InferenceBackend::Vllm { base_url } = &state.config.backend {
        match state.forward_get(base_url, "/v1/models", &headers).await {
            Ok(res) => return res,
            Err(e) => {
                warn!(%e, "vLLM /v1/models failed; falling back to registry catalog");
            }
        }
    }

    let mut ids: Vec<_> = state.registry.models.keys().cloned().collect();
    ids.sort();
    let mut data = Vec::with_capacity(ids.len());
    for id in ids {
        let Some(worker) = state.workers.get(&id) else {
            continue;
        };
        let runtime = worker.runtime.lock().await;
        let running = runtime.child.is_some();
        data.push(json!({
            "id": id,
            "object": "model",
            "created": unix_now(),
            "owned_by": "rbitnet",
            "ready": running,
            "loaded": running,
            "metadata": {
                "backend": backend_label(&state.config.backend),
                "model_path": redact_display_path(&worker.entry.gguf.display().to_string()),
                "tokenizer_path": worker
                    .entry
                    .tokenizer
                    .as_ref()
                    .map(|p| redact_display_path(&p.display().to_string())),
                "architecture": worker.entry.architecture.as_deref().unwrap_or("auto"),
                "child_url": runtime.base_url,
                "restart_failures": runtime.failures,
            }
        }));
    }
    Json(json!({ "object": "list", "data": data })).into_response()
}

async fn chat_completions(
    State(state): State<ProxyState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_openai_post(state, headers, body, "/v1/chat/completions").await
}

async fn completions(State(state): State<ProxyState>, headers: HeaderMap, body: Bytes) -> Response {
    proxy_openai_post(state, headers, body, "/v1/completions").await
}

async fn proxy_openai_post(
    state: ProxyState,
    headers: HeaderMap,
    body: Bytes,
    path: &'static str,
) -> Response {
    if let Err(r) = check_auth(&state, &headers) {
        return *r;
    }

    let request_model = match request_model(&body, &state.registry) {
        Ok(model) => model,
        Err(r) => return *r,
    };

    let base_url = match &state.config.backend {
        InferenceBackend::Vllm { base_url } => base_url.clone(),
        InferenceBackend::LocalWorkers => match state.ensure_worker(&request_model).await {
            Ok(url) => url,
            Err(r) => return *r,
        },
    };

    match state.forward_post(&base_url, path, &headers, body).await {
        Ok(res) => res,
        Err(e) => {
            if matches!(state.config.backend, InferenceBackend::LocalWorkers) {
                state.mark_worker_failed(&request_model).await;
            }
            error!(model = %request_model, %e, "upstream request failed");
            json_error(
                StatusCode::BAD_GATEWAY,
                format!("upstream model '{request_model}' request failed: {e}"),
                "rbitnet_proxy_error",
            )
        }
    }
}

impl ProxyState {
    async fn ensure_worker(&self, model: &str) -> Result<String, Box<Response>> {
        let Some(worker) = self.workers.get(model).cloned() else {
            return Err(Box::new(json_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "unknown model '{model}'; allowed ids are defined in RBITNET_MODEL_REGISTRY"
                ),
                "invalid_request_error",
            )));
        };

        let mut runtime = worker.runtime.lock().await;
        if let Some(until) = runtime.backoff_until {
            if Instant::now() < until {
                return Err(Box::new(json_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("model '{model}' runner is in restart backoff"),
                    "rbitnet_proxy_error",
                )));
            }
        }

        let current_base_url = runtime.base_url.clone();
        if let Some(child) = runtime.child.as_mut() {
            match child.try_wait() {
                Ok(None) => {
                    if let Some(base_url) = current_base_url {
                        if self.health_check(&base_url).await {
                            runtime.failures = 0;
                            runtime.backoff_until = None;
                            return Ok(base_url);
                        }
                    }
                    warn!(model, "runner health check failed; recycling child");
                    let _ = child.start_kill();
                    runtime.child = None;
                    runtime.base_url = None;
                    record_failure(&mut runtime);
                    return Err(Box::new(json_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        format!("model '{model}' runner failed health check"),
                        "rbitnet_proxy_error",
                    )));
                }
                Ok(Some(status)) => {
                    warn!(model, %status, "runner exited");
                    runtime.child = None;
                    runtime.base_url = None;
                    record_failure(&mut runtime);
                    return Err(Box::new(json_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        format!("model '{model}' runner exited; restart backoff active"),
                        "rbitnet_proxy_error",
                    )));
                }
                Err(e) => {
                    runtime.child = None;
                    runtime.base_url = None;
                    record_failure(&mut runtime);
                    return Err(Box::new(json_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        format!("model '{model}' runner status check failed: {e}"),
                        "rbitnet_proxy_error",
                    )));
                }
            }
        }

        let (child, base_url) = match self.spawn_worker(&worker).await {
            Ok(v) => v,
            Err(e) => {
                record_failure(&mut runtime);
                return Err(Box::new(json_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("failed to start model '{model}' runner: {e}"),
                    "rbitnet_proxy_error",
                )));
            }
        };
        runtime.child = Some(child);
        runtime.base_url = Some(base_url.clone());
        runtime.failures = 0;
        runtime.backoff_until = None;
        Ok(base_url)
    }

    async fn spawn_worker(&self, worker: &Worker) -> Result<(Child, String), String> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| format!("reserve localhost port: {e}"))?;
        let addr: SocketAddr = listener
            .local_addr()
            .map_err(|e| format!("read reserved port: {e}"))?;
        drop(listener);

        let bind = addr.to_string();
        let base_url = format!("http://{bind}");
        let mut cmd = Command::new(&self.config.runner_bin);
        cmd.env("RBITNET_MODEL", &worker.entry.gguf)
            .env("RBITNET_BIND", &bind)
            .env("RBITNET_ACTIVE_MODEL_ID", &worker.id)
            .env("RBITNET_REQUIRE_MODEL_MATCH", "1")
            .env("RBITNET_INFERENCE_BACKEND", "local")
            .env_remove("RBITNET_MODEL_REGISTRY")
            .env_remove("RBITNET_VLLM_BASE_URL")
            .env_remove("RBITNET_STUB")
            .env_remove("RBITNET_TOY")
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        if let Some(tokenizer) = worker.entry.tokenizer.as_ref() {
            cmd.env("RBITNET_TOKENIZER", tokenizer);
        }
        if let Some(architecture) = worker.entry.architecture.as_ref() {
            cmd.env("RBITNET_ARCHITECTURE", architecture);
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", self.config.runner_bin))?;
        if self.wait_ready(&base_url, &mut child).await {
            info!(model = %worker.id, %base_url, "runner ready");
            return Ok((child, base_url));
        }
        let _ = child.start_kill();
        Err(format!("runner did not become ready at {base_url}"))
    }

    async fn wait_ready(&self, base_url: &str, child: &mut Child) -> bool {
        let deadline = Instant::now() + self.config.ready_timeout;
        while Instant::now() < deadline {
            match child.try_wait() {
                Ok(Some(status)) => {
                    warn!(%status, "runner exited before ready");
                    return false;
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(%e, "runner status check failed before ready");
                    return false;
                }
            }
            if self.health_check(base_url).await {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        false
    }

    async fn health_check(&self, base_url: &str) -> bool {
        let url = format!("{base_url}/ready");
        matches!(
            self.client
                .get(url)
                .timeout(Duration::from_secs(2))
                .send()
                .await,
            Ok(res) if res.status().is_success()
        )
    }

    async fn forward_get(
        &self,
        base_url: &str,
        path: &str,
        headers: &HeaderMap,
    ) -> Result<Response, reqwest::Error> {
        let mut req = self.client.get(format!("{base_url}{path}"));
        req = copy_forward_headers(req, headers);
        response_to_axum(req.send().await?).await
    }

    async fn forward_post(
        &self,
        base_url: &str,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Result<Response, reqwest::Error> {
        let mut req = self.client.post(format!("{base_url}{path}")).body(body);
        req = copy_forward_headers(req, headers);
        response_to_axum(req.send().await?).await
    }

    async fn mark_worker_failed(&self, model: &str) {
        let Some(worker) = self.workers.get(model) else {
            return;
        };
        let mut runtime = worker.runtime.lock().await;
        if let Some(child) = runtime.child.as_mut() {
            let _ = child.start_kill();
        }
        runtime.child = None;
        runtime.base_url = None;
        record_failure(&mut runtime);
    }

    async fn shutdown_children(&self) {
        for worker in self.workers.values() {
            let mut runtime = worker.runtime.lock().await;
            if let Some(child) = runtime.child.as_mut() {
                let _ = child.start_kill();
                let _ = child.wait().await;
            }
            runtime.child = None;
            runtime.base_url = None;
        }
    }
}

fn copy_forward_headers(
    mut req: reqwest::RequestBuilder,
    headers: &HeaderMap,
) -> reqwest::RequestBuilder {
    for (name, value) in headers {
        if *name == axum::http::header::HOST
            || *name == axum::http::header::CONTENT_LENGTH
            || *name == axum::http::header::TRANSFER_ENCODING
        {
            continue;
        }
        req = req.header(name, value);
    }
    req
}

async fn response_to_axum(upstream: reqwest::Response) -> Result<Response, reqwest::Error> {
    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let headers = upstream.headers().clone();
    let bytes = upstream.bytes().await?;
    let mut response = (status, bytes).into_response();
    for (name, value) in &headers {
        if *name == axum::http::header::CONTENT_LENGTH
            || *name == axum::http::header::TRANSFER_ENCODING
        {
            continue;
        }
        response.headers_mut().insert(name.clone(), value.clone());
    }
    Ok(response)
}

fn record_failure(runtime: &mut WorkerRuntime) {
    runtime.failures = runtime.failures.saturating_add(1);
    let pow = 1u64 << runtime.failures.min(5);
    let secs = pow.min(30);
    runtime.backoff_until = Some(Instant::now() + Duration::from_secs(secs));
}

fn request_model(body: &[u8], registry: &ProxyRegistry) -> Result<String, Box<Response>> {
    let value: serde_json::Value = serde_json::from_slice(body).map_err(|e| {
        Box::new(json_error(
            StatusCode::BAD_REQUEST,
            format!("invalid JSON request body: {e}"),
            "invalid_request_error",
        ))
    })?;
    if let Some(model) = value
        .get("model")
        .and_then(|m| m.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return Ok(model.to_string());
    }
    registry.default_or_first_model().ok_or_else(|| {
        Box::new(json_error(
            StatusCode::BAD_REQUEST,
            "request omitted model and registry is empty",
            "invalid_request_error",
        ))
    })
}

fn check_auth(state: &ProxyState, headers: &HeaderMap) -> Result<(), Box<Response>> {
    let Some(key) = state.config.api_key.as_ref() else {
        return Ok(());
    };
    let ok = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| {
            let mut parts = s.splitn(2, ' ');
            let scheme = parts.next()?;
            let token = parts.next()?.trim();
            scheme.eq_ignore_ascii_case("bearer").then_some(token)
        })
        .map(|token| token == key)
        .unwrap_or(false)
        || headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .map(|token| token == key)
            .unwrap_or(false);
    if ok {
        Ok(())
    } else {
        Err(Box::new(json_error(
            StatusCode::UNAUTHORIZED,
            "invalid or missing API key",
            "authentication_error",
        )))
    }
}

fn json_error(status: StatusCode, message: impl Into<String>, error_type: &str) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message.into(),
                "type": error_type
            }
        })),
    )
        .into_response()
}

fn parse_inference_backend() -> Result<InferenceBackend, String> {
    let raw = std::env::var("RBITNET_INFERENCE_BACKEND")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "local".into());
    match raw.as_str() {
        "local" | "local_workers" | "local-workers" | "rbitnet" | "bitnet-core" => {
            Ok(InferenceBackend::LocalWorkers)
        }
        "vllm" => {
            let base_url = std::env::var("RBITNET_VLLM_BASE_URL")
                .map_err(|_| "RBITNET_INFERENCE_BACKEND=vllm requires RBITNET_VLLM_BASE_URL")?;
            Ok(InferenceBackend::Vllm {
                base_url: normalize_base_url(&base_url)?,
            })
        }
        other => Err(format!(
            "unsupported RBITNET_INFERENCE_BACKEND='{other}'; default runtime is local native workers"
        )),
    }
}

fn backend_label(backend: &InferenceBackend) -> &'static str {
    match backend {
        InferenceBackend::LocalWorkers => "rbitnet-runner",
        InferenceBackend::Vllm { .. } => "vllm",
    }
}

fn normalize_base_url(url: &str) -> Result<String, String> {
    let trimmed = url.trim().trim_end_matches('/');
    if !(trimmed.starts_with("http://") || trimmed.starts_with("https://")) {
        return Err("RBITNET_VLLM_BASE_URL must start with http:// or https://".into());
    }
    Ok(trimmed.to_string())
}

fn parse_u64_env(key: &str, default: u64) -> Result<u64, String> {
    match std::env::var(key) {
        Ok(s) if s.trim().is_empty() => Ok(default),
        Ok(s) => s
            .parse::<u64>()
            .map_err(|_| format!("{key}: expected a non-negative integer")),
        Err(_) => Ok(default),
    }
}

fn parse_usize_env(key: &str, default: usize) -> Result<usize, String> {
    let raw = parse_u64_env(key, default as u64)?;
    usize::try_from(raw).map_err(|_| format!("{key}: value too large for usize"))
}

fn default_runner_bin() -> String {
    let exe = if cfg!(windows) {
        "rbitnet-runner.exe"
    } else {
        "rbitnet-runner"
    };
    if let Ok(current) = std::env::current_exe() {
        if let Some(dir) = current.parent() {
            let sibling = dir.join(exe);
            if sibling.is_file() {
                return sibling.display().to_string();
            }
        }
    }
    exe.into()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn redact_display_path(path: &str) -> String {
    let lower = path.to_ascii_lowercase();
    if lower.contains("hf_")
        || lower.contains("token=")
        || lower.contains("apikey")
        || lower.contains("api_key")
    {
        return "<redacted>".into();
    }
    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[test]
    fn registry_parses_without_default_and_selects_first() {
        let reg = ProxyRegistry::from_json(
            r#"{
                "models": {
                    "zeta": { "gguf": "z.gguf" },
                    "alpha": { "gguf": "a.gguf", "tokenizer": "tok.json", "architecture": "llama" }
                }
            }"#,
        )
        .expect("registry");
        assert_eq!(reg.default_or_first_model().as_deref(), Some("alpha"));
        assert_eq!(
            reg.models["alpha"].tokenizer.as_deref(),
            Some(Path::new("tok.json"))
        );
    }

    #[test]
    fn registry_rejects_unknown_default() {
        let err = ProxyRegistry::from_json(
            r#"{
                "default": "missing",
                "models": { "alpha": { "gguf": "a.gguf" } }
            }"#,
        )
        .unwrap_err();
        assert!(err.contains("default model"));
    }

    #[test]
    fn request_model_uses_body_or_registry_default() {
        let reg = ProxyRegistry::from_json(
            r#"{
                "default": "alpha",
                "models": { "alpha": { "gguf": "a.gguf" }, "beta": { "gguf": "b.gguf" } }
            }"#,
        )
        .unwrap();
        assert_eq!(
            request_model(br#"{"model":"beta","messages":[]}"#, &reg).unwrap(),
            "beta"
        );
        assert_eq!(request_model(br#"{"messages":[]}"#, &reg).unwrap(), "alpha");
    }

    #[test]
    fn vllm_base_url_is_normalized() {
        assert_eq!(
            normalize_base_url("http://127.0.0.1:8000/").unwrap(),
            "http://127.0.0.1:8000"
        );
        assert!(normalize_base_url("127.0.0.1:8000").is_err());
    }

    fn test_config() -> ProxyConfig {
        ProxyConfig {
            bind: "127.0.0.1:0".into(),
            registry_path: None,
            runner_bin: "rbitnet-runner".into(),
            api_key: None,
            max_body_bytes: 1024 * 1024,
            ready_timeout: Duration::from_secs(1),
            request_timeout: Duration::from_secs(1),
            backend: InferenceBackend::LocalWorkers,
        }
    }

    #[tokio::test]
    async fn models_route_lists_registry_without_spawning() {
        let reg = ProxyRegistry::from_json(
            r#"{
                "default": "alpha",
                "models": { "alpha": { "gguf": "a.gguf" }, "beta": { "gguf": "b.gguf" } }
            }"#,
        )
        .unwrap();
        let app = create_proxy_app(test_config(), reg).unwrap();
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(res.status().is_success());
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["data"].as_array().unwrap().len(), 2);
        assert_eq!(v["data"][0]["id"], "alpha");
        assert_eq!(v["data"][0]["loaded"], false);
    }

    #[tokio::test]
    async fn chat_route_rejects_unknown_model_before_spawn() {
        let reg = ProxyRegistry::from_json(
            r#"{
                "default": "alpha",
                "models": { "alpha": { "gguf": "a.gguf" } }
            }"#,
        )
        .unwrap();
        let app = create_proxy_app(test_config(), reg).unwrap();
        let body = serde_json::json!({
            "model": "missing",
            "messages": [{ "role": "user", "content": "hello" }]
        });
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }
}
