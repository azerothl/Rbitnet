//! Terminal chat UI for quick Rbitnet smoke tests.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use base64::Engine as _;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::{DefaultTerminal, Frame};
use serde_json::json;

const STATUS_HELP: &str = "Enter: send | /image PATH: attach | Ctrl+I: attach typed path | Ctrl+R reload | Ctrl+U unload | F2/F3 tokens | F4 models | +/- temp | Esc quit";

#[derive(Debug, Clone)]
pub struct ChatOptions {
    pub base_url: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub admin_token: Option<String>,
    pub serve: bool,
    pub bind: String,
    pub server_bin: Option<PathBuf>,
    pub model_path: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
    pub mmproj: Option<PathBuf>,
    pub chat_format: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub transcript: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct ChatParams {
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub stop: Option<String>,
}

#[derive(Debug, Clone)]
struct AttachedImage {
    path: PathBuf,
    data_url: String,
}

#[derive(Debug, Clone)]
struct ChatTurn {
    role: String,
    /// Display / transcript text (prompt without slash commands).
    content: String,
    /// Optional OpenAI data-URL for vision turns.
    image_data_url: Option<String>,
    /// Path label shown in the conversation pane.
    image_label: Option<String>,
}

struct ChatApp {
    base_url: String,
    api_key: Option<String>,
    admin_token: Option<String>,
    params: ChatParams,
    turns: Vec<ChatTurn>,
    input: String,
    pending_image: Option<AttachedImage>,
    status: String,
    transcript: Option<PathBuf>,
}

impl ChatApp {
    fn new(opts: ChatOptions) -> Self {
        let model = opts.model.unwrap_or_else(|| "rbitnet-stub".into());
        Self {
            base_url: normalize_base_url(&opts.base_url),
            api_key: opts.api_key,
            admin_token: opts.admin_token,
            params: ChatParams {
                model,
                max_tokens: opts.max_tokens,
                temperature: opts.temperature,
                top_p: opts.top_p,
                seed: opts.seed,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                stop: None,
            },
            turns: Vec::new(),
            input: String::new(),
            pending_image: None,
            status: STATUS_HELP.into(),
            transcript: opts.transcript,
        }
    }

    fn attach_image_path(&mut self, raw: &str) {
        let path = PathBuf::from(raw.trim());
        if path.as_os_str().is_empty() {
            self.pending_image = None;
            self.status = "Image cleared".into();
            return;
        }
        match load_image_data_url(&path) {
            Ok(data_url) => {
                self.status = format!("Attached image: {}", path.display());
                self.pending_image = Some(AttachedImage { path, data_url });
            }
            Err(e) => self.status = format!("Image error: {e}"),
        }
    }

    fn attach_from_input(&mut self) {
        let path = self.input.trim().to_string();
        if path.is_empty() {
            self.status = "Type an image path, then Ctrl+I (or /image PATH)".into();
            return;
        }
        self.input.clear();
        self.attach_image_path(&path);
    }

    fn send(&mut self) {
        let raw = self.input.trim().to_string();
        if raw.is_empty() && self.pending_image.is_none() {
            return;
        }

        if let Some(path) = parse_image_command(&raw) {
            self.input.clear();
            self.attach_image_path(path);
            return;
        }

        self.input.clear();
        let image = self.pending_image.take();
        let prompt = if raw.is_empty() {
            "Describe this image.".to_string()
        } else {
            raw
        };
        self.turns.push(ChatTurn {
            role: "user".into(),
            content: prompt.clone(),
            image_data_url: image.as_ref().map(|i| i.data_url.clone()),
            image_label: image.as_ref().map(|i| i.path.display().to_string()),
        });
        self.status = "Sending...".into();
        match post_chat(
            &self.base_url,
            self.api_key.as_deref(),
            &self.params,
            &self.turns,
        ) {
            Ok(reply) => {
                self.turns.push(ChatTurn {
                    role: "assistant".into(),
                    content: reply.clone(),
                    image_data_url: None,
                    image_label: None,
                });
                self.status = "OK".into();
                if let Err(e) = append_transcript(
                    &self.transcript,
                    &prompt,
                    image.as_ref().map(|i| i.path.as_path()),
                    &reply,
                    &self.params,
                ) {
                    self.status = format!("Transcript error: {e}");
                }
            }
            Err(e) => {
                // Keep the failed user turn visible; re-attach image so the user can retry.
                if let Some(img) = image {
                    self.pending_image = Some(img);
                }
                self.status = format!("Chat error: {e}");
            }
        }
    }

    fn reload(&mut self) {
        match post_admin(&self.base_url, "reload", self.admin_token.as_deref()) {
            Ok(s) => self.status = format!("Reload: {s}"),
            Err(e) => self.status = format!("Reload error: {e}"),
        }
    }

    fn unload(&mut self) {
        match post_admin(&self.base_url, "unload", self.admin_token.as_deref()) {
            Ok(s) => self.status = format!("Unload: {s}"),
            Err(e) => self.status = format!("Unload error: {e}"),
        }
    }

    fn list_models(&mut self) {
        match get_models(&self.base_url, self.api_key.as_deref()) {
            Ok(models) => self.status = format!("Models: {}", models.join(", ")),
            Err(e) => self.status = format!("Models error: {e}"),
        }
    }
}

pub fn run_chat_tui(mut opts: ChatOptions) -> Result<(), String> {
    let mut child = if opts.serve {
        let child = start_managed_server(&opts)?;
        opts.base_url = format!("http://{}/v1", opts.bind);
        wait_ready(&opts.bind, Duration::from_secs(60))?;
        Some(child)
    } else {
        None
    };

    let result = run_terminal(ChatApp::new(opts));
    if let Some(child) = child.as_mut() {
        let _ = child.kill();
        let _ = child.wait();
    }
    result
}

fn run_terminal(mut app: ChatApp) -> Result<(), String> {
    let mut terminal = ratatui::init();
    let result = run_loop(&mut terminal, &mut app);
    ratatui::restore();
    result
}

fn run_loop(terminal: &mut DefaultTerminal, app: &mut ChatApp) -> Result<(), String> {
    loop {
        terminal.draw(|f| draw(f, app)).map_err(|e| e.to_string())?;
        if !event::poll(Duration::from_millis(100)).map_err(|e| e.to_string())? {
            continue;
        }
        let Event::Key(key) = event::read().map_err(|e| e.to_string())? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        match key.code {
            KeyCode::Esc => return Ok(()),
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return Ok(()),
            KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => app.reload(),
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => app.unload(),
            KeyCode::Char('i') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.attach_from_input()
            }
            KeyCode::Enter => app.send(),
            KeyCode::Backspace => {
                app.input.pop();
            }
            KeyCode::F(4) => app.list_models(),
            KeyCode::F(2) => {
                app.params.max_tokens = app.params.max_tokens.saturating_sub(1).max(1);
            }
            KeyCode::F(3) => {
                app.params.max_tokens = app.params.max_tokens.saturating_add(1);
            }
            KeyCode::Char('+') => {
                app.params.temperature = (app.params.temperature + 0.1).min(2.0);
            }
            KeyCode::Char('-') => {
                app.params.temperature = (app.params.temperature - 0.1).max(0.0);
            }
            KeyCode::Char(_) if key.modifiers.contains(KeyModifiers::CONTROL) => {}
            KeyCode::Char(c) => app.input.push(c),
            _ => {}
        }
    }
}

fn draw(f: &mut Frame<'_>, app: &ChatApp) {
    let chunks = Layout::vertical([
        Constraint::Min(8),
        Constraint::Length(5),
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Length(3),
    ])
    .split(f.area());

    let mut text = Text::default();
    for turn in &app.turns {
        let line = match &turn.image_label {
            Some(label) => format!("{}: [image: {}] {}", turn.role, label, turn.content),
            None => format!("{}: {}", turn.role, turn.content),
        };
        text.lines.push(Line::from(line));
        text.lines.push(Line::from(""));
    }
    f.render_widget(
        Paragraph::new(text)
            .block(Block::new().title("Conversation").borders(Borders::ALL))
            .wrap(Wrap { trim: false }),
        chunks[0],
    );

    let params = format!(
        "model={} max_tokens={} temperature={:.2} top_p={} seed={}",
        app.params.model,
        app.params.max_tokens,
        app.params.temperature,
        app.params
            .top_p
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "-".into()),
        app.params
            .seed
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".into())
    );
    f.render_widget(
        Paragraph::new(params).block(Block::new().title("Params").borders(Borders::ALL)),
        chunks[1],
    );

    let image_line = match &app.pending_image {
        Some(img) => format!("pending: {}", img.path.display()),
        None => "none (use /image PATH or Ctrl+I)".into(),
    };
    f.render_widget(
        Paragraph::new(image_line).block(Block::new().title("Image").borders(Borders::ALL)),
        chunks[2],
    );
    f.render_widget(
        Paragraph::new(app.input.as_str())
            .block(Block::new().title("Prompt").borders(Borders::ALL)),
        chunks[3],
    );
    f.render_widget(
        Paragraph::new(app.status.as_str())
            .block(Block::new().title("Status").borders(Borders::ALL)),
        chunks[4],
    );
}

pub fn normalize_base_url(raw: &str) -> String {
    let mut base = raw.trim().trim_end_matches('/').to_string();
    if !base.ends_with("/v1") {
        base.push_str("/v1");
    }
    base
}

/// `/image PATH`, `/image clear`, or `/image` (clear). Returns `Some` when the line is that command.
pub fn parse_image_command(raw: &str) -> Option<&str> {
    let trimmed = raw.trim();
    let rest = trimmed.strip_prefix("/image")?;
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let path = rest.trim();
    if path.is_empty() || path.eq_ignore_ascii_case("clear") || path.eq_ignore_ascii_case("none") {
        Some("")
    } else {
        Some(path)
    }
}

pub fn mime_for_image_path(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        _ => "application/octet-stream",
    }
}

pub fn load_image_data_url(path: &Path) -> Result<String, String> {
    if !path.is_file() {
        return Err(format!("not a file: {}", path.display()));
    }
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.is_empty() {
        return Err(format!("empty file: {}", path.display()));
    }
    let mime = mime_for_image_path(path);
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(format!("data:{mime};base64,{b64}"))
}

fn message_content_json(turn: &ChatTurn) -> serde_json::Value {
    match &turn.image_data_url {
        Some(url) => json!([
            { "type": "text", "text": turn.content },
            { "type": "image_url", "image_url": { "url": url } }
        ]),
        None => json!(turn.content),
    }
}

fn build_chat_payload(params: &ChatParams, turns: &[ChatTurn]) -> serde_json::Value {
    let messages: Vec<_> = turns
        .iter()
        .map(|turn| {
            json!({
                "role": turn.role,
                "content": message_content_json(turn),
            })
        })
        .collect();
    json!({
        "model": params.model,
        "messages": messages,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "seed": params.seed,
        "frequency_penalty": params.frequency_penalty,
        "presence_penalty": params.presence_penalty,
        "stop": params.stop
    })
}

/// Convenience for tests that only have text turns.
fn build_chat_payload_text(params: &ChatParams, turns: &[(&str, &str)]) -> serde_json::Value {
    let owned: Vec<ChatTurn> = turns
        .iter()
        .map(|(role, content)| ChatTurn {
            role: (*role).into(),
            content: (*content).into(),
            image_data_url: None,
            image_label: None,
        })
        .collect();
    build_chat_payload(params, &owned)
}

fn post_chat(
    base_url: &str,
    api_key: Option<&str>,
    params: &ChatParams,
    turns: &[ChatTurn],
) -> Result<String, String> {
    let payload = build_chat_payload(params, turns);
    let url = format!("{base_url}/chat/completions");
    let mut req = ureq::post(&url).set("content-type", "application/json");
    if let Some(key) = api_key {
        req = req.set("authorization", &format!("Bearer {key}"));
    }
    let value: serde_json::Value = req
        .send_json(payload)
        .map_err(|e| e.to_string())?
        .into_json()
        .map_err(|e| e.to_string())?;
    if let Some(err) = value.get("error") {
        let msg = err
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("chat request failed");
        let code = err
            .get("code")
            .and_then(|v| v.as_str())
            .unwrap_or("error");
        return Err(format!("{code}: {msg}"));
    }
    Ok(value["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string())
}

fn post_admin(base_url: &str, action: &str, admin_token: Option<&str>) -> Result<String, String> {
    let url = format!("{base_url}/admin/{action}");
    let mut req = ureq::post(&url).set("content-type", "application/json");
    if let Some(token) = admin_token {
        req = req.set("x-rbitnet-admin-token", token);
    }
    let response = req.send_string("{}").map_err(|e| e.to_string())?;
    response.into_string().map_err(|e| e.to_string())
}

fn get_models(base_url: &str, api_key: Option<&str>) -> Result<Vec<String>, String> {
    let url = format!("{base_url}/models");
    let mut req = ureq::get(&url);
    if let Some(key) = api_key {
        req = req.set("authorization", &format!("Bearer {key}"));
    }
    let value: serde_json::Value = req
        .call()
        .map_err(|e| e.to_string())?
        .into_json()
        .map_err(|e| e.to_string())?;
    Ok(value["data"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .filter_map(|row| row["id"].as_str().map(ToString::to_string))
                .collect()
        })
        .unwrap_or_default())
}

pub fn managed_server_command(opts: &ChatOptions) -> (PathBuf, Vec<(String, String)>) {
    let bin = opts.server_bin.clone().unwrap_or_else(default_server_bin);
    let mut envs = vec![("RBITNET_BIND".into(), opts.bind.clone())];
    if let Some(path) = &opts.model_path {
        envs.push(("RBITNET_MODEL".into(), path.display().to_string()));
    }
    if let Some(path) = &opts.tokenizer {
        envs.push(("RBITNET_TOKENIZER".into(), path.display().to_string()));
    }
    if let Some(path) = &opts.mmproj {
        envs.push(("RBITNET_MMPROJ".into(), path.display().to_string()));
    }
    if let Some(fmt) = &opts.chat_format {
        envs.push(("RBITNET_CHAT_FORMAT".into(), fmt.clone()));
    }
    if let Some(key) = &opts.api_key {
        envs.push(("RBITNET_API_KEY".into(), key.clone()));
    }
    if let Some(token) = &opts.admin_token {
        envs.push(("RBITNET_ADMIN_TOKEN".into(), token.clone()));
    }
    (bin, envs)
}

fn start_managed_server(opts: &ChatOptions) -> Result<Child, String> {
    let (bin, envs) = managed_server_command(opts);
    let mut cmd = Command::new(&bin);
    for (key, value) in envs {
        cmd.env(key, value);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("spawn {}: {e}", bin.display()))
}

fn default_server_bin() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let name = if cfg!(windows) {
                "rbitnet-server.exe"
            } else {
                "rbitnet-server"
            };
            let sibling = dir.join(name);
            if sibling.is_file() {
                return sibling;
            }
        }
    }
    PathBuf::from(if cfg!(windows) {
        "rbitnet-server.exe"
    } else {
        "rbitnet-server"
    })
}

fn wait_ready(bind: &str, timeout: Duration) -> Result<(), String> {
    let url = format!("http://{bind}/ready");
    let start = Instant::now();
    while start.elapsed() < timeout {
        if ureq::get(&url)
            .call()
            .map(|r| r.status() < 500)
            .unwrap_or(false)
        {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    Err(format!("server did not become ready at {url}"))
}

fn append_transcript(
    path: &Option<PathBuf>,
    prompt: &str,
    image: Option<&Path>,
    reply: &str,
    params: &ChatParams,
) -> Result<(), String> {
    let Some(path) = path else {
        return Ok(());
    };
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let row = json!({
        "prompt": prompt,
        "image": image.map(|p| p.display().to_string()),
        "reply": reply,
        "model": params.model,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature
    });
    writeln!(file, "{row}").map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> ChatOptions {
        ChatOptions {
            base_url: "http://127.0.0.1:8080".into(),
            model: Some("m".into()),
            api_key: None,
            admin_token: None,
            serve: false,
            bind: "127.0.0.1:8080".into(),
            server_bin: None,
            model_path: None,
            tokenizer: None,
            mmproj: None,
            chat_format: None,
            max_tokens: 4,
            temperature: 0.2,
            top_p: None,
            seed: Some(7),
            transcript: None,
        }
    }

    #[test]
    fn base_url_adds_v1() {
        assert_eq!(
            normalize_base_url("http://127.0.0.1:8080"),
            "http://127.0.0.1:8080/v1"
        );
        assert_eq!(
            normalize_base_url("http://127.0.0.1:8080/v1/"),
            "http://127.0.0.1:8080/v1"
        );
    }

    #[test]
    fn payload_contains_live_params() {
        let options = opts();
        let params = ChatParams {
            model: options.model.unwrap(),
            max_tokens: options.max_tokens,
            temperature: options.temperature,
            top_p: Some(0.9),
            seed: options.seed,
            frequency_penalty: 0.1,
            presence_penalty: 0.2,
            stop: Some("</s>".into()),
        };
        let value = build_chat_payload_text(&params, &[("user", "hello")]);
        assert_eq!(value["model"], "m");
        assert_eq!(value["max_tokens"], 4);
        assert_eq!(value["messages"][0]["content"], "hello");
    }

    #[test]
    fn payload_includes_image_url_parts() {
        let params = ChatParams {
            model: "vision".into(),
            max_tokens: 8,
            temperature: 0.0,
            top_p: None,
            seed: None,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop: None,
        };
        let turns = [ChatTurn {
            role: "user".into(),
            content: "what color?".into(),
            image_data_url: Some("data:image/png;base64,aaa".into()),
            image_label: Some("red.png".into()),
        }];
        let value = build_chat_payload(&params, &turns);
        let content = &value["messages"][0]["content"];
        assert!(content.is_array());
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "what color?");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,aaa");
    }

    #[test]
    fn image_command_parses_path_and_clear() {
        assert_eq!(parse_image_command("/image ./cat.png"), Some("./cat.png"));
        assert_eq!(parse_image_command("  /image   /tmp/a.jpg  "), Some("/tmp/a.jpg"));
        assert_eq!(parse_image_command("/image"), Some(""));
        assert_eq!(parse_image_command("/image clear"), Some(""));
        assert_eq!(parse_image_command("/images foo"), None);
        assert_eq!(parse_image_command("hello"), None);
    }

    #[test]
    fn mime_from_extension() {
        assert_eq!(mime_for_image_path(Path::new("a.PNG")), "image/png");
        assert_eq!(mime_for_image_path(Path::new("a.jpeg")), "image/jpeg");
        assert_eq!(mime_for_image_path(Path::new("a.webp")), "image/webp");
    }

    #[test]
    fn load_image_builds_data_url() {
        let dir = std::env::temp_dir().join(format!(
            "rbitnet-chat-image-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dot.png");
        // Minimal 1x1 PNG
        let png: &[u8] = &[
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x90, 0x77, 0x53, 0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41, 0x54, 0x08,
            0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00, 0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x05, 0xfe,
            0xd4, 0xef, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
        ];
        fs::write(&path, png).unwrap();
        let url = load_image_data_url(&path).unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn managed_command_sets_model_and_mmproj_env() {
        let mut options = opts();
        options.server_bin = Some(PathBuf::from("server-bin"));
        options.model_path = Some(PathBuf::from("model.gguf"));
        options.tokenizer = Some(PathBuf::from("tokenizer.json"));
        options.mmproj = Some(PathBuf::from("mmproj.gguf"));
        let (bin, envs) = managed_server_command(&options);
        assert_eq!(bin, PathBuf::from("server-bin"));
        assert!(envs
            .iter()
            .any(|(k, v)| k == "RBITNET_MODEL" && v.contains("model.gguf")));
        assert!(envs
            .iter()
            .any(|(k, v)| k == "RBITNET_TOKENIZER" && v.contains("tokenizer.json")));
        assert!(envs
            .iter()
            .any(|(k, v)| k == "RBITNET_MMPROJ" && v.contains("mmproj.gguf")));
    }
}
