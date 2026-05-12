//! `rbitnet` — list curated models, search Hugging Face for `.gguf` repos, download weights, or run the HTTP server.

mod bitnet_install;
mod catalog;
mod chat_tui;
mod download;
mod hf_search;
mod hub_http;
mod interactive_models;
mod train_cli;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};

use download::HubPlaceMode;

fn hub_place_mode(symlink: bool) -> HubPlaceMode {
    if symlink {
        HubPlaceMode::Symlink
    } else {
        HubPlaceMode::HardLinkOrCopy
    }
}

#[derive(Parser)]
#[command(
    name = "rbitnet",
    version,
    about = "Rbitnet CLI: Hugging Face models, download, optional Python train helper, and HTTP server"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download/resolve a GGUF repo and print the exact environment + server/curl commands.
    Quickstart(QuickstartCmd),
    /// Download/resolve a GGUF repo and write runnable local config defaults.
    Up(UpCmd),
    #[command(subcommand, about = "Curated catalog, HF search, and downloads")]
    Models(ModelsCmd),
    /// Run an optional LoRA/SFT Python recipe under `training/` (requires a repo checkout + Python).
    Train(TrainCmd),
    /// Print steps to convert a Hugging Face checkpoint directory to GGUF for Rbitnet (see llama.cpp).
    ExportGguf(ExportGgufCmd),
    /// Run the OpenAI-compatible HTTP server (same as `rbitnet-server`).
    Serve(ServeCmd),
    /// Run the server and open the bundled local web UI.
    Ui(UiCmd),
    /// Open a terminal chatbot for quick inference tests.
    Chat(ChatCmd),
}

#[derive(Args)]
struct ServeCmd {
    /// API key for protected routes (only if `RBITNET_API_KEY` is not already set).
    #[arg(long, env = "RBITNET_API_KEY")]
    api_key: Option<String>,
    /// Listen address host:port (only if `RBITNET_BIND` is not already set).
    #[arg(long, env = "RBITNET_BIND")]
    bind: Option<String>,
    /// Open the bundled local web UI (`/ui`) in the default browser before serving.
    #[arg(long)]
    open_ui: bool,
}

#[derive(Args)]
struct UiCmd {
    /// API key for protected routes (only if `RBITNET_API_KEY` is not already set).
    #[arg(long, env = "RBITNET_API_KEY")]
    api_key: Option<String>,
    /// Listen address host:port (only if `RBITNET_BIND` is not already set).
    #[arg(long, env = "RBITNET_BIND")]
    bind: Option<String>,
}

#[derive(Args)]
struct ChatCmd {
    /// OpenAI-compatible base URL. `/v1` is added when omitted.
    #[arg(
        long,
        env = "RBITNET_CHAT_BASE_URL",
        default_value = "http://127.0.0.1:8080/v1"
    )]
    base_url: String,
    /// Model id sent in chat requests.
    #[arg(long)]
    model: Option<String>,
    /// API key for protected `/v1/*` routes.
    #[arg(long, env = "RBITNET_API_KEY")]
    api_key: Option<String>,
    /// Admin token for reload/unload actions.
    #[arg(long, env = "RBITNET_ADMIN_TOKEN")]
    admin_token: Option<String>,
    /// Launch and manage a local `rbitnet-server` child process.
    #[arg(long)]
    serve: bool,
    /// Listen address for `--serve`.
    #[arg(long, env = "RBITNET_BIND", default_value = "127.0.0.1:8080")]
    bind: String,
    /// Server binary used by `--serve` (defaults to sibling `rbitnet-server`).
    #[arg(long)]
    server_bin: Option<PathBuf>,
    /// GGUF path passed to the managed server.
    #[arg(long = "model-path", env = "RBITNET_MODEL")]
    model_path: Option<PathBuf>,
    /// Tokenizer path passed to the managed server.
    #[arg(long, env = "RBITNET_TOKENIZER")]
    tokenizer: Option<PathBuf>,
    /// Chat format passed to the managed server.
    #[arg(long, env = "RBITNET_CHAT_FORMAT")]
    chat_format: Option<String>,
    /// Initial max_tokens.
    #[arg(long, default_value_t = 64)]
    max_tokens: u32,
    /// Initial temperature.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    /// Initial top_p.
    #[arg(long)]
    top_p: Option<f32>,
    /// Optional seed.
    #[arg(long)]
    seed: Option<u64>,
    /// Append prompt/reply rows as JSONL.
    #[arg(long)]
    transcript: Option<PathBuf>,
}

impl From<ChatCmd> for chat_tui::ChatOptions {
    fn from(cmd: ChatCmd) -> Self {
        Self {
            base_url: cmd.base_url,
            model: cmd.model,
            api_key: cmd.api_key,
            admin_token: cmd.admin_token,
            serve: cmd.serve,
            bind: cmd.bind,
            server_bin: cmd.server_bin,
            model_path: cmd.model_path,
            tokenizer: cmd.tokenizer,
            chat_format: cmd.chat_format,
            max_tokens: cmd.max_tokens,
            temperature: cmd.temperature,
            top_p: cmd.top_p,
            seed: cmd.seed,
            transcript: cmd.transcript,
        }
    }
}

#[derive(Args)]
struct QuickstartCmd {
    /// Hugging Face model repo id, e.g. TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF.
    model_id: String,
    /// Files to fetch (repeatable). If omitted, downloads all `.gguf` plus tokenizer files when present.
    #[arg(long = "file", short = 'f', action = clap::ArgAction::Append)]
    files: Vec<String>,
    /// Destination directory for downloaded files.
    #[arg(long, default_value = "models")]
    dir: PathBuf,
    /// Print commands without downloading files.
    #[arg(long)]
    no_download: bool,
    /// Listen address to use in printed commands.
    #[arg(long, env = "RBITNET_BIND", default_value = "127.0.0.1:8080")]
    bind: String,
    /// Optional chat prompt format: raw, llama3, or chatml.
    #[arg(long)]
    chat_format: Option<String>,
    #[arg(long, env = "HF_TOKEN")]
    token: Option<String>,
    #[arg(long)]
    symlink: bool,
    /// Write a local `rbitnet.toml` with the resolved model/tokenizer defaults.
    #[arg(long)]
    write_config: bool,
    /// Config path for `--write-config` (default: `rbitnet.toml`, or user config with `--user-config`).
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,
    /// Write config to the user config directory instead of the current directory.
    #[arg(long)]
    user_config: bool,
}

#[derive(Args)]
struct UpCmd {
    /// Hugging Face model repo id, e.g. TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF.
    model_id: String,
    /// Files to fetch (repeatable). If omitted, downloads all `.gguf` plus tokenizer files when present.
    #[arg(long = "file", short = 'f', action = clap::ArgAction::Append)]
    files: Vec<String>,
    /// Destination directory for downloaded files.
    #[arg(long, default_value = "models")]
    dir: PathBuf,
    /// Print commands and write config without downloading files.
    #[arg(long)]
    no_download: bool,
    /// Listen address written to config and shown in commands.
    #[arg(long, env = "RBITNET_BIND", default_value = "127.0.0.1:8080")]
    bind: String,
    /// Optional chat prompt format: raw, llama3, or chatml.
    #[arg(long)]
    chat_format: Option<String>,
    #[arg(long, env = "HF_TOKEN")]
    token: Option<String>,
    #[arg(long)]
    symlink: bool,
    /// Config path to write (default: `rbitnet.toml`, or user config with `--user-config`).
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,
    /// Write config to the user config directory instead of the current directory.
    #[arg(long)]
    user_config: bool,
}

fn apply_serve_cli_env(cmd: &ServeCmd) {
    if let Some(k) = &cmd.api_key {
        if std::env::var_os("RBITNET_API_KEY").is_none() {
            std::env::set_var("RBITNET_API_KEY", k);
        }
    }
    if let Some(b) = &cmd.bind {
        if std::env::var_os("RBITNET_BIND").is_none() {
            std::env::set_var("RBITNET_BIND", b);
        }
    }
}

fn apply_ui_cli_env(cmd: &UiCmd) {
    if let Some(k) = &cmd.api_key {
        if std::env::var_os("RBITNET_API_KEY").is_none() {
            std::env::set_var("RBITNET_API_KEY", k);
        }
    }
    if let Some(b) = &cmd.bind {
        if std::env::var_os("RBITNET_BIND").is_none() {
            std::env::set_var("RBITNET_BIND", b);
        }
    }
}

fn ui_url_from_env() -> String {
    let bind = std::env::var("RBITNET_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
    format!("http://{bind}/ui")
}

fn open_url_in_browser(url: &str) {
    #[cfg(target_os = "windows")]
    let result = Command::new("cmd").args(["/C", "start", "", url]).spawn();

    #[cfg(target_os = "macos")]
    let result = Command::new("open").arg(url).spawn();

    #[cfg(all(unix, not(target_os = "macos")))]
    let result = Command::new("xdg-open").arg(url).spawn();

    if let Err(e) = result {
        eprintln!("could not open browser automatically: {e}");
    }
}

fn open_url_in_browser_soon(url: String) {
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(900));
        open_url_in_browser(&url);
    });
}

fn user_config_path() -> Result<PathBuf, String> {
    if let Some(dir) = std::env::var_os("RBITNET_CONFIG_DIR") {
        return Ok(PathBuf::from(dir).join("rbitnet.toml"));
    }
    #[cfg(windows)]
    {
        std::env::var_os("APPDATA")
            .map(|d| PathBuf::from(d).join("Rbitnet").join("rbitnet.toml"))
            .ok_or_else(|| "APPDATA is not set; pass --config instead".to_string())
    }
    #[cfg(not(windows))]
    {
        if let Some(dir) = std::env::var_os("XDG_CONFIG_HOME") {
            return Ok(PathBuf::from(dir).join("rbitnet").join("rbitnet.toml"));
        }
        std::env::var_os("HOME")
            .map(|h| {
                PathBuf::from(h)
                    .join(".config")
                    .join("rbitnet")
                    .join("rbitnet.toml")
            })
            .ok_or_else(|| "HOME is not set; pass --config instead".to_string())
    }
}

fn config_path(explicit: &Option<PathBuf>, user_config: bool) -> Result<PathBuf, String> {
    if let Some(path) = explicit {
        Ok(path.clone())
    } else if user_config {
        user_config_path()
    } else {
        Ok(PathBuf::from("rbitnet.toml"))
    }
}

fn toml_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn write_rbitnet_config(
    path: &Path,
    model_path: &Path,
    tokenizer_path: Option<&Path>,
    bind: &str,
    chat_format: Option<&str>,
) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let mut body = String::new();
    body.push_str("# Generated by rbitnet quickstart/up. Env vars still take precedence.\n");
    body.push_str(&format!(
        "model = {}\n",
        toml_string(&model_path.display().to_string())
    ));
    if let Some(tok) = tokenizer_path {
        body.push_str(&format!(
            "tokenizer = {}\n",
            toml_string(&tok.display().to_string())
        ));
    }
    body.push_str(&format!("bind = {}\n", toml_string(bind)));
    if let Some(fmt) = chat_format.map(str::trim).filter(|s| !s.is_empty()) {
        body.push_str(&format!("chat_format = {}\n", toml_string(fmt)));
    }
    fs::write(path, body).map_err(|e| format!("write {}: {e}", path.display()))
}

impl From<UpCmd> for QuickstartCmd {
    fn from(cmd: UpCmd) -> Self {
        Self {
            model_id: cmd.model_id,
            files: cmd.files,
            dir: cmd.dir,
            no_download: cmd.no_download,
            bind: cmd.bind,
            chat_format: cmd.chat_format,
            token: cmd.token,
            symlink: cmd.symlink,
            write_config: true,
            config: cmd.config,
            user_config: cmd.user_config,
        }
    }
}

fn path_for_downloaded_file(dir: &Path, file: &str) -> Result<PathBuf, String> {
    let rel = Path::new(file);
    for component in rel.components() {
        match component {
            std::path::Component::Normal(_) | std::path::Component::CurDir => {}
            _ => {
                return Err(format!(
                "unsafe path component in '{file}': only relative paths without '..' are allowed"
            ))
            }
        }
    }
    Ok(dir.join(rel))
}

fn print_quickstart(cmd: QuickstartCmd) -> Result<(), String> {
    let resolved =
        download::resolve_download_files(&cmd.model_id, &cmd.files, cmd.token.as_deref())?;
    if resolved.is_empty() {
        return Err(format!(
            "no downloadable files resolved for {}",
            cmd.model_id
        ));
    }

    let paths = if cmd.no_download {
        resolved
            .iter()
            .map(|f| path_for_downloaded_file(&cmd.dir, f))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        eprintln!(
            "Downloading {} file(s) from {} -> {}",
            resolved.len(),
            cmd.model_id,
            cmd.dir.display()
        );
        download::download_files(
            &cmd.model_id,
            &resolved,
            &cmd.dir,
            cmd.token.as_deref(),
            hub_place_mode(cmd.symlink),
        )?
    };

    let model_path = paths
        .iter()
        .find(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
        })
        .ok_or_else(|| "resolved files did not include a .gguf file".to_string())?;
    let tokenizer_path = paths.iter().find(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .map(|n| {
                let n = n.to_ascii_lowercase();
                n == "tokenizer.json" || n == "tokenizer.model"
            })
            .unwrap_or(false)
    });

    println!("Rbitnet quickstart for {}", cmd.model_id);
    println!();
    if cmd.no_download {
        println!(
            "(download skipped; paths below assume files are present under {})",
            cmd.dir.display()
        );
        println!();
    }
    println!("PowerShell:");
    println!("  $env:RBITNET_MODEL=\"{}\"", model_path.display());
    if let Some(tok) = tokenizer_path {
        println!("  $env:RBITNET_TOKENIZER=\"{}\"", tok.display());
    }
    if let Some(fmt) = &cmd.chat_format {
        println!("  $env:RBITNET_CHAT_FORMAT=\"{}\"", fmt);
    }
    println!("  $env:RBITNET_BIND=\"{}\"", cmd.bind);
    println!("  rbitnet serve");
    println!();
    println!("bash/zsh:");
    println!("  export RBITNET_MODEL=\"{}\"", model_path.display());
    if let Some(tok) = tokenizer_path {
        println!("  export RBITNET_TOKENIZER=\"{}\"", tok.display());
    }
    if let Some(fmt) = &cmd.chat_format {
        println!("  export RBITNET_CHAT_FORMAT=\"{}\"", fmt);
    }
    println!("  export RBITNET_BIND=\"{}\"", cmd.bind);
    println!("  rbitnet serve");
    println!();
    println!("Local URL: http://{}", cmd.bind);
    println!("Models:    curl -s http://{}/v1/models", cmd.bind);
    println!("Chat:");
    println!(
        "  curl -s http://{}/v1/chat/completions -H \"Content-Type: application/json\" -d '{{\"model\":\"rbitnet-llama\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello from Rbitnet\"}}],\"max_tokens\":64,\"temperature\":0.7}}'",
        cmd.bind
    );
    if tokenizer_path.is_none() {
        println!();
        println!(
            "Note: no tokenizer.json/tokenizer.model was found in the resolved repo file list."
        );
        println!("Place a tokenizer beside the GGUF or set RBITNET_TOKENIZER before launching.");
    }
    if cmd.write_config {
        let path = config_path(&cmd.config, cmd.user_config)?;
        write_rbitnet_config(
            &path,
            model_path,
            tokenizer_path.map(PathBuf::as_path),
            &cmd.bind,
            cmd.chat_format.as_deref(),
        )?;
        println!();
        println!("Wrote config: {}", path.display());
        println!("Next: rbitnet serve --open-ui");
    } else {
        println!();
        println!("Tip: add --write-config (or use `rbitnet up`) so `rbitnet serve` works without env vars.");
    }
    Ok(())
}

#[derive(Args)]
struct TrainCmd {
    /// Root of the Rbitnet repository (must contain `training/recipes/`).
    #[arg(
        long,
        env = "RBITNET_REPO_ROOT",
        default_value = ".",
        value_name = "DIR"
    )]
    repo_root: PathBuf,
    /// Path under `training/` to the recipe script.
    #[arg(long, default_value = "recipes/sft_lora.py", value_name = "REL_PATH")]
    recipe: PathBuf,
    /// Forwarded to the Python script (place `--` before flags if clap mis-parses).
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    passthrough: Vec<String>,
}

#[derive(Args)]
struct ExportGgufCmd {
    /// Hugging Face export directory (`config.json` + model weights) to mention in the example command.
    #[arg(long, value_name = "DIR")]
    checkpoint: Option<PathBuf>,
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// List curated / project-tested models (HTTPS JSON catalog).
    List {
        /// Override the default GitHub raw URL for `compatible_models.json`.
        #[arg(long, env = "RBITNET_MODELS_INDEX_URL")]
        index_url: Option<String>,
        /// Tableau interactif (ratatui) : détail par ligne et téléchargement (touche `d`).
        #[arg(short, long)]
        interactive: bool,
        /// Répertoire cible pour `d` dans le mode interactif (défaut : `models`).
        #[arg(long, env = "RBITNET_DOWNLOAD_DIR", default_value = "models")]
        download_dir: PathBuf,
        /// Jeton Hugging Face (modèles privés / rate limit) pour le téléchargement depuis le TUI.
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
        /// Créer un lien symbolique vers le fichier en cache Hub au lieu d’un hard link / copie.
        #[arg(long)]
        symlink: bool,
    },
    /// Search Hugging Face for model repos that expose at least one `.gguf` file.
    Search {
        query: String,
        #[arg(long, default_value_t = 50)]
        search_limit: usize,
        #[arg(long, default_value_t = 120)]
        max_inspect: usize,
        /// Disable strict BitNet filtering and show all GGUF repos.
        /// By default, only repos heuristically matching BitNet (`likely`/`possible`) are shown.
        #[arg(long)]
        all_gguf: bool,
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
        #[arg(short, long)]
        interactive: bool,
        #[arg(long, env = "RBITNET_DOWNLOAD_DIR", default_value = "models")]
        download_dir: PathBuf,
        #[arg(long)]
        symlink: bool,
    },
    /// Install a curated BitNet-related bundle (paired GGUF + tokenizer repos) and write `rbitnet.manifest.json`.
    Install {
        /// Print known bundle ids and exit.
        #[arg(long, conflicts_with = "bundle_id")]
        list: bool,
        /// Bundle id (see `--list`), e.g. `microsoft-bitnet-b1.58-2b-4t`.
        #[arg(required_unless_present = "list")]
        bundle_id: Option<String>,
        #[arg(long, default_value = ".", env = "RBITNET_DOWNLOAD_DIR")]
        dir: PathBuf,
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
        #[arg(long)]
        symlink: bool,
    },
    /// Download files from a Hugging Face model repo (uses HF cache, then hard link or copy into `--dir`).
    Download {
        /// Repository id, e.g. `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`.
        repo_id: String,
        /// Files to fetch (repeatable). If omitted, downloads all `.gguf` plus tokenizer files when present.
        #[arg(long = "file", short = 'f', action = clap::ArgAction::Append)]
        files: Vec<String>,
        #[arg(long, default_value = ".")]
        dir: PathBuf,
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
        #[arg(long)]
        symlink: bool,
    },
    /// Inspect a local model file or directory and print paths Rbitnet would use.
    Inspect {
        /// Local `.gguf` file or directory containing downloaded model files.
        path: PathBuf,
    },
    /// Remove a local model file or directory downloaded under your model cache.
    #[command(alias = "remove")]
    Rm {
        /// Local `.gguf` file or directory to delete.
        path: PathBuf,
        /// Do not ask for confirmation.
        #[arg(short, long)]
        yes: bool,
    },
    /// Resolve GGUF/tokenizer files for one Hugging Face repo and print suggested `RBITNET_*`.
    Resolve {
        /// Repository id, e.g. `microsoft/bitnet-b1.58-2B-4T-gguf`.
        repo_id: String,
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
        /// Print pretty JSON.
        #[arg(long)]
        json: bool,
    },
    /// Build a `compatible_models.json` skeleton from Hugging Face (one GGUF + tokenizer per repo when found).
    ///
    /// Output is meant to be reviewed and committed; it does not replace project testing.
    GenerateCatalog {
        /// Hub search string. Default `gguf` surfaces repos that usually ship `.gguf` files;
        /// `llama` alone tends to return Safetensors-only Meta repos first.
        #[arg(long, default_value = "gguf")]
        query: String,
        #[arg(long, default_value_t = 100)]
        search_limit: usize,
        /// Max `/api/models/{repo}` fetches (skips non-GGUF repos without counting toward `--max-entries`).
        #[arg(long, default_value_t = 250)]
        max_inspect: usize,
        #[arg(long, default_value_t = 40)]
        max_entries: usize,
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
        /// Write JSON to this path instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

fn print_catalog_list(url: &str) -> Result<(), String> {
    let cat = catalog::fetch_catalog(url)?;
    println!("Catalog URL: {url}");
    println!("Schema version: {}", cat.version);
    println!();
    if cat.models.is_empty() {
        println!("(no curated models in this index)");
        return Ok(());
    }
    for m in &cat.models {
        println!("id: {}", m.id);
        println!("  repo: {}", m.repo);
        println!("  description: {}", m.description);
        if let Some(f) = &m.file {
            println!("  file: {f}");
        }
        if !m.files.is_empty() {
            println!("  files: {}", m.files.join(", "));
        }
        if let Some(ram) = &m.min_ram {
            println!("  min_ram: {ram}");
        }
        if let Some(tier) = &m.tier {
            println!("  tier: {tier}");
        }
        if !m.use_case.is_empty() {
            println!("  use_case: {}", m.use_case.join(", "));
        }
        if let Some(ram_gb) = m.min_ram_gb {
            println!("  min_ram_gb: {ram_gb}");
        }
        if let Some(verified) = m.verified {
            println!("  verified: {verified}");
        }
        if let Some(notes) = &m.notes {
            println!("  notes: {notes}");
        }
        if let Some(tested) = m.tested {
            println!("  tested: {tested}");
        }
        if let Some(v) = &m.min_rbitnet_version {
            println!("  min_rbitnet_version: {v}");
        }
        if let Some(v) = &m.smoke_test {
            println!("  smoke_test: {v}");
        }
        println!();
    }
    Ok(())
}

fn find_named_file(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_named_file(&path, names) {
                return Some(found);
            }
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if names
            .iter()
            .any(|candidate| name.eq_ignore_ascii_case(candidate))
        {
            return Some(path);
        }
    }
    None
}

fn find_first_gguf(dir: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    let mut dirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            dirs.push(path);
            continue;
        }
        if path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
        {
            return Some(path);
        }
    }
    for dir in dirs {
        if let Some(found) = find_first_gguf(&dir) {
            return Some(found);
        }
    }
    None
}

fn inspect_model_path(path: &Path) -> Result<(), String> {
    let meta = fs::metadata(path).map_err(|e| format!("inspect {}: {e}", path.display()))?;
    let (model, root) = if meta.is_dir() {
        (find_first_gguf(path), path.to_path_buf())
    } else {
        (
            Some(path.to_path_buf()),
            path.parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf(),
        )
    };
    let tokenizer = find_named_file(&root, &["tokenizer.json", "tokenizer.model"]);
    let template = find_named_file(&root, &["tokenizer_config.json"]);
    println!("root: {}", root.display());
    match model {
        Some(m) => println!("model: {}", m.display()),
        None => println!("model: (no .gguf found)"),
    }
    match tokenizer {
        Some(t) => println!("tokenizer: {}", t.display()),
        None => {
            println!("tokenizer: (not found; set RBITNET_TOKENIZER or place one beside the GGUF)")
        }
    }
    match template {
        Some(t) => println!("template_source: {}", t.display()),
        None => println!(
            "template_source: env RBITNET_CHAT_TEMPLATE / RBITNET_CHAT_FORMAT / raw fallback"
        ),
    }
    Ok(())
}

fn remove_model_path(path: &Path, yes: bool) -> Result<(), String> {
    let meta = fs::metadata(path).map_err(|e| format!("remove {}: {e}", path.display()))?;
    if !yes {
        return Err(format!(
            "refusing to remove {} without --yes ({}).",
            path.display(),
            if meta.is_dir() { "directory" } else { "file" }
        ));
    }
    if meta.is_dir() {
        fs::remove_dir_all(path).map_err(|e| format!("remove dir {}: {e}", path.display()))?;
    } else {
        fs::remove_file(path).map_err(|e| format!("remove file {}: {e}", path.display()))?;
    }
    println!("removed: {}", path.display());
    Ok(())
}

fn run_models(cmd: ModelsCmd) -> Result<(), String> {
    match cmd {
        ModelsCmd::List {
            index_url,
            interactive,
            download_dir,
            token,
            symlink,
        } => {
            let url = index_url.unwrap_or_else(|| catalog::DEFAULT_MODELS_INDEX_URL.to_string());
            if interactive {
                interactive_models::run_catalog_interactive(
                    &url,
                    token,
                    download_dir,
                    hub_place_mode(symlink),
                )
            } else {
                print_catalog_list(&url)
            }
        }
        ModelsCmd::Install {
            list,
            bundle_id,
            dir,
            token,
            symlink,
        } => {
            if list {
                print!("{}", bitnet_install::list_bundles_text());
                return Ok(());
            }
            // clap guarantees bundle_id is Some (required_unless_present = "list")
            let id =
                bundle_id.expect("bundle_id guaranteed by clap (required_unless_present = list)");
            bitnet_install::install_bundle(&id, &dir, token.as_deref(), hub_place_mode(symlink))
        }
        ModelsCmd::Search {
            query,
            search_limit,
            max_inspect,
            all_gguf,
            token,
            interactive,
            download_dir,
            symlink,
        } => {
            let strict_bitnet = !all_gguf;
            eprintln!("{}", hf_search::SEARCH_WARNING);
            if strict_bitnet {
                eprintln!("strict-bitnet: enabled (keeps likely/possible BitNet candidates only).");
            }
            eprintln!();
            if interactive {
                interactive_models::run_search_interactive(
                    &query,
                    search_limit,
                    max_inspect,
                    strict_bitnet,
                    token,
                    download_dir,
                    hub_place_mode(symlink),
                )
            } else {
                let hits = hf_search::search_gguf_models(
                    &query,
                    search_limit,
                    max_inspect,
                    strict_bitnet,
                    token.as_deref(),
                )?;
                if hits.is_empty() {
                    println!(
                        "No repos with .gguf files found (try --query gguf / TheBloke, or raise --max-inspect / --search-limit)."
                    );
                    return Ok(());
                }
                for h in hits {
                    println!(
                        "{} [{}:{} rbitnet={}]",
                        h.id,
                        h.confidence.label(),
                        h.confidence_score,
                        h.readiness.label()
                    );
                    for f in &h.gguf_files {
                        println!("  {f}");
                    }
                    if let Some(t) = &h.tokenizer_json {
                        println!("  {t}");
                    }
                    if let Some(t) = &h.tokenizer_model {
                        println!("  {t}");
                    }
                    if h.tokenizer_json.is_none() && h.tokenizer_model.is_none() {
                        if let Some(t) = &h.tokenizer_config_json {
                            println!("  {t}  (config only — see USAGE tokenizer note)");
                        } else {
                            println!("  (no tokenizer.json / tokenizer.model in repo file list)");
                        }
                    }
                    println!();
                }
                Ok(())
            }
        }
        ModelsCmd::Download {
            repo_id,
            files,
            dir,
            token,
            symlink,
        } => {
            let resolved = download::resolve_download_files(&repo_id, &files, token.as_deref())?;
            eprintln!(
                "Downloading {} file(s) from {} -> {}",
                resolved.len(),
                repo_id,
                dir.display()
            );
            let paths = download::download_files(
                &repo_id,
                &resolved,
                &dir,
                token.as_deref(),
                hub_place_mode(symlink),
            )?;
            for p in paths {
                println!("{}", p.display());
            }
            Ok(())
        }
        ModelsCmd::Inspect { path } => inspect_model_path(&path),
        ModelsCmd::Rm { path, yes } => remove_model_path(&path, yes),
        ModelsCmd::Resolve {
            repo_id,
            token,
            json,
        } => {
            let resolved = bitnet_install::resolve_model(&repo_id, token.as_deref())?;
            if json {
                let body = serde_json::to_string_pretty(&resolved)
                    .map_err(|e| format!("serialize resolve output: {e}"))?;
                println!("{body}");
                return Ok(());
            }
            println!("repo: {}", resolved.repo_id);
            println!("readiness: {}", resolved.readiness);
            if resolved.gguf.is_empty() {
                println!("gguf: (none)");
            } else {
                println!("gguf:");
                for g in &resolved.gguf {
                    println!("  {g}");
                }
            }
            if let Some(v) = &resolved.tokenizer_json {
                println!("tokenizer.json: {v}");
            }
            if let Some(v) = &resolved.tokenizer_model {
                println!("tokenizer.model: {v}");
            }
            if let Some(v) = &resolved.tokenizer_config_json {
                println!("tokenizer_config.json: {v}");
            }
            if !resolved.suggested_env.is_empty() {
                println!("suggested env:");
                for (k, v) in &resolved.suggested_env {
                    println!("  {k}={v}");
                }
            }
            Ok(())
        }
        ModelsCmd::GenerateCatalog {
            query,
            search_limit,
            max_inspect,
            max_entries,
            token,
            output,
        } => {
            eprintln!("{}", hf_search::GENERATE_CATALOG_WARNING);
            eprintln!();
            let repos = hf_search::discover_gguf_repos(
                &query,
                search_limit,
                max_inspect,
                max_entries,
                None,
                token.as_deref(),
            )?;
            let mut models = Vec::new();
            for r in repos {
                let Some(primary) = catalog::pick_primary_gguf(&r.gguf_files) else {
                    continue;
                };
                let mut files = vec![primary];
                if let Some(tj) = r.tokenizer_json {
                    files.push(tj);
                } else if let Some(tm) = r.tokenizer_model {
                    files.push(tm);
                }
                models.push(catalog::CatalogModel {
                    id: catalog::catalog_id_from_repo(&r.id),
                    repo: r.id.clone(),
                    description: format!(
                        "Auto-discovered on Hugging Face (search query: {query}). Primary GGUF chosen heuristically; not Rbitnet-CI-tested."
                    ),
                    file: None,
                    files,
                    notes: None,
                    tier: Some("power".into()),
                    use_case: vec!["chat".into()],
                    min_ram_gb: None,
                    verified: Some(false),
                    min_ram: None,
                    tested: None,
                    min_rbitnet_version: None,
                    smoke_test: Some("cargo test -p rbitnet-cli catalog".into()),
                });
            }
            let cat = catalog::Catalog { version: 1, models };
            if cat.models.is_empty() {
                eprintln!(
                    "hint: aucun dépôt avec fichiers .gguf trouvé pour cette requête. Essayez `--query gguf` ou `--query tinyllama`, ou augmentez `--max-inspect` / `--search-limit`."
                );
            }
            let json = serde_json::to_string_pretty(&cat)
                .map_err(|e| format!("serialize catalog: {e}"))?;
            if let Some(path) = output {
                fs::write(&path, json).map_err(|e| format!("write {}: {e}", path.display()))?;
                eprintln!("Wrote {} model(s) to {}", cat.models.len(), path.display());
            } else {
                println!("{json}");
            }
            Ok(())
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Quickstart(cmd) => print_quickstart(cmd),
        Commands::Up(cmd) => print_quickstart(cmd.into()),
        Commands::Models(m) => run_models(m),
        Commands::Train(cmd) => train_cli::run_train(&cmd.repo_root, &cmd.recipe, &cmd.passthrough),
        Commands::ExportGguf(cmd) => {
            train_cli::print_export_gguf_hint(cmd.checkpoint.as_deref());
            Ok(())
        }
        Commands::Serve(cmd) => {
            apply_serve_cli_env(&cmd);
            if cmd.open_ui {
                let url = ui_url_from_env();
                eprintln!("Opening Rbitnet UI: {url}");
                open_url_in_browser_soon(url);
            }
            eprintln!("Rbitnet UI: {}", ui_url_from_env());
            bitnet_server::run_server().await.map_err(|e| e.to_string())
        }
        Commands::Ui(cmd) => {
            apply_ui_cli_env(&cmd);
            let url = ui_url_from_env();
            eprintln!("Opening Rbitnet UI: {url}");
            open_url_in_browser_soon(url);
            bitnet_server::run_server().await.map_err(|e| e.to_string())
        }
        Commands::Chat(cmd) => chat_tui::run_chat_tui(cmd.into()),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
