//! `rbitnet` — list curated models, search Hugging Face for `.gguf` repos, download weights, or run the HTTP server.

mod bitnet_install;
mod catalog;
mod download;
mod hf_search;
mod hub_http;
mod interactive_models;

use std::fs;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "rbitnet", version, about = "Rbitnet CLI: Hugging Face models, download, and optional HTTP server")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(subcommand, about = "Curated catalog, HF search, and downloads")]
    Models(ModelsCmd),
    /// Run the OpenAI-compatible HTTP server (same as `rbitnet-server`).
    Serve,
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
    },
    /// Download files from a Hugging Face model repo (uses HF cache, then copies into `--dir`).
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
        if !m.files.is_empty() {
            println!("  files: {}", m.files.join(", "));
        }
        if let Some(v) = &m.min_rbitnet_version {
            println!("  min_rbitnet_version: {v}");
        }
        println!();
    }
    Ok(())
}

fn run_models(cmd: ModelsCmd) -> Result<(), String> {
    match cmd {
        ModelsCmd::List {
            index_url,
            interactive,
            download_dir,
            token,
        } => {
            let url = index_url
                .unwrap_or_else(|| catalog::DEFAULT_MODELS_INDEX_URL.to_string());
            if interactive {
                interactive_models::run_catalog_interactive(&url, token, download_dir)
            } else {
                print_catalog_list(&url)
            }
        }
        ModelsCmd::Install {
            list,
            bundle_id,
            dir,
            token,
        } => {
            if list {
                print!("{}", bitnet_install::list_bundles_text());
                return Ok(());
            }
            // clap guarantees bundle_id is Some (required_unless_present = "list")
            let id = bundle_id.expect("bundle_id guaranteed by clap (required_unless_present = list)");
            bitnet_install::install_bundle(&id, &dir, token.as_deref())
        }
        ModelsCmd::Search {
            query,
            search_limit,
            max_inspect,
            all_gguf,
            token,
            interactive,
            download_dir,
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
        } => {
            let resolved = download::resolve_download_files(&repo_id, &files, token.as_deref())?;
            eprintln!(
                "Downloading {} file(s) from {} -> {}",
                resolved.len(),
                repo_id,
                dir.display()
            );
            let paths = download::download_files(&repo_id, &resolved, &dir, token.as_deref())?;
            for p in paths {
                println!("{}", p.display());
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
                    files,
                    min_rbitnet_version: None,
                });
            }
            let cat = catalog::Catalog {
                version: 1,
                models,
            };
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
        Commands::Models(m) => run_models(m),
        Commands::Serve => bitnet_server::run_server().await.map_err(|e| e.to_string()),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
