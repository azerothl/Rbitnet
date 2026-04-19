//! `rbitnet` — list curated models, search Hugging Face for `.gguf` repos, download weights, or run the HTTP server.

mod catalog;
mod download;
mod hf_search;

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
    },
    /// Search Hugging Face for model repos that expose at least one `.gguf` file.
    Search {
        query: String,
        #[arg(long, default_value_t = 25)]
        search_limit: usize,
        #[arg(long, default_value_t = 20)]
        max_inspect: usize,
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
        ModelsCmd::List { index_url } => {
            let url = index_url
                .unwrap_or_else(|| catalog::DEFAULT_MODELS_INDEX_URL.to_string());
            print_catalog_list(&url)
        }
        ModelsCmd::Search {
            query,
            search_limit,
            max_inspect,
            token,
        } => {
            eprintln!("{}", hf_search::SEARCH_WARNING);
            eprintln!();
            let hits = hf_search::search_gguf_models(
                &query,
                search_limit,
                max_inspect,
                token.as_deref(),
            )?;
            if hits.is_empty() {
                println!("No repos with .gguf files found in the first inspected candidates (try a different query or increase --max-inspect).");
                return Ok(());
            }
            for h in hits {
                println!("{}", h.id);
                for f in &h.gguf_files {
                    println!("  {f}");
                }
                println!();
            }
            Ok(())
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
