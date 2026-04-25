//! Navigateur terminal interactif (tableau + détails + téléchargement).

use std::path::PathBuf;
use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState, Wrap};
use ratatui::{DefaultTerminal, Frame};

use crate::catalog;
use crate::download;
use crate::hf_search;

/// Une ligne affichable (catalogue curaté ou résultats de recherche HF).
#[derive(Clone)]
pub struct ModelBrowserRow {
    pub id: String,
    pub repo_id: String,
    pub description: String,
    pub confidence: Option<String>,
    /// Best-effort Rbitnet readiness (`models search` only).
    pub readiness: Option<String>,
    /// Texte multi-lignes pour le panneau détail (fichiers, version min, etc.).
    pub detail_body: String,
    /// `Some` = fichiers explicites (catalogue) ; `None` = résolution auto comme `models download` sans `--file`.
    pub explicit_files: Option<Vec<String>>,
}

impl ModelBrowserRow {
    pub fn from_catalog(m: &catalog::CatalogModel) -> Self {
        let mut detail = String::new();
        detail.push_str(&format!("Repo: {}\n", m.repo));
        detail.push_str(&format!("Description:\n{}\n", m.description));
        if let Some(v) = &m.min_rbitnet_version {
            detail.push_str(&format!("min_rbitnet_version: {v}\n"));
        }
        detail.push_str(&format!("\nFichiers ({}):\n", m.files.len()));
        for f in &m.files {
            detail.push_str(f);
            detail.push('\n');
        }
        Self {
            id: m.id.clone(),
            repo_id: m.repo.clone(),
            description: m.description.clone(),
            confidence: None,
            readiness: None,
            detail_body: detail,
            explicit_files: Some(m.files.clone()),
        }
    }

    pub fn from_search_hit(h: &hf_search::GgufSearchHit) -> Self {
        let mut detail = String::new();
        let mut explicit_files = h.gguf_files.clone();
        detail.push_str(&format!("Repo: {}\n", h.id));
        detail.push_str(
            "Recherche HF: dépôts contenant des fichiers .gguf. Attention: GGUF ne veut PAS dire modèle BitNet 1-bit ni compatibilité Rbitnet.\n\n",
        );
        detail.push_str(&format!(
            "Confiance heuristique BitNet: {} (score {})\n",
            h.confidence.label(),
            h.confidence_score
        ));
        detail.push_str(&format!(
            "Lisibilité Rbitnet (heuristique siblings): {}\n\n",
            h.readiness.label()
        ));
        for f in &h.gguf_files {
            detail.push_str(f);
            detail.push('\n');
        }
        detail.push_str("\nTokenizer (fichiers listés par l’API Hub / siblings) :\n");
        if let Some(tok) = &h.tokenizer_json {
            detail.push_str(tok);
            detail.push('\n');
            explicit_files.push(tok.clone());
        }
        if let Some(tok) = &h.tokenizer_model {
            detail.push_str(tok);
            detail.push('\n');
            explicit_files.push(tok.clone());
        }
        if h.tokenizer_json.is_none() && h.tokenizer_model.is_none() {
            if let Some(cfg) = &h.tokenizer_config_json {
                detail.push_str(cfg);
                detail.push('\n');
                detail.push_str(
                    "(tokenizer_config.json seul : métadonnées Transformers ; Rbitnet attend tokenizer.json ou tokenizer.model, ou RBITNET_TOKENIZER.)\n",
                );
            } else {
                detail.push_str(
                    "(aucun tokenizer.json / tokenizer.model / tokenizer_config.json dans les siblings — le README peut charger le tokenizer via AutoTokenizer depuis un autre dépôt.)\n",
                );
            }
        }
        detail.push_str(
            "\nTéléchargement (d) : fichiers listés ci-dessus (GGUF + tokenizer.json/model si présents). tokenizer_config.json n’est pas téléchargé : inutile pour le moteur Rbitnet tel quel.",
        );
        explicit_files.sort();
        explicit_files.dedup();
        let n = explicit_files.len();
        Self {
            // Keep the exact Hub repo id to avoid any accidental truncation/mangling.
            id: h.id.clone(),
            repo_id: h.id.clone(),
            description: format!("{n} fichier(s) (gguf/tokenizer)"),
            confidence: Some(format!("{}:{}", h.confidence.label(), h.confidence_score)),
            readiness: Some(h.readiness.label().to_string()),
            detail_body: detail,
            explicit_files: Some(explicit_files),
        }
    }
}

#[derive(Clone, Copy)]
enum SearchFilterMode {
    AllGguf,
    StrictBitnet,
}

#[derive(Clone)]
struct SearchContext {
    query: String,
    search_limit: usize,
    max_inspect: usize,
}

struct BrowserApp {
    rows: Vec<ModelBrowserRow>,
    table_state: TableState,
    /// Ligne de détail : scroll vertical (offset en caractères / lignes simplifiées).
    detail_scroll: usize,
    status: String,
    token: Option<String>,
    download_dir: PathBuf,
    title: String,
    /// Hauteur utile du panneau détail (lignes), mise à jour à chaque frame.
    detail_viewport_lines: usize,
    search_filter_mode: Option<SearchFilterMode>,
    search_ctx: Option<SearchContext>,
}

impl BrowserApp {
    fn new(
        rows: Vec<ModelBrowserRow>,
        token: Option<String>,
        download_dir: PathBuf,
        title: String,
        search_filter_mode: Option<SearchFilterMode>,
        search_ctx: Option<SearchContext>,
    ) -> Self {
        Self {
            rows,
            table_state: TableState::default().with_selected(0),
            detail_scroll: 0,
            status: format!(
                "↑/↓ ou j/k : ligne | PgUp/PgDn : détail | d : télécharger vers {} | q/Esc : quitter",
                download_dir.display()
            ),
            token,
            download_dir,
            title,
            detail_viewport_lines: 8,
            search_filter_mode,
            search_ctx,
        }
    }

    fn selected(&self) -> Option<usize> {
        self.table_state.selected()
    }

    fn selected_row(&self) -> Option<&ModelBrowserRow> {
        self.selected().and_then(|i| self.rows.get(i))
    }

    fn next_row(&mut self) {
        if self.rows.is_empty() {
            return;
        }
        let i = match self.table_state.selected() {
            Some(i) if i + 1 < self.rows.len() => i + 1,
            Some(_) => 0,
            None => 0,
        };
        self.table_state.select(Some(i));
        self.detail_scroll = 0;
    }

    fn prev_row(&mut self) {
        if self.rows.is_empty() {
            return;
        }
        let i = match self.table_state.selected() {
            Some(i) if i > 0 => i - 1,
            Some(_) => self.rows.len() - 1,
            None => 0,
        };
        self.table_state.select(Some(i));
        self.detail_scroll = 0;
    }

    fn detail_line_count(&self) -> usize {
        self.selected_row()
            .map(|r| r.detail_body.lines().count())
            .unwrap_or(0)
    }

    fn scroll_detail_up(&mut self, page: usize) {
        self.detail_scroll = self.detail_scroll.saturating_sub(page);
    }

    fn scroll_detail_down(&mut self, page: usize, viewport_lines: usize) {
        let total = self.detail_line_count();
        let max_scroll = total.saturating_sub(viewport_lines);
        self.detail_scroll = (self.detail_scroll + page).min(max_scroll);
    }

    fn download_selected(&mut self) -> Result<(), String> {
        let Some(row) = self.selected_row().cloned() else {
            self.status = "Aucune ligne sélectionnée.".into();
            return Ok(());
        };
        self.status = format!("Téléchargement {} …", row.repo_id);
        let files = match &row.explicit_files {
            Some(f) if !f.is_empty() => {
                download::resolve_download_files(&row.repo_id, f, self.token.as_deref())?
            }
            _ => download::resolve_download_files(&row.repo_id, &[], self.token.as_deref())?,
        };
        let paths = download::download_files(
            &row.repo_id,
            &files,
            &self.download_dir,
            self.token.as_deref(),
        )?;
        self.status = format!(
            "OK — {} fichier(s) écrit(s) sous {}",
            paths.len(),
            self.download_dir.display()
        );
        Ok(())
    }

    fn toggle_search_filter(&mut self) {
        let Some(mode) = self.search_filter_mode else {
            self.status = "Filtre non disponible (catalogue curaté).".into();
            return;
        };
        self.search_filter_mode = Some(match mode {
            SearchFilterMode::AllGguf => SearchFilterMode::StrictBitnet,
            SearchFilterMode::StrictBitnet => SearchFilterMode::AllGguf,
        });
        let strict = matches!(self.search_filter_mode, Some(SearchFilterMode::StrictBitnet));
        let Some(ctx) = &self.search_ctx else {
            self.status = "Contexte de recherche indisponible.".into();
            return;
        };
        let hits = match hf_search::search_gguf_models(
            &ctx.query,
            ctx.search_limit,
            ctx.max_inspect,
            strict,
            self.token.as_deref(),
        ) {
            Ok(v) => v,
            Err(e) => {
                self.status = format!("Erreur reload filtre: {e}");
                return;
            }
        };
        self.rows = hits.iter().map(ModelBrowserRow::from_search_hit).collect();
        self.table_state.select(if self.rows.is_empty() { None } else { Some(0) });
        self.detail_scroll = 0;
        let label = match self.search_filter_mode {
            Some(SearchFilterMode::StrictBitnet) => "strict-bitnet",
            Some(SearchFilterMode::AllGguf) => "all-gguf",
            None => "catalog",
        };
        self.status = format!("Filtre actif: {label} ({} résultat(s))", self.rows.len());
    }

    fn draw(&mut self, frame: &mut Frame) {
        let area = frame.area();
        let vchunks = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(8),
            Constraint::Length(2),
        ])
        .split(area);
        let title_area = vchunks[0];
        let main_area = vchunks[1];
        let status_area = vchunks[2];

        let help_hint = if self.search_filter_mode.is_some() {
            "(↑↓ j/k row · PgUp/PgDn detail · d download · f filter · q quit)"
        } else {
            "(↑↓ j/k row · PgUp/PgDn detail · d download · q quit)"
        };

        let title = Paragraph::new(Line::from(vec![
            self.title.as_str().bold(),
            "  ".into(),
            help_hint.dim().into(),
        ]))
        .style(Style::default().fg(Color::Cyan));
        frame.render_widget(title, title_area);

        let hchunks = Layout::horizontal([
            Constraint::Percentage(48),
            Constraint::Percentage(52),
        ])
        .split(main_area);
        let table_area = hchunks[0];
        let detail_area = hchunks[1];

        let header = Row::new(vec![
            Cell::from("id").style(Style::default().add_modifier(Modifier::BOLD)),
            Cell::from("repo").style(Style::default().add_modifier(Modifier::BOLD)),
            Cell::from("conf").style(Style::default().add_modifier(Modifier::BOLD)),
            Cell::from("rbitnet").style(Style::default().add_modifier(Modifier::BOLD)),
            Cell::from("#f").style(Style::default().add_modifier(Modifier::BOLD)),
            Cell::from("résumé").style(Style::default().add_modifier(Modifier::BOLD)),
        ])
        .style(Style::default().fg(Color::Yellow));
        let rows: Vec<Row> = self
            .rows
            .iter()
            .map(|r| {
                let n_files = r
                    .explicit_files
                    .as_ref()
                    .map(|f| f.len())
                    .unwrap_or_else(|| {
                        r.detail_body
                            .lines()
                            .filter(|l| l.to_ascii_lowercase().ends_with(".gguf"))
                            .count()
                            .max(1)
                    });
                Row::new(vec![
                    Cell::from(truncate(&r.id, 18)),
                    Cell::from(truncate(&r.repo_id, 24)),
                    Cell::from(truncate(
                        r.confidence.as_deref().unwrap_or("-"),
                        14,
                    )),
                    Cell::from(truncate(
                        r.readiness.as_deref().unwrap_or("-"),
                        18,
                    )),
                    Cell::from(format!("{n_files}")),
                    Cell::from(truncate(&r.description, 28)),
                ])
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Max(20),
                Constraint::Min(14),
                Constraint::Length(14),
                Constraint::Length(18),
                Constraint::Length(4),
                Constraint::Min(10),
            ],
        )
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Modèles ")
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .row_highlight_style(Style::default().reversed())
        .column_spacing(1);
        frame.render_stateful_widget(table, table_area, &mut self.table_state);

        let detail_h = detail_area.height.saturating_sub(2).max(1) as usize;
        self.detail_viewport_lines = detail_h;
        let body = self
            .selected_row()
            .map(|r| {
                let lines: Vec<&str> = r.detail_body.lines().collect();
                let start = self.detail_scroll.min(lines.len().saturating_sub(1));
                let end = (start + detail_h).min(lines.len());
                lines[start..end].join("\n")
            })
            .unwrap_or_else(|| "—".into());

        let detail = Paragraph::new(Text::raw(body))
            .wrap(Wrap { trim: true })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Détail (ligne sélectionnée) ")
                    .border_style(Style::default().fg(Color::DarkGray)),
            );
        frame.render_widget(detail, detail_area);

        let status = Paragraph::new(Text::raw(&self.status))
            .style(Style::default().fg(Color::Green))
            .block(Block::default().borders(Borders::TOP));
        frame.render_widget(status, status_area);
    }

    fn run(mut self, terminal: &mut DefaultTerminal) -> Result<(), String> {
        loop {
            terminal
                .draw(|f| self.draw(f))
                .map_err(|e| format!("affichage: {e}"))?;

            if !event::poll(Duration::from_millis(200))
                .map_err(|e| format!("poll: {e}"))?
            {
                continue;
            }
            let Event::Key(key) = event::read().map_err(|e| format!("event: {e}"))? else {
                continue;
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }

            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                KeyCode::Down | KeyCode::Char('j') => self.next_row(),
                KeyCode::Up | KeyCode::Char('k') => self.prev_row(),
                KeyCode::PageDown => {
                    let v = self.detail_viewport_lines.max(1);
                    self.scroll_detail_down(v, v);
                }
                KeyCode::PageUp => {
                    let v = self.detail_viewport_lines.max(1);
                    self.scroll_detail_up(v);
                }
                KeyCode::Char('d') => {
                    if let Err(e) = self.download_selected() {
                        self.status = format!("Erreur: {e}");
                    }
                }
                KeyCode::Char('f') => self.toggle_search_filter(),
                _ => {}
            }
        }
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    let count = s.chars().count();
    if count <= max_chars {
        return s.to_string();
    }
    let take = max_chars.saturating_sub(1);
    format!("{}…", s.chars().take(take).collect::<String>())
}

/// Ouvre le TUI pour le catalogue chargé depuis `url`.
pub fn run_catalog_interactive(
    url: &str,
    token: Option<String>,
    download_dir: PathBuf,
) -> Result<(), String> {
    let cat = catalog::fetch_catalog(url)?;
    if cat.models.is_empty() {
        return Err("le catalogue ne contient aucun modèle.".into());
    }
    let rows: Vec<ModelBrowserRow> = cat.models.iter().map(ModelBrowserRow::from_catalog).collect();
    let title = format!("Catalogue ({url})");
    run_browser(rows, token, download_dir, title, None, None)
}

/// Ouvre le TUI pour les résultats de recherche HF.
pub fn run_search_interactive(
    query: &str,
    search_limit: usize,
    max_inspect: usize,
    strict_bitnet: bool,
    token: Option<String>,
    download_dir: PathBuf,
) -> Result<(), String> {
    let hits = hf_search::search_gguf_models(
        query,
        search_limit,
        max_inspect,
        strict_bitnet,
        token.as_deref(),
    )?;
    if hits.is_empty() {
        return Err(
            "aucun dépôt avec .gguf (essayez --query gguf ou augmentez --max-inspect).".into(),
        );
    }
    let rows: Vec<ModelBrowserRow> = hits.iter().map(ModelBrowserRow::from_search_hit).collect();
    let mode = if strict_bitnet { "strict-bitnet" } else { "all-gguf" };
    let title = format!("Recherche HF (.gguf, non garanti 1-bit, {mode}) « {query} »");
    let initial_mode = if strict_bitnet {
        Some(SearchFilterMode::StrictBitnet)
    } else {
        Some(SearchFilterMode::AllGguf)
    };
    let search_ctx = Some(SearchContext {
        query: query.to_string(),
        search_limit,
        max_inspect,
    });
    run_browser(rows, token, download_dir, title, initial_mode, search_ctx)
}

fn run_browser(
    rows: Vec<ModelBrowserRow>,
    token: Option<String>,
    download_dir: PathBuf,
    title: String,
    search_filter_mode: Option<SearchFilterMode>,
    search_ctx: Option<SearchContext>,
) -> Result<(), String> {
    let mut terminal = ratatui::init();
    let app = BrowserApp::new(rows, token, download_dir, title, search_filter_mode, search_ctx);
    let r = app.run(&mut terminal);
    ratatui::restore();
    r
}
