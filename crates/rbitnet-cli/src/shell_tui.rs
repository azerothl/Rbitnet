//! One ratatui screen for `rbitnet` commands. Chat and the model browser stay in their own screens.

use std::collections::VecDeque;
use std::io::{self, IsTerminal, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::{DefaultTerminal, Frame};

/// Result shown on the shared terminal screen.
#[derive(Debug, Clone)]
pub struct Screen {
    pub title: String,
    pub body: String,
    pub action: Option<String>,
    pub error: Option<String>,
}

impl Screen {
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            title: "Rbitnet".into(),
            body: String::new(),
            action: None,
            error: Some(message.into()),
        }
    }
}

pub fn use_terminal(plain: bool) -> bool {
    !plain && io::stdout().is_terminal()
}

pub fn home_screen() -> Screen {
    let model = std::env::var("RBITNET_MODEL").unwrap_or_else(|_| "aucun".into());
    Screen {
        title: "Rbitnet".into(),
        body: format!(
            "\
Commandes
  quickstart    résoudre un tag et préparer le serveur
  up            écrire la config locale
  serve         lancer le serveur HTTP
  chat          conversation
  models list   catalogue
  models search dépôts GGUF
  welcome       guide
  tune          profil de service
  recipe        recette JSON
  export-gguf   export GGUF

Modèle résolu
  {model}

Prochaine action
  rbitnet quickstart bitnet:2b"
        ),
        action: Some("Entrée : quitter  ·  lancez `rbitnet quickstart bitnet:2b`".into()),
        error: None,
    }
}

pub fn quickstart_screen(
    model_id: &str,
    gguf: &Path,
    tokenizer: Option<&Path>,
    bind: &str,
) -> Screen {
    let tokenizer_line = match tokenizer {
        Some(path) => format!("Tokenizer\n  {}", path.display()),
        None => "Tokenizer\n  absent".into(),
    };
    let error = tokenizer.map(|_| None).unwrap_or_else(|| {
        Some(
            "tokenizer manquant : placez tokenizer.json à côté du GGUF ou définissez RBITNET_TOKENIZER"
                .into(),
        )
    });
    Screen {
        title: format!("Quickstart {model_id}"),
        body: format!(
            "\
GGUF
  {}

{tokenizer_line}

URL
  http://{bind}",
            gguf.display()
        ),
        action: if error.is_none() {
            Some("Entrée : rbitnet serve".into())
        } else {
            None
        },
        error,
    }
}

/// Print the screen for scripts, or open it until Esc / q. Enter returns true when an action is set.
pub fn present(screen: &Screen, plain: bool) -> Result<bool, String> {
    if !use_terminal(plain) {
        if let Some(err) = &screen.error {
            eprintln!("error: {err}");
        }
        if !screen.body.is_empty() {
            println!("{}", screen.body);
        }
        if let Some(action) = &screen.action {
            println!("{action}");
        }
        return Ok(false);
    }
    run(screen)
}

pub fn run(screen: &Screen) -> Result<bool, String> {
    let mut terminal = ratatui::init();
    let result = loop {
        draw(&mut terminal, screen, None, None);
        if event::poll(Duration::from_millis(200)).map_err(|e| e.to_string())? {
            if let Event::Key(key) = event::read().map_err(|e| e.to_string())? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Esc | KeyCode::Char('q') => break Ok(false),
                    KeyCode::Enter if screen.action.is_some() && screen.error.is_none() => {
                        break Ok(true)
                    }
                    _ => {}
                }
            }
        }
    };
    ratatui::restore();
    result
}

pub fn run_serve(title: &str, url: &str, log: Arc<Mutex<VecDeque<String>>>) -> Result<(), String> {
    let mut terminal = ratatui::init();
    let result = loop {
        let lines: Vec<String> = log
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .cloned()
            .collect();
        let status = serve_status(&lines, url);
        let body = if lines.is_empty() {
            "journal en attente".to_string()
        } else {
            lines.into_iter().collect::<Vec<_>>().join("\n")
        };
        let screen = Screen {
            title: title.into(),
            body,
            action: Some("Esc : arrêter".into()),
            error: status.error,
        };
        draw(&mut terminal, &screen, Some(url), status.ready.as_deref());
        if event::poll(Duration::from_millis(200)).map_err(|e| e.to_string())? {
            if let Event::Key(key) = event::read().map_err(|e| e.to_string())? {
                if key.kind == KeyEventKind::Press
                    && matches!(key.code, KeyCode::Esc | KeyCode::Char('q'))
                {
                    break Ok(());
                }
            }
        }
    };
    ratatui::restore();
    result
}

struct ServeStatus {
    ready: Option<String>,
    error: Option<String>,
}

fn serve_status(lines: &[String], url: &str) -> ServeStatus {
    let joined = lines.join("\n").to_ascii_lowercase();
    if joined.contains("error") || joined.contains("erreur") {
        let last = lines
            .iter()
            .rev()
            .find(|l| {
                let n = l.to_ascii_lowercase();
                n.contains("error") || n.contains("erreur")
            })
            .cloned();
        return ServeStatus {
            ready: Some(format!("chargement en erreur · {url}")),
            error: last,
        };
    }
    if joined.contains("listening") || joined.contains("ready") || joined.contains("started") {
        ServeStatus {
            ready: Some(format!("prêt · {url}")),
            error: None,
        }
    } else {
        ServeStatus {
            ready: Some(format!("démarrage · {url}")),
            error: None,
        }
    }
}

fn draw(terminal: &mut DefaultTerminal, screen: &Screen, url: Option<&str>, ready: Option<&str>) {
    let _ = terminal.draw(|frame| render(frame, screen, url, ready));
}

fn render(frame: &mut Frame, screen: &Screen, url: Option<&str>, ready: Option<&str>) {
    let area = frame.area();
    let chunks = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(6),
        Constraint::Length(if screen.error.is_some() { 5 } else { 0 }),
        Constraint::Length(3),
    ])
    .split(area);

    let header = match (url, ready) {
        (Some(url), Some(ready)) => format!("{}  {ready}  {url}", screen.title),
        _ => screen.title.clone(),
    };
    frame.render_widget(
        Paragraph::new(header).block(
            Block::default()
                .title(" Rbitnet ")
                .borders(Borders::ALL),
        ),
        chunks[0],
    );
    frame.render_widget(
        Paragraph::new(Text::from(screen.body.clone()))
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .title(" détail ")
                    .borders(Borders::ALL),
            ),
        chunks[1],
    );
    if let Some(err) = &screen.error {
        frame.render_widget(
            Paragraph::new(Line::from(err.clone()).style(
                Style::default()
                    .fg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            ))
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .title(" erreur ")
                    .borders(Borders::ALL),
            ),
            chunks[2],
        );
    }
    let action = screen.action.clone().unwrap_or_else(|| "Esc : quitter".into());
    frame.render_widget(
        Paragraph::new(action).block(Block::default().title(" action ").borders(Borders::ALL)),
        chunks[3],
    );
}

/// Line buffer used as the tracing writer while `serve` owns the terminal.
#[derive(Clone)]
pub struct LogLines(pub Arc<Mutex<VecDeque<String>>>);

pub struct LogWriter {
    lines: Arc<Mutex<VecDeque<String>>>,
    pending: Vec<u8>,
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogLines {
    type Writer = LogWriter;

    fn make_writer(&'a self) -> Self::Writer {
        LogWriter {
            lines: Arc::clone(&self.0),
            pending: Vec::new(),
        }
    }
}

impl Write for LogWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.pending.extend_from_slice(buf);
        while let Some(pos) = self.pending.iter().position(|b| *b == b'\n') {
            let line = String::from_utf8_lossy(&self.pending[..=pos])
                .trim_end_matches(['\r', '\n'])
                .to_string();
            self.pending.drain(..=pos);
            if line.is_empty() {
                continue;
            }
            let mut guard = self.lines.lock().unwrap_or_else(|p| p.into_inner());
            guard.push_back(line);
            while guard.len() > 200 {
                guard.pop_front();
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn rendered(screen: &Screen) -> String {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| render(frame, screen, None, None))
            .expect("draw");
        let buffer = terminal.backend().buffer().clone();
        let mut out = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                out.push_str(buffer[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn home_screen_lists_commands_and_next_action() {
        let text = rendered(&home_screen());
        assert!(text.contains("quickstart"), "{text}");
        assert!(text.contains("serve"), "{text}");
        assert!(text.contains("chat"), "{text}");
        assert!(text.contains("models search"), "{text}");
        assert!(text.contains("bitnet:2b"), "{text}");
    }

    #[test]
    fn quickstart_screen_shows_gguf_tokenizer_and_one_serve_action() {
        let screen = quickstart_screen(
            "bitnet:2b",
            Path::new("models/ggml-model-i2_s.gguf"),
            Some(Path::new("models/tokenizer.json")),
            "127.0.0.1:8080",
        );
        let text = rendered(&screen);
        assert!(text.contains("ggml-model-i2_s.gguf"), "{text}");
        assert!(text.contains("tokenizer.json"), "{text}");
        assert!(text.contains("rbitnet serve"), "{text}");
        assert!(!text.contains("curl"), "{text}");
        assert!(!text.to_ascii_lowercase().contains("powershell"), "{text}");
        assert!(screen.error.is_none());
    }

    #[test]
    fn missing_tokenizer_stays_in_the_error_panel() {
        let screen = quickstart_screen(
            "bitnet:2b",
            Path::new("models/ggml-model-i2_s.gguf"),
            None,
            "127.0.0.1:8080",
        );
        let text = rendered(&screen);
        assert!(text.contains("tokenizer manquant"), "{text}");
        assert!(screen.action.is_none());
    }

    #[test]
    fn serve_log_keeps_the_url_and_names_a_load_error() {
        let status = serve_status(
            &[
                "listening on 127.0.0.1:8080".into(),
                "error: tokenizer missing".into(),
            ],
            "http://127.0.0.1:8080",
        );
        let ready = status.ready.expect("status line");
        assert!(ready.contains("http://127.0.0.1:8080"), "{ready}");
        assert!(ready.contains("erreur"), "{ready}");
        assert!(status.error.unwrap().contains("tokenizer missing"));
    }

    #[test]
    fn command_error_is_a_panel() {
        let screen = Screen::error(
            "unknown model ref 'not-a-model:zzz'. Known tags: bitnet:2b, tinyllama:q4",
        );
        let text = rendered(&screen);
        assert!(text.contains("not-a-model:zzz"), "{text}");
        assert!(text.contains("bitnet:2b"), "{text}");
    }
}
