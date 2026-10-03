//! Retain only a suffix that may become a stop sequence in a later UTF-8 delta.
pub(crate) struct StreamStop {
    stops: Vec<String>,
    pending: String,
    stopped: bool,
}
impl StreamStop {
    pub fn new(stop: Option<&crate::StopSequence>) -> Self {
        Self {
            stops: stop
                .map(|s| {
                    s.as_strings()
                        .into_iter()
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default(),
            pending: String::new(),
            stopped: false,
        }
    }
    pub fn push(&mut self, text: &str) -> String {
        if self.stopped {
            return String::new();
        }
        self.pending.push_str(text);
        if let Some(cut) = self.stops.iter().filter_map(|s| self.pending.find(s)).min() {
            self.stopped = true;
            let result = self.pending[..cut].to_owned();
            self.pending.clear();
            return result;
        }
        let mut keep = 0;
        for stop in &self.stops {
            for (end, _) in stop.char_indices().skip(1) {
                if end > keep && self.pending.ends_with(&stop[..end]) {
                    keep = end;
                }
            }
        }
        let release = self.pending.len() - keep;
        self.pending.drain(..release).collect()
    }
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_utf8_partition_matches_the_nonstreaming_stop_result() {
        for text in [
            "avant été après",
            "avant 🙂 après",
            "avant été🙂 après",
            "fin incomplète ét",
            "pas d'arrêt",
        ] {
            let stop = crate::StopSequence::Many(vec!["été".into(), "🙂".into(), "".into()]);
            let expected = crate::apply_stop_sequences(text.to_owned(), Some(&stop));
            let boundaries: Vec<_> = text
                .char_indices()
                .map(|(i, _)| i)
                .chain(std::iter::once(text.len()))
                .collect();
            for &a in &boundaries {
                for &b in &boundaries {
                    if b < a {
                        continue;
                    }
                    let mut filter = StreamStop::new(Some(&stop));
                    let result = [&text[..a], &text[a..b], &text[b..]]
                        .into_iter()
                        .map(|part| filter.push(part))
                        .collect::<String>()
                        + &filter.finish();
                    assert_eq!(result, expected, "boundaries {a}/{b}: {text}");
                }
            }
        }
    }
    #[test]
    fn pending_suffix_is_bounded_and_released_when_no_stop_completes() {
        let stop = crate::StopSequence::One("résumé".into());
        let mut filter = StreamStop::new(Some(&stop));
        assert_eq!(filter.push("a long response ré"), "a long response ");
        assert_eq!(filter.pending, "ré");
        assert_eq!(filter.push("sultat"), "résultat");
        assert_eq!(filter.push("résu"), "");
        assert_eq!(filter.finish(), "résu");
    }
}
