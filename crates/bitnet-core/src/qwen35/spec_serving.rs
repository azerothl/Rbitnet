//! Optional real draft model; target sampling remains the source of emitted IDs.
use super::*;
use crate::timings::GenerationFinishReason;
use std::path::PathBuf;

pub(super) struct DraftRuntime {
    pub(super) runtime: Box<Qwen35Runtime>,
    pub(super) depth: usize,
}

struct DraftOptions {
    path: PathBuf,
    depth: usize,
}

impl DraftOptions {
    fn parse(
        enabled: Option<&str>,
        path: Option<&str>,
        depth: Option<&str>,
    ) -> Result<Option<Self>> {
        match enabled {
            None | Some("0") => return Ok(None),
            Some("1") => {}
            _ => {
                return Err(BitNetError::Inference(
                    "RBITNET_QWEN_SPECULATIVE must be 0 or 1".into(),
                ))
            }
        }
        let path = path.filter(|s| !s.trim().is_empty()).ok_or_else(|| {
            BitNetError::Inference(
                "RBITNET_QWEN_DRAFT_GGUF is required for speculative Qwen".into(),
            )
        })?;
        let depth = depth
            .unwrap_or("4")
            .parse::<usize>()
            .ok()
            .filter(|&n| (1..=8).contains(&n))
            .ok_or_else(|| BitNetError::Inference("RBITNET_QWEN_SPEC_DEPTH must be 1..8".into()))?;
        Ok(Some(Self {
            path: PathBuf::from(path),
            depth,
        }))
    }
    fn from_env() -> Result<Option<Self>> {
        let enabled = std::env::var("RBITNET_QWEN_SPECULATIVE").ok();
        let path = std::env::var("RBITNET_QWEN_DRAFT_GGUF").ok();
        let depth = std::env::var("RBITNET_QWEN_SPEC_DEPTH").ok();
        Self::parse(enabled.as_deref(), path.as_deref(), depth.as_deref())
    }
}

impl Qwen35Runtime {
    pub(super) fn load_with_optional_draft(
        archive: Arc<GgufArchive>,
        tokenizer_path: &Path,
        backend_kind: BackendKind,
    ) -> Result<Self> {
        let options = DraftOptions::from_env()?;
        if options.is_some() {
            if backend_kind != BackendKind::Cuda {
                return Err(BitNetError::Inference("RBITNET_QWEN_SPECULATIVE requires CUDA, two dense Qwen models and the full resident pipeline".into()));
            }
            if prefix::enabled() {
                return Err(BitNetError::Inference("RBITNET_QWEN_SPECULATIVE with RBITNET_PREFIX_KV is not yet validated; disable prefix reuse for this experiment".into()));
            }
            if Qwen35Config::from_gguf(archive.as_ref())?.n_expert != 0 {
                return Err(BitNetError::Inference(
                    "RBITNET_QWEN_SPECULATIVE supports dense Qwen only".into(),
                ));
            }
        }
        let mut target = Self::load_without_draft(archive, tokenizer_path, backend_kind)?;
        if let Some(options) = options {
            let archive = Arc::new(GgufArchive::mmap_path(&options.path)?);
            if Qwen35Config::from_gguf(archive.as_ref())?.n_expert != 0 {
                return Err(BitNetError::Inference(
                    "RBITNET_QWEN_DRAFT_GGUF must be a dense Qwen checkpoint".into(),
                ));
            }
            let mut draft = Self::load_without_draft(archive, tokenizer_path, backend_kind)?;
            target.spec_check_draft(&draft)?;
            target.gpu_full.as_mut().unwrap().spec_configure()?;
            draft.gpu_full.as_mut().unwrap().spec_configure()?;
            target.draft = Some(DraftRuntime {
                runtime: Box::new(draft),
                depth: options.depth,
            });
        }
        Ok(target)
    }

    pub(crate) fn speculative_enabled(&self) -> bool {
        self.draft.is_some()
    }
    pub(crate) fn draft_resident_weights_bytes(&self) -> usize {
        self.draft
            .as_ref()
            .map_or(0, |draft| draft.runtime.resident_weights_bytes())
    }
    pub(crate) fn draft_summary(&self) -> Option<String> {
        self.draft.as_ref().map(|draft| format!("; optional Qwen draft depth {}, {} MiB additional resident weights; prefix reuse disabled; target-only sampling",draft.depth,draft.runtime.resident_weights_bytes()/(1024*1024)))
    }

    pub(super) fn generate_with_owned_draft(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        sampling: SamplingOptions,
        events: Option<&mut (dyn FnMut(crate::stream::StreamEvent) -> Result<()> + Send)>,
    ) -> Result<(String, PhaseTimings)> {
        let mut draft = self
            .draft
            .take()
            .expect("draft route selected only with owned draft");
        let result = (|| {
            let encoding = Instant::now();
            let ids = self.tokenizer.encode_ids(prompt, true)?;
            let encode_ms = encoding.elapsed().as_millis() as u64;
            let run = self.spec_generate_ids_with_draft(
                &mut draft.runtime,
                &ids,
                max_tokens,
                draft.depth,
                sampling,
                events,
            )?;
            crate::perf::record_speculative(
                run.proposed as u32,
                (run.proposed + run.rounds) as u32,
                run.accepted as u32,
            );
            tracing::debug!(
                rounds = run.rounds,
                proposed = run.proposed,
                accepted = run.accepted,
                replayed = run.replayed,
                "Qwen speculative request complete"
            );
            let text = self.tokenizer.decode_ids(&run.ids, true)?;
            Ok((
                text,
                PhaseTimings {
                    encode_ms,
                    prefill_ms: run.prefill_ms,
                    decode_ms: run.decode_ms,
                    prompt_tokens: ids.len() as u32,
                    completion_tokens: run.ids.len() as u32,
                    finish_reason: if run.eos {
                        GenerationFinishReason::Stop
                    } else {
                        GenerationFinishReason::Length
                    },
                },
            ))
        })();
        // Keep both owners available after EOS, cancellation, callback failure or
        // an error. The next prompt begins at position zero and resets Native state.
        self.draft = Some(draft);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn draft_options_fail_closed_and_disabled_options_do_not_load_a_model() {
        assert!(DraftOptions::parse(None, None, Some("invalid"))
            .unwrap()
            .is_none());
        assert!(
            DraftOptions::parse(Some("0"), Some("missing.gguf"), Some("0"))
                .unwrap()
                .is_none()
        );
        assert!(DraftOptions::parse(Some("yes"), Some("draft.gguf"), None).is_err());
        for path in [None, Some(""), Some("  ")] {
            assert!(DraftOptions::parse(Some("1"), path, None).is_err());
        }
        for depth in ["0", "9", "invalid", "-1"] {
            assert!(DraftOptions::parse(Some("1"), Some("draft.gguf"), Some(depth)).is_err());
        }
        let options = DraftOptions::parse(Some("1"), Some("draft.gguf"), None)
            .unwrap()
            .unwrap();
        assert_eq!(options.depth, 4);
        for depth in ["1", "4", "8"] {
            assert_eq!(
                DraftOptions::parse(Some("1"), Some("draft.gguf"), Some(depth))
                    .unwrap()
                    .unwrap()
                    .depth,
                depth.parse::<usize>().unwrap()
            );
        }
    }
}
