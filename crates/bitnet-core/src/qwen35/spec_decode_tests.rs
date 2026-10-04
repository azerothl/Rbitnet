//! Actual draft coupling to an independently run serial target sampler.
use super::spec_decode::DraftRun;
use super::*;

fn serial_ids(
    rt: &mut Qwen35Runtime,
    ids: &[u32],
    limit: u32,
    options: SamplingOptions,
) -> Vec<u32> {
    let archive = Arc::clone(&rt.archive);
    let greedy = options.device_greedy_eligible();
    let capacity = rt
        .gpu_prefill_capacity()
        .min(rt.prefill_chunk_tokens.max(1));
    let mut logits = Vec::new();
    let mut next = None;
    for (chunk, part) in ids.chunks(capacity).enumerate() {
        let position = chunk * capacity;
        let output = position + part.len() == ids.len();
        (logits, next) = if part.len() > 1 {
            rt.forward_block(part, position, &archive, output, greedy)
                .unwrap()
        } else {
            rt.forward_inner(part[0], position, &archive, output, greedy)
                .unwrap()
        };
    }
    let mut rng = seeded_rng(options.seed);
    let mut generated = Vec::new();
    let eos = rt.tokenizer.eos_token_ids();
    for step in 0..limit {
        let id = next
            .take()
            .unwrap_or_else(|| sample_token(&logits, &options, &generated, &mut rng));
        if eos.contains(&id) {
            break;
        }
        generated.push(id);
        if step + 1 < limit {
            (logits, next) = rt
                .forward_inner(id, ids.len() + step as usize, &archive, true, greedy)
                .unwrap();
        }
    }
    generated
}
fn setup(archive: Arc<GgufArchive>, tokenizer: &Path) -> Qwen35Runtime {
    let mut model = Qwen35Runtime::load(archive, tokenizer, BackendKind::Cuda).unwrap();
    model.gpu_full.as_mut().unwrap().spec_configure().unwrap();
    model
}
fn describe(run: &DraftRun, fixture: &str, depth: usize, options: SamplingOptions) {
    eprintln!(
        "QWEN_SPEC_DECODE {}",
        serde_json::json!({"fixture":fixture,"depth":depth,"temperature":options.temperature,"top_p":options.top_p,"seed":options.seed,
        "frequency_penalty":options.frequency_penalty,"presence_penalty":options.presence_penalty,"structured_json":options.structured_json,"ids":run.ids,
        "rounds":run.rounds,"proposed":run.proposed,"accepted":run.accepted,"replayed_tokens":run.replayed,"prefill_ms":run.prefill_ms,"decode_ms":run.decode_ms})
    );
}
#[test]
fn optional_actual_qwen_draft_seed_penalties_stream_cancel_and_rejection_exact() {
    if std::env::var("RBITNET_QWEN_DRAFT_TEST").as_deref() != Ok("1") {
        return;
    }
    std::env::set_var("RBITNET_PREFIX_KV", "0");
    std::env::remove_var("RBITNET_QWEN_ORDERED_BLOCK_TEST");
    std::env::set_var("RBITNET_CUDA_QWEN_PREFILL", "1");
    std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", "0");
    crate::cancel::clear_inference_cancel();
    let tokenizer = std::env::var("RBITNET_QWEN_TEST_TOKENIZER").unwrap();
    let tokenizer = Path::new(&tokenizer);
    let target = Arc::new(
        GgufArchive::mmap_path(Path::new(&std::env::var("RBITNET_QWEN_TEST_GGUF").unwrap()))
            .unwrap(),
    );
    let draft = Arc::new(
        GgufArchive::mmap_path(Path::new(
            &std::env::var("RBITNET_QWEN_DRAFT_GGUF").unwrap(),
        ))
        .unwrap(),
    );
    let mut checked = 0;
    let mut total_replayed = 0;
    let mut saw_eos = false;
    for graphs in ["0", "1"] {
        std::env::set_var("RBITNET_CUDA_QWEN_FULL_GRAPH", graphs);
        let mut reference = setup(Arc::clone(&target), tokenizer);
        let mut coupled = setup(Arc::clone(&target), tokenizer);
        let mut proposal = setup(Arc::clone(&draft), tokenizer);
        coupled.spec_check_draft(&proposal).unwrap();
        let vocab = proposal.cfg.n_vocab;
        proposal.cfg.n_vocab -= 1;
        assert!(coupled.spec_check_draft(&proposal).is_err());
        proposal.cfg.n_vocab = vocab;
        // A byte/ASCII mask cannot constrain this model's subword vocabulary.
        // Reject before prefill or checkpoint mutation, with both flag paths.
        let saved_structured = std::env::var_os("RBITNET_STRUCTURED_OUTPUT");
        std::env::remove_var("RBITNET_STRUCTURED_OUTPUT");
        let before_refusal = crate::perf::snapshot();
        let mut structured = SamplingOptions::from_temperature(0.0);
        structured.structured_json = true;
        assert!(coupled
            .spec_generate_ids_with_draft(&mut proposal, &[1], 1, 1, structured, None)
            .is_err());
        for value in ["json", "tool", "tool-call", "tool_call"] {
            std::env::set_var("RBITNET_STRUCTURED_OUTPUT", value);
            assert!(coupled
                .spec_generate_ids_with_draft(
                    &mut proposal,
                    &[1],
                    1,
                    1,
                    SamplingOptions::from_temperature(0.0),
                    None
                )
                .is_err());
        }
        match saved_structured {
            Some(value) => std::env::set_var("RBITNET_STRUCTURED_OUTPUT", value),
            None => std::env::remove_var("RBITNET_STRUCTURED_OUTPUT"),
        }
        let after_refusal = crate::perf::snapshot();
        assert_eq!(
            before_refusal.gpu_qwen_full_tokens,
            after_refusal.gpu_qwen_full_tokens
        );
        assert_eq!(
            before_refusal.gpu_prefill_blocks,
            after_refusal.gpu_prefill_blocks
        );
        let cases=[("story","Continue une histoire détaillée : Un robot entre dans une bibliothèque ancienne et découvre",32u32),
            ("code","Écris une fonction Python qui calcule la moyenne en ignorant les valeurs None, avec un exemple.",32),
            ("short-capital","Quelle est la capitale de la France ? Réponds en un mot.",16),
            ("context-tail","Poursuis le texte suivant en conservant les détails :",32)];
        for (fixture, question, limit) in cases {
            let notes = if fixture == "context-tail" {
                "La bibliothèque possède un jardin, des livres de science et une horloge bleue.\n"
                    .repeat(24)
            } else {
                String::new()
            };
            let prompt=format!("<|im_start|>system\n{notes}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
            let ids = coupled.tokenizer.encode_ids(&prompt, true).unwrap();
            assert!(ids.len() + limit as usize <= coupled.cfg.max_seq);
            for options in [
                SamplingOptions::from_temperature(0.0),
                SamplingOptions {
                    temperature: 0.8,
                    top_p: Some(0.9),
                    seed: Some(42),
                    frequency_penalty: 0.4,
                    presence_penalty: 0.2,
                    structured_json: false,
                },
                SamplingOptions {
                    temperature: 0.0,
                    seed: Some(13),
                    frequency_penalty: 0.6,
                    presence_penalty: 0.3,
                    ..SamplingOptions::default()
                },
                SamplingOptions {
                    temperature: 0.7,
                    top_p: Some(0.95),
                    seed: Some(95),
                    structured_json: false,
                    ..SamplingOptions::default()
                },
            ] {
                let expected = serial_ids(&mut reference, &ids, limit, options);
                saw_eos |= expected.len() < limit as usize;
                for depth in [1usize, 4, 8] {
                    let mut deltas = String::new();
                    let mut callback = |event| -> Result<()> {
                        if let crate::stream::StreamEvent::Delta { text } = event {
                            deltas.push_str(&text);
                        }
                        Ok(())
                    };
                    let actual = coupled
                        .spec_generate_ids_with_draft(
                            &mut proposal,
                            &ids,
                            limit,
                            depth,
                            options,
                            Some(&mut callback),
                        )
                        .unwrap();
                    assert_eq!(
                        actual.ids, expected,
                        "graphs={graphs} fixture={fixture} depth={depth} seed/penalty target IDs"
                    );
                    assert_eq!(
                        deltas,
                        coupled.tokenizer.decode_ids(&expected, true).unwrap(),
                        "stream concatenation"
                    );
                    assert!(actual.accepted <= actual.proposed);
                    total_replayed += actual.replayed;
                    checked += 1;
                    describe(&actual, fixture, depth, options);
                    // The unchanged production serial generator is a second text
                    // control at one depth, including EOS and UTF-8 decoding.
                    if depth == 1 {
                        let text = reference
                            .generate_inner(&prompt, limit, options, None)
                            .unwrap()
                            .0;
                        assert_eq!(
                            text,
                            coupled.tokenizer.decode_ids(&actual.ids, true).unwrap(),
                            "production serial text"
                        );
                    }
                }
            }
        }
        let prompt="<|im_start|>user\nÉcris une histoire sur un robot qui traverse une ville.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let ids = coupled.tokenizer.encode_ids(prompt, true).unwrap();
        let options = SamplingOptions {
            seed: Some(51),
            temperature: 0.7,
            top_p: Some(0.95),
            ..SamplingOptions::default()
        };
        let expected = serial_ids(&mut reference, &ids, 32, options);
        let mut deltas = 0;
        let mut callback = |event| -> Result<()> {
            if matches!(event, crate::stream::StreamEvent::Delta { .. }) {
                deltas += 1;
                if deltas == 3 {
                    crate::cancel::request_inference_cancel();
                }
            }
            Ok(())
        };
        let cancelled = coupled.spec_generate_ids_with_draft(
            &mut proposal,
            &ids,
            32,
            8,
            options,
            Some(&mut callback),
        );
        assert!(cancelled.is_err() && deltas == 3);
        crate::cancel::clear_inference_cancel();
        let resumed = coupled
            .spec_generate_ids_with_draft(&mut proposal, &ids, 32, 8, options, None)
            .unwrap();
        assert_eq!(resumed.ids, expected, "cancel/reset resumed seed");
        let mut callback = |_: crate::stream::StreamEvent| -> Result<()> {
            Err(BitNetError::Inference(
                "stream channel closed: test disconnect".into(),
            ))
        };
        assert!(coupled
            .spec_generate_ids_with_draft(&mut proposal, &ids, 32, 4, options, Some(&mut callback))
            .is_err());
        let resumed = coupled
            .spec_generate_ids_with_draft(&mut proposal, &ids, 32, 4, options, None)
            .unwrap();
        assert_eq!(resumed.ids, expected, "callback error/reset resumed seed");
        for limit in [0u32, 1, 2, 7, 8, 9] {
            let before = crate::perf::snapshot();
            let actual = coupled
                .spec_generate_ids_with_draft(&mut proposal, &ids, limit, 8, options, None)
                .unwrap();
            if limit == 0 {
                let after = crate::perf::snapshot();
                assert_eq!(actual.prefill_ms, 0);
                assert_eq!(actual.decode_ms, 0);
                assert_eq!(actual.rounds, 0);
                assert_eq!(after.gpu_qwen_full_tokens, before.gpu_qwen_full_tokens);
                assert_eq!(after.gpu_prefill_blocks, before.gpu_prefill_blocks);
            }
            assert_eq!(
                actual.ids,
                serial_ids(&mut reference, &ids, limit, options),
                "max-output tail {limit}"
            );
        }
        assert!(coupled
            .spec_generate_ids_with_draft(&mut proposal, &ids, 32, 0, options, None)
            .is_err());
        assert!(coupled
            .spec_generate_ids_with_draft(&mut proposal, &ids, 32, 9, options, None)
            .is_err());
        assert!(coupled
            .spec_generate_ids_with_draft(
                &mut proposal,
                &vec![ids[0]; coupled.cfg.max_seq],
                1,
                4,
                options,
                None
            )
            .is_err());
        // Identical draft/target gives an independent all-acceptance branch;
        // rejections must also have been exercised naturally with the 0.8B draft.
        drop(proposal);
        let mut same = setup(Arc::clone(&target), tokenizer);
        coupled.spec_check_draft(&same).unwrap();
        let options = SamplingOptions::from_temperature(0.0);
        let actual = coupled
            .spec_generate_ids_with_draft(&mut same, &ids, 32, 8, options, None)
            .unwrap();
        assert_eq!(actual.ids, serial_ids(&mut reference, &ids, 32, options));
        assert_eq!(
            actual.replayed, 0,
            "identical deterministic draft accepts every used proposal"
        );
        describe(
            &actual,
            "identical-model-all-acceptance-control",
            8,
            options,
        );
    }
    assert!(
        total_replayed > 0,
        "at least one actual rejected draft tail/replay is required"
    );
    assert!(saw_eos, "at least one actual target EOS must be exercised");
    eprintln!("QWEN_SPEC_DECODE_DONE exact_runs={checked} replayed_tokens={total_replayed}; sampled target IDs, penalties, stream, EOS/limits and cancel/reset checked; no throughput claim");
}
