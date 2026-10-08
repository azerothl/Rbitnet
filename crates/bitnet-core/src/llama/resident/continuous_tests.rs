//! Actual independent serial Native reference, including per-request RNG.
use super::continuous::{ContinuousLlama, WaveOutput};
use super::*;
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::sampling::{sample_token, SamplingOptions};
use crate::stream::{StreamCallback, StreamEvent};
use crate::timings::GenerationFinishReason;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
#[derive(Clone)]
struct Job {
    id: u64,
    prompt: String,
    maximum: u32,
    sampling: SamplingOptions,
}
fn serial(
    model: &LlamaModel,
    tokenizer: &LoadedPromptTokenizer,
    job: &Job,
) -> (Vec<u32>, String, GenerationFinishReason) {
    crate::clear_inference_cancel();
    let ids = tokenizer
        .encode_ids(&job.prompt, crate::llama::llama_encode_add_special_tokens())
        .unwrap();
    if job.maximum == 0 {
        return (Vec::new(), String::new(), GenerationFinishReason::Length);
    }
    let mut r = Resident::new_with_pages(model, None, None).unwrap();
    let greedy = job.sampling.device_greedy_eligible();
    let (mut logits, mut next) = r.prefill(model, &ids, 0, greedy).unwrap();
    let mut rng = StdRng::seed_from_u64(job.sampling.seed.unwrap());
    let eos = tokenizer.eos_token_ids();
    let mut generated = Vec::new();
    let mut reason = GenerationFinishReason::Length;
    for offset in 0..job.maximum {
        if !greedy {
            next = sample_token(&logits, &job.sampling, &generated, &mut rng);
        }
        if eos.contains(&next) {
            reason = GenerationFinishReason::Stop;
            break;
        }
        generated.push(next);
        if offset + 1 < job.maximum {
            if greedy {
                next = r
                    .greedy(model, next, ids.len() + offset as usize, true)
                    .unwrap();
            } else {
                logits = r
                    .forward(model, next, ids.len() + offset as usize, true)
                    .unwrap();
            }
        }
    }
    let text = tokenizer.decode_ids(&generated, true).unwrap();
    (generated, text, reason)
}
fn jobs() -> Vec<Job> {
    let prompts=[
        "<|start_header_id|>user<|end_header_id|>\n\nQuelle est la capitale de la France ? Réponds uniquement Paris.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".to_owned(),
        format!("{}Écris un récit sur un robot qui explore une bibliothèque.\n", "Note : les villes ont des jardins, des musées et des bibliothèques.\n".repeat(18)),
        format!("{}Return a Python checked_sum function with an example.\n","fn add(a: i32, b: i32) -> i32 { a + b }\n".repeat(11)),
        "Répète les mots suivants : été, café, résumé, 🙂. Puis explique leur sens.".to_owned(),
    ];
    (0..13)
        .map(|i| Job {
            id: i,
            prompt: prompts[i as usize % 4].clone(),
            maximum: [33, 17, 65, 0][i as usize % 4],
            sampling: SamplingOptions {
                temperature: if i % 3 == 1 { 0.7 } else { 0. },
                top_p: if i % 3 == 1 { Some(0.9) } else { None },
                seed: Some(42 + i),
                frequency_penalty: if i % 3 == 2 { 0.2 } else { 0. },
                presence_penalty: if i % 3 == 2 { 0.1 } else { 0. },
                structured_json: false,
            },
        })
        .collect()
}
fn load() -> (Arc<LlamaModel>, Arc<LoadedPromptTokenizer>) {
    let archive = Arc::new(
        crate::gguf::GgufArchive::mmap_path(std::path::Path::new(
            &std::env::var("RBITNET_TEST_GGUF").unwrap(),
        ))
        .unwrap(),
    );
    let tokenizer = Arc::new(
        LoadedPromptTokenizer::from_path_for_gguf(
            std::path::Path::new(&std::env::var("RBITNET_TOKENIZER").unwrap()),
            &archive,
        )
        .unwrap(),
    );
    let model = Arc::new(
        LlamaModel::from_gguf_arc_for_backend(archive, crate::backend::BackendKind::Cuda).unwrap(),
    );
    (model, tokenizer)
}
fn validate(
    job: &Job,
    actual: WaveOutput,
    expected: &(Vec<u32>, String, GenerationFinishReason),
    events: &[StreamEvent],
) {
    assert_eq!(
        &actual.ids, &expected.0,
        "request {} changed sampled IDs",
        job.id
    );
    assert_eq!(actual.text, expected.1);
    assert_eq!(actual.phases.finish_reason, expected.2);
    assert_eq!(actual.phases.completion_tokens as usize, actual.ids.len());
    let text: String = events
        .iter()
        .filter_map(|ev| match ev {
            StreamEvent::Delta { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, actual.text);
    let done: Vec<_> = events
        .iter()
        .filter_map(|ev| match ev {
            StreamEvent::Done(out) => Some(out),
            _ => None,
        })
        .collect();
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].text, actual.text);
    assert_eq!(done[0].stats.finish_reason, expected.2);
    assert_eq!(
        events
            .iter()
            .filter(|ev| matches!(ev, StreamEvent::FirstToken { .. }))
            .count(),
        usize::from(job.maximum > 0)
    );
    assert_eq!(
        actual.inter_token_us.len(),
        actual.ids.len().saturating_sub(1)
    );
    assert!(actual.queued_ms >= 0. && actual.total_wall_ms >= actual.queued_ms);
    assert_eq!(actual.ttft_wall_ms.is_some(), !actual.ids.is_empty());
}
#[test]
fn optional_actual_llama_continuous_arrivals_departures_sampling_and_request_local_cancel_exact() {
    if std::env::var("RBITNET_LLAMA_CONTINUOUS_TEST").as_deref() != Ok("1") {
        return;
    }
    std::env::set_var("RBITNET_MAX_SEQ", "1024");
    std::env::set_var("RBITNET_CUDA_PREFILL", "1");
    std::env::set_var("RBITNET_CUDA_PREFILL_TOKENS", "128");
    std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", "0");
    std::env::set_var("RBITNET_CUDA_SPLIT_KV", "0");
    std::env::set_var("RBITNET_PREFIX_KV", "0");
    std::env::set_var("RBITNET_CUDA_KV_FORMAT", "f32");
    std::env::remove_var("RBITNET_CUDA_KV_PAGE_LIMIT");
    let (model, tokenizer) = load();
    let jobs = jobs();
    let mut cases = 0;
    let mut genuine_eos = 0;
    for graph in ["0", "1"] {
        std::env::set_var("RBITNET_CUDA_RESIDENT_GRAPH", graph);
        let expected: BTreeMap<_, _> = jobs
            .iter()
            .map(|j| (j.id, serial(&model, &tokenizer, j)))
            .collect();
        for pages in [None, Some(256)] {
            for maximum in [1usize, 4, 8] {
                for ordering in [0, 1] {
                    let baseline = crate::backend::cuda_managed_memory_stats().unwrap();
                    let mut driver = ContinuousLlama::new(
                        Arc::clone(&model),
                        Arc::clone(&tokenizer),
                        maximum,
                        16,
                        256,
                        pages,
                        ordering,
                    )
                    .unwrap();
                    let workspace = crate::backend::cuda_managed_memory_stats()
                        .unwrap()
                        .categories[5];
                    let events: Arc<Mutex<BTreeMap<u64, Vec<StreamEvent>>>> =
                        Arc::new(Mutex::new(BTreeMap::new()));
                    let mut sent = 0;
                    let mut finished = 0;
                    let mut ticks = Vec::new();
                    for tick in 0..10000 {
                        if tick == 0 || tick % 7 == 0 {
                            let count = if tick == 0 { maximum } else { 1 };
                            for _ in 0..count {
                                let Some(j) = jobs.get(sent) else { break };
                                let id = j.id;
                                let capture = Arc::clone(&events);
                                let callback: StreamCallback = Box::new(move |ev| {
                                    capture.lock().unwrap().entry(id).or_default().push(ev);
                                    Ok(())
                                });
                                driver
                                    .submit(
                                        id,
                                        &j.prompt,
                                        j.maximum,
                                        j.sampling,
                                        Arc::new(AtomicBool::new(false)),
                                        Some(callback),
                                    )
                                    .unwrap();
                                sent += 1;
                            }
                        }
                        let wave = driver.tick().unwrap();
                        assert!(
                            wave.decode_rows <= maximum
                                && wave.decode_rows + wave.prefill_tokens <= 256
                        );
                        ticks.push(serde_json::json!({"decode_rows":wave.decode_rows,"prefill_tokens":wave.prefill_tokens,"admitted":wave.admitted,"retired":wave.retired}));
                        for (id, result) in driver.take_completed() {
                            let output = result.unwrap();
                            genuine_eos += usize::from(
                                output.phases.finish_reason == GenerationFinishReason::Stop,
                            );
                            validate(
                                &jobs[id as usize],
                                output,
                                &expected[&id],
                                events.lock().unwrap().get(&id).unwrap(),
                            );
                            finished += 1;
                        }
                        assert_eq!(
                            crate::backend::cuda_managed_memory_stats()
                                .unwrap()
                                .categories[5],
                            workspace,
                            "warm scratch must not grow"
                        );
                        if finished == jobs.len() {
                            break;
                        }
                        assert!(
                            tick + 1 < 10000,
                            "continuous scheduler did not retire every request"
                        );
                    }
                    assert_eq!(sent, 13);
                    assert_eq!(finished, 13);
                    assert!(driver.is_idle());
                    let stats = driver.stats().unwrap();
                    assert!(stats[0] > 0 && stats[1] > 0);
                    // Once all KV owners are gone, new owners must still be admitted to
                    // the same workspace without stale-pointer use or scratch growth.
                    let j = &jobs[0];
                    driver
                        .submit(
                            100,
                            &j.prompt,
                            j.maximum,
                            j.sampling,
                            Arc::new(AtomicBool::new(false)),
                            None,
                        )
                        .unwrap();
                    for _ in 0..1000 {
                        driver.tick().unwrap();
                        if driver.is_idle() {
                            break;
                        }
                    }
                    let fresh = driver.take_completed();
                    assert_eq!(fresh.len(), 1);
                    let output = fresh.into_iter().next().unwrap().1.unwrap();
                    assert_eq!(output.ids, expected[&0].0);
                    assert_eq!(output.text, expected[&0].1);
                    assert_eq!(
                        crate::backend::cuda_managed_memory_stats()
                            .unwrap()
                            .categories[5],
                        workspace
                    );
                    drop(driver);
                    let after = crate::backend::cuda_managed_memory_stats().unwrap();
                    for cat in [1, 2, 3, 5] {
                        assert_eq!(
                            after.categories[cat], baseline.categories[cat],
                            "category {cat} leaked"
                        );
                    }
                    eprintln!(
                        "LLAMA_CONTINUOUS_CASE {}",
                        serde_json::json!({"graphs":graph,"pages":pages,"capacity":maximum,"ordering":ordering,"requests":14,"ticks":ticks,"native_stats":stats})
                    );
                    cases += 1;
                }
            }
        }
    }
    assert!(
        genuine_eos > 0,
        "actual EOS must be observed, not inferred from output length"
    );
    for pages in [None, Some(256)] {
        let mut driver = ContinuousLlama::new(
            Arc::clone(&model),
            Arc::clone(&tokenizer),
            4,
            16,
            256,
            pages,
            0,
        )
        .unwrap();
        let mut controls = BTreeMap::new();
        for (id, kind) in [
            (201, "cancel-before"),
            (202, "cancel-during"),
            (203, "disconnect"),
            (204, "survivor"),
        ] {
            let flag = Arc::new(AtomicBool::new(kind == "cancel-before"));
            controls.insert(id, Arc::clone(&flag));
            let mut deltas = 0;
            let callback: StreamCallback = Box::new(move |ev| {
                if matches!(ev, StreamEvent::Delta { .. }) {
                    deltas += 1;
                    if kind == "cancel-during" && deltas == 3 {
                        flag.store(true, Ordering::Release);
                    }
                    if kind == "disconnect" && deltas == 3 {
                        return Err(BitNetError::Inference("owned client disconnected".into()));
                    }
                }
                Ok(())
            });
            let j = &jobs[1];
            driver
                .submit(
                    id,
                    &j.prompt,
                    65,
                    j.sampling,
                    Arc::clone(&controls[&id]),
                    Some(callback),
                )
                .unwrap();
        }
        let survivor = Job {
            id: 204,
            maximum: 65,
            ..jobs[1].clone()
        };
        let reference = serial(&model, &tokenizer, &survivor);
        // The current global legacy cancel flag must not cancel this private
        // request-local engine. Each cancellation above is scoped to one ID.
        crate::request_inference_cancel();
        let mut completed = BTreeMap::new();
        for _ in 0..1000 {
            driver.tick().unwrap();
            completed.extend(driver.take_completed());
            if driver.is_idle() {
                break;
            }
        }
        crate::clear_inference_cancel();
        assert_eq!(completed.len(), 4);
        for id in [201, 202, 203] {
            assert!(completed.remove(&id).unwrap().is_err());
        }
        let output = completed.remove(&204).unwrap().unwrap();
        assert_eq!(output.ids, reference.0);
        assert_eq!(output.text, reference.1);
        assert_eq!(output.phases.finish_reason, reference.2);
        assert!(!driver.cancel(999));
    }
    {
        let before = crate::backend::cuda_managed_memory_stats().unwrap().live;
        for (slots, queued, budget) in [
            (0, 1, 256),
            (9, 1, 256),
            (1, 0, 256),
            (1, 65, 256),
            (1, 1, 128),
        ] {
            assert!(ContinuousLlama::new(
                Arc::clone(&model),
                Arc::clone(&tokenizer),
                slots,
                queued,
                budget,
                None,
                0
            )
            .is_err());
        }
        std::env::set_var("RBITNET_CUDA_KV_FORMAT", "f16");
        assert!(ContinuousLlama::new(
            Arc::clone(&model),
            Arc::clone(&tokenizer),
            1,
            2,
            256,
            None,
            0
        )
        .is_err());
        std::env::set_var("RBITNET_CUDA_KV_FORMAT", "f32");
        assert_eq!(
            crate::backend::cuda_managed_memory_stats().unwrap().live,
            before
        );
        let mut driver = ContinuousLlama::new(
            Arc::clone(&model),
            Arc::clone(&tokenizer),
            1,
            2,
            256,
            None,
            0,
        )
        .unwrap();
        let memory_before_refusal = crate::backend::cuda_managed_memory_stats().unwrap().live;
        let unsupported = SamplingOptions {
            structured_json: true,
            ..jobs[0].sampling
        };
        assert!(driver
            .submit(
                899,
                "Le jardin.",
                2,
                unsupported,
                Arc::new(AtomicBool::new(false)),
                None
            )
            .is_err());
        assert!(driver.is_idle());
        assert_eq!(
            crate::backend::cuda_managed_memory_stats().unwrap().live,
            memory_before_refusal
        );
        assert!(driver
            .submit(
                900,
                "Le jardin.",
                1024,
                jobs[0].sampling,
                Arc::new(AtomicBool::new(false)),
                None
            )
            .is_err());
        let long = "Les jardins et les bibliothèques accueillent des visiteurs. ".repeat(40);
        let length = tokenizer
            .encode_ids(&long, crate::llama::llama_encode_add_special_tokens())
            .unwrap()
            .len();
        assert!(length > 128 && length + 33 <= 1024);
        let first = Arc::new(AtomicBool::new(false));
        let capture = Arc::clone(&first);
        driver
            .submit(
                205,
                &long,
                33,
                jobs[0].sampling,
                Arc::new(AtomicBool::new(false)),
                Some(Box::new(move |event| {
                    if matches!(event, StreamEvent::FirstToken { .. }) {
                        capture.store(true, Ordering::Release);
                    }
                    Ok(())
                })),
            )
            .unwrap();
        driver
            .submit(
                206,
                &jobs[0].prompt,
                jobs[0].maximum,
                jobs[0].sampling,
                Arc::new(AtomicBool::new(false)),
                None,
            )
            .unwrap();
        assert!(driver
            .submit(
                207,
                "Le jardin.",
                2,
                jobs[0].sampling,
                Arc::new(AtomicBool::new(false)),
                None
            )
            .is_err());
        let tick = driver.tick().unwrap();
        assert_eq!(tick.prefill_tokens, 128);
        assert_eq!(tick.decode_rows, 0);
        assert!(!first.load(Ordering::Acquire));
        assert!(driver.cancel(205));
        let reference = serial(&model, &tokenizer, &jobs[0]);
        let mut results = BTreeMap::new();
        for _ in 0..1000 {
            driver.tick().unwrap();
            results.extend(driver.take_completed());
            if driver.is_idle() {
                break;
            }
        }
        assert!(!first.load(Ordering::Acquire));
        assert!(results.remove(&205).unwrap().is_err());
        let output = results.remove(&206).unwrap().unwrap();
        assert_eq!(output.ids, reference.0);
        assert_eq!(output.text, reference.1);
        assert!(driver.is_idle());
        assert_eq!(results.len(), 0);
    }
    eprintln!("LLAMA_CONTINUOUS_DONE cases={cases} local_cancel_layouts=2 partial_prefill_cancel=1 admission_refusals=true genuine_eos={genuine_eos}");
    assert_eq!(cases, 24);
}

#[test]
fn optional_actual_llama_continuous_heterogeneous_mid_wave_cancel_and_admit() {
    if std::env::var("RBITNET_LLAMA_CONTINUOUS_TEST").as_deref() != Ok("1") {
        return;
    }
    std::env::set_var("RBITNET_MAX_SEQ", "1024");
    std::env::set_var("RBITNET_CUDA_PREFILL", "1");
    std::env::set_var("RBITNET_CUDA_PREFILL_TOKENS", "128");
    std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", "0");
    std::env::set_var("RBITNET_CUDA_SPLIT_KV", "0");
    std::env::set_var("RBITNET_PREFIX_KV", "0");
    std::env::set_var("RBITNET_CUDA_KV_FORMAT", "f32");
    std::env::set_var("RBITNET_CUDA_RESIDENT_GRAPH", "1");
    let (model, tokenizer) = load();
    let jobs = jobs();

    for pages in [None, Some(256)] {
        let cancel = Arc::new(AtomicBool::new(false));
        let cancellation = Arc::clone(&cancel);
        let mut cancelled_deltas = 0;
        let cancel_callback: StreamCallback = Box::new(move |event| {
            if matches!(event, StreamEvent::Delta { .. }) {
                cancelled_deltas += 1;
                if cancelled_deltas == 2 {
                    cancellation.store(true, Ordering::Release);
                }
            }
            Ok(())
        });
        let mut driver = ContinuousLlama::new(
            Arc::clone(&model),
            Arc::clone(&tokenizer),
            2,
            8,
            256,
            pages,
            0,
        )
        .unwrap();
        let cancelled_job = &jobs[2];
        let survivor = &jobs[1];
        let replacement = &jobs[0];
        let survivor_expected = serial(&model, &tokenizer, survivor);
        let replacement_expected = serial(&model, &tokenizer, replacement);
        driver
            .submit(
                301,
                &cancelled_job.prompt,
                65,
                cancelled_job.sampling,
                Arc::clone(&cancel),
                Some(cancel_callback),
            )
            .unwrap();
        driver
            .submit(
                302,
                &survivor.prompt,
                survivor.maximum,
                survivor.sampling,
                Arc::new(AtomicBool::new(false)),
                None,
            )
            .unwrap();

        let mut replacement_submitted = false;
        let mut shared_decode_after_replacement = false;
        let mut completed = BTreeMap::new();
        for tick in 0..2000 {
            let wave = driver.tick().unwrap();
            completed.extend(driver.take_completed());
            if cancel.load(Ordering::Acquire) && !replacement_submitted {
                driver
                    .submit(
                        303,
                        &replacement.prompt,
                        replacement.maximum,
                        replacement.sampling,
                        Arc::new(AtomicBool::new(false)),
                        None,
                    )
                    .unwrap();
                replacement_submitted = true;
            }
            shared_decode_after_replacement |= replacement_submitted && wave.decode_rows == 2;
            if driver.is_idle() {
                break;
            }
            assert!(tick + 1 < 2000, "heterogeneous wave did not drain");
        }
        completed.extend(driver.take_completed());
        assert!(
            replacement_submitted,
            "mid-wave cancellation never occurred"
        );
        assert!(
            shared_decode_after_replacement,
            "replacement owner never joined its survivor in a shared decode wave"
        );
        assert_eq!(completed.len(), 3);
        assert!(completed.remove(&301).unwrap().is_err());
        let survivor_output = completed.remove(&302).unwrap().unwrap();
        assert_eq!(survivor_output.ids, survivor_expected.0);
        assert_eq!(survivor_output.text, survivor_expected.1);
        assert_eq!(survivor_output.phases.finish_reason, survivor_expected.2);
        let replacement_output = completed.remove(&303).unwrap().unwrap();
        assert_eq!(replacement_output.ids, replacement_expected.0);
        assert_eq!(replacement_output.text, replacement_expected.1);
        assert_eq!(
            replacement_output.phases.finish_reason,
            replacement_expected.2
        );
        assert!(driver.is_idle());
        eprintln!(
            "LLAMA_CONTINUOUS_HETEROGENEOUS {}",
            serde_json::json!({
                "pages": pages,
                "cancelled_owner": 301,
                "survivor_owner": 302,
                "replacement_owner": 303,
                "replacement_shared_decode": shared_decode_after_replacement,
            })
        );
    }
}

#[test]
fn optional_actual_llama_live_mux_heterogeneous_deadline_cancel_and_admit_parity() {
    if std::env::var("RBITNET_LLAMA_CONTINUOUS_TEST").as_deref() != Ok("1") {
        return;
    }
    use super::controller::{BatchController, BatchOptions};
    std::env::set_var("RBITNET_MAX_SEQ", "1024");
    std::env::set_var("RBITNET_CUDA_PREFILL", "1");
    std::env::set_var("RBITNET_CUDA_PREFILL_TOKENS", "128");
    std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", "0");
    std::env::set_var("RBITNET_CUDA_SPLIT_KV", "0");
    std::env::set_var("RBITNET_PREFIX_KV", "0");
    std::env::set_var("RBITNET_CUDA_KV_FORMAT", "f32");
    std::env::set_var("RBITNET_CUDA_RESIDENT_GRAPH", "1");
    std::env::set_var("RBITNET_CUDA_CONTINUOUS", "1");
    std::env::set_var("RBITNET_CUDA_LIVE_SSE_MUX", "1");
    std::env::set_var("RBITNET_CONTINUOUS_BATCHING", "1");
    std::env::set_var("RBITNET_FUSED_MULTI_SEQ", "1");
    std::env::remove_var("RBITNET_CUDA_KV_PAGE_LIMIT");
    assert!(BatchOptions::configured(crate::backend::BackendKind::Cuda)
        .unwrap()
        .is_some());

    let (model, tokenizer) = load();
    let jobs = jobs();
    let deadline = Job {
        id: 401,
        maximum: 65,
        ..jobs[2].clone()
    };
    let survivor = Job {
        id: 402,
        maximum: 65,
        ..jobs[1].clone()
    };
    let replacement = Job {
        id: 403,
        maximum: 33,
        ..jobs[0].clone()
    };
    let survivor_expected = serial(&model, &tokenizer, &survivor);
    let replacement_expected = serial(&model, &tokenizer, &replacement);

    for pages in [None, Some(256)] {
        let controller = Arc::new(
            BatchController::start(
                Arc::clone(&model),
                Arc::clone(&tokenizer),
                BatchOptions {
                    slots: 2,
                    queued: 8,
                    token_budget: 256,
                    pages,
                    ordering: 0,
                },
            )
            .unwrap(),
        );
        let start = Arc::new(std::sync::Barrier::new(2));
        let deadline_fired = Arc::new(AtomicBool::new(false));
        let survivor_done = Arc::new(AtomicBool::new(false));

        let deadline_engine = Arc::clone(&controller);
        let deadline_start = Arc::clone(&start);
        let deadline_signal = Arc::clone(&deadline_fired);
        let deadline_job = deadline.clone();
        let deadline_handle = std::thread::spawn(move || {
            deadline_start.wait();
            let mut deltas = 0;
            let result = deadline_engine.generate_streaming(
                &deadline_job.prompt,
                deadline_job.maximum,
                deadline_job.sampling,
                &mut |event| {
                    if matches!(event, StreamEvent::Delta { .. }) {
                        deltas += 1;
                        if deltas == 2 {
                            deadline_signal.store(true, Ordering::Release);
                            return Err(BitNetError::Inference(
                                "simulated streaming deadline".into(),
                            ));
                        }
                    }
                    Ok(())
                },
            );
            (result, deltas)
        });

        let survivor_engine = Arc::clone(&controller);
        let survivor_start = Arc::clone(&start);
        let survivor_finished = Arc::clone(&survivor_done);
        let survivor_job = survivor.clone();
        let survivor_handle = std::thread::spawn(move || {
            survivor_start.wait();
            let mut events = Vec::new();
            let result = survivor_engine.generate_streaming(
                &survivor_job.prompt,
                survivor_job.maximum,
                survivor_job.sampling,
                &mut |event| {
                    events.push(event);
                    Ok(())
                },
            );
            survivor_finished.store(true, Ordering::Release);
            (result, events)
        });

        for _ in 0..10_000 {
            if deadline_fired.load(Ordering::Acquire) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(
            deadline_fired.load(Ordering::Acquire),
            "deadline owner never received two live token deltas"
        );

        let replacement_overlapped = Arc::new(AtomicBool::new(false));
        let replacement_engine = Arc::clone(&controller);
        let survivor_finished = Arc::clone(&survivor_done);
        let replacement_overlap = Arc::clone(&replacement_overlapped);
        let replacement_job = replacement.clone();
        let replacement_handle = std::thread::spawn(move || {
            let mut events = Vec::new();
            let result = replacement_engine.generate_streaming(
                &replacement_job.prompt,
                replacement_job.maximum,
                replacement_job.sampling,
                &mut |event| {
                    if matches!(
                        event,
                        StreamEvent::FirstToken { .. } | StreamEvent::Delta { .. }
                    ) && !survivor_finished.load(Ordering::Acquire)
                    {
                        replacement_overlap.store(true, Ordering::Release);
                    }
                    events.push(event);
                    Ok(())
                },
            );
            (result, events)
        });

        let (deadline_result, deadline_deltas) = deadline_handle.join().unwrap();
        assert!(deadline_result.is_err());
        assert_eq!(deadline_deltas, 2);
        let (survivor_result, survivor_events) = survivor_handle.join().unwrap();
        survivor_result.unwrap();
        let (replacement_result, replacement_events) = replacement_handle.join().unwrap();
        replacement_result.unwrap();
        assert!(
            replacement_overlapped.load(Ordering::Acquire),
            "replacement did not emit before its heterogeneous survivor completed"
        );

        for (events, expected) in [
            (&survivor_events, &survivor_expected),
            (&replacement_events, &replacement_expected),
        ] {
            let output = events
                .iter()
                .find_map(|event| match event {
                    StreamEvent::Done(output) => Some(output),
                    _ => None,
                })
                .expect("live owner must receive one terminal event");
            let deltas: String = events
                .iter()
                .filter_map(|event| match event {
                    StreamEvent::Delta { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(deltas, output.text);
            assert_eq!(output.text, expected.1);
            assert_eq!(output.stats.finish_reason, expected.2);
        }
        let stats = controller.native_stats().unwrap();
        assert!(
            stats[0] > 0 && stats[1] > stats[0],
            "live mux must execute shared multi-owner projections: {stats:?}"
        );
        eprintln!(
            "LLAMA_LIVE_MUX_DEADLINE_CASE {}",
            serde_json::json!({
                "pages": pages,
                "deadline_owner": deadline.id,
                "survivor_owner": survivor.id,
                "replacement_owner": replacement.id,
                "replacement_overlapped": replacement_overlapped.load(Ordering::Acquire),
                "native_stats": stats,
            })
        );
    }
}

#[test]
fn optional_actual_llama_continuous_thread_controller_owned_disconnect_and_shutdown() {
    if std::env::var("RBITNET_LLAMA_CONTINUOUS_TEST").as_deref() != Ok("1") {
        return;
    }
    use super::controller::{BatchController, BatchOptions};
    std::env::set_var("RBITNET_MAX_SEQ", "1024");
    std::env::set_var("RBITNET_CUDA_PREFILL", "1");
    std::env::set_var("RBITNET_CUDA_PREFILL_TOKENS", "128");
    std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", "0");
    std::env::set_var("RBITNET_CUDA_SPLIT_KV", "0");
    std::env::set_var("RBITNET_PREFIX_KV", "0");
    std::env::set_var("RBITNET_CUDA_KV_FORMAT", "f32");
    std::env::set_var("RBITNET_CUDA_RESIDENT_GRAPH", "1");
    std::env::remove_var("RBITNET_CUDA_KV_PAGE_LIMIT");
    let (model, tokenizer) = load();
    let selected = vec![
        jobs()[1].clone(),
        jobs()[2].clone(),
        jobs()[4].clone(),
        jobs()[6].clone(),
    ];
    let expected: Vec<_> = selected
        .iter()
        .map(|j| serial(&model, &tokenizer, j))
        .collect();
    for pages in [None, Some(256)] {
        let baseline = crate::backend::cuda_managed_memory_stats().unwrap();
        let controller = Arc::new(
            BatchController::start(
                Arc::clone(&model),
                Arc::clone(&tokenizer),
                BatchOptions {
                    slots: 4,
                    queued: 16,
                    token_budget: 256,
                    pages,
                    ordering: 0,
                },
            )
            .unwrap(),
        );
        let barrier = Arc::new(std::sync::Barrier::new(selected.len()));
        let mut handles = Vec::new();
        for (index, job) in selected.iter().cloned().enumerate() {
            let engine = Arc::clone(&controller);
            let ready = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                ready.wait();
                let mut events = Vec::new();
                let mut deltas = 0;
                let result = engine.generate_streaming(
                    &job.prompt,
                    job.maximum,
                    job.sampling,
                    &mut |event| {
                        if matches!(event, StreamEvent::Delta { .. }) {
                            deltas += 1;
                            if index == 0 && deltas == 3 {
                                return Err(BitNetError::Inference(
                                    "controller client disconnected".into(),
                                ));
                            }
                        }
                        events.push(event);
                        Ok(())
                    },
                );
                (index, result, events)
            }));
        }
        for handle in handles {
            let (index, result, events) = handle.join().unwrap();
            if index == 0 {
                assert!(result.is_err());
                assert_eq!(
                    events
                        .iter()
                        .filter(|e| matches!(e, StreamEvent::Done(_)))
                        .count(),
                    0
                );
                continue;
            }
            result.unwrap();
            let terminals: Vec<_> = events
                .iter()
                .filter_map(|event| match event {
                    StreamEvent::Done(output) => Some(output),
                    _ => None,
                })
                .collect();
            assert_eq!(terminals.len(), 1);
            let actual = terminals[0];
            assert_eq!(actual.text, expected[index].1);
            assert_eq!(
                actual.stats.completion_tokens as usize,
                expected[index].0.len()
            );
            assert_eq!(actual.stats.finish_reason, expected[index].2);
            assert!(actual.stats.total_wall_ms >= actual.stats.ttft_ms);
            let deltas: String = events
                .iter()
                .filter_map(|event| match event {
                    StreamEvent::Delta { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(deltas, actual.text);
        }
        let stats = controller.native_stats().unwrap();
        assert!(
            stats[0] > 0 && stats[1] > stats[0],
            "actual shared multi-owner forward required: {stats:?}"
        );
        // A failed consumer must leave the worker available for another request.
        let job = &selected[1];
        let actual = controller
            .generate_output(&job.prompt, job.maximum, job.sampling)
            .unwrap();
        assert_eq!(actual.text, expected[1].1);
        assert_eq!(actual.stats.finish_reason, expected[1].2);
        drop(controller);
        let after = crate::backend::cuda_managed_memory_stats().unwrap();
        for category in 0..baseline.categories.len() {
            assert_eq!(
                after.categories[category], baseline.categories[category],
                "controller shutdown leaked category {category}"
            );
        }
        eprintln!(
            "LLAMA_CONTINUOUS_THREAD_CASE {}",
            serde_json::json!({"pages":pages,"native_stats":stats,"requests":5,"disconnects":1})
        );
    }
    eprintln!("LLAMA_CONTINUOUS_THREAD_DONE layouts=2 actual_shared_rows=true survivor_exact=true shutdown_categories_exact=true");
}
