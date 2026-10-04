//! Cross-feature ABI refusals must preserve an actual model's live KV state.
use super::*;

#[test]
fn optional_actual_combined_encoded_native_consumers_refused_without_mutation() {
    if std::env::var("RBITNET_STACK_GUARD_TEST").as_deref() != Ok("1") {
        return;
    }
    std::env::set_var("RBITNET_MAX_SEQ", "256");
    std::env::set_var("RBITNET_CUDA_PREFILL", "1");
    std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", "0");
    std::env::set_var("RBITNET_CUDA_SPLIT_KV", "0");
    std::env::remove_var("RBITNET_CUDA_KV_PAGE_LIMIT");
    let model = LlamaModel::from_gguf_arc_for_backend(
        std::sync::Arc::new(
            crate::gguf::GgufArchive::mmap_path(std::path::Path::new(
                &std::env::var("RBITNET_TEST_GGUF").unwrap(),
            ))
            .unwrap(),
        ),
        crate::backend::BackendKind::Cuda,
    )
    .unwrap();
    type Create = unsafe extern "C" fn(*const c_void, u32, u32) -> *mut c_void;
    type Destroy = unsafe extern "C" fn(*mut c_void);
    type Step = unsafe extern "C" fn(
        *mut c_void,
        *const *mut c_void,
        *const u32,
        *const f32,
        u32,
        u32,
        *mut f32,
        *mut u32,
    ) -> i32;
    type Stats = unsafe extern "C" fn(*const c_void, *mut u64, u32) -> i32;
    type Bytes = unsafe extern "C" fn(*mut c_void, u32) -> usize;
    type Export = unsafe extern "C" fn(*mut c_void, u32, *mut f32, usize) -> i32;
    type Import = unsafe extern "C" fn(*mut c_void, u32, *const f32, usize) -> i32;
    let library = crate::ggml::load_cuda_quant_library().unwrap();
    let create: Create = unsafe { *library.get(b"rbitnet_cuda_llama_batch_create\0").unwrap() };
    let destroy: Destroy = unsafe { *library.get(b"rbitnet_cuda_llama_batch_destroy\0").unwrap() };
    let step: Step = unsafe { *library.get(b"rbitnet_cuda_llama_batch_step\0").unwrap() };
    let stats: Stats = unsafe { *library.get(b"rbitnet_cuda_llama_batch_stats\0").unwrap() };
    let size: Bytes = unsafe { *library.get(b"rbitnet_cuda_llama_portable_bytes\0").unwrap() };
    let export: Export = unsafe {
        *library
            .get(b"rbitnet_cuda_llama_portable_export\0")
            .unwrap()
    };
    let import: Import = unsafe {
        *library
            .get(b"rbitnet_cuda_llama_portable_import\0")
            .unwrap()
    };
    struct Batch(*mut c_void, Destroy);
    impl Drop for Batch {
        fn drop(&mut self) {
            unsafe { (self.1)(self.0) };
        }
    }
    let mut cases = 0;
    for graph in ["0", "1"] {
        std::env::set_var("RBITNET_CUDA_RESIDENT_GRAPH", graph);
        for pages in [None, Some(24)] {
            std::env::set_var("RBITNET_CUDA_KV_FORMAT", "f32");
            let f32_seed = Resident::new_with_pages(&model, pages, None).unwrap();
            let batch = Batch(
                unsafe { create(f32_seed.context as *const c_void, 1, 0) },
                destroy,
            );
            assert!(!batch.0.is_null(), "F32 batch positive control");
            for format in ["f16", "q8"] {
                std::env::set_var("RBITNET_CUDA_KV_FORMAT", format);
                let mut reference = Resident::new_with_pages(&model, pages, None).unwrap();
                let mut refused = Resident::new_with_pages(&model, pages, None).unwrap();
                for (position, token) in [128000, 300, 5000, 19].into_iter().enumerate() {
                    let expected = reference.forward(&model, token, position, true).unwrap();
                    let actual = refused.forward(&model, token, position, true).unwrap();
                    assert!(actual
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()));
                }
                let context = refused.context as *mut c_void;
                let pages_before = refused.page_stats().unwrap();
                assert!(unsafe { create(context as *const c_void, 1, 0) }.is_null());
                let mut before = [0u64; 3];
                assert_eq!(unsafe { stats(batch.0, before.as_mut_ptr(), 3) }, 0);
                let embedding = vec![0.0; model.cfg.n_embd];
                assert_ne!(
                    unsafe {
                        step(
                            batch.0,
                            [context].as_ptr(),
                            [4].as_ptr(),
                            embedding.as_ptr(),
                            1,
                            0,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                        )
                    },
                    0
                );
                let mut after = [0u64; 3];
                assert_eq!(unsafe { stats(batch.0, after.as_mut_ptr(), 3) }, 0);
                assert_eq!(before, after, "refusal must not execute a batch wave");
                assert_eq!(unsafe { size(context, 4) }, 0);
                let mut sentinel = [42.0f32];
                assert_ne!(unsafe { export(context, 4, sentinel.as_mut_ptr(), 4) }, 0);
                assert_eq!(sentinel, [42.0]);
                assert_ne!(unsafe { import(context, 4, sentinel.as_ptr(), 4) }, 0);
                let pages_after = refused.page_stats().unwrap();
                if let (Some(a), Some(b)) = (pages_before, pages_after) {
                    assert_eq!(a.tokens, b.tokens);
                    assert_eq!(a.allocations, b.allocations);
                    assert_eq!(a.references, b.references);
                    assert_eq!(a.active_pages, b.active_pages);
                }
                for (position, token) in [420, 9, 128001].into_iter().enumerate() {
                    let expected = reference
                        .forward(&model, token, position + 4, true)
                        .unwrap();
                    let actual = refused.forward(&model, token, position + 4, true).unwrap();
                    assert!(actual
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits()));
                }
                cases += 1;
            }
        }
    }
    eprintln!("STACK_ENCODED_GUARDS_DONE cases={cases} batch_and_portable_refusal_exact=true pages_preserved=true F32_control=true");
}
