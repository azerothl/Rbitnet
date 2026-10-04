//! Mutable Native TF32 refusal must preserve encoded state and shared-pool mode.
use super::*;

#[test]
fn optional_actual_encoded_tf32_mutable_configuration_refused() {
    if std::env::var("RBITNET_ENCODED_GUARD_TEST").as_deref() != Ok("1") {
        return;
    }
    std::env::set_var("RBITNET_MAX_SEQ", "512");
    std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", "0");
    std::env::set_var("RBITNET_CUDA_PREFILL", "1");
    std::env::remove_var("RBITNET_CUDA_KV_PAGE_LIMIT");
    let archive = std::sync::Arc::new(
        crate::gguf::GgufArchive::mmap_path(std::path::Path::new(
            &std::env::var("RBITNET_TEST_GGUF").unwrap(),
        ))
        .unwrap(),
    );
    let model =
        LlamaModel::from_gguf_arc_for_backend(archive, crate::backend::BackendKind::Cuda).unwrap();
    type Configure = unsafe extern "C" fn(*mut c_void, u32) -> i32;
    let library = crate::ggml::load_cuda_quant_library().unwrap();
    let configure = unsafe {
        *library
            .get::<Configure>(b"rbitnet_cuda_llama_configure_tensor_prefill\0")
            .unwrap()
    };
    assert_ne!(unsafe { configure(std::ptr::null_mut(), 1) }, 0);
    let mut cases = 0;
    for graph in ["0", "1"] {
        std::env::set_var("RBITNET_CUDA_RESIDENT_GRAPH", graph);
        for format in ["f16", "q8"] {
            std::env::set_var("RBITNET_CUDA_KV_FORMAT", format);
            for pages in [None, Some(64)] {
                let mut pristine = Resident::new_with_pages(&model, pages, None).unwrap();
                let mut rejected = Resident::new_with_pages(&model, pages, None).unwrap();
                let context = rejected.context as *mut c_void;
                assert_ne!(unsafe { configure(context, 2) }, 0);
                assert_ne!(unsafe { configure(context, 1) }, 0);
                assert_eq!(unsafe { configure(context, 0) }, 0);
                // Refusal must not mutate the shared pool's F32 arithmetic mode.
                let sibling = pages.map(|limit| {
                    Resident::new_with_pages(&model, Some(limit), Some(&rejected)).unwrap()
                });
                for (position, token) in [128000, 300, 5000, 19, 420, 9, 128001]
                    .into_iter()
                    .enumerate()
                {
                    let expected = pristine.forward(&model, token, position, true).unwrap();
                    let actual = rejected.forward(&model, token, position, true).unwrap();
                    assert!(actual.iter().all(|value| value.is_finite()));
                    assert_eq!(actual.len(), expected.len());
                    assert!(
                        actual
                            .iter()
                            .zip(expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "mutable refusal changed {format} pages={pages:?} graph={graph}"
                    );
                }
                assert_ne!(
                    unsafe { configure(context, 0) },
                    0,
                    "filled state configuration must stay sealed"
                );
                assert_ne!(unsafe { configure(context, 1) }, 0);
                drop(sibling);
                cases += 1;
            }
        }
    }
    // Positive control: a fresh dense F32 owner still accepts this API.
    std::env::set_var("RBITNET_CUDA_KV_FORMAT", "f32");
    let control = Resident::new_with_pages(&model, None, None).unwrap();
    assert_eq!(unsafe { configure(control.context as *mut c_void, 1) }, 0);
    assert_eq!(unsafe { configure(control.context as *mut c_void, 0) }, 0);
    eprintln!("ENCODED_GUARD_DONE cases={cases} F16/Q8 dense/paged graph0/1 refusal_exact=true shared_pool_preserved=true F32_control=true");
}
