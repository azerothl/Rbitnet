//! Direct ABI refusal prevents a mutable TF32 setting on encoded KV state.
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
    for format in ["f16", "q8"] {
        std::env::set_var("RBITNET_CUDA_KV_FORMAT", format);
        let mut r = Resident::new_with_pages(&model, None, None).unwrap();
        assert_ne!(unsafe { configure(r.context as *mut c_void, 1) }, 0);
        assert_eq!(unsafe { configure(r.context as *mut c_void, 0) }, 0);
        let logits = r.forward(&model, 128000, 0, true).unwrap();
        assert!(logits.iter().all(|x| x.is_finite()));
        assert_ne!(
            unsafe { configure(r.context as *mut c_void, 0) },
            0,
            "filled state configuration is sealed"
        );
    }
    eprintln!("ENCODED_GUARD_DONE F16/Q8 mutable TF32 requests refused before any sequence");
}
