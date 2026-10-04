//! Actual transfer/unload/reload checks, separate from a future tier policy.
use super::*;
use crate::portable_envelope::{self, Compatibility};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::sync::Arc;
type Bytes = unsafe extern "C" fn(*mut c_void, u32) -> usize;
type Export = unsafe extern "C" fn(*mut c_void, u32, *mut f32, usize) -> i32;
type Import = unsafe extern "C" fn(*mut c_void, u32, *const f32, usize) -> i32;
fn digest_file(path: &std::path::Path) -> [u8; 32] {
    let mut input = std::fs::File::open(path).unwrap();
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = input.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    hash.finalize().into()
}
fn exact(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
}
#[test]
fn optional_native_portable_llama_f32_dense_pages_unload_reload_and_sealed_files() {
    if std::env::var("RBITNET_PORTABLE_TEST").as_deref() != Ok("1") {
        return;
    }
    let path = std::path::PathBuf::from(std::env::var("RBITNET_TEST_GGUF").unwrap());
    let token_path = std::path::PathBuf::from(std::env::var("RBITNET_TOKENIZER").unwrap());
    let library_path = std::path::PathBuf::from(std::env::var("RBITNET_CUDA_QUANT_LIB").unwrap());
    let lib = crate::ggml::load_cuda_quant_library().unwrap();
    let size: Bytes = unsafe { *lib.get(b"rbitnet_cuda_llama_portable_bytes\0").unwrap() };
    let export: Export = unsafe { *lib.get(b"rbitnet_cuda_llama_portable_export\0").unwrap() };
    let import: Import = unsafe { *lib.get(b"rbitnet_cuda_llama_portable_import\0").unwrap() };
    let model_sha256 = digest_file(&path);
    let tokenizer_sha256 = digest_file(&token_path);
    let native_library_sha256 = digest_file(&library_path);
    let tokenizer = tokenizers::Tokenizer::from_file(&token_path).unwrap();
    let text =
        "Le robot visite une bibliothèque calme et lit un livre dans le jardin. ".repeat(100);
    let encoded = tokenizer.encode(text, false).unwrap();
    let ids = encoded.get_ids();
    assert!(ids.len() > 160);
    let rt = crate::backend::CudaRuntime::try_load().unwrap();
    let baseline = rt.managed_memory_stats().unwrap();
    std::env::set_var("RBITNET_MAX_SEQ", "256");
    std::env::set_var("RBITNET_PREFIX_KV", "0");
    std::env::set_var("RBITNET_CUDA_PREFILL", "1");
    std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", "0");
    for graphs in ["0", "1"] {
        std::env::set_var("RBITNET_CUDA_RESIDENT_GRAPH", graphs);
        for pages in [None, Some(24)] {
            for length in [1usize, 33, 129] {
                let directory = tempfile::tempdir().unwrap();
                let load = || {
                    LlamaModel::from_gguf_arc_for_backend(
                        Arc::new(crate::gguf::GgufArchive::mmap_path(&path).unwrap()),
                        crate::backend::BackendKind::Cuda,
                    )
                    .unwrap()
                };
                let (file, key, reference, bytes, old_context) = {
                    let model = load();
                    let mut runtime = Resident::new_with_pages(&model, pages, None).unwrap();
                    runtime.prefill(&model, &ids[..length], 0, false).unwrap();
                    let bytes = unsafe { size(runtime.context as *mut c_void, length as u32) };
                    assert!(bytes > 0 && bytes % 4 == 0);
                    let mut values = vec![0.0; bytes / 4];
                    assert_eq!(
                        unsafe {
                            export(
                                runtime.context as *mut c_void,
                                length as u32,
                                values.as_mut_ptr(),
                                bytes,
                            )
                        },
                        0
                    );
                    assert_ne!(
                        unsafe {
                            export(
                                runtime.context as *mut c_void,
                                length as u32 + 1,
                                values.as_mut_ptr(),
                                bytes,
                            )
                        },
                        0
                    );
                    let key = Compatibility {
                        model_sha256,
                        tokenizer_sha256,
                        native_library_sha256,
                        execution_config_sha256: Sha256::digest(
                            format!(
                                "{:?};graphs={graphs};split={}",
                                model.cfg,
                                std::env::var("RBITNET_CUDA_SPLIT_KV").unwrap()
                            )
                            .as_bytes(),
                        )
                        .into(),
                        layout: "llama-native-f32-v1".into(),
                    };
                    assert!(portable_envelope::write_checkpoint(
                        directory.path(),
                        &key,
                        &ids[..length],
                        &values,
                        bytes - 1
                    )
                    .is_err());
                    let file = portable_envelope::write_checkpoint(
                        directory.path(),
                        &key,
                        &ids[..length],
                        &values,
                        bytes,
                    )
                    .unwrap();
                    let identical = portable_envelope::write_checkpoint(
                        directory.path(),
                        &key,
                        &ids[..length],
                        &values,
                        bytes,
                    )
                    .unwrap();
                    assert_eq!(file, identical);
                    let mut other = key.clone();
                    other.model_sha256[0] ^= 1;
                    assert!(
                        portable_envelope::read_checkpoint(&file, &other, bytes, bytes).is_err()
                    );
                    let damaged_directory = directory.path().join("corrupt");
                    std::fs::create_dir(&damaged_directory).unwrap();
                    let damaged = damaged_directory.join(file.file_name().unwrap());
                    let mut raw = std::fs::read(&file).unwrap();
                    let last = raw.len() - 1;
                    raw[last] ^= 1;
                    std::fs::write(&damaged, &raw).unwrap();
                    assert!(
                        portable_envelope::read_checkpoint(&damaged, &key, bytes, bytes).is_err()
                    );
                    let reference: Vec<_> = ids[length..length + 8]
                        .iter()
                        .enumerate()
                        .map(|(i, &token)| {
                            runtime.forward(&model, token, length + i, true).unwrap()
                        })
                        .collect();
                    (file, key, reference, bytes, runtime.context)
                };
                let unloaded = rt.managed_memory_stats().unwrap();
                for category in 0..baseline.categories.len() {
                    assert_eq!(
                        unloaded.categories[category], baseline.categories[category],
                        "category {category} after unload"
                    );
                }
                let checkpoint =
                    portable_envelope::read_checkpoint(&file, &key, bytes, bytes).unwrap();
                assert_eq!(checkpoint.tokens, &ids[..length]);
                let model = load();
                let mut runtime = Resident::new_with_pages(&model, pages, None).unwrap();
                // Allocator addresses may be reused; serialized state carries no raw pointer.
                assert_eq!(
                    unsafe { size(runtime.context as *mut c_void, length as u32) },
                    bytes
                );
                assert_ne!(
                    unsafe {
                        import(
                            runtime.context as *mut c_void,
                            length as u32,
                            checkpoint.values.as_ptr(),
                            bytes - 4,
                        )
                    },
                    0
                );
                let mut nonfinite = checkpoint.values.clone();
                nonfinite[0] = f32::NAN;
                assert_ne!(
                    unsafe {
                        import(
                            runtime.context as *mut c_void,
                            length as u32,
                            nonfinite.as_ptr(),
                            bytes,
                        )
                    },
                    0
                );
                assert_eq!(
                    unsafe {
                        import(
                            runtime.context as *mut c_void,
                            length as u32,
                            checkpoint.values.as_ptr(),
                            bytes,
                        )
                    },
                    0
                );
                for (i, &token) in ids[length..length + 8].iter().enumerate() {
                    exact(
                        &runtime.forward(&model, token, length + i, true).unwrap(),
                        &reference[i],
                    );
                }
                println!("PORTABLE_LLAMA graphs={graphs} pages={pages:?} length={length} bytes={bytes} old_address={old_context} new_address={} exact=true",runtime.context);
            }
        }
    }
    println!("PORTABLE_LLAMA actual F32 planes, sealed files and full model unload/reload passed; tier serving still pending");
}
