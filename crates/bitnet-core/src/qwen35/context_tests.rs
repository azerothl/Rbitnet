//! Actual Qwen F32 KV/GDN/convolution round-trip across full model owners.
use super::*;
use crate::portable_envelope::{self, Compatibility};
use sha2::{Digest, Sha256};
use std::ffi::c_void;
use std::io::Read;
type Bytes = unsafe extern "C" fn(*mut c_void, u32) -> usize;
type Export = unsafe extern "C" fn(*mut c_void, u32, *mut f32, usize) -> i32;
type Import = unsafe extern "C" fn(*mut c_void, u32, *const f32, usize) -> i32;
fn digest_file(path: &Path) -> [u8; 32] {
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
#[test]
fn optional_native_portable_qwen_kv_gdn_convolution_full_unload_reload_and_sealed_files() {
    if std::env::var("RBITNET_PORTABLE_TEST").as_deref() != Ok("1") {
        return;
    }
    let path = std::path::PathBuf::from(std::env::var("RBITNET_QWEN_TEST_GGUF").unwrap());
    let tokenizer_path =
        std::path::PathBuf::from(std::env::var("RBITNET_QWEN_TEST_TOKENIZER").unwrap());
    let library_path = std::path::PathBuf::from(std::env::var("RBITNET_CUDA_QUANT_LIB").unwrap());
    let model_sha256 = digest_file(&path);
    let tokenizer_sha256 = digest_file(&tokenizer_path);
    let native_library_sha256 = digest_file(&library_path);
    let lib = crate::ggml::load_cuda_quant_library().unwrap();
    let size: Bytes = unsafe { *lib.get(b"rbitnet_cuda_qwen_portable_bytes\0").unwrap() };
    let export: Export = unsafe { *lib.get(b"rbitnet_cuda_qwen_portable_export\0").unwrap() };
    let import: Import = unsafe { *lib.get(b"rbitnet_cuda_qwen_portable_import\0").unwrap() };
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap();
    let encoded = tokenizer
        .encode(
            "Un robot explore une bibliothèque et apprend à lire des livres dans un jardin. "
                .repeat(100),
            false,
        )
        .unwrap();
    let ids = encoded.get_ids();
    assert!(ids.len() > 160);
    std::env::set_var("RBITNET_MAX_SEQ", "256");
    std::env::set_var("RBITNET_PREFIX_KV", "0");
    std::env::set_var("RBITNET_CUDA_QWEN_FULL", "1");
    std::env::set_var("RBITNET_REQUIRE_QWEN_FULL", "1");
    std::env::set_var("RBITNET_CUDA_QWEN_PREFILL", "0");
    std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", "0");
    let cuda = crate::backend::CudaRuntime::try_load().unwrap();
    let baseline = cuda.managed_memory_stats().unwrap();
    for graphs in ["0", "1"] {
        std::env::set_var("RBITNET_CUDA_QWEN_FULL_GRAPH", graphs);
        for length in [1usize, 33, 129] {
            let directory = tempfile::tempdir().unwrap();
            let load = || {
                let archive = Arc::new(GgufArchive::mmap_path(&path).unwrap());
                let runtime =
                    Qwen35Runtime::load(Arc::clone(&archive), &tokenizer_path, BackendKind::Cuda)
                        .unwrap();
                (archive, runtime)
            };
            let (file, key, expected, bytes) = {
                let (archive, mut runtime) = load();
                assert!(runtime.gpu_full.is_some());
                for (pos, &token) in ids[..length].iter().enumerate() {
                    runtime.forward_one(token, pos, &archive, false).unwrap();
                }
                let context = runtime.gpu_full.as_ref().unwrap().portable_context() as *mut c_void;
                let bytes = unsafe { size(context, length as u32) };
                assert!(bytes > 0 && bytes % 4 == 0);
                let mut values = vec![0.0; bytes / 4];
                assert_eq!(
                    unsafe { export(context, length as u32, values.as_mut_ptr(), bytes) },
                    0
                );
                assert_ne!(
                    unsafe { export(context, length as u32 - 1, values.as_mut_ptr(), bytes) },
                    0,
                    "GDN checkpoint cannot be truncated"
                );
                let key = Compatibility {
                    model_sha256,
                    tokenizer_sha256,
                    native_library_sha256,
                    execution_config_sha256: Sha256::digest(
                        format!(
                            "{:?};graphs={graphs};split={}",
                            runtime.cfg,
                            std::env::var("RBITNET_CUDA_SPLIT_KV").unwrap()
                        )
                        .as_bytes(),
                    )
                    .into(),
                    layout: "qwen-full-kv-gdn-conv-f32-v1".into(),
                };
                let file = portable_envelope::write_checkpoint(
                    directory.path(),
                    &key,
                    &ids[..length],
                    &values,
                    bytes,
                )
                .unwrap();
                for field in 0..4 {
                    let mut other = key.clone();
                    match field {
                        0 => other.model_sha256[0] ^= 1,
                        1 => other.tokenizer_sha256[0] ^= 1,
                        2 => other.native_library_sha256[0] ^= 1,
                        _ => other.execution_config_sha256[0] ^= 1,
                    };
                    assert!(
                        portable_envelope::read_checkpoint(&file, &other, bytes, bytes).is_err()
                    );
                }
                let expected: Vec<_> = ids[length..length + 8]
                    .iter()
                    .enumerate()
                    .map(|(i, &token)| {
                        runtime
                            .forward_one(token, length + i, &archive, true)
                            .unwrap()
                    })
                    .collect();
                (file, key, expected, bytes)
            };
            let unloaded = cuda.managed_memory_stats().unwrap();
            for category in 0..baseline.categories.len() {
                assert_eq!(
                    unloaded.categories[category], baseline.categories[category],
                    "category {category} after Qwen unload"
                );
            }
            let checkpoint = portable_envelope::read_checkpoint(&file, &key, bytes, bytes).unwrap();
            assert_eq!(checkpoint.tokens, &ids[..length]);
            let (archive, mut runtime) = load();
            let context = runtime.gpu_full.as_ref().unwrap().portable_context() as *mut c_void;
            assert_eq!(unsafe { size(context, length as u32) }, bytes);
            assert_ne!(
                unsafe {
                    import(
                        context,
                        length as u32,
                        checkpoint.values.as_ptr(),
                        bytes - 4,
                    )
                },
                0
            );
            assert_eq!(
                unsafe { import(context, length as u32, checkpoint.values.as_ptr(), bytes) },
                0
            );
            for (i, &token) in ids[length..length + 8].iter().enumerate() {
                let actual = runtime
                    .forward_one(token, length + i, &archive, true)
                    .unwrap();
                assert_eq!(actual.len(), expected[i].len());
                assert!(actual
                    .iter()
                    .zip(&expected[i])
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
            }
            println!("PORTABLE_QWEN graphs={graphs} length={length} bytes={bytes} KV_GDN_convolution=true exact=true");
        }
    }
    println!("PORTABLE_QWEN actual full checkpoint, sealed identity and model unload/reload passed; tier serving still pending");
}
