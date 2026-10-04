//! Actual regression: block size / final-token recomputation may not change encoded state.
use super::*;
use std::sync::Arc;
fn same(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len());
    let count = a
        .iter()
        .zip(b)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let worst = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("KV_CANONICAL_COMPARE {label} changed_logits={count} max_abs={worst}");
    assert_eq!(
        count, 0,
        "{label}: all same-format logits must be bit exact"
    );
}
#[test]
fn optional_actual_encoded_prefill_partition_prefix_restore_exact() {
    if std::env::var("RBITNET_KV_CANONICAL_TEST").as_deref() != Ok("1") {
        return;
    }
    std::env::set_var("RBITNET_MAX_SEQ", "512");
    std::env::set_var("RBITNET_CUDA_PREFILL", "1");
    std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", "0");
    std::env::set_var("RBITNET_PREFIX_KV", "1");
    std::env::set_var("RBITNET_CUDA_PREFIX_ENTRIES", "8");
    std::env::set_var("RBITNET_CUDA_PREFIX_MB", "128");
    std::env::remove_var("RBITNET_CUDA_KV_PAGE_LIMIT");
    let archive = Arc::new(
        crate::gguf::GgufArchive::mmap_path(std::path::Path::new(
            &std::env::var("RBITNET_TEST_GGUF").unwrap(),
        ))
        .unwrap(),
    );
    let model =
        LlamaModel::from_gguf_arc_for_backend(archive, crate::backend::BackendKind::Cuda).unwrap();
    let tokenizer =
        tokenizers::Tokenizer::from_file(std::env::var("RBITNET_TOKENIZER").unwrap()).unwrap();
    for format in ["f16", "q8"] {
        std::env::set_var("RBITNET_CUDA_KV_FORMAT", format);
        for graph in ["0", "1"] {
            std::env::set_var("RBITNET_CUDA_RESIDENT_GRAPH", graph);
            for pages in [None, Some(64u32)] {
                let mut serial = Resident::new_with_pages(&model, pages, None).unwrap();
                let mut block = Resident::new_with_pages(&model, pages, None).unwrap();
                for text in ["Les villes ont des bibliothèques, des jardins et des musées. ","fn checked_sum(xs: &[i32]) -> Option<i32> { xs.iter().try_fold(0, |a, &b| a.checked_add(b)) } "] {
                    let encoded=tokenizer.encode(format!("<|begin_of_text|>{}",text.repeat(60)),false).unwrap();let ids=encoded.get_ids();assert!(ids.len()>300);
                    for prefix in [33usize,129,257] {
                        let mut expected=Vec::new();
                        for (pos,&id)in ids[..prefix].iter().enumerate(){expected=serial.forward(&model,id,pos,pos+1==prefix).unwrap();}
                        for chunk in [2usize,16,128] {
                            std::env::set_var("RBITNET_CUDA_PREFILL_TOKENS",chunk.to_string());
                            let actual=block.prefill(&model,&ids[..prefix],0,false).unwrap().0;
                            same(&actual,&expected,&format!("format={format} graph={graph} pages={pages:?} prefix={prefix} chunk={chunk} cold vs serial"));
                            block.save_prefix(&ids[..prefix]);
                            block.forward(&model,ids[prefix+13],prefix,false).unwrap();
                            let restored=block.restore_prefix(&ids[..prefix]).unwrap();assert_eq!(restored,prefix-1);
                            let warm=block.prefill(&model,&ids[restored..prefix],restored,false).unwrap().0;
                            same(&warm,&expected,"same-format restored last token vs cold serial");
                            for offset in 0..4 {
                                let a=serial.forward(&model,ids[prefix+offset],prefix+offset,true).unwrap();
                                let b=block.forward(&model,ids[prefix+offset],prefix+offset,true).unwrap();same(&a,&b,"restored successor logits");
                            }
                            // Reference returns to the exact original prompt for
                            // the next independent block partition.
                            for (pos,&id)in ids[..prefix].iter().enumerate(){serial.forward(&model,id,pos,false).unwrap();}
                        }
                    }
                }
            }
        }
    }
    eprintln!("KV_CANONICAL_DONE encoded F16/Q8 dense/page, block partitions and restored prefix/successors exactly match serial");
}
