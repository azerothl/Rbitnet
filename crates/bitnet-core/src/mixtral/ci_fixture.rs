//! Shared tiny Mixtral MoE fixture for CI / HTTP e2e (Refs #25).

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

const N_EMBD: u64 = 16;
const N_VOCAB: u64 = 8;
const N_HEAD: u64 = 2;
const N_KV: u64 = 1;
const HEAD_DIM: u64 = 8;
const N_FF: u64 = 32;
const N_EXPERT: u64 = 4;
const N_EXPERT_USED: u32 = 2;
const N_LAYER: u64 = 1;
const MAX_SEQ: u64 = 16;

fn write_u32(w: &mut File, x: u32) -> std::io::Result<()> {
    w.write_all(&x.to_le_bytes())
}

fn write_u64(w: &mut File, x: u64) -> std::io::Result<()> {
    w.write_all(&x.to_le_bytes())
}

fn write_str(w: &mut File, s: &str) -> std::io::Result<()> {
    write_u64(w, s.len() as u64)?;
    w.write_all(s.as_bytes())
}

fn write_kv_str(w: &mut File, key: &str, val: &str) -> std::io::Result<()> {
    write_str(w, key)?;
    write_u32(w, 8)?;
    write_str(w, val)
}

fn write_kv_u32(w: &mut File, key: &str, val: u32) -> std::io::Result<()> {
    write_str(w, key)?;
    write_u32(w, 4)?;
    write_u32(w, val)
}

fn write_kv_f32(w: &mut File, key: &str, val: f32) -> std::io::Result<()> {
    write_str(w, key)?;
    write_u32(w, 6)?;
    w.write_all(&val.to_le_bytes())
}

fn fill_f32(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mixed = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(seed)
                .wrapping_mul(2246822519);
            let unit = (mixed as f32) / (u32::MAX as f32);
            (unit - 0.5) * 0.05
        })
        .collect()
}

fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

struct TensorWrite {
    name: &'static str,
    dims: Vec<u64>,
    data: Vec<f32>,
}

fn nbytes(dims: &[u64]) -> usize {
    dims.iter().fold(1usize, |a, &d| a * d as usize) * 4
}

fn write_tensor_info(w: &mut File, t: &TensorWrite, offset: u64) -> std::io::Result<()> {
    write_str(w, t.name)?;
    write_u32(w, t.dims.len() as u32)?;
    for &d in &t.dims {
        write_u64(w, d)?;
    }
    write_u32(w, 0)?;
    write_u64(w, offset)
}

/// Write a 1-layer F32 Mixtral MoE GGUF used by default-CI goldens and `/v1` e2e.
pub fn write_tiny_mixtral_gguf(path: &Path) -> std::io::Result<()> {
    let n_q = N_HEAD * HEAD_DIM;
    let n_kv = N_KV * HEAD_DIM;
    let tensors = vec![
        TensorWrite {
            name: "token_embd.weight",
            dims: vec![N_EMBD, N_VOCAB],
            data: fill_f32((N_EMBD * N_VOCAB) as usize, 11),
        },
        TensorWrite {
            name: "output_norm.weight",
            dims: vec![N_EMBD],
            data: ones(N_EMBD as usize),
        },
        TensorWrite {
            name: "output.weight",
            dims: vec![N_EMBD, N_VOCAB],
            data: fill_f32((N_EMBD * N_VOCAB) as usize, 22),
        },
        TensorWrite {
            name: "blk.0.attn_norm.weight",
            dims: vec![N_EMBD],
            data: ones(N_EMBD as usize),
        },
        TensorWrite {
            name: "blk.0.attn_q.weight",
            dims: vec![N_EMBD, n_q],
            data: fill_f32((N_EMBD * n_q) as usize, 31),
        },
        TensorWrite {
            name: "blk.0.attn_k.weight",
            dims: vec![N_EMBD, n_kv],
            data: fill_f32((N_EMBD * n_kv) as usize, 32),
        },
        TensorWrite {
            name: "blk.0.attn_v.weight",
            dims: vec![N_EMBD, n_kv],
            data: fill_f32((N_EMBD * n_kv) as usize, 33),
        },
        TensorWrite {
            name: "blk.0.attn_output.weight",
            dims: vec![n_q, N_EMBD],
            data: fill_f32((n_q * N_EMBD) as usize, 34),
        },
        TensorWrite {
            name: "blk.0.ffn_norm.weight",
            dims: vec![N_EMBD],
            data: ones(N_EMBD as usize),
        },
        TensorWrite {
            name: "blk.0.ffn_gate_inp.weight",
            dims: vec![N_EMBD, N_EXPERT],
            data: fill_f32((N_EMBD * N_EXPERT) as usize, 40),
        },
        TensorWrite {
            name: "blk.0.ffn_up_exps.weight",
            dims: vec![N_EMBD, N_FF, N_EXPERT],
            data: fill_f32((N_EMBD * N_FF * N_EXPERT) as usize, 41),
        },
        TensorWrite {
            name: "blk.0.ffn_gate_exps.weight",
            dims: vec![N_EMBD, N_FF, N_EXPERT],
            data: fill_f32((N_EMBD * N_FF * N_EXPERT) as usize, 42),
        },
        TensorWrite {
            name: "blk.0.ffn_down_exps.weight",
            dims: vec![N_FF, N_EMBD, N_EXPERT],
            data: fill_f32((N_FF * N_EMBD * N_EXPERT) as usize, 43),
        },
    ];

    let mut f = File::create(path)?;
    let kv_count = 13u64;
    f.write_all(b"GGUF")?;
    write_u32(&mut f, 3)?;
    write_u64(&mut f, tensors.len() as u64)?;
    write_u64(&mut f, kv_count)?;
    write_kv_str(&mut f, "general.architecture", "mixtral")?;
    write_kv_u32(&mut f, "mixtral.embedding_length", N_EMBD as u32)?;
    write_kv_u32(&mut f, "mixtral.vocab_size", N_VOCAB as u32)?;
    write_kv_u32(&mut f, "mixtral.block_count", N_LAYER as u32)?;
    write_kv_u32(&mut f, "mixtral.attention.head_count", N_HEAD as u32)?;
    write_kv_u32(&mut f, "mixtral.attention.head_count_kv", N_KV as u32)?;
    write_kv_u32(&mut f, "mixtral.feed_forward_length", N_FF as u32)?;
    write_kv_u32(&mut f, "mixtral.expert_count", N_EXPERT as u32)?;
    write_kv_u32(&mut f, "mixtral.expert_used_count", N_EXPERT_USED)?;
    write_kv_u32(&mut f, "mixtral.context_length", MAX_SEQ as u32)?;
    write_kv_f32(&mut f, "mixtral.rope.freq_base", 1_000_000.0)?;
    write_kv_f32(&mut f, "mixtral.attention.layer_norm_rms_epsilon", 1e-5)?;
    write_kv_u32(&mut f, "general.alignment", 32)?;

    let mut offset = 0u64;
    for t in &tensors {
        write_tensor_info(&mut f, t, offset)?;
        offset += nbytes(&t.dims) as u64;
    }

    let pos = f.metadata()?.len() as usize;
    let pad = (32 - (pos % 32)) % 32;
    f.write_all(&vec![0u8; pad])?;

    for t in &tensors {
        assert_eq!(t.data.len(), nbytes(&t.dims) / 4);
        for &v in &t.data {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

/// WordLevel tokenizer: prompt `"Hello"` → token id `1`.
pub fn write_wordlevel_tokenizer(path: &Path) -> std::io::Result<()> {
    let json = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {
      "id": 0,
      "content": "[UNK]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
  ],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "unk_token": "[UNK]",
    "vocab": {
      "[UNK]": 0,
      "Hello": 1,
      "world": 2,
      "a": 3,
      "b": 4,
      "c": 5,
      "d": 6,
      "e": 7
    }
  }
}"#;
    fs::write(path, json)
}
