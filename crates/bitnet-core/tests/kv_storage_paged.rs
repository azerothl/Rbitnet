//! Dense vs paged KV layout equivalence (Inference stack v2 phase A.1).
//! Also covers KV Q8 compact pages (`RBITNET_KV_QUANT=q8`).

use bitnet_core::llama::kv_storage::{
    KvCache, KvQuantFormat, KvStorage, PagedKvPool, PagedSeqKv,
};
use bitnet_core::llama::LlamaConfig;

fn tiny_cfg() -> LlamaConfig {
    LlamaConfig {
        n_vocab: 100,
        n_embd: 32,
        n_layer: 2,
        n_head: 4,
        n_kv: 4,
        head_dim: 8,
        rope_rot_dims: 8,
        n_ff: 64,
        max_seq: 128,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_neox: false,
        sliding_window: None,
    }
}

#[test]
fn paged_seq_roundtrip_matches_dense_offsets() {
    let cfg = tiny_cfg();
    let stride = cfg.n_kv * cfg.head_dim;
    let mut dense = KvCache::new(&cfg);
    let mut paged = PagedSeqKv::new(&cfg, 4, 64).expect("paged new");

    let k: Vec<f32> = (0..stride).map(|i| i as f32 * 0.01).collect();
    let v: Vec<f32> = (0..stride).map(|i| i as f32 * 0.02).collect();

    for pos in 0..15usize {
        let off = pos * stride;
        dense.k[0][off..off + stride].copy_from_slice(&k);
        dense.v[0][off..off + stride].copy_from_slice(&v);
        paged.write_kv_layer(0, pos, &k, &v).expect("paged write");
    }

    for pos in 0..15 {
        let off = pos * stride;
        for kv_h in 0..cfg.n_kv {
            let hd = cfg.head_dim;
            let d_slice = &dense.k[0][off + kv_h * hd..off + (kv_h + 1) * hd];
            let p_slice = paged.k_head_slice(0, pos, kv_h, hd);
            assert_eq!(d_slice, p_slice, "k layer0 pos {pos} head {kv_h}");
            let dv = &dense.v[0][off + kv_h * hd..off + (kv_h + 1) * hd];
            let pv = paged.v_head_slice(0, pos, kv_h, hd);
            assert_eq!(dv, pv, "v layer0 pos {pos} head {kv_h}");
        }
    }
}

#[test]
fn paged_pool_reuses_phys_after_clear() {
    let cfg = tiny_cfg();
    let stride = cfg.n_kv * cfg.head_dim;
    let mut paged = PagedSeqKv::new(&cfg, 2, 32).expect("paged");
    let k: Vec<f32> = (0..stride).map(|i| i as f32).collect();
    let v = k.clone();
    for pos in 0..12usize {
        paged.write_kv_layer(0, pos, &k, &v).expect("write");
    }
    let before_clear = paged.pool_stats();
    assert!(before_clear.new_phys_pages >= 1);
    paged.clear();
    for pos in 0..12usize {
        paged
            .write_kv_layer(0, pos, &k, &v)
            .expect("write after clear");
    }
    let after = paged.pool_stats();
    assert!(
        after.reused_phys_pages >= 1,
        "expected free-list reuse: {:?}",
        after
    );
}

#[test]
fn kv_storage_dense_gpu_fill_matches_rows() {
    let cfg = tiny_cfg();
    let stride = cfg.n_kv * cfg.head_dim;
    let mut ks = KvStorage::new_dense(&cfg);
    let pos = 3usize;
    let kv_h = 1usize;
    let k_row: Vec<f32> = (0..stride).map(|i| (i + pos * 7) as f32).collect();
    let v_row = k_row.clone();
    ks.write_layer_kv(0, pos, &k_row, &v_row, stride).unwrap();

    let mut dst = vec![0.0f32; (pos + 1) * cfg.head_dim];
    ks.fill_k_rows_gpu(0, pos, kv_h, cfg.head_dim, stride, &mut dst);
    for p in 0..=pos {
        let src = ks.k_head_slice(0, p, kv_h, cfg.head_dim, stride);
        let row_off = p * cfg.head_dim;
        assert_eq!(&dst[row_off..row_off + cfg.head_dim], src);
    }
}

#[test]
fn paged_kv_pool_opens_multiple_sequences() {
    let cfg = tiny_cfg();
    let mut pool = PagedKvPool::from_env(&cfg).expect("pool");
    let a = pool.open_sequence().expect("seq a");
    let b = pool.open_sequence().expect("seq b");
    assert_ne!(a, b);
    assert_eq!(pool.active_sequences(), 2);
    let stride = cfg.n_kv * cfg.head_dim;
    let k: Vec<f32> = (0..stride).map(|i| i as f32).collect();
    let v = k.clone();
    pool.sequence_mut(a)
        .expect("seq a")
        .write_kv_layer(0, 0, &k, &v)
        .expect("write a");
    pool.close_sequence(a);
    assert_eq!(pool.active_sequences(), 1);
    pool.sequence_mut(b)
        .expect("seq b")
        .write_kv_layer(0, 1, &k, &v)
        .expect("write b");
    assert!(pool.aggregate_pool_stats().new_phys_pages >= 1);
}

#[test]
fn shared_pool_clear_returns_pages_and_reuses() {
    let cfg = tiny_cfg();
    let mut pool = PagedKvPool::from_env(&cfg).expect("pool");
    let stride = cfg.n_kv * cfg.head_dim;
    let k: Vec<f32> = (0..stride).map(|i| (i as f32) * 0.1).collect();
    let v = k.clone();

    let a = pool.open_sequence().expect("a");
    {
        let seq = pool.sequence_mut(a).expect("seq");
        for pos in 0..10usize {
            seq.write_kv_layer(0, pos, &k, &v).expect("write");
        }
    }
    let allocated_after_a = pool.allocated_phys_pages();
    assert!(allocated_after_a >= 1);
    pool.close_sequence(a);
    assert!(pool.free_phys_pages() >= 1);
    assert!(pool.fragmentation_ratio() > 0.0);

    let b = pool.open_sequence().expect("b");
    {
        let seq = pool.sequence_mut(b).expect("seq");
        for pos in 0..10usize {
            seq.write_kv_layer(0, pos, &k, &v).expect("write reuse");
        }
    }
    let stats = pool.aggregate_pool_stats();
    assert!(
        stats.reused_phys_pages >= 1,
        "expected shared free-list reuse after close: {:?}",
        stats
    );
    // New allocations should not grow past the first wave when lengths match.
    assert_eq!(pool.allocated_phys_pages(), allocated_after_a);
}

#[test]
fn shared_pool_concurrency_uses_fewer_pages_than_dense_estimate() {
    let cfg = tiny_cfg();
    let mut pool = PagedKvPool::from_env(&cfg).expect("pool");
    let stride = cfg.n_kv * cfg.head_dim;
    let k: Vec<f32> = (0..stride).map(|i| i as f32).collect();
    let v = k.clone();
    let tokens_per_seq = 20usize;
    let concurrency = 4usize;

    let mut ids = Vec::new();
    for _ in 0..concurrency {
        let id = pool.open_sequence().expect("open");
        let seq = pool.sequence_mut(id).expect("seq");
        for pos in 0..tokens_per_seq {
            seq.write_kv_layer(0, pos, &k, &v).expect("write");
            seq.write_kv_layer(1, pos, &k, &v).expect("write L1");
        }
        ids.push(id);
    }

    let page_tokens = pool.page_tokens().max(1);
    let pages_per_seq_layer = (tokens_per_seq + page_tokens - 1) / page_tokens;
    let paged_pages = pool.allocated_phys_pages();
    // Dense would reserve max_seq * n_layer logical rows; paged only pages for live tokens.
    let dense_equiv_pages = concurrency * cfg.n_layer * ((cfg.max_seq + page_tokens - 1) / page_tokens);
    assert!(
        paged_pages <= concurrency * cfg.n_layer * pages_per_seq_layer,
        "paged={paged_pages} expected_cap={}",
        concurrency * cfg.n_layer * pages_per_seq_layer
    );
    assert!(
        paged_pages < dense_equiv_pages,
        "paged pages {paged_pages} should beat dense-equivalent {dense_equiv_pages}"
    );

    for id in ids {
        pool.close_sequence(id);
    }
    assert_eq!(pool.active_sequences(), 0);
    assert!(pool.free_phys_pages() >= paged_pages);
}

#[test]
fn shared_kv_storage_attention_scores_match_dense() {
    let cfg = tiny_cfg();
    let pool = PagedKvPool::from_env(&cfg).expect("pool");
    let shared = pool.shared_phys();
    let mut dense = KvStorage::new_dense(&cfg);
    let mut paged = KvStorage::new_paged_shared(
        &cfg,
        pool.page_tokens(),
        pool.max_pages_per_seq(),
        shared,
    )
    .expect("paged shared");

    let stride = cfg.n_kv * cfg.head_dim;
    let k: Vec<f32> = (0..stride).map(|i| (i as f32) * 0.01).collect();
    let v: Vec<f32> = (0..stride).map(|i| (i as f32) * 0.02).collect();
    let pos = 7usize;
    dense.write_layer_kv(0, pos, &k, &v, stride).unwrap();
    paged.write_layer_kv(0, pos, &k, &v, stride).unwrap();

    // Fill earlier positions so attention over 0..=pos is defined.
    for p in 0..pos {
        let kk: Vec<f32> = (0..stride).map(|i| (i + p) as f32 * 0.001).collect();
        let vv = kk.clone();
        dense.write_layer_kv(0, p, &kk, &vv, stride).unwrap();
        paged.write_layer_kv(0, p, &kk, &vv, stride).unwrap();
    }

    let q: Vec<f32> = (0..cfg.head_dim).map(|i| i as f32 * 0.05).collect();
    let mut out_d = vec![0.0f32; pos + 1];
    let mut out_p = vec![0.0f32; pos + 1];
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    dense.attention_scores_cpu(0, pos, 0, cfg.head_dim, stride, &q, scale, &mut out_d);
    paged.attention_scores_cpu(0, pos, 0, cfg.head_dim, stride, &q, scale, &mut out_p);
    for (i, (a, b)) in out_d.iter().zip(out_p.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "score mismatch at {i}: dense={a} paged={b}"
        );
    }
}

/// Q8 gate: decode error stays within abs-max scale / 127 (symmetric INT8).
#[test]
fn q8_paged_roundtrip_within_int8_error() {
    let cfg = tiny_cfg();
    let stride = cfg.n_kv * cfg.head_dim;
    let mut paged =
        PagedSeqKv::new_with_quant(&cfg, 4, 64, KvQuantFormat::Q8).expect("q8 paged");
    assert_eq!(paged.quant_format(), KvQuantFormat::Q8);

    let k: Vec<f32> = (0..stride)
        .map(|i| ((i as f32) * 0.07 - 1.1).sin())
        .collect();
    let v: Vec<f32> = (0..stride)
        .map(|i| ((i as f32) * 0.05 - 0.3).cos())
        .collect();

    for pos in 0..15usize {
        paged.write_kv_layer(0, pos, &k, &v).expect("q8 write");
    }

    let max_abs_k = k.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    let max_abs_v = v.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    let tol_k = (max_abs_k / 127.0) + 1e-5;
    let tol_v = (max_abs_v / 127.0) + 1e-5;

    let mut out = vec![0.0f32; cfg.head_dim];
    for pos in 0..15 {
        for kv_h in 0..cfg.n_kv {
            let start = kv_h * cfg.head_dim;
            paged.fill_k_head_values(0, pos, kv_h, cfg.head_dim, &mut out);
            for (i, &got) in out.iter().enumerate() {
                let want = k[start + i];
                assert!(
                    (got - want).abs() <= tol_k,
                    "k pos={pos} head={kv_h} i={i}: got={got} want={want} tol={tol_k}"
                );
            }
            paged.fill_v_head_values(0, pos, kv_h, cfg.head_dim, &mut out);
            for (i, &got) in out.iter().enumerate() {
                let want = v[start + i];
                assert!(
                    (got - want).abs() <= tol_v,
                    "v pos={pos} head={kv_h} i={i}: got={got} want={want} tol={tol_v}"
                );
            }
        }
    }
}

#[test]
fn q8_paged_resident_bytes_beat_f32() {
    let cfg = tiny_cfg();
    let stride = cfg.n_kv * cfg.head_dim;
    let tokens = 32usize;
    let page_tokens = 4usize;

    let mut f32_kv =
        PagedSeqKv::new_with_quant(&cfg, page_tokens, 64, KvQuantFormat::F32).expect("f32");
    let mut q8_kv =
        PagedSeqKv::new_with_quant(&cfg, page_tokens, 64, KvQuantFormat::Q8).expect("q8");

    let k: Vec<f32> = (0..stride).map(|i| i as f32 * 0.01).collect();
    let v = k.clone();
    for pos in 0..tokens {
        f32_kv.write_kv_layer(0, pos, &k, &v).expect("f32 write");
        f32_kv.write_kv_layer(1, pos, &k, &v).expect("f32 L1");
        q8_kv.write_kv_layer(0, pos, &k, &v).expect("q8 write");
        q8_kv.write_kv_layer(1, pos, &k, &v).expect("q8 L1");
    }

    let f32_bytes = f32_kv.resident_bytes();
    let q8_bytes = q8_kv.resident_bytes();
    assert!(f32_bytes > 0, "f32 resident should be > 0");
    assert!(q8_bytes > 0, "q8 resident should be > 0");
    // Q8 row = 4 + stride bytes vs 4*stride for F32 → ~4× smaller for large stride.
    assert!(
        q8_bytes * 2 < f32_bytes,
        "q8={q8_bytes} should be < half of f32={f32_bytes}"
    );

    let expected_q8 = KvQuantFormat::Q8.resident_bytes_for_pages(
        q8_kv.physical_counts().iter().sum(),
        page_tokens,
        stride,
    );
    assert_eq!(q8_bytes, expected_q8);
}

#[test]
fn q8_attention_scores_close_to_f32() {
    let cfg = tiny_cfg();
    let stride = cfg.n_kv * cfg.head_dim;
    let mut f32_store = KvStorage::Paged(
        PagedSeqKv::new_with_quant(&cfg, 4, 64, KvQuantFormat::F32).expect("f32"),
    );
    let mut q8_store = KvStorage::Paged(
        PagedSeqKv::new_with_quant(&cfg, 4, 64, KvQuantFormat::Q8).expect("q8"),
    );

    let pos = 11usize;
    for p in 0..=pos {
        let k: Vec<f32> = (0..stride)
            .map(|i| ((i + p) as f32 * 0.03).sin())
            .collect();
        let v: Vec<f32> = (0..stride)
            .map(|i| ((i + p) as f32 * 0.02).cos())
            .collect();
        f32_store.write_layer_kv(0, p, &k, &v, stride).unwrap();
        q8_store.write_layer_kv(0, p, &k, &v, stride).unwrap();
    }

    let q: Vec<f32> = (0..cfg.head_dim).map(|i| i as f32 * 0.04).collect();
    let mut out_f = vec![0.0f32; pos + 1];
    let mut out_q = vec![0.0f32; pos + 1];
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    f32_store.attention_scores_cpu(0, pos, 0, cfg.head_dim, stride, &q, scale, &mut out_f);
    q8_store.attention_scores_cpu(0, pos, 0, cfg.head_dim, stride, &q, scale, &mut out_q);

    let mut max_rel = 0.0f32;
    for (a, b) in out_f.iter().zip(out_q.iter()) {
        let denom = a.abs().max(1e-3);
        max_rel = max_rel.max((a - b).abs() / denom);
    }
    // Gate: relative score drift under ~5% for this tiny synthetic case (INT8 KV).
    assert!(
        max_rel < 0.05,
        "Q8 attention relative drift {max_rel} exceeds 5% gate"
    );
}
