//! Microbench TQ1_0 / TQ2_0 row dots: stack-scratch `dot_row` vs heap decode-then-dot.
//!
//! Usage:
//!   TQ_ITERS=200 cargo run -p bitnet-core --example tq_dot_microbench --release --locked

use half::f16;
use std::time::Instant;

fn heap_decode_then_dot(ty: u32, row: &[u8], x: &[f32]) -> f32 {
    let mut buf = vec![0.0f32; x.len()];
    bitnet_core::ggml::decode_row_to_f32(ty, row, &mut buf).expect("tq decode");
    buf.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum()
}

fn main() {
    let iters: usize = std::env::var("TQ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let rows: usize = std::env::var("TQ_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    let mut payload_tq2 = Vec::with_capacity(rows * 66);
    let mut payload_tq1 = Vec::with_capacity(rows * 54);
    for r in 0..rows {
        let mut blk2 = vec![0u8; 66];
        blk2[0..2].copy_from_slice(&f16::from_f32(0.125).to_bits().to_le_bytes());
        for i in 2..66 {
            blk2[i] = ((i + r) as u8).wrapping_mul(19).wrapping_add(7);
        }
        payload_tq2.extend_from_slice(&blk2);

        let mut blk1 = vec![0u8; 54];
        blk1[0..2].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
        for i in 2..54 {
            blk1[i] = ((i + r) as u8).wrapping_mul(17).wrapping_add(5);
        }
        payload_tq1.extend_from_slice(&blk1);
    }
    let x: Vec<f32> = (0..256).map(|i| i as f32 * 0.01 - 1.0).collect();

    let stack2 = bitnet_core::ggml::dot_row(35, &payload_tq2[..66], &x).unwrap();
    let heap2 = heap_decode_then_dot(35, &payload_tq2[..66], &x);
    assert!((stack2 - heap2).abs() < 1e-5, "tq2 stack={stack2} heap={heap2}");
    let stack1 = bitnet_core::ggml::dot_row(34, &payload_tq1[..54], &x).unwrap();
    let heap1 = heap_decode_then_dot(34, &payload_tq1[..54], &x);
    assert!((stack1 - heap1).abs() < 1e-5, "tq1 stack={stack1} heap={heap1}");

    let t0 = Instant::now();
    for _ in 0..iters {
        for r in 0..rows {
            let _ = bitnet_core::ggml::dot_row(35, &payload_tq2[r * 66..(r + 1) * 66], &x).unwrap();
        }
    }
    let stack_tq2_ns = t0.elapsed().as_nanos() / (iters * rows) as u128;

    let t1 = Instant::now();
    for _ in 0..iters {
        for r in 0..rows {
            let _ = heap_decode_then_dot(35, &payload_tq2[r * 66..(r + 1) * 66], &x);
        }
    }
    let heap_tq2_ns = t1.elapsed().as_nanos() / (iters * rows) as u128;

    let t2 = Instant::now();
    for _ in 0..iters {
        for r in 0..rows {
            let _ = bitnet_core::ggml::dot_row(34, &payload_tq1[r * 54..(r + 1) * 54], &x).unwrap();
        }
    }
    let stack_tq1_ns = t2.elapsed().as_nanos() / (iters * rows) as u128;

    let t3 = Instant::now();
    for _ in 0..iters {
        for r in 0..rows {
            let _ = heap_decode_then_dot(34, &payload_tq1[r * 54..(r + 1) * 54], &x);
        }
    }
    let heap_tq1_ns = t3.elapsed().as_nanos() / (iters * rows) as u128;

    let speedup2 = heap_tq2_ns as f64 / stack_tq2_ns.max(1) as f64;
    let speedup1 = heap_tq1_ns as f64 / stack_tq1_ns.max(1) as f64;

    println!(
        "| TQ row dots {rows}x256 | tq2_stack={stack_tq2_ns}ns tq2_heap={heap_tq2_ns}ns (~{speedup2:.2}x) tq1_stack={stack_tq1_ns}ns tq1_heap={heap_tq1_ns}ns (~{speedup1:.2}x) | bit_exact≈true | stack vs heap scratch | NATIVE_FIRST mmap GEMV |"
    );
    eprintln!("tq2_speedup={speedup2:.2}x tq1_speedup={speedup1:.2}x stack_ok=true");
}
