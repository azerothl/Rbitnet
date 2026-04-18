//! Golden-style checks for reference kernels (no external data files).
//! See `docs/GOLDEN_TESTS.md` for the full bitnet.cpp export workflow.

use bitnet_core::kernels::matvec_ternary_i8;

#[test]
fn golden_matvec_3x4() {
    // Hand-computed: W [[1,-1,0,1],[0,1,1,-1],[1,0,-1,0]], x = [0.5, -0.25, 2.0, 1.0]
    // row0: 0.5*1 + 0.25 + 0 + 1 = 1.75
    // row1: 0 -0.25 + 2 -1 = 0.75
    // row2: 0.5 + 0 -2 + 0 = -1.5
    let w: Vec<i8> = vec![1, -1, 0, 1, 0, 1, 1, -1, 1, 0, -1, 0];
    let x = vec![0.5f32, -0.25, 2.0, 1.0];
    let mut y = vec![0.0f32; 3];
    matvec_ternary_i8(&w, &x, &mut y, 3, 4);
    assert!((y[0] - 1.75).abs() < 1e-5, "y0={}", y[0]);
    assert!((y[1] - 0.75).abs() < 1e-5, "y1={}", y[1]);
    assert!((y[2] - (-1.5)).abs() < 1e-5, "y2={}", y[2]);
}
