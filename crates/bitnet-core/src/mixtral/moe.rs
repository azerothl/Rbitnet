//! Top-k MoE FFN for Mixtral-style expert tensors (CPU F32 / mmap quant).

use crate::error::{BitNetError, Result};
use crate::ggml::{ggml_nbytes, matvec_embd_out_mmap};
use crate::gguf::{GgufArchive, GgufTensorInfo};
use crate::qwen35::qmatvec::{quant_matmul_vec, quant_matmul_vec_offset};

use super::config::MixtralConfig;

fn softmax_vec(s: &mut [f32]) {
    let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for z in s.iter_mut() {
        *z = (*z - m).exp();
        sum += *z;
    }
    if sum > 0.0 {
        for z in s.iter_mut() {
            *z /= sum;
        }
    }
}

fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(scores.len());
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    idx.truncate(k);
    idx
}

/// Routed experts + SiLU gated FF; returns delta hidden `[n_embd]`.
pub fn moe_forward(
    archive: &GgufArchive,
    cfg: &MixtralConfig,
    x: &[f32],
    gate_inp: &GgufTensorInfo,
    up_exps: &GgufTensorInfo,
    gate_exps: &GgufTensorInfo,
    down_exps: &GgufTensorInfo,
) -> Result<Vec<f32>> {
    if gate_inp.dimensions.len() != 2 {
        return Err(BitNetError::Inference(
            "ffn_gate_inp: expected 2D tensor".into(),
        ));
    }
    let n_emb_g = usize::try_from(gate_inp.dimensions[0])
        .map_err(|_| BitNetError::Inference("ne0 gate".into()))?;
    let n_exp_g = usize::try_from(gate_inp.dimensions[1])
        .map_err(|_| BitNetError::Inference("ne1 gate".into()))?;
    if n_emb_g != cfg.n_embd || n_exp_g != cfg.n_expert {
        return Err(BitNetError::Inference(format!(
            "ffn_gate_inp dims [{n_emb_g},{n_exp_g}] != config embd/experts"
        )));
    }

    let mut router = if gate_inp.ggml_type == 0 {
        matvec_embd_out_mmap(archive, gate_inp, x, n_emb_g, n_exp_g)?
    } else {
        let gi_py = archive.tensor_payload(gate_inp)?;
        quant_matmul_vec(gi_py, gate_inp.ggml_type, n_emb_g, n_exp_g, x)?
    };
    softmax_vec(&mut router);
    let top = top_k_indices(&router, cfg.n_expert_used);

    for t in &[up_exps, gate_exps, down_exps] {
        if t.dimensions.len() != 3 {
            return Err(BitNetError::Inference(
                "MoE expert tensor: expected rank-3 GGUF layout".into(),
            ));
        }
    }

    let stride_up = ggml_nbytes(
        &[up_exps.dimensions[0], up_exps.dimensions[1]],
        up_exps.ggml_type,
    )?;
    let stride_gate = ggml_nbytes(
        &[gate_exps.dimensions[0], gate_exps.dimensions[1]],
        gate_exps.ggml_type,
    )?;
    let stride_down = ggml_nbytes(
        &[down_exps.dimensions[0], down_exps.dimensions[1]],
        down_exps.ggml_type,
    )?;
    let n_embd_up = usize::try_from(up_exps.dimensions[0])
        .map_err(|_| BitNetError::Inference("n_embd experts".into()))?;
    let n_ff_up = usize::try_from(up_exps.dimensions[1])
        .map_err(|_| BitNetError::Inference("n_ff up".into()))?;
    let n_ff_dn = usize::try_from(down_exps.dimensions[0])
        .map_err(|_| BitNetError::Inference("n_ff dn".into()))?;
    let n_embd_dn = usize::try_from(down_exps.dimensions[1])
        .map_err(|_| BitNetError::Inference("n_embd dn".into()))?;

    let mut acc = vec![0f32; cfg.n_embd];
    let up_payload = archive.tensor_payload(up_exps)?;
    let gate_payload = archive.tensor_payload(gate_exps)?;
    let down_payload = archive.tensor_payload(down_exps)?;

    for &e in &top {
        let w_router = router.get(e).copied().unwrap_or(0f32);
        if w_router <= 1e-8 {
            continue;
        }
        let up_off = e * stride_up;
        let gate_off = e * stride_gate;
        let down_off = e * stride_down;

        let up = quant_matmul_vec_offset(
            up_payload,
            up_exps.ggml_type,
            n_embd_up,
            n_ff_up,
            up_off,
            x,
        )?;
        let gate = quant_matmul_vec_offset(
            gate_payload,
            gate_exps.ggml_type,
            n_embd_up,
            n_ff_up,
            gate_off,
            x,
        )?;
        let silu_g = silu(&gate);
        let hidden: Vec<f32> = up.iter().zip(silu_g.iter()).map(|(u, g)| u * g).collect();
        let y = quant_matmul_vec_offset(
            down_payload,
            down_exps.ggml_type,
            n_ff_dn,
            n_embd_dn,
            down_off,
            &hidden,
        )?;
        for i in 0..cfg.n_embd.min(y.len()) {
            acc[i] += y[i] * w_router;
        }
    }

    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_picks_highest_scores() {
        let scores = [0.1f32, 0.9, 0.3, 0.8];
        assert_eq!(top_k_indices(&scores, 2), vec![1, 3]);
    }
}
