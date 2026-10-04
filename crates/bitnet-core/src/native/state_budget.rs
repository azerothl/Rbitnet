//! Conservative state reservation before the MoE weight/cache planner. The
//! allocator still enforces the exact live cap, including optional snapshots.
use super::{Config, Family};
use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;

fn overflow() -> BitNetError {
    BitNetError::Inference("CUDA state reservation overflow".into())
}
fn product(v: &[usize]) -> Result<usize> {
    v.iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v).ok_or_else(overflow))
}
fn sum(v: &[usize]) -> Result<usize> {
    v.iter()
        .try_fold(0usize, |n, &v| n.checked_add(v).ok_or_else(overflow))
}
pub(super) fn reserve(a: &GgufArchive, c: &Config) -> Result<usize> {
    let attention = !matches!(
        std::env::var("RBITNET_CUDA_ATTENTION").as_deref(),
        Ok("0" | "false" | "no")
    );
    let (kv_heads, key, value) = if c.family == Family::Mla {
        (1, sum(&[c.kv_rank, c.rotary])?, c.kv_rank)
    } else {
        (c.kv_heads, c.head, c.value)
    };
    let kv = if attention {
        product(&[c.layers, c.max_seq, kv_heads, sum(&[key, value])?])?
    } else {
        0
    };
    let blocks = c.vocab.div_ceil(256);
    let head = sum(&[product(&[3, c.embd])?, c.vocab, product(&[2, blocks])?, 2])?;
    let partial = sum(&[
        kv,
        if attention {
            product(&[c.layers, c.heads, sum(&[key, value, 1])?])?
        } else {
            0
        },
        if std::env::var("RBITNET_CUDA_HEAD").as_deref() == Ok("1") {
            head
        } else {
            0
        },
    ])?;
    let full = if attention
        && c.family == Family::GptOss
        && std::env::var("RBITNET_CUDA_GPT_FULL").as_deref() == Ok("1")
    {
        let qs = product(&[c.heads, c.head])?;
        let ks = product(&[c.kv_heads, c.head])?;
        let common = sum(&[
            product(&[4, c.embd])?,
            product(&[2, qs])?,
            product(&[2, ks])?,
            c.experts,
            c.rotary / 2,
            product(&[c.max_seq, c.rotary])?,
            c.vocab,
            product(&[2, blocks])?,
            3,
        ])?;
        // Reserve optional selection bias too; the actual ledger counts only
        // copies that were made, without counting borrowed matrices again.
        let per_layer = sum(&[
            product(&[3, c.embd])?,
            qs,
            product(&[2, ks])?,
            product(&[2, c.experts])?,
            c.heads,
        ])?;
        let split = if std::env::var("RBITNET_CUDA_SPLIT_KV").as_deref() == Ok("1") {
            product(&[c.heads, c.max_seq.div_ceil(256), sum(&[c.head, 2])?])?
        } else {
            0
        };
        sum(&[kv, common, product(&[c.layers, per_layer])?, split])?
    } else {
        0
    };
    let mut moe = 0usize;
    if std::env::var("RBITNET_CUDA_MOE").as_deref() != Ok("0") {
        for il in c.dense_layers..c.layers {
            let name = format!("blk.{il}.ffn_gate_exps.weight");
            let t = a
                .tensor_by_name(&name)
                .ok_or_else(|| BitNetError::Inference(format!("missing {name}")))?;
            let ff = usize::try_from(t.dimensions[1]).map_err(|_| overflow())?;
            let words = sum(&[
                product(&[2, c.embd])?,
                product(&[2, c.used, ff])?,
                product(&[c.used, c.embd])?,
                product(&[2, c.used])?,
            ])?;
            moe = sum(&[
                moe,
                product(&[words, 4])?,
                product(&[c.used, 3, std::mem::size_of::<usize>()])?,
            ])?;
            for p in ["gate", "up", "down"] {
                if let Some(t) = a.tensor_by_name(&format!("blk.{il}.ffn_{p}_exps.bias")) {
                    let dims = t
                        .dimensions
                        .iter()
                        .map(|&d| usize::try_from(d).map_err(|_| overflow()))
                        .collect::<Result<Vec<_>>>()?;
                    moe = sum(&[moe, product(&[product(&dims)?, 4])?])?;
                }
            }
        }
    }
    // The partial graph can execute independent slab GEMVs on Rayon workers.
    // Reserve their small input/output scratch, never a full expert-bank copy.
    let mut input = c.embd;
    let mut output = c.vocab;
    for t in &a.tensors {
        if t.dimensions.len() < 2 || t.name.contains("nextn") {
            continue;
        }
        let dims = t
            .dimensions
            .iter()
            .map(|&d| usize::try_from(d).map_err(|_| overflow()))
            .collect::<Result<Vec<_>>>()?;
        let batches = if dims.len() == 3 && !t.name.contains("_exps.") {
            dims[2]
        } else {
            1
        };
        input = input.max(product(&[dims[0], batches])?);
        output = output.max(product(&[dims[1], batches])?);
    }
    let scratch = product(&[sum(&[input, output])?, 4, rayon::current_num_threads()])?;
    sum(&[moe, product(&[partial.max(full), 4])?, scratch])
}

#[cfg(test)]
mod tests {
    #[test]
    fn reservation_overflow_is_rejected() {
        assert!(super::product(&[usize::MAX, 2]).is_err());
        assert!(super::sum(&[usize::MAX, 1]).is_err());
    }
}
