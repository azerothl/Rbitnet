//! CPU CLIP ViT + LLaVA MLP projector encoder.

use std::path::Path;

use rayon::prelude::*;

use crate::error::{BitNetError, Result};
use crate::ggml::tensor_to_f32;
use crate::gguf::{GgufArchive, GgufTensorInfo};

use super::config::{MmprojConfig, VitFfnOp};
use super::image_io::{decode_and_normalize, NormalizedImage};

/// Dense linear layer: `y = x @ W^T + b` with GGML layout `W[ne0=in, ne1=out]`.
#[derive(Debug, Clone)]
struct Linear {
    /// Row-major `[out, in]` packed as `w[o * in + i]` (same as GGML `ne[0]=in`, `ne[1]=out`).
    w: Vec<f32>,
    bias: Option<Vec<f32>>,
    n_in: usize,
    n_out: usize,
}

impl Linear {
    fn from_tensors(
        archive: &GgufArchive,
        weight: &GgufTensorInfo,
        bias: Option<&GgufTensorInfo>,
    ) -> Result<Self> {
        if weight.dimensions.len() < 2 {
            return Err(BitNetError::InvalidGguf(format!(
                "linear weight {} expected ≥2 dims, got {:?}",
                weight.name, weight.dimensions
            )));
        }
        let n_in = weight.dimensions[0] as usize;
        let n_out = weight.dimensions[1] as usize;
        let payload = archive.tensor_payload(weight)?;
        let w = tensor_to_f32(payload, weight.ggml_type, &weight.dimensions)?;
        if w.len() != n_in * n_out {
            return Err(BitNetError::InvalidGguf(format!(
                "linear {}: dequant len {} != {}*{}",
                weight.name,
                w.len(),
                n_in,
                n_out
            )));
        }
        let bias = match bias {
            Some(b) => {
                let bp = archive.tensor_payload(b)?;
                let bv = tensor_to_f32(bp, b.ggml_type, &b.dimensions)?;
                if bv.len() != n_out {
                    return Err(BitNetError::InvalidGguf(format!(
                        "bias {}: len {} != out {n_out}",
                        b.name,
                        bv.len()
                    )));
                }
                Some(bv)
            }
            None => None,
        };
        Ok(Self {
            w,
            bias,
            n_in,
            n_out,
        })
    }

    fn apply_row(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.n_in);
        debug_assert_eq!(y.len(), self.n_out);
        for o in 0..self.n_out {
            let mut acc = 0.0f32;
            let row = &self.w[o * self.n_in..(o + 1) * self.n_in];
            for i in 0..self.n_in {
                acc += row[i] * x[i];
            }
            if let Some(ref b) = self.bias {
                acc += b[o];
            }
            y[o] = acc;
        }
    }

    fn apply_rows(&self, xs: &[f32], n_rows: usize, ys: &mut [f32]) {
        debug_assert_eq!(xs.len(), n_rows * self.n_in);
        debug_assert_eq!(ys.len(), n_rows * self.n_out);
        ys.par_chunks_mut(self.n_out)
            .zip(xs.par_chunks(self.n_in))
            .for_each(|(y, x)| self.apply_row(x, y));
    }
}

#[derive(Debug, Clone)]
struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    eps: f32,
}

impl LayerNorm {
    fn from_tensors(
        archive: &GgufArchive,
        w: &GgufTensorInfo,
        b: &GgufTensorInfo,
        eps: f32,
    ) -> Result<Self> {
        let wp = archive.tensor_payload(w)?;
        let bp = archive.tensor_payload(b)?;
        let weight = tensor_to_f32(wp, w.ggml_type, &w.dimensions)?;
        let bias = tensor_to_f32(bp, b.ggml_type, &b.dimensions)?;
        if weight.len() != bias.len() {
            return Err(BitNetError::InvalidGguf(format!(
                "layernorm {} / {} length mismatch",
                w.name, b.name
            )));
        }
        Ok(Self { weight, bias, eps })
    }

    fn apply_inplace(&self, x: &mut [f32]) {
        let n = x.len();
        debug_assert_eq!(n, self.weight.len());
        let mean = x.iter().sum::<f32>() / n as f32;
        let mut var = 0.0f32;
        for v in x.iter() {
            let d = *v - mean;
            var += d * d;
        }
        var /= n as f32;
        let inv = 1.0 / (var + self.eps).sqrt();
        for i in 0..n {
            x[i] = (x[i] - mean) * inv * self.weight[i] + self.bias[i];
        }
    }

    fn apply_rows_inplace(&self, xs: &mut [f32], n_embd: usize) {
        xs.par_chunks_mut(n_embd).for_each(|row| self.apply_inplace(row));
    }
}

#[derive(Debug, Clone)]
struct VitBlock {
    ln1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln2: LayerNorm,
    ffn_up: Linear,
    ffn_down: Linear,
}

/// Loaded mmproj encoder: CLIP ViT-L + LLaVA MLP projector.
#[derive(Debug, Clone)]
pub struct MmprojEncoder {
    pub config: MmprojConfig,
    patch_embd: Vec<f32>, // [n_embd, patch_dim] with patch_dim = p*p*3
    patch_dim: usize,
    class_embd: Vec<f32>,
    position_embd: Vec<f32>, // [n_pos, n_embd]
    pre_ln: Option<LayerNorm>,
    blocks: Vec<VitBlock>,
    mm0: Linear,
    mm2: Linear,
}

impl MmprojEncoder {
    pub fn load(path: &Path) -> Result<Self> {
        let archive = GgufArchive::mmap_path(path).map_err(|e| {
            BitNetError::InvalidGguf(format!("mmproj mmap {}: {e}", path.display()))
        })?;
        Self::from_archive(&archive)
    }

    pub fn from_archive(archive: &GgufArchive) -> Result<Self> {
        let config = MmprojConfig::from_gguf(archive)?;
        if !config.has_llava_projector {
            return Err(BitNetError::InvalidGguf(
                "mmproj missing clip.has_llava_projector (LLaVA MLP path required)".into(),
            ));
        }

        let patch_info = must_tensor(archive, "v.patch_embd.weight")?;
        // dims [p, p, 3, n_embd]
        if patch_info.dimensions.len() != 4 {
            return Err(BitNetError::InvalidGguf(format!(
                "v.patch_embd.weight expected 4 dims, got {:?}",
                patch_info.dimensions
            )));
        }
        let p0 = patch_info.dimensions[0] as usize;
        let p1 = patch_info.dimensions[1] as usize;
        let c = patch_info.dimensions[2] as usize;
        let n_embd_w = patch_info.dimensions[3] as usize;
        if p0 != config.patch_size || p1 != config.patch_size || c != 3 || n_embd_w != config.n_embd
        {
            return Err(BitNetError::InvalidGguf(format!(
                "v.patch_embd.weight dims {:?} mismatch patch_size={} n_embd={}",
                patch_info.dimensions, config.patch_size, config.n_embd
            )));
        }
        let patch_dim = p0 * p1 * c;
        let patch_payload = archive.tensor_payload(patch_info)?;
        let patch_embd = tensor_to_f32(
            patch_payload,
            patch_info.ggml_type,
            &patch_info.dimensions,
        )?;

        let class_info = must_tensor(archive, "v.class_embd")?;
        let class_payload = archive.tensor_payload(class_info)?;
        let class_embd = tensor_to_f32(
            class_payload,
            class_info.ggml_type,
            &class_info.dimensions,
        )?;
        if class_embd.len() != config.n_embd {
            return Err(BitNetError::InvalidGguf(format!(
                "v.class_embd len {} != n_embd {}",
                class_embd.len(),
                config.n_embd
            )));
        }

        let pos_info = must_tensor(archive, "v.position_embd.weight")?;
        let pos_payload = archive.tensor_payload(pos_info)?;
        let position_embd =
            tensor_to_f32(pos_payload, pos_info.ggml_type, &pos_info.dimensions)?;
        // dims [n_embd, n_pos]
        if pos_info.dimensions.len() < 2
            || pos_info.dimensions[0] as usize != config.n_embd
        {
            return Err(BitNetError::InvalidGguf(format!(
                "v.position_embd.weight unexpected dims {:?}",
                pos_info.dimensions
            )));
        }
        let n_pos_w = pos_info.dimensions[1] as usize;
        let n_patches = config.n_patches();
        let n_pos = n_patches + 1;
        if n_pos_w < n_pos {
            return Err(BitNetError::InvalidGguf(format!(
                "position_embd has {n_pos_w} positions, need at least {n_pos}"
            )));
        }

        let pre_ln = match (
            archive.tensor_by_name("v.pre_ln.weight"),
            archive.tensor_by_name("v.pre_ln.bias"),
        ) {
            (Some(w), Some(b)) => Some(LayerNorm::from_tensors(
                archive,
                w,
                b,
                config.layer_norm_eps,
            )?),
            (None, None) => None,
            _ => {
                return Err(BitNetError::InvalidGguf(
                    "v.pre_ln weight/bias pair incomplete".into(),
                ))
            }
        };

        let mut blocks = Vec::with_capacity(config.n_layer);
        for il in 0..config.n_layer {
            let p = format!("v.blk.{il}");
            let ln1 = LayerNorm::from_tensors(
                archive,
                must_tensor(archive, &format!("{p}.ln1.weight"))?,
                must_tensor(archive, &format!("{p}.ln1.bias"))?,
                config.layer_norm_eps,
            )?;
            let ln2 = LayerNorm::from_tensors(
                archive,
                must_tensor(archive, &format!("{p}.ln2.weight"))?,
                must_tensor(archive, &format!("{p}.ln2.bias"))?,
                config.layer_norm_eps,
            )?;
            let q = Linear::from_tensors(
                archive,
                must_tensor(archive, &format!("{p}.attn_q.weight"))?,
                archive.tensor_by_name(&format!("{p}.attn_q.bias")),
            )?;
            let k = Linear::from_tensors(
                archive,
                must_tensor(archive, &format!("{p}.attn_k.weight"))?,
                archive.tensor_by_name(&format!("{p}.attn_k.bias")),
            )?;
            let v = Linear::from_tensors(
                archive,
                must_tensor(archive, &format!("{p}.attn_v.weight"))?,
                archive.tensor_by_name(&format!("{p}.attn_v.bias")),
            )?;
            let o = Linear::from_tensors(
                archive,
                must_tensor(archive, &format!("{p}.attn_out.weight"))?,
                archive.tensor_by_name(&format!("{p}.attn_out.bias")),
            )?;

            let mut ffn_up = Linear::from_tensors(
                archive,
                must_tensor(archive, &format!("{p}.ffn_up.weight"))?,
                archive.tensor_by_name(&format!("{p}.ffn_up.bias")),
            )?;
            let mut ffn_down = Linear::from_tensors(
                archive,
                must_tensor(archive, &format!("{p}.ffn_down.weight"))?,
                archive.tensor_by_name(&format!("{p}.ffn_down.bias")),
            )?;
            // Legacy LLaVA export: names swapped when ffn_down.ne[0] == n_embd
            // (llama.cpp clip.cpp).
            if ffn_down.n_in == config.n_embd {
                std::mem::swap(&mut ffn_up, &mut ffn_down);
            }
            if ffn_up.n_in != config.n_embd || ffn_up.n_out != config.n_ff {
                return Err(BitNetError::InvalidGguf(format!(
                    "blk.{il} ffn_up shape [{}→{}] expected [{}→{}]",
                    ffn_up.n_in, ffn_up.n_out, config.n_embd, config.n_ff
                )));
            }
            if ffn_down.n_in != config.n_ff || ffn_down.n_out != config.n_embd {
                return Err(BitNetError::InvalidGguf(format!(
                    "blk.{il} ffn_down shape [{}→{}] expected [{}→{}]",
                    ffn_down.n_in, ffn_down.n_out, config.n_ff, config.n_embd
                )));
            }

            blocks.push(VitBlock {
                ln1,
                q,
                k,
                v,
                o,
                ln2,
                ffn_up,
                ffn_down,
            });
        }

        let mm0 = Linear::from_tensors(
            archive,
            must_tensor(archive, "mm.0.weight")?,
            archive.tensor_by_name("mm.0.bias"),
        )?;
        let mm2 = Linear::from_tensors(
            archive,
            must_tensor(archive, "mm.2.weight")?,
            archive.tensor_by_name("mm.2.bias"),
        )?;
        if mm0.n_in != config.n_embd {
            return Err(BitNetError::InvalidGguf(format!(
                "mm.0.weight in_dim {} != n_embd {}",
                mm0.n_in, config.n_embd
            )));
        }

        Ok(Self {
            config,
            patch_embd,
            patch_dim,
            class_embd,
            position_embd,
            pre_ln,
            blocks,
            mm0,
            mm2,
        })
    }

    /// Projector output embedding width (e.g. 4096 for LLaVA-7B).
    pub fn proj_out_dim(&self) -> usize {
        self.mm2.n_out
    }

    pub fn n_patches(&self) -> usize {
        self.config.n_patches()
    }

    /// Encode JPEG/PNG bytes or a data-URL into row-major `[n_patches × proj_out]` f32.
    pub fn encode_image_bytes(&self, bytes_or_data_url: &[u8]) -> Result<Vec<f32>> {
        let img = decode_and_normalize(bytes_or_data_url, &self.config)?;
        self.encode_normalized(&img)
    }

    pub fn encode_normalized(&self, img: &NormalizedImage) -> Result<Vec<f32>> {
        let cfg = &self.config;
        if img.width != cfg.image_size || img.height != cfg.image_size {
            return Err(BitNetError::Inference(format!(
                "normalized image {}x{} != image_size {}",
                img.width, img.height, cfg.image_size
            )));
        }
        if img.chw.len() != 3 * cfg.image_size * cfg.image_size {
            return Err(BitNetError::Inference("normalized CHW length mismatch".into()));
        }

        let n_patches = cfg.n_patches();
        let n_embd = cfg.n_embd;
        let n_pos = n_patches + 1;
        let patch_side = cfg.image_size / cfg.patch_size;

        // Patch embed: im2col + matmul with flattened kernels [n_embd, patch_dim].
        let mut hidden = vec![0.0f32; n_pos * n_embd];
        // CLS at row 0
        hidden[..n_embd].copy_from_slice(&self.class_embd);

        let patch_dim = self.patch_dim;
        let patch_size = cfg.patch_size;
        let chw = &img.chw;
        let plane = cfg.image_size * cfg.image_size;

        // Fill patch rows 1..n_patches
        hidden[n_embd..]
            .par_chunks_mut(n_embd)
            .enumerate()
            .for_each(|(pi, out)| {
                let py = pi / patch_side;
                let px = pi % patch_side;
                let mut col = vec![0.0f32; patch_dim];
                // Order matches GGML weight [p,p,3,n_embd]: kx + p*(ky + p*(c + 3*oc))
                let mut idx = 0usize;
                for c in 0..3 {
                    let base = c * plane;
                    for ky in 0..patch_size {
                        let y = py * patch_size + ky;
                        for kx in 0..patch_size {
                            let x = px * patch_size + kx;
                            col[idx] = chw[base + y * cfg.image_size + x];
                            idx += 1;
                        }
                    }
                }
                // y[oc] = dot(W[oc], col)
                for oc in 0..n_embd {
                    let wrow = &self.patch_embd[oc * patch_dim..(oc + 1) * patch_dim];
                    let mut acc = 0.0f32;
                    for i in 0..patch_dim {
                        acc += wrow[i] * col[i];
                    }
                    out[oc] = acc;
                }
            });

        // Add position embeddings (rows 0..n_pos-1).
        for p in 0..n_pos {
            let pe = &self.position_embd[p * n_embd..(p + 1) * n_embd];
            let h = &mut hidden[p * n_embd..(p + 1) * n_embd];
            for i in 0..n_embd {
                h[i] += pe[i];
            }
        }

        if let Some(ref pre) = self.pre_ln {
            pre.apply_rows_inplace(&mut hidden, n_embd);
        }

        let max_feature_layer = cfg.max_feature_layer();
        let n_head = cfg.n_head;
        let d_head = cfg.head_dim();
        let scale = 1.0 / (d_head as f32).sqrt();

        for il in 0..max_feature_layer {
            let block = &self.blocks[il];
            let residual = hidden.clone();

            let mut normed = residual.clone();
            block.ln1.apply_rows_inplace(&mut normed, n_embd);

            let mut q = vec![0.0f32; n_pos * n_embd];
            let mut k = vec![0.0f32; n_pos * n_embd];
            let mut v = vec![0.0f32; n_pos * n_embd];
            block.q.apply_rows(&normed, n_pos, &mut q);
            block.k.apply_rows(&normed, n_pos, &mut k);
            block.v.apply_rows(&normed, n_pos, &mut v);

            let mut attn_out = vec![0.0f32; n_pos * n_embd];
            let per_head: Vec<Vec<f32>> = (0..n_head)
                .into_par_iter()
                .map(|h| {
                    let mut head_out = vec![0.0f32; n_pos * d_head];
                    let mut scores = vec![0.0f32; n_pos];
                    for qi in 0..n_pos {
                        let qh = &q[qi * n_embd + h * d_head..qi * n_embd + (h + 1) * d_head];
                        for kj in 0..n_pos {
                            let kh =
                                &k[kj * n_embd + h * d_head..kj * n_embd + (h + 1) * d_head];
                            let mut dot = 0.0f32;
                            for i in 0..d_head {
                                dot += qh[i] * kh[i];
                            }
                            scores[kj] = dot * scale;
                        }
                        softmax_inplace(&mut scores);
                        let out = &mut head_out[qi * d_head..(qi + 1) * d_head];
                        out.fill(0.0);
                        for kj in 0..n_pos {
                            let vh =
                                &v[kj * n_embd + h * d_head..kj * n_embd + (h + 1) * d_head];
                            let s = scores[kj];
                            for i in 0..d_head {
                                out[i] += s * vh[i];
                            }
                        }
                    }
                    head_out
                })
                .collect();
            for h in 0..n_head {
                for qi in 0..n_pos {
                    let src = &per_head[h][qi * d_head..(qi + 1) * d_head];
                    let dst = &mut attn_out
                        [qi * n_embd + h * d_head..qi * n_embd + (h + 1) * d_head];
                    dst.copy_from_slice(src);
                }
            }

            let mut proj = vec![0.0f32; n_pos * n_embd];
            block.o.apply_rows(&attn_out, n_pos, &mut proj);
            for i in 0..hidden.len() {
                hidden[i] = residual[i] + proj[i];
            }

            let residual2 = hidden.clone();
            let mut ffn_in = residual2.clone();
            block.ln2.apply_rows_inplace(&mut ffn_in, n_embd);
            let mut up = vec![0.0f32; n_pos * cfg.n_ff];
            block.ffn_up.apply_rows(&ffn_in, n_pos, &mut up);
            match cfg.ffn_op {
                VitFfnOp::GeluQuick => gelu_quick_inplace(&mut up),
                VitFfnOp::Gelu => gelu_inplace(&mut up),
            }
            let mut down = vec![0.0f32; n_pos * n_embd];
            block.ffn_down.apply_rows(&up, n_pos, &mut down);
            for i in 0..hidden.len() {
                hidden[i] = residual2[i] + down[i];
            }
        }

        // Skip CLS (row 0); project patch rows 1..n_patches.
        let patches = &hidden[n_embd..];
        let mut mid = vec![0.0f32; n_patches * self.mm0.n_out];
        self.mm0.apply_rows(patches, n_patches, &mut mid);
        gelu_inplace(&mut mid);
        let mut out = vec![0.0f32; n_patches * self.mm2.n_out];
        self.mm2.apply_rows(&mid, n_patches, &mut out);
        Ok(out)
    }
}

fn must_tensor<'a>(archive: &'a GgufArchive, name: &str) -> Result<&'a GgufTensorInfo> {
    archive.tensor_by_name(name).ok_or_else(|| {
        BitNetError::InvalidGguf(format!("mmproj missing tensor '{name}'"))
    })
}

fn softmax_inplace(scores: &mut [f32]) {
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        sum += *s;
    }
    if sum > 0.0 {
        for s in scores.iter_mut() {
            *s /= sum;
        }
    }
}

/// GGML `ggml_gelu_quick`: `x * sigmoid(1.702 * x)`.
pub(crate) fn gelu_quick_inplace(v: &mut [f32]) {
    for x in v {
        *x *= 1.0 / (1.0 + (-1.702 * *x).exp());
    }
}

/// GGML standard GELU (tanh approximation), used by the LLaVA MLP projector.
pub(crate) fn gelu_inplace(v: &mut [f32]) {
    const K: f32 = 0.797_884_560_802_865_4; // sqrt(2/pi)
    for x in v {
        let x3 = *x * *x * *x;
        *x = 0.5 * *x * (1.0 + (K * (*x + 0.044_715 * x3)).tanh());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_quick_zero_and_positive() {
        let mut v = vec![0.0f32, 1.0];
        gelu_quick_inplace(&mut v);
        assert!((v[0] - 0.0).abs() < 1e-6);
        // sigmoid(1.702) ≈ 0.8458 → ≈ 0.8458
        assert!((v[1] - 0.845_8).abs() < 1e-3);
    }

    #[test]
    fn layer_norm_unit_variance() {
        let ln = LayerNorm {
            weight: vec![1.0, 1.0, 1.0, 1.0],
            bias: vec![0.0, 0.0, 0.0, 0.0],
            eps: 1e-5,
        };
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        ln.apply_inplace(&mut x);
        let mean = x.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4);
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-3);
    }

    /// Optional end-to-end encode against a real mmproj GGUF.
    ///
    /// ```text
    /// RBITNET_TEST_MMPROJ=/tmp/rbitnet-vision/mmproj-model-f16.gguf \
    ///   cargo test -p bitnet-core mmproj_encode_real -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn mmproj_encode_real_llava_smoke() {
        let path = std::env::var("RBITNET_TEST_MMPROJ")
            .expect("set RBITNET_TEST_MMPROJ to an mmproj GGUF path");
        let enc = MmprojEncoder::load(Path::new(&path)).expect("load mmproj");
        // Tiny PNG
        let mut img = image::RgbImage::new(8, 8);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([(x * 17) as u8, (y * 31) as u8, 128]);
        }
        let mut png = Vec::new();
        let e = image::codecs::png::PngEncoder::new(&mut png);
        use image::ImageEncoder;
        e.write_image(img.as_raw(), 8, 8, image::ExtendedColorType::Rgb8)
            .unwrap();
        let out = enc.encode_image_bytes(&png).expect("encode");
        let n_patches = enc.n_patches();
        let dim = enc.proj_out_dim();
        assert_eq!(n_patches, 576);
        assert_eq!(dim, 4096);
        assert_eq!(out.len(), n_patches * dim);
        assert!(out.iter().all(|v| v.is_finite()));
        eprintln!("mmproj smoke: n_patches={n_patches} proj_out={dim}");
    }
}
