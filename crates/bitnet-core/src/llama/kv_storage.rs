//! KV layouts for Llama: dense (legacy) and paged (Inference stack v2 phase A).

use crate::error::{BitNetError, Result};

use super::config::LlamaConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum KvBackendKind {
    DenseCpu,
    PagedCpu,
    PagedGpuPlanned,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct KvBackendStats {
    pub backend: &'static str,
    pub quant_format: &'static str,
    pub page_tokens: Option<usize>,
    pub max_pages: Option<usize>,
    pub physical_pages: usize,
    pub new_phys_pages: usize,
    pub reused_phys_pages: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum KvQuantFormat {
    F32,
    Q8,
    Q4,
}

impl KvQuantFormat {
    pub fn from_env() -> Self {
        match std::env::var("RBITNET_KV_QUANT")
            .unwrap_or_else(|_| "off".into())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "q8" | "int8" => Self::Q8,
            "q4" | "int4" => Self::Q4,
            _ => Self::F32,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q8 => "q8",
            Self::Q4 => "q4",
        }
    }

    fn row_bytes(self, len: usize) -> usize {
        match self {
            Self::F32 => len * std::mem::size_of::<f32>(),
            Self::Q8 => 4 + len,
            Self::Q4 => 4 + (len + 1) / 2,
        }
    }
}

/// Dense per-layer KV buffer (legacy layout).
pub struct KvCache {
    /// Per layer: flattened `k` / `v` with stride `n_kv * head_dim` per sequence position.
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(cfg: &LlamaConfig) -> Self {
        let stride = cfg.n_kv * cfg.head_dim;
        let len = stride * cfg.max_seq;
        let k = (0..cfg.n_layer).map(|_| vec![0.0f32; len]).collect();
        let v = (0..cfg.n_layer).map(|_| vec![0.0f32; len]).collect();
        Self { k, v }
    }

    pub fn clear(&mut self) {
        for row in &mut self.k {
            row.fill(0.0);
        }
        for row in &mut self.v {
            row.fill(0.0);
        }
    }
}

/// Allocation counters for paged KV (phase A.2 observability).
#[derive(Debug, Clone, Default)]
pub struct KvPoolStats {
    pub new_phys_pages: usize,
    pub reused_phys_pages: usize,
}

/// Single-sequence paged KV: logical token positions map to fixed-size physical pages **per layer**.
#[derive(Debug)]
pub struct PagedSeqKv {
    n_layer: usize,
    stride: usize,
    page_tokens: usize,
    max_pages: usize,
    stats: KvPoolStats,
    quant_format: KvQuantFormat,
    /// Per layer: physical pages; each slab holds `stride * page_tokens` floats.
    phys_k: Vec<Vec<Vec<f32>>>,
    phys_v: Vec<Vec<Vec<f32>>>,
    /// Quantized page storage, used when `quant_format != F32`; row layout stores scale then packed values.
    phys_k_q: Vec<Vec<Vec<u8>>>,
    phys_v_q: Vec<Vec<Vec<u8>>>,
    free_ids: Vec<Vec<usize>>,
    /// Per layer: logical_block_index -> physical page index (`usize::MAX` = unassigned).
    block_phys: Vec<Vec<usize>>,
}

impl PagedSeqKv {
    pub fn new(cfg: &LlamaConfig, page_tokens: usize, max_pages: usize) -> Result<Self> {
        if page_tokens == 0 {
            return Err(BitNetError::Inference(
                "paged KV: page_tokens must be >= 1".into(),
            ));
        }
        if max_pages == 0 {
            return Err(BitNetError::Inference(
                "paged KV: max_pages must be >= 1".into(),
            ));
        }
        let stride = cfg.n_kv * cfg.head_dim;
        let n_layer = cfg.n_layer;
        let quant_format = KvQuantFormat::from_env();
        Ok(Self {
            n_layer,
            stride,
            page_tokens,
            max_pages,
            quant_format,
            phys_k: (0..n_layer).map(|_| Vec::new()).collect(),
            phys_v: (0..n_layer).map(|_| Vec::new()).collect(),
            phys_k_q: (0..n_layer).map(|_| Vec::new()).collect(),
            phys_v_q: (0..n_layer).map(|_| Vec::new()).collect(),
            free_ids: (0..n_layer).map(|_| Vec::new()).collect(),
            block_phys: (0..n_layer).map(|_| Vec::new()).collect(),
            stats: KvPoolStats::default(),
        })
    }

    fn alloc_phys(&mut self, layer: usize) -> Result<usize> {
        if let Some(id) = self.free_ids[layer].pop() {
            self.phys_k[layer][id].fill(0.0);
            self.phys_v[layer][id].fill(0.0);
            if let Some(slab) = self.phys_k_q[layer].get_mut(id) {
                slab.fill(0);
            }
            if let Some(slab) = self.phys_v_q[layer].get_mut(id) {
                slab.fill(0);
            }
            self.stats.reused_phys_pages += 1;
            return Ok(id);
        }
        if self.phys_k[layer].len() >= self.max_pages {
            return Err(BitNetError::Inference(format!(
                "paged KV: exhausted physical pages for layer {layer} (max_pages={})",
                self.max_pages
            )));
        }
        let len = self.stride * self.page_tokens;
        let id = self.phys_k[layer].len();
        self.phys_k[layer].push(vec![0.0f32; len]);
        self.phys_v[layer].push(vec![0.0f32; len]);
        let q_len = self.quant_format.row_bytes(self.stride) * self.page_tokens;
        self.phys_k_q[layer].push(vec![0u8; q_len]);
        self.phys_v_q[layer].push(vec![0u8; q_len]);
        self.stats.new_phys_pages += 1;
        Ok(id)
    }

    fn ensure_logical_block(&mut self, layer: usize, logical_block: usize) -> Result<usize> {
        while self.block_phys[layer].len() <= logical_block {
            self.block_phys[layer].push(usize::MAX);
        }
        if self.block_phys[layer][logical_block] == usize::MAX {
            let pid = self.alloc_phys(layer)?;
            self.block_phys[layer][logical_block] = pid;
        }
        Ok(self.block_phys[layer][logical_block])
    }

    /// Write full K/V rows for `pos` (stride-wide slices).
    pub fn write_kv_layer(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> Result<()> {
        if k.len() != self.stride || v.len() != self.stride {
            return Err(BitNetError::Inference(
                "paged KV write: bad slice len".into(),
            ));
        }
        let lb = pos / self.page_tokens;
        let pid = self.ensure_logical_block(layer, lb)?;
        let off_in_page = (pos % self.page_tokens) * self.stride;
        match self.quant_format {
            KvQuantFormat::F32 => {
                let slab_k = &mut self.phys_k[layer][pid];
                let slab_v = &mut self.phys_v[layer][pid];
                slab_k[off_in_page..off_in_page + self.stride].copy_from_slice(k);
                slab_v[off_in_page..off_in_page + self.stride].copy_from_slice(v);
            }
            fmt => {
                let row_bytes = fmt.row_bytes(self.stride);
                let q_off = (pos % self.page_tokens) * row_bytes;
                encode_quant_row(
                    fmt,
                    k,
                    &mut self.phys_k_q[layer][pid][q_off..q_off + row_bytes],
                );
                encode_quant_row(
                    fmt,
                    v,
                    &mut self.phys_v_q[layer][pid][q_off..q_off + row_bytes],
                );
            }
        }
        Ok(())
    }

    pub fn k_head_slice(
        &self,
        layer: usize,
        pos: usize,
        kv_head: usize,
        head_dim: usize,
    ) -> &[f32] {
        let lb = pos / self.page_tokens;
        let pid = self.block_phys[layer][lb];
        if self.quant_format != KvQuantFormat::F32 {
            panic!("k_head_slice is only available for f32 KV; use fill_k_head_values");
        }
        let off_in_page = (pos % self.page_tokens) * self.stride + kv_head * head_dim;
        &self.phys_k[layer][pid][off_in_page..off_in_page + head_dim]
    }

    pub fn v_head_slice(
        &self,
        layer: usize,
        pos: usize,
        kv_head: usize,
        head_dim: usize,
    ) -> &[f32] {
        let lb = pos / self.page_tokens;
        let pid = self.block_phys[layer][lb];
        if self.quant_format != KvQuantFormat::F32 {
            panic!("v_head_slice is only available for f32 KV; use fill_v_head_values");
        }
        let off_in_page = (pos % self.page_tokens) * self.stride + kv_head * head_dim;
        &self.phys_v[layer][pid][off_in_page..off_in_page + head_dim]
    }

    pub fn fill_k_head_values(
        &self,
        layer: usize,
        pos: usize,
        kv_head: usize,
        head_dim: usize,
        out: &mut [f32],
    ) {
        match self.quant_format {
            KvQuantFormat::F32 => {
                out.copy_from_slice(self.k_head_slice(layer, pos, kv_head, head_dim))
            }
            fmt => self.decode_quant_head(layer, pos, kv_head, head_dim, true, fmt, out),
        }
    }

    pub fn fill_v_head_values(
        &self,
        layer: usize,
        pos: usize,
        kv_head: usize,
        head_dim: usize,
        out: &mut [f32],
    ) {
        match self.quant_format {
            KvQuantFormat::F32 => {
                out.copy_from_slice(self.v_head_slice(layer, pos, kv_head, head_dim))
            }
            fmt => self.decode_quant_head(layer, pos, kv_head, head_dim, false, fmt, out),
        }
    }

    fn decode_quant_head(
        &self,
        layer: usize,
        pos: usize,
        kv_head: usize,
        head_dim: usize,
        is_k: bool,
        fmt: KvQuantFormat,
        out: &mut [f32],
    ) {
        let lb = pos / self.page_tokens;
        let pid = self.block_phys[layer][lb];
        let row_bytes = fmt.row_bytes(self.stride);
        let row_off = (pos % self.page_tokens) * row_bytes;
        let slab = if is_k {
            &self.phys_k_q[layer][pid]
        } else {
            &self.phys_v_q[layer][pid]
        };
        let mut row = vec![0.0f32; self.stride];
        decode_quant_row(fmt, &slab[row_off..row_off + row_bytes], &mut row);
        let start = kv_head * head_dim;
        out.copy_from_slice(&row[start..start + head_dim]);
    }

    pub fn clear(&mut self) {
        for layer in 0..self.n_layer {
            for &pid in self.block_phys[layer].iter() {
                if pid != usize::MAX && pid < self.phys_k[layer].len() {
                    self.free_ids[layer].push(pid);
                }
            }
            self.phys_k[layer].iter_mut().for_each(|s| s.fill(0.0));
            self.phys_v[layer].iter_mut().for_each(|s| s.fill(0.0));
            self.block_phys[layer].clear();
        }
    }

    pub fn page_tokens(&self) -> usize {
        self.page_tokens
    }

    pub fn max_pages(&self) -> usize {
        self.max_pages
    }

    pub fn physical_counts(&self) -> Vec<usize> {
        (0..self.n_layer).map(|l| self.phys_k[l].len()).collect()
    }

    pub fn pool_stats(&self) -> KvPoolStats {
        self.stats.clone()
    }

    pub fn quant_format(&self) -> KvQuantFormat {
        self.quant_format
    }
}

/// Llama KV backing store: dense mmap-style buffer or paged slabs.
pub enum KvStorage {
    Dense(KvCache),
    Paged(PagedSeqKv),
}

impl KvStorage {
    pub fn new_dense(cfg: &LlamaConfig) -> Self {
        Self::Dense(KvCache::new(cfg))
    }

    pub fn new_paged(cfg: &LlamaConfig, page_tokens: usize, max_pages: usize) -> Result<Self> {
        Ok(Self::Paged(PagedSeqKv::new(cfg, page_tokens, max_pages)?))
    }

    pub fn clear(&mut self) {
        match self {
            Self::Dense(kv) => kv.clear(),
            Self::Paged(p) => p.clear(),
        }
    }

    pub fn write_layer_kv(
        &mut self,
        layer: usize,
        pos: usize,
        k: &[f32],
        v: &[f32],
        stride: usize,
    ) -> Result<()> {
        match self {
            Self::Dense(kv) => {
                let off = pos * stride;
                kv.k[layer][off..off + stride].copy_from_slice(k);
                kv.v[layer][off..off + stride].copy_from_slice(v);
                crate::perf::record_kv_write(
                    2usize
                        .saturating_mul(stride)
                        .saturating_mul(std::mem::size_of::<f32>()),
                );
                crate::perf::record_kv_backend(kv.k.len(), 0, 0, 0);
                Ok(())
            }
            Self::Paged(p) => {
                p.write_kv_layer(layer, pos, k, v)?;
                let pool = p.pool_stats();
                crate::perf::record_kv_backend(
                    p.physical_counts().iter().sum(),
                    pool.new_phys_pages,
                    pool.reused_phys_pages,
                    match p.quant_format() {
                        KvQuantFormat::F32 => 0,
                        KvQuantFormat::Q8 => 1,
                        KvQuantFormat::Q4 => 2,
                    },
                );
                crate::perf::record_kv_write(
                    2usize.saturating_mul(p.quant_format().row_bytes(stride)),
                );
                Ok(())
            }
        }
    }

    pub fn k_head_slice(
        &self,
        layer: usize,
        pos: usize,
        kv_head: usize,
        head_dim: usize,
        stride: usize,
    ) -> &[f32] {
        match self {
            Self::Dense(kv) => {
                let off = pos * stride + kv_head * head_dim;
                &kv.k[layer][off..off + head_dim]
            }
            Self::Paged(p) => p.k_head_slice(layer, pos, kv_head, head_dim),
        }
    }

    pub fn v_head_slice(
        &self,
        layer: usize,
        pos: usize,
        kv_head: usize,
        head_dim: usize,
        stride: usize,
    ) -> &[f32] {
        match self {
            Self::Dense(kv) => {
                let off = pos * stride + kv_head * head_dim;
                &kv.v[layer][off..off + head_dim]
            }
            Self::Paged(p) => p.v_head_slice(layer, pos, kv_head, head_dim),
        }
    }

    pub fn fill_k_head_values(
        &self,
        layer: usize,
        pos: usize,
        kv_head: usize,
        head_dim: usize,
        stride: usize,
        out: &mut [f32],
    ) {
        match self {
            Self::Dense(kv) => {
                let off = pos * stride + kv_head * head_dim;
                out.copy_from_slice(&kv.k[layer][off..off + head_dim]);
            }
            Self::Paged(p) => p.fill_k_head_values(layer, pos, kv_head, head_dim, out),
        }
    }

    pub fn fill_v_head_values(
        &self,
        layer: usize,
        pos: usize,
        kv_head: usize,
        head_dim: usize,
        stride: usize,
        out: &mut [f32],
    ) {
        match self {
            Self::Dense(kv) => {
                let off = pos * stride + kv_head * head_dim;
                out.copy_from_slice(&kv.v[layer][off..off + head_dim]);
            }
            Self::Paged(p) => p.fill_v_head_values(layer, pos, kv_head, head_dim, out),
        }
    }

    /// Copy rows `0..=pos` for `kv_head` into `dst` layout `[pos+1][head_dim]` row-major (matches former GPU path).
    pub fn fill_k_rows_gpu(
        &self,
        layer: usize,
        pos: usize,
        kv_head: usize,
        head_dim: usize,
        stride: usize,
        dst: &mut [f32],
    ) {
        for p in 0..=pos {
            let row_off = p * head_dim;
            self.fill_k_head_values(
                layer,
                p,
                kv_head,
                head_dim,
                stride,
                &mut dst[row_off..row_off + head_dim],
            );
        }
    }

    pub fn attention_scores_cpu(
        &self,
        layer: usize,
        pos: usize,
        kv_head: usize,
        head_dim: usize,
        stride: usize,
        q: &[f32],
        scale: f32,
        out: &mut [f32],
    ) {
        let mut k_tmp = vec![0.0f32; head_dim];
        for p in 0..=pos {
            self.fill_k_head_values(layer, p, kv_head, head_dim, stride, &mut k_tmp);
            let dot: f32 = q.iter().zip(k_tmp.iter()).map(|(a, b)| a * b).sum();
            out[p] = dot * scale;
        }
    }

    pub fn backend_kind(&self) -> KvBackendKind {
        match self {
            Self::Dense(_) => KvBackendKind::DenseCpu,
            Self::Paged(_) => {
                if matches!(
                    std::env::var("RBITNET_KV_BACKEND").as_deref(),
                    Ok("gpu") | Ok("cuda")
                ) {
                    KvBackendKind::PagedGpuPlanned
                } else {
                    KvBackendKind::PagedCpu
                }
            }
        }
    }

    pub fn stats(&self) -> KvBackendStats {
        match self {
            Self::Dense(kv) => KvBackendStats {
                backend: "dense_cpu",
                quant_format: "f32",
                physical_pages: kv.k.len(),
                ..Default::default()
            },
            Self::Paged(p) => {
                let pool = p.pool_stats();
                KvBackendStats {
                    backend: match self.backend_kind() {
                        KvBackendKind::PagedGpuPlanned => "paged_gpu_planned",
                        _ => "paged_cpu",
                    },
                    page_tokens: Some(p.page_tokens()),
                    max_pages: Some(p.max_pages()),
                    quant_format: p.quant_format().as_str(),
                    physical_pages: p.physical_counts().iter().sum(),
                    new_phys_pages: pool.new_phys_pages,
                    reused_phys_pages: pool.reused_phys_pages,
                }
            }
        }
    }

    pub fn as_paged_mut(&mut self) -> Option<&mut PagedSeqKv> {
        match self {
            Self::Paged(p) => Some(p),
            Self::Dense(_) => None,
        }
    }
}

fn encode_quant_row(fmt: KvQuantFormat, src: &[f32], dst: &mut [u8]) {
    let max_abs = src.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let levels = match fmt {
        KvQuantFormat::F32 => unreachable!(),
        KvQuantFormat::Q8 => 127.0,
        KvQuantFormat::Q4 => 7.0,
    };
    let scale = if max_abs > 0.0 { max_abs / levels } else { 1.0 };
    dst[..4].copy_from_slice(&scale.to_le_bytes());
    match fmt {
        KvQuantFormat::Q8 => {
            for (i, &v) in src.iter().enumerate() {
                dst[4 + i] = ((v / scale).round().clamp(-127.0, 127.0) as i8) as u8;
            }
        }
        KvQuantFormat::Q4 => {
            for (i, pair) in src.chunks(2).enumerate() {
                let a = quant4(pair[0], scale);
                let b = pair.get(1).map(|&v| quant4(v, scale)).unwrap_or(0);
                dst[4 + i] = (a & 0x0f) | ((b & 0x0f) << 4);
            }
        }
        KvQuantFormat::F32 => unreachable!(),
    }
}

fn decode_quant_row(fmt: KvQuantFormat, src: &[u8], dst: &mut [f32]) {
    let scale = f32::from_le_bytes(src[..4].try_into().unwrap_or([0, 0, 128, 63]));
    match fmt {
        KvQuantFormat::Q8 => {
            for i in 0..dst.len() {
                dst[i] = (src[4 + i] as i8 as f32) * scale;
            }
        }
        KvQuantFormat::Q4 => {
            for i in 0..dst.len() {
                let byte = src[4 + i / 2];
                let nibble = if i % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                dst[i] = dequant4(nibble as i8) as f32 * scale;
            }
        }
        KvQuantFormat::F32 => unreachable!(),
    }
}

fn quant4(v: f32, scale: f32) -> u8 {
    ((v / scale).round().clamp(-7.0, 7.0) as i8 as u8) & 0x0f
}

fn dequant4(v: i8) -> i8 {
    if v >= 8 {
        v - 16
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_roundtrip_preserves_row_shape() {
        let src = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let mut packed = vec![0u8; KvQuantFormat::Q8.row_bytes(src.len())];
        let mut out = vec![0.0f32; src.len()];
        encode_quant_row(KvQuantFormat::Q8, &src, &mut packed);
        decode_quant_row(KvQuantFormat::Q8, &packed, &mut out);
        assert_eq!(out.len(), src.len());
        assert!((out[0] + 1.0).abs() < 0.02);
        assert!((out[4] - 1.0).abs() < 0.02);
    }

    #[test]
    fn q4_roundtrip_preserves_sign() {
        let src = [-1.0, -0.25, 0.25, 1.0];
        let mut packed = vec![0u8; KvQuantFormat::Q4.row_bytes(src.len())];
        let mut out = vec![0.0f32; src.len()];
        encode_quant_row(KvQuantFormat::Q4, &src, &mut packed);
        decode_quant_row(KvQuantFormat::Q4, &packed, &mut out);
        assert!(out[0] < 0.0);
        assert!(out[1] < 0.0);
        assert!(out[2] > 0.0);
        assert!(out[3] > 0.0);
    }
}
