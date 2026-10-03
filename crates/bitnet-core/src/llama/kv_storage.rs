//! KV layouts for Llama: dense (legacy) and paged (Inference stack v2 phase A).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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

    /// Bytes for one token position (K or V row) at the given stride.
    pub fn bytes_per_token_row(self, stride: usize) -> usize {
        self.row_bytes(stride)
    }

    /// Approximate RSS for `n_pages` physical pages of K+V at this format.
    pub fn resident_bytes_for_pages(
        self,
        n_pages: usize,
        page_tokens: usize,
        stride: usize,
    ) -> usize {
        let per_page = self
            .row_bytes(stride)
            .saturating_mul(page_tokens)
            .saturating_mul(2);
        per_page.saturating_mul(n_pages)
    }
}

/// Dense per-layer KV buffer (legacy layout).
pub struct KvCache {
    /// Per layer: flattened `k` / `v` with stride `n_kv * head_dim` per sequence position.
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    cuda_attention: Vec<Option<crate::native::attention::CudaAttention>>,
}

impl KvCache {
    pub fn new(cfg: &LlamaConfig) -> Self {
        let stride = cfg.n_kv * cfg.head_dim;
        let len = stride * cfg.max_seq;
        let k = (0..cfg.n_layer).map(|_| vec![0.0f32; len]).collect();
        let v = (0..cfg.n_layer).map(|_| vec![0.0f32; len]).collect();
        Self {
            k,
            v,
            cuda_attention: (0..cfg.n_layer).map(|_| None).collect(),
        }
    }

    pub fn clear(&mut self) {
        for attention in self.cuda_attention.iter_mut().flatten() {
            attention.clear();
        }
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

/// Physical page slabs shared across sequences in a [`PagedKvPool`].
#[derive(Debug)]
pub struct SharedPhysKvStore {
    n_layer: usize,
    stride: usize,
    page_tokens: usize,
    max_phys_pages: usize,
    quant_format: KvQuantFormat,
    phys_k: Vec<Vec<Vec<f32>>>,
    phys_v: Vec<Vec<Vec<f32>>>,
    phys_k_q: Vec<Vec<Vec<u8>>>,
    phys_v_q: Vec<Vec<Vec<u8>>>,
    free_ids: Vec<Vec<usize>>,
    stats: KvPoolStats,
}

impl SharedPhysKvStore {
    pub fn new(cfg: &LlamaConfig, page_tokens: usize, max_phys_pages: usize) -> Result<Self> {
        Self::new_with_quant(cfg, page_tokens, max_phys_pages, KvQuantFormat::from_env())
    }

    pub fn new_with_quant(
        cfg: &LlamaConfig,
        page_tokens: usize,
        max_phys_pages: usize,
        quant_format: KvQuantFormat,
    ) -> Result<Self> {
        if page_tokens == 0 || max_phys_pages == 0 {
            return Err(BitNetError::Inference(
                "shared paged KV: bad page config".into(),
            ));
        }
        let stride = cfg.n_kv * cfg.head_dim;
        let n_layer = cfg.n_layer;
        Ok(Self {
            n_layer,
            stride,
            page_tokens,
            max_phys_pages,
            quant_format,
            phys_k: (0..n_layer).map(|_| Vec::new()).collect(),
            phys_v: (0..n_layer).map(|_| Vec::new()).collect(),
            phys_k_q: (0..n_layer).map(|_| Vec::new()).collect(),
            phys_v_q: (0..n_layer).map(|_| Vec::new()).collect(),
            free_ids: (0..n_layer).map(|_| Vec::new()).collect(),
            stats: KvPoolStats::default(),
        })
    }

    pub fn alloc_phys(&mut self, layer: usize) -> Result<usize> {
        if let Some(id) = self.free_ids[layer].pop() {
            zero_phys_page(
                id,
                &mut self.phys_k[layer],
                &mut self.phys_v[layer],
                &mut self.phys_k_q[layer],
                &mut self.phys_v_q[layer],
            );
            self.stats.reused_phys_pages += 1;
            return Ok(id);
        }
        if self.phys_k[layer].len() >= self.max_phys_pages {
            return Err(BitNetError::Inference(format!(
                "shared paged KV: exhausted physical pages for layer {layer} (max={})",
                self.max_phys_pages
            )));
        }
        let id = self.phys_k[layer].len();
        push_phys_page(
            self.quant_format,
            self.stride,
            self.page_tokens,
            &mut self.phys_k[layer],
            &mut self.phys_v[layer],
            &mut self.phys_k_q[layer],
            &mut self.phys_v_q[layer],
        );
        self.stats.new_phys_pages += 1;
        Ok(id)
    }

    pub fn free_phys(&mut self, layer: usize, id: usize) {
        if layer >= self.n_layer || id >= self.phys_k[layer].len() {
            return;
        }
        if !self.free_ids[layer].contains(&id) {
            self.free_ids[layer].push(id);
        }
    }

    pub fn pool_stats(&self) -> KvPoolStats {
        self.stats.clone()
    }

    /// Resident bytes for allocated K+V pages (compact when quantized).
    pub fn resident_bytes(&self) -> usize {
        resident_bytes_for_slabs(
            self.quant_format,
            &self.phys_k,
            &self.phys_v,
            &self.phys_k_q,
            &self.phys_v_q,
        )
    }

    pub fn quant_format(&self) -> KvQuantFormat {
        self.quant_format
    }

    /// Sum of allocated physical pages across layers (live slabs, including free-listed).
    pub fn allocated_pages(&self) -> usize {
        self.phys_k.iter().map(|l| l.len()).sum()
    }

    /// Sum of free-list entries across layers.
    pub fn free_pages(&self) -> usize {
        self.free_ids.iter().map(|l| l.len()).sum()
    }

    /// Fraction of allocated pages currently on the free list (0.0–1.0). Higher = more reclaimable.
    pub fn fragmentation_ratio(&self) -> f64 {
        let alloc = self.allocated_pages();
        if alloc == 0 {
            return 0.0;
        }
        self.free_pages() as f64 / alloc as f64
    }
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
    shared: Option<Arc<Mutex<SharedPhysKvStore>>>,
}

impl PagedSeqKv {
    pub fn new(cfg: &LlamaConfig, page_tokens: usize, max_pages: usize) -> Result<Self> {
        Self::new_inner(cfg, page_tokens, max_pages, None, KvQuantFormat::from_env())
    }

    /// Construct with an explicit KV quant format (tests / tune profiles).
    pub fn new_with_quant(
        cfg: &LlamaConfig,
        page_tokens: usize,
        max_pages: usize,
        quant_format: KvQuantFormat,
    ) -> Result<Self> {
        Self::new_inner(cfg, page_tokens, max_pages, None, quant_format)
    }

    pub fn new_with_shared(
        cfg: &LlamaConfig,
        page_tokens: usize,
        max_logical_pages: usize,
        shared: Arc<Mutex<SharedPhysKvStore>>,
    ) -> Result<Self> {
        let quant = shared
            .lock()
            .map(|g| g.quant_format())
            .unwrap_or_else(|_| KvQuantFormat::from_env());
        Self::new_inner(cfg, page_tokens, max_logical_pages, Some(shared), quant)
    }

    fn new_inner(
        cfg: &LlamaConfig,
        page_tokens: usize,
        max_pages: usize,
        shared: Option<Arc<Mutex<SharedPhysKvStore>>>,
        quant_format: KvQuantFormat,
    ) -> Result<Self> {
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
            shared,
        })
    }

    fn alloc_phys(&mut self, layer: usize) -> Result<usize> {
        if let Some(shared) = &self.shared {
            return shared
                .lock()
                .map_err(|_| BitNetError::Inference("shared phys KV lock poisoned".into()))?
                .alloc_phys(layer);
        }
        if let Some(id) = self.free_ids[layer].pop() {
            zero_phys_page(
                id,
                &mut self.phys_k[layer],
                &mut self.phys_v[layer],
                &mut self.phys_k_q[layer],
                &mut self.phys_v_q[layer],
            );
            self.stats.reused_phys_pages += 1;
            return Ok(id);
        }
        if self.phys_k[layer].len() >= self.max_pages {
            return Err(BitNetError::Inference(format!(
                "paged KV: exhausted physical pages for layer {layer} (max_pages={})",
                self.max_pages
            )));
        }
        let id = self.phys_k[layer].len();
        push_phys_page(
            self.quant_format,
            self.stride,
            self.page_tokens,
            &mut self.phys_k[layer],
            &mut self.phys_v[layer],
            &mut self.phys_k_q[layer],
            &mut self.phys_v_q[layer],
        );
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
        let stride = self.stride;
        match self.quant_format {
            KvQuantFormat::F32 => {
                if let Some(shared) = &self.shared {
                    let mut g = shared.lock().expect("shared phys KV lock");
                    g.phys_k[layer][pid][off_in_page..off_in_page + stride].copy_from_slice(k);
                    g.phys_v[layer][pid][off_in_page..off_in_page + stride].copy_from_slice(v);
                } else {
                    self.phys_k[layer][pid][off_in_page..off_in_page + stride].copy_from_slice(k);
                    self.phys_v[layer][pid][off_in_page..off_in_page + stride].copy_from_slice(v);
                }
            }
            fmt => {
                let row_bytes = fmt.row_bytes(stride);
                let q_off = (pos % self.page_tokens) * row_bytes;
                if let Some(shared) = &self.shared {
                    let mut g = shared.lock().expect("shared phys KV lock");
                    encode_quant_row(
                        fmt,
                        k,
                        &mut g.phys_k_q[layer][pid][q_off..q_off + row_bytes],
                    );
                    encode_quant_row(
                        fmt,
                        v,
                        &mut g.phys_v_q[layer][pid][q_off..q_off + row_bytes],
                    );
                } else {
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
        if self.shared.is_some() {
            panic!("k_head_slice unavailable with shared phys pages; use fill_k_head_values");
        }
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
        if self.shared.is_some() {
            panic!("v_head_slice unavailable with shared phys pages; use fill_v_head_values");
        }
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
                let lb = pos / self.page_tokens;
                let pid = self.block_phys[layer][lb];
                let off_in_page = (pos % self.page_tokens) * self.stride + kv_head * head_dim;
                if let Some(shared) = &self.shared {
                    let g = shared.lock().expect("shared phys KV lock");
                    out.copy_from_slice(&g.phys_k[layer][pid][off_in_page..off_in_page + head_dim]);
                } else {
                    out.copy_from_slice(self.k_head_slice(layer, pos, kv_head, head_dim));
                }
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
                let lb = pos / self.page_tokens;
                let pid = self.block_phys[layer][lb];
                let off_in_page = (pos % self.page_tokens) * self.stride + kv_head * head_dim;
                if let Some(shared) = &self.shared {
                    let g = shared.lock().expect("shared phys KV lock");
                    out.copy_from_slice(&g.phys_v[layer][pid][off_in_page..off_in_page + head_dim]);
                } else {
                    out.copy_from_slice(self.v_head_slice(layer, pos, kv_head, head_dim));
                }
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
        let mut row = vec![0.0f32; self.stride];
        if let Some(shared) = &self.shared {
            let g = shared.lock().expect("shared phys KV lock");
            let slab = if is_k {
                &g.phys_k_q[layer][pid]
            } else {
                &g.phys_v_q[layer][pid]
            };
            decode_quant_row(fmt, &slab[row_off..row_off + row_bytes], &mut row);
        } else {
            let slab = if is_k {
                &self.phys_k_q[layer][pid]
            } else {
                &self.phys_v_q[layer][pid]
            };
            decode_quant_row(fmt, &slab[row_off..row_off + row_bytes], &mut row);
        }
        let start = kv_head * head_dim;
        out.copy_from_slice(&row[start..start + head_dim]);
    }

    pub fn clear(&mut self) {
        if let Some(shared) = &self.shared {
            let mut g = shared.lock().expect("shared phys KV lock");
            for layer in 0..self.n_layer {
                for &pid in self.block_phys[layer].iter() {
                    if pid != usize::MAX {
                        g.free_phys(layer, pid);
                    }
                }
                self.block_phys[layer].clear();
            }
            return;
        }
        for layer in 0..self.n_layer {
            for &pid in self.block_phys[layer].iter() {
                if pid != usize::MAX && pid < self.phys_k[layer].len() {
                    self.free_ids[layer].push(pid);
                }
            }
            self.phys_k[layer].iter_mut().for_each(|s| s.fill(0.0));
            self.phys_v[layer].iter_mut().for_each(|s| s.fill(0.0));
            self.phys_k_q[layer].iter_mut().for_each(|s| s.fill(0));
            self.phys_v_q[layer].iter_mut().for_each(|s| s.fill(0));
            self.block_phys[layer].clear();
        }
    }

    pub fn page_tokens(&self) -> usize {
        self.page_tokens
    }

    pub fn max_pages(&self) -> usize {
        self.max_pages
    }

    pub fn block_table_snapshot(&self) -> Vec<Vec<usize>> {
        self.block_phys.clone()
    }

    pub fn restore_block_table(&mut self, tables: &[Vec<usize>], _token_count: usize) -> bool {
        if tables.len() != self.n_layer {
            return false;
        }
        self.block_phys.clone_from_slice(tables);
        true
    }

    pub fn physical_counts(&self) -> Vec<usize> {
        if let Some(shared) = &self.shared {
            let g = shared.lock().expect("shared phys KV lock");
            return (0..self.n_layer).map(|l| g.phys_k[l].len()).collect();
        }
        (0..self.n_layer).map(|l| self.phys_k[l].len()).collect()
    }

    pub fn pool_stats(&self) -> KvPoolStats {
        if let Some(shared) = &self.shared {
            return shared.lock().map(|g| g.pool_stats()).unwrap_or_default();
        }
        self.stats.clone()
    }

    pub fn uses_shared_phys(&self) -> bool {
        self.shared.is_some()
    }

    pub fn quant_format(&self) -> KvQuantFormat {
        self.quant_format
    }

    /// Resident bytes for this sequence's (or shared) physical K+V pages.
    pub fn resident_bytes(&self) -> usize {
        if let Some(shared) = &self.shared {
            return shared.lock().map(|g| g.resident_bytes()).unwrap_or(0);
        }
        resident_bytes_for_slabs(
            self.quant_format,
            &self.phys_k,
            &self.phys_v,
            &self.phys_k_q,
            &self.phys_v_q,
        )
    }
}

/// Llama KV backing store: dense mmap-style buffer or paged slabs.
pub enum KvStorage {
    Dense(KvCache),
    Paged(PagedSeqKv),
}

impl KvStorage {
    pub(crate) fn attention_cuda(
        &mut self,
        layer: usize,
        pos: usize,
        first: usize,
        q: &[f32],
        heads: usize,
        kv_heads: usize,
        head: usize,
        scale: f32,
        out: &mut [f32],
    ) -> Result<bool> {
        let Self::Dense(kv) = self else {
            return Ok(false);
        };
        let slot = &mut kv.cuda_attention[layer];
        if slot.is_none() {
            *slot = crate::native::attention::CudaAttention::new(
                kv.k[layer].len() / (kv_heads * head),
                kv_heads,
                head,
                head,
                heads,
            );
        }
        if let Some(attention) = slot {
            attention.run(q, &kv.k[layer], &kv.v[layer], pos, first, scale, None, out)?;
            return Ok(true);
        }
        Ok(false)
    }
    pub fn new_dense(cfg: &LlamaConfig) -> Self {
        Self::Dense(KvCache::new(cfg))
    }

    pub fn new_paged(cfg: &LlamaConfig, page_tokens: usize, max_pages: usize) -> Result<Self> {
        Ok(Self::Paged(PagedSeqKv::new(cfg, page_tokens, max_pages)?))
    }

    pub fn new_paged_shared(
        cfg: &LlamaConfig,
        page_tokens: usize,
        max_pages: usize,
        shared: Arc<Mutex<SharedPhysKvStore>>,
    ) -> Result<Self> {
        Ok(Self::Paged(PagedSeqKv::new_with_shared(
            cfg,
            page_tokens,
            max_pages,
            shared,
        )?))
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
        let mut k_tmp = Vec::new();
        if !matches!(self, Self::Dense(_)) {
            k_tmp.resize(head_dim, 0.0);
        }
        for p in 0..=pos {
            let k = if matches!(self, Self::Dense(_)) {
                self.k_head_slice(layer, p, kv_head, head_dim, stride)
            } else {
                self.fill_k_head_values(layer, p, kv_head, head_dim, stride, &mut k_tmp);
                &k_tmp
            };
            out[p] = crate::ggml::simd::dot(q, k) * scale;
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
                let physical_pages: usize = p.physical_counts().iter().sum();
                KvBackendStats {
                    backend: match self.backend_kind() {
                        KvBackendKind::PagedGpuPlanned => "paged_gpu_planned",
                        _ => {
                            if p.uses_shared_phys() {
                                "paged_cpu_pool"
                            } else {
                                "paged_cpu"
                            }
                        }
                    },
                    page_tokens: Some(p.page_tokens()),
                    max_pages: Some(p.max_pages()),
                    quant_format: p.quant_format().as_str(),
                    physical_pages,
                    new_phys_pages: pool.new_phys_pages,
                    reused_phys_pages: pool.reused_phys_pages,
                }
            }
        }
    }

    pub fn as_paged(&self) -> Option<&PagedSeqKv> {
        match self {
            Self::Paged(p) => Some(p),
            Self::Dense(_) => None,
        }
    }

    pub fn as_paged_mut(&mut self) -> Option<&mut PagedSeqKv> {
        match self {
            Self::Paged(p) => Some(p),
            Self::Dense(_) => None,
        }
    }
}

/// Multi-sequence paged KV pool: one physical page free-list budget shared across sequences.
#[derive(Debug)]
pub struct PagedKvPool {
    cfg: LlamaConfig,
    page_tokens: usize,
    max_pages_per_seq: usize,
    shared_phys: Arc<Mutex<SharedPhysKvStore>>,
    sequences: HashMap<u64, PagedSeqKv>,
    next_seq_id: u64,
}

impl PagedKvPool {
    pub fn from_env(cfg: &LlamaConfig) -> Result<Self> {
        let p = crate::paged_kv::PagedKvCache::from_env();
        let max_per = std::env::var("RBITNET_KV_POOL_MAX_SEQS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(8);
        let pages_each = p.max_pages / max_per.max(1);
        let shared_phys = Arc::new(Mutex::new(SharedPhysKvStore::new(
            cfg,
            p.page_size_tokens,
            p.max_pages,
        )?));
        Ok(Self {
            cfg: cfg.clone(),
            page_tokens: p.page_size_tokens,
            max_pages_per_seq: pages_each.max(1),
            shared_phys,
            sequences: HashMap::new(),
            next_seq_id: 1,
        })
    }

    pub fn open_sequence(&mut self) -> Result<u64> {
        let id = self.next_seq_id;
        self.next_seq_id = self.next_seq_id.saturating_add(1);
        let kv = PagedSeqKv::new_with_shared(
            &self.cfg,
            self.page_tokens,
            self.max_pages_per_seq,
            Arc::clone(&self.shared_phys),
        )?;
        self.sequences.insert(id, kv);
        Ok(id)
    }

    pub fn close_sequence(&mut self, seq_id: u64) {
        if let Some(mut seq) = self.sequences.remove(&seq_id) {
            seq.clear();
        }
    }

    pub fn shared_phys(&self) -> Arc<Mutex<SharedPhysKvStore>> {
        Arc::clone(&self.shared_phys)
    }

    pub fn page_tokens(&self) -> usize {
        self.page_tokens
    }

    pub fn max_pages_per_seq(&self) -> usize {
        self.max_pages_per_seq
    }

    pub fn fragmentation_ratio(&self) -> f64 {
        self.shared_phys
            .lock()
            .map(|g| g.fragmentation_ratio())
            .unwrap_or(0.0)
    }

    pub fn allocated_phys_pages(&self) -> usize {
        self.shared_phys
            .lock()
            .map(|g| g.allocated_pages())
            .unwrap_or(0)
    }

    pub fn free_phys_pages(&self) -> usize {
        self.shared_phys.lock().map(|g| g.free_pages()).unwrap_or(0)
    }

    pub fn sequence_mut(&mut self, seq_id: u64) -> Option<&mut PagedSeqKv> {
        self.sequences.get_mut(&seq_id)
    }

    pub fn active_sequences(&self) -> usize {
        self.sequences.len()
    }

    pub fn aggregate_pool_stats(&self) -> KvPoolStats {
        let shared = self
            .shared_phys
            .lock()
            .map(|g| g.pool_stats())
            .unwrap_or_default();
        let mut stats = shared;
        for seq in self.sequences.values() {
            let s = seq.pool_stats();
            stats.new_phys_pages = stats.new_phys_pages.max(s.new_phys_pages);
            stats.reused_phys_pages += s.reused_phys_pages;
        }
        stats
    }
}

fn push_phys_page(
    fmt: KvQuantFormat,
    stride: usize,
    page_tokens: usize,
    phys_k: &mut Vec<Vec<f32>>,
    phys_v: &mut Vec<Vec<f32>>,
    phys_k_q: &mut Vec<Vec<u8>>,
    phys_v_q: &mut Vec<Vec<u8>>,
) {
    match fmt {
        KvQuantFormat::F32 => {
            let len = stride.saturating_mul(page_tokens);
            phys_k.push(vec![0.0f32; len]);
            phys_v.push(vec![0.0f32; len]);
            // Keep Q index parallel with empty slabs (unused for F32).
            phys_k_q.push(Vec::new());
            phys_v_q.push(Vec::new());
        }
        KvQuantFormat::Q8 | KvQuantFormat::Q4 => {
            // Compact path: no F32 residency; empty f32 slots preserve page ids.
            phys_k.push(Vec::new());
            phys_v.push(Vec::new());
            let q_len = fmt.row_bytes(stride).saturating_mul(page_tokens);
            phys_k_q.push(vec![0u8; q_len]);
            phys_v_q.push(vec![0u8; q_len]);
        }
    }
}

fn zero_phys_page(
    id: usize,
    phys_k: &mut [Vec<f32>],
    phys_v: &mut [Vec<f32>],
    phys_k_q: &mut [Vec<u8>],
    phys_v_q: &mut [Vec<u8>],
) {
    if let Some(slab) = phys_k.get_mut(id) {
        slab.fill(0.0);
    }
    if let Some(slab) = phys_v.get_mut(id) {
        slab.fill(0.0);
    }
    if let Some(slab) = phys_k_q.get_mut(id) {
        slab.fill(0);
    }
    if let Some(slab) = phys_v_q.get_mut(id) {
        slab.fill(0);
    }
}

fn resident_bytes_for_slabs(
    fmt: KvQuantFormat,
    phys_k: &[Vec<Vec<f32>>],
    phys_v: &[Vec<Vec<f32>>],
    phys_k_q: &[Vec<Vec<u8>>],
    phys_v_q: &[Vec<Vec<u8>>],
) -> usize {
    let mut bytes = 0usize;
    match fmt {
        KvQuantFormat::F32 => {
            for layer in phys_k {
                for slab in layer {
                    bytes =
                        bytes.saturating_add(slab.len().saturating_mul(std::mem::size_of::<f32>()));
                }
            }
            for layer in phys_v {
                for slab in layer {
                    bytes =
                        bytes.saturating_add(slab.len().saturating_mul(std::mem::size_of::<f32>()));
                }
            }
        }
        KvQuantFormat::Q8 | KvQuantFormat::Q4 => {
            for layer in phys_k_q {
                for slab in layer {
                    bytes = bytes.saturating_add(slab.len());
                }
            }
            for layer in phys_v_q {
                for slab in layer {
                    bytes = bytes.saturating_add(slab.len());
                }
            }
        }
    }
    bytes
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
