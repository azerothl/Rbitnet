//! KV layouts for Llama: dense (legacy) and paged (Inference stack v2 phase A).

use crate::error::{BitNetError, Result};

use super::config::LlamaConfig;

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
    /// Per layer: physical pages; each slab holds `stride * page_tokens` floats.
    phys_k: Vec<Vec<Vec<f32>>>,
    phys_v: Vec<Vec<Vec<f32>>>,
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
        Ok(Self {
            n_layer,
            stride,
            page_tokens,
            max_pages,
            phys_k: (0..n_layer).map(|_| Vec::new()).collect(),
            phys_v: (0..n_layer).map(|_| Vec::new()).collect(),
            free_ids: (0..n_layer).map(|_| Vec::new()).collect(),
            block_phys: (0..n_layer).map(|_| Vec::new()).collect(),
            stats: KvPoolStats::default(),
        })
    }

    fn alloc_phys(&mut self, layer: usize) -> Result<usize> {
        if let Some(id) = self.free_ids[layer].pop() {
            self.phys_k[layer][id].fill(0.0);
            self.phys_v[layer][id].fill(0.0);
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
            return Err(BitNetError::Inference("paged KV write: bad slice len".into()));
        }
        let lb = pos / self.page_tokens;
        let pid = self.ensure_logical_block(layer, lb)?;
        let off_in_page = (pos % self.page_tokens) * self.stride;
        let slab_k = &mut self.phys_k[layer][pid];
        let slab_v = &mut self.phys_v[layer][pid];
        slab_k[off_in_page..off_in_page + self.stride].copy_from_slice(k);
        slab_v[off_in_page..off_in_page + self.stride].copy_from_slice(v);
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
        let off_in_page = (pos % self.page_tokens) * self.stride + kv_head * head_dim;
        &self.phys_v[layer][pid][off_in_page..off_in_page + head_dim]
    }

    pub fn clear(&mut self) {
        for layer in 0..self.n_layer {
            for &pid in self.block_phys[layer].iter() {
                if pid != usize::MAX && pid < self.phys_k[layer].len() {
                    self.free_ids[layer].push(pid);
                }
            }
            self.phys_k[layer]
                .iter_mut()
                .for_each(|s| s.fill(0.0));
            self.phys_v[layer]
                .iter_mut()
                .for_each(|s| s.fill(0.0));
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
        (0..self.n_layer)
            .map(|l| self.phys_k[l].len())
            .collect()
    }

    pub fn pool_stats(&self) -> KvPoolStats {
        self.stats.clone()
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
                Ok(())
            }
            Self::Paged(p) => p.write_kv_layer(layer, pos, k, v),
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
            let src = self.k_head_slice(layer, p, kv_head, head_dim, stride);
            dst[row_off..row_off + head_dim].copy_from_slice(src);
        }
    }

    pub fn as_paged_mut(&mut self) -> Option<&mut PagedSeqKv> {
        match self {
            Self::Paged(p) => Some(p),
            Self::Dense(_) => None,
        }
    }
}
