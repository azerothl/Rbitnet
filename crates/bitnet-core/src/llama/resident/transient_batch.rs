//! A batch workspace owns weights, but borrows sequence owners only during a wave.
use super::*;
use std::sync::Arc;

type BatchCreate = unsafe extern "C" fn(*const c_void, u32, u32) -> *mut c_void;
type BatchStep = unsafe extern "C" fn(
    *mut c_void,
    *const *mut c_void,
    *const u32,
    *const f32,
    u32,
    u32,
    *mut f32,
    *mut u32,
) -> i32;
type BatchQuery = unsafe extern "C" fn(*const c_void, *mut u64, u32) -> i32;
pub(super) struct NativeBatchWorkspace {
    model: Arc<LlamaModel>,
    context: usize,
    maximum: usize,
    step: BatchStep,
    destroy: Destroy,
    query: BatchQuery,
    embeddings: Vec<f32>,
    logits: Vec<f32>,
    tokens: Vec<u32>,
    pointers: Vec<*mut c_void>,
}
impl Drop for NativeBatchWorkspace {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.context as *mut c_void) }
    }
}
impl NativeBatchWorkspace {
    pub(super) fn new(
        model: Arc<LlamaModel>,
        seed: &Resident,
        maximum: usize,
        ordering: u32,
    ) -> Result<Self> {
        if !(1..=8).contains(&maximum) || ordering > 1 {
            return Err(BitNetError::Inference(
                "batch workspace capacity/order refused".into(),
            ));
        }
        let addresses: Vec<_> = std::iter::once(&model.output)
            .chain(model.layers.iter().flat_map(|l| {
                [
                    &l.wq,
                    &l.wk,
                    &l.wv,
                    &l.wo,
                    &l.ffn_gate,
                    &l.ffn_up,
                    &l.ffn_down,
                ]
            }))
            .map(|w| match w {
                MatrixWeights::CudaQuant { device, .. } => device.device_address(),
                _ => None,
            })
            .collect();
        if addresses.iter().any(Option::is_none)
            || seed.vocab != model.cfg.n_vocab
            || seed
                ._weights
                .iter()
                .map(CudaDeviceQuantMatrix::device_address)
                .collect::<Vec<_>>()
                != addresses
        {
            return Err(BitNetError::Inference(
                "batch embedding and seed weights have different owners".into(),
            ));
        }
        let lib = crate::ggml::load_cuda_quant_library()
            .ok_or_else(|| BitNetError::Inference("Native batch library unavailable".into()))?;
        let create = unsafe {
            *lib.get::<BatchCreate>(b"rbitnet_cuda_llama_batch_create\0")
                .map_err(|_| BitNetError::Inference("Native batch create unavailable".into()))?
        };
        let step = unsafe {
            *lib.get::<BatchStep>(b"rbitnet_cuda_llama_batch_step\0")
                .map_err(|_| BitNetError::Inference("Native batch step unavailable".into()))?
        };
        let destroy = unsafe {
            *lib.get::<Destroy>(b"rbitnet_cuda_llama_batch_destroy\0")
                .map_err(|_| BitNetError::Inference("Native batch destroy unavailable".into()))?
        };
        let query = unsafe {
            *lib.get::<BatchQuery>(b"rbitnet_cuda_llama_batch_stats\0")
                .map_err(|_| BitNetError::Inference("Native batch stats unavailable".into()))?
        };
        let context =
            unsafe { create(seed.context as *const c_void, maximum as u32, ordering) } as usize;
        if context == 0 {
            return Err(BitNetError::Inference(
                "Native F32 mono batch workspace allocation/configuration refused".into(),
            ));
        }
        Ok(Self {
            embeddings: vec![0.; maximum * model.cfg.n_embd],
            logits: vec![0.; maximum * model.cfg.n_vocab],
            tokens: vec![0; maximum],
            pointers: Vec::with_capacity(maximum),
            model,
            context,
            maximum,
            step,
            destroy,
            query,
        })
    }
    fn execute(
        &mut self,
        contexts: &mut [&mut Resident],
        ids: &[u32],
        positions: &[u32],
        mode: u32,
    ) -> Result<()> {
        let count = contexts.len();
        if count == 0
            || count > self.maximum
            || ids.len() != count
            || positions.len() != count
            || mode > 2
        {
            return Err(BitNetError::Inference(
                "transient batch wave shape refused".into(),
            ));
        }
        self.pointers.clear();
        for (row, r) in contexts.iter().enumerate() {
            if ids[row] as usize >= self.model.cfg.n_vocab
                || r.vocab != self.model.cfg.n_vocab
                || self.pointers.contains(&(r.context as *mut c_void))
            {
                return Err(BitNetError::Inference(
                    "transient batch duplicate owner/token refused".into(),
                ));
            }
            self.model.token_embd.embed_row(
                ids[row] as usize,
                self.model.cfg.n_embd,
                self.model.cfg.n_vocab,
                &mut self.embeddings
                    [row * self.model.cfg.n_embd..(row + 1) * self.model.cfg.n_embd],
            )?;
            self.pointers.push(r.context as *mut c_void);
        }
        // The Native API validates the immutable model key, pool, positions and
        // layout before publishing work, then synchronizes its own wave stream.
        // No context pointer is dereferenced after this blocking call returns.
        let status = unsafe {
            (self.step)(
                self.context as *mut c_void,
                self.pointers.as_ptr(),
                positions.as_ptr(),
                self.embeddings.as_ptr(),
                count as u32,
                mode,
                self.logits.as_mut_ptr(),
                self.tokens.as_mut_ptr(),
            )
        };
        self.pointers.clear();
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "transient Native batch failed/refused status {status}"
            )));
        }
        crate::perf::record_native_llama_batch(
            count,
            (7 * self.model.cfg.n_layer + usize::from(mode != 0)) as u64,
        );
        if mode != 0 {
            crate::perf::record_scheduler_decode_wave(1);
        }
        crate::perf::record_gpu_transfer(
            (count * (self.model.cfg.n_embd * 4 + 4 * std::mem::size_of::<*mut c_void>() + 8))
                as u64,
            (count
                * if mode == 1 {
                    self.model.cfg.n_vocab * 4
                } else if mode == 2 {
                    4
                } else {
                    0
                }) as u64,
            (7 * self.model.cfg.n_layer + usize::from(mode != 0)) as u64,
        );
        for _ in 0..self.model.cfg.n_layer {
            crate::perf::record_gpu_attention();
        }
        Ok(())
    }
    pub(super) fn full_logits(
        &mut self,
        contexts: &mut [&mut Resident],
        ids: &[u32],
        positions: &[u32],
    ) -> Result<&[f32]> {
        self.execute(contexts, ids, positions, 1)?;
        Ok(&self.logits[..contexts.len() * self.model.cfg.n_vocab])
    }
    pub(super) fn greedy_ids(
        &mut self,
        contexts: &mut [&mut Resident],
        ids: &[u32],
        positions: &[u32],
    ) -> Result<&[u32]> {
        self.execute(contexts, ids, positions, 2)?;
        Ok(&self.tokens[..contexts.len()])
    }
    pub(super) fn advance(
        &mut self,
        contexts: &mut [&mut Resident],
        ids: &[u32],
        positions: &[u32],
    ) -> Result<()> {
        self.execute(contexts, ids, positions, 0)
    }
    pub(super) fn stats(&self) -> Result<[u64; 3]> {
        let mut out = [0; 3];
        let status = unsafe { (self.query)(self.context as *const c_void, out.as_mut_ptr(), 3) };
        if status != 0 {
            return Err(BitNetError::Inference("Native batch stats refused".into()));
        }
        Ok(out)
    }
    /// Exactly one normal 128-token prefill partition. Cancellation is supplied
    /// by the caller per request; this path never changes the global cancel flag.
    pub(super) fn prefill_chunk(
        &self,
        r: &mut Resident,
        ids: &[u32],
        base: usize,
        last: bool,
        greedy: bool,
    ) -> Result<(Vec<f32>, u32)> {
        if ids.is_empty()
            || ids.len() > 128
            || base
                .checked_add(ids.len())
                .is_none_or(|end| end > r.capacity)
        {
            return Err(BitNetError::Inference(
                "continuous prefill partition refused".into(),
            ));
        }
        let prefill = r.prefill.ok_or_else(|| {
            BitNetError::Inference("continuous CUDA block prefill unavailable".into())
        })?;
        r.prefill_embeddings
            .resize(ids.len() * self.model.cfg.n_embd, 0.);
        for (&id, row) in ids
            .iter()
            .zip(r.prefill_embeddings.chunks_exact_mut(self.model.cfg.n_embd))
        {
            if id as usize >= r.vocab {
                return Err(BitNetError::Inference(
                    "continuous prefill token refused".into(),
                ));
            }
            self.model
                .token_embd
                .embed_row(id as usize, self.model.cfg.n_embd, r.vocab, row)?;
        }
        let mode = if !last {
            0
        } else if greedy {
            2
        } else {
            1
        };
        let mut logits = if mode == 1 {
            vec![0.; r.vocab]
        } else {
            Vec::new()
        };
        let mut token = 0;
        let status = unsafe {
            prefill(
                r.context as *mut c_void,
                r.prefill_embeddings.as_ptr(),
                base as u32,
                ids.len() as u32,
                mode,
                logits.as_mut_ptr(),
                &mut token,
            )
        };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "continuous CUDA prefill failed status {status}"
            )));
        }
        crate::perf::record_gpu_prefill(ids.len(), if ids.len() > 1 { r.matrix_count } else { 0 });
        crate::perf::record_scheduler_prefill_chunk(1);
        for _ in 0..r.layers {
            crate::perf::record_gpu_attention();
        }
        crate::perf::record_gpu_transfer(
            (r.prefill_embeddings.len() * 4 + 4) as u64,
            if mode == 1 {
                (r.vocab * 4) as u64
            } else if mode == 2 {
                4
            } else {
                0
            },
            u64::from(mode > 0) + if ids.len() == 1 { r.matrix_count } else { 0 },
        );
        Ok((logits, token))
    }
}
