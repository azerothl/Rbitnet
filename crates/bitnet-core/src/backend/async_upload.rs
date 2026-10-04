//! Bounded pinned transfers with unpublished device owners.
//! Each transfer retains its source and destination until its event completes.
use super::*;
use crate::error::{BitNetError, Result};
use std::time::Instant;

#[derive(Clone, Copy)]
struct Api {
    stream_create: unsafe extern "C" fn(*mut *mut c_void, u32) -> i32,
    stream_destroy: unsafe extern "C" fn(*mut c_void) -> i32,
    stream_wait: unsafe extern "C" fn(*mut c_void) -> i32,
    host_alloc: unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32,
    host_free: unsafe extern "C" fn(*mut c_void) -> i32,
    copy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32, *mut c_void) -> i32,
    event_create: unsafe extern "C" fn(*mut *mut c_void, u32) -> i32,
    event_destroy: unsafe extern "C" fn(*mut c_void) -> i32,
    event_record: unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32,
    event_query: unsafe extern "C" fn(*mut c_void) -> i32,
    event_wait: unsafe extern "C" fn(*mut c_void) -> i32,
    elapsed: unsafe extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> i32,
}
impl Api {
    unsafe fn load(rt: &CudaRuntime) -> Option<Self> {
        Some(Self {
            stream_create: *rt._lib.get(b"cudaStreamCreateWithFlags\0").ok()?,
            stream_destroy: *rt._lib.get(b"cudaStreamDestroy\0").ok()?,
            stream_wait: *rt._lib.get(b"cudaStreamSynchronize\0").ok()?,
            host_alloc: *rt._lib.get(b"cudaHostAlloc\0").ok()?,
            host_free: *rt._lib.get(b"cudaFreeHost\0").ok()?,
            copy: *rt._lib.get(b"cudaMemcpyAsync\0").ok()?,
            event_create: *rt._lib.get(b"cudaEventCreateWithFlags\0").ok()?,
            event_destroy: *rt._lib.get(b"cudaEventDestroy\0").ok()?,
            event_record: *rt._lib.get(b"cudaEventRecord\0").ok()?,
            event_query: *rt._lib.get(b"cudaEventQuery\0").ok()?,
            event_wait: *rt._lib.get(b"cudaEventSynchronize\0").ok()?,
            elapsed: *rt._lib.get(b"cudaEventElapsedTime\0").ok()?,
        })
    }
}
fn checked(status: i32, what: &str) -> Result<()> {
    if status == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(BitNetError::Inference(format!(
            "CUDA {what} failed: {status}"
        )))
    }
}
fn staging_span(payload: &[u8], bytes: usize) -> Result<&[u8]> {
    payload
        .get(..bytes)
        .filter(|span| !span.is_empty())
        .ok_or_else(|| BitNetError::Inference("invalid async matrix span".into()))
}
pub(crate) struct CopyStream {
    api: Api,
    rt: Arc<CudaRuntime>,
    stream: usize,
}
// A CUDA stream supports concurrent API calls. Cache scheduling still owns its
// per-slot mutable host memory/events; no pending upload publishes a matrix.
unsafe impl Send for CopyStream {}
unsafe impl Sync for CopyStream {}
impl CopyStream {
    pub(crate) fn new(rt: Arc<CudaRuntime>) -> Result<Option<Arc<Self>>> {
        let Some(api) = (unsafe { Api::load(&rt) }) else {
            return Ok(None);
        };
        let mut stream = null_mut();
        checked(
            unsafe { (api.stream_create)(&mut stream, 1) },
            "copy stream creation",
        )?;
        Ok(Some(Arc::new(Self {
            api,
            rt,
            stream: stream as usize,
        })))
    }
}
impl Drop for CopyStream {
    fn drop(&mut self) {
        unsafe {
            (self.api.stream_wait)(self.stream as *mut c_void);
            (self.api.stream_destroy)(self.stream as *mut c_void);
        }
    }
}

pub(crate) struct CopySlot {
    channel: Arc<CopyStream>,
    pinned: usize,
    capacity: usize,
    start: usize,
    done: usize,
    in_flight: bool,
}
unsafe impl Send for CopySlot {}
impl CopySlot {
    pub(crate) fn new(channel: Arc<CopyStream>, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(BitNetError::Inference(
                "pinned staging capacity must be positive".into(),
            ));
        }
        let mut slot = Self {
            channel,
            pinned: 0,
            capacity,
            start: 0,
            done: 0,
            in_flight: false,
        };
        let a = slot.channel.api;
        let mut pointer = null_mut();
        checked(
            unsafe { (a.host_alloc)(&mut pointer, capacity, 0) },
            "pinned staging allocation",
        )?;
        slot.pinned = pointer as usize;
        checked(
            unsafe { (a.event_create)(&mut pointer, 0) },
            "copy start event",
        )?;
        slot.start = pointer as usize;
        checked(
            unsafe { (a.event_create)(&mut pointer, 0) },
            "copy completion event",
        )?;
        slot.done = pointer as usize;
        Ok(slot)
    }
    fn drain(&mut self) {
        if self.in_flight {
            unsafe {
                (self.channel.api.stream_wait)(self.channel.stream as *mut c_void);
            };
            self.in_flight = false;
        }
    }
    pub(crate) fn ready(&self) -> Result<bool> {
        if !self.in_flight {
            return Ok(true);
        }
        let status = unsafe { (self.channel.api.event_query)(self.done as *mut c_void) };
        match status {
            0 => Ok(true),
            600 => Ok(false),
            _ => {
                checked(status, "completion query")?;
                unreachable!()
            }
        }
    }
    fn wait(&mut self) -> Result<(u64, u64)> {
        if !self.in_flight {
            return Err(BitNetError::Inference(
                "copy slot has no pending upload".into(),
            ));
        }
        let waiting = Instant::now();
        let a = self.channel.api;
        let status = unsafe { (a.event_wait)(self.done as *mut c_void) };
        let wait_ns = waiting.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        if status != 0 {
            self.drain();
            checked(status, "selected upload wait")?;
        }
        self.in_flight = false;
        let mut ms = 0.;
        checked(
            unsafe { (a.elapsed)(&mut ms, self.start as *mut c_void, self.done as *mut c_void) },
            "copy timing",
        )?;
        if !ms.is_finite() || ms < 0. {
            return Err(BitNetError::Inference("invalid CUDA copy timing".into()));
        }
        Ok(((ms as f64 * 1e6).min(u64::MAX as f64) as u64, wait_ns))
    }
    /// Device matrices remain private until completion; the slot owns pinned
    /// source bytes before enqueue and retains them through its completion event.
    pub(crate) fn submit(mut self, matrices: Vec<UnpublishedQuant>) -> Result<PendingUpload> {
        if self.in_flight {
            return Err(BitNetError::Inference(
                "staging slot is already pending".into(),
            ));
        }
        let bytes = matrices
            .iter()
            .try_fold(0usize, |n, m| n.checked_add(m.matrix.bytes()))
            .ok_or_else(|| BitNetError::Inference("staging byte count overflow".into()))?;
        if bytes == 0 || bytes > self.capacity {
            return Err(BitNetError::Inference(
                "expert group exceeds pinned staging capacity".into(),
            ));
        }
        let staging = Instant::now();
        let mut offset = 0;
        for m in &matrices {
            // Owned host payloads may contain trailing bytes. Staging and the
            // device allocation both cover exactly the matrix's validated span.
            let payload = staging_span(m.matrix.host_payload(), m.matrix.bytes())?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    payload.as_ptr(),
                    (self.pinned as *mut u8).add(offset),
                    payload.len(),
                );
            }
            offset += payload.len();
        }
        let stage_ns = staging.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let a = self.channel.api;
        let stream = self.channel.stream as *mut c_void;
        self.in_flight = true;
        let enqueue = (|| {
            checked(
                unsafe { (a.event_record)(self.start as *mut c_void, stream) },
                "copy start record",
            )?;
            let mut offset = 0;
            for m in &matrices {
                checked(
                    unsafe {
                        (a.copy)(
                            m.matrix.device.as_ref().unwrap().as_device_ptr(),
                            (self.pinned as *const u8).add(offset).cast(),
                            m.matrix.bytes(),
                            CUDA_MEMCPY_HOST_TO_DEVICE,
                            stream,
                        )
                    },
                    "pinned expert upload",
                )?;
                offset += m.matrix.bytes();
            }
            checked(
                unsafe { (a.event_record)(self.done as *mut c_void, stream) },
                "completion record",
            )
        })();
        if let Err(e) = enqueue {
            // matrices are otherwise dropped before this by-value self on error.
            self.drain();
            return Err(e);
        }
        Ok(PendingUpload {
            slot: Some(self),
            matrices,
            bytes: bytes as u64,
            stage_ns,
        })
    }
}
impl Drop for CopySlot {
    fn drop(&mut self) {
        self.drain();
        let a = self.channel.api;
        unsafe {
            if self.start != 0 {
                (a.event_destroy)(self.start as *mut c_void);
            }
            if self.done != 0 {
                (a.event_destroy)(self.done as *mut c_void);
            }
            if self.pinned != 0 {
                (a.host_free)(self.pinned as *mut c_void);
            }
        }
    }
}

pub(crate) struct UnpublishedQuant {
    matrix: CudaDeviceQuantMatrix,
}
impl UnpublishedQuant {
    pub(crate) fn allocate(
        rt: &Arc<CudaRuntime>,
        archive: Arc<crate::gguf::GgufArchive>,
        tensor: &crate::gguf::GgufTensorInfo,
        start: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Option<Self>> {
        let mut matrix = CudaDeviceQuantMatrix::from_archive_range(
            None, archive, tensor, start, rows, cols, true,
        )?;
        if !crate::ggml::ggml_type_supports_cuda_quant(tensor.ggml_type) {
            return Ok(None);
        }
        let Some(ptr) = rt.alloc_device_category(matrix.bytes(), device_memory::EXPERTS) else {
            return Ok(None);
        };
        matrix.device = Some(Arc::new(CudaDeviceBuffer {
            rt: Arc::clone(rt),
            ptr: ptr as usize,
            nbytes: matrix.bytes(),
        }));
        matrix.rt = Some(Arc::clone(rt));
        Ok(Some(Self { matrix }))
    }
    pub(crate) fn recycle(matrix: CudaDeviceQuantMatrix) -> Result<Self> {
        if matrix
            .device
            .as_ref()
            .is_none_or(|buffer| Arc::strong_count(buffer) != 1)
        {
            return Err(BitNetError::Inference(
                "asynchronous refill requires an exclusive device allocation".into(),
            ));
        }
        Ok(Self { matrix })
    }
    pub(crate) fn bind(
        &mut self,
        archive: Arc<crate::gguf::GgufArchive>,
        tensor: &crate::gguf::GgufTensorInfo,
        start: usize,
    ) -> Result<()> {
        let row_bytes = crate::ggml::ggml_row_size(tensor.ggml_type, self.matrix.in_cols as u64)?;
        let bytes = row_bytes
            .checked_mul(self.matrix.out_rows)
            .ok_or_else(|| BitNetError::Inference("async matrix bytes overflow".into()))?;
        if !crate::ggml::ggml_type_supports_cuda_quant(tensor.ggml_type)
            || tensor.dimensions.len() != 3
            || tensor.dimensions[0] != self.matrix.in_cols as u64
            || tensor.dimensions[1] != self.matrix.out_rows as u64
            || self
                .matrix
                .device
                .as_ref()
                .is_none_or(|buffer| buffer.nbytes < bytes)
            || row_bytes == 0
            || start % row_bytes != 0
        {
            return Err(BitNetError::Inference(
                "asynchronous expert span geometry mismatch".into(),
            ));
        }
        let host = Arc::new(QuantHostBacking::mapped(archive, tensor, start, bytes)?);
        self.matrix.host = host;
        self.matrix.row_bytes = row_bytes;
        self.matrix.ggml_type = tensor.ggml_type;
        Ok(())
    }
    pub(crate) fn bytes(&self) -> usize {
        self.matrix.bytes()
    }
}
pub(crate) struct CompletedUpload {
    pub(crate) matrices: Vec<CudaDeviceQuantMatrix>,
    pub(crate) slot: CopySlot,
    pub(crate) bytes: u64,
    pub(crate) stage_ns: u64,
    pub(crate) dma_ns: u64,
    pub(crate) wait_ns: u64,
}
pub(crate) struct PendingUpload {
    slot: Option<CopySlot>,
    matrices: Vec<UnpublishedQuant>,
    bytes: u64,
    stage_ns: u64,
}
impl PendingUpload {
    pub(crate) fn ready(&self) -> Result<bool> {
        self.slot.as_ref().unwrap().ready()
    }
    pub(crate) fn complete(mut self) -> Result<CompletedUpload> {
        let (dma_ns, wait_ns) = self.slot.as_mut().unwrap().wait()?;
        let slot = self.slot.take().unwrap();
        // This is the only publication of previously uninitialized buffers.
        let matrices = std::mem::take(&mut self.matrices)
            .into_iter()
            .map(|m| m.matrix)
            .collect();
        slot.channel
            .rt
            .upload_bytes
            .fetch_add(self.bytes, Ordering::Relaxed);
        crate::perf::record_gpu_transfer(self.bytes, 0, 0);
        Ok(CompletedUpload {
            matrices,
            slot,
            bytes: self.bytes,
            stage_ns: self.stage_ns,
            dma_ns,
            wait_ns,
        })
    }
}
impl Drop for PendingUpload {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.as_mut() {
            slot.drain();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_staging_uses_exact_matrix_span_even_with_trailing_host_bytes() {
        let host = [1, 2, 3, 4, 5, 6];
        assert_eq!(staging_span(&host, 4).unwrap(), &host[..4]);
        assert!(staging_span(&host, 7).is_err());
        assert!(staging_span(&host, usize::MAX).is_err());
        assert!(staging_span(&host, 0).is_err());
    }
    #[test]
    fn optional_async_upload_publishes_only_after_events_and_drains_on_drop() {
        if std::env::var("RBITNET_CUDA_ASYNC_TEST").as_deref() != Ok("1") {
            return;
        }
        let rt = CudaRuntime::try_load().expect("CUDA required");
        let archive = Arc::new(
            crate::gguf::GgufArchive::mmap_path(std::path::Path::new(
                &std::env::var("RBITNET_TEST_GGUF").unwrap(),
            ))
            .unwrap(),
        );
        let tensor = archive.tensor_by_name("blk.0.ffn_up_exps.weight").unwrap();
        let rows = tensor.dimensions[1] as usize;
        let cols = tensor.dimensions[0] as usize;
        let bytes = crate::ggml::ggml_row_size(tensor.ggml_type, cols as u64).unwrap() * rows;
        let before = rt.managed_memory_stats().unwrap();
        let channel = CopyStream::new(Arc::clone(&rt)).unwrap().unwrap();
        let slot = CopySlot::new(channel, bytes).unwrap();
        let blank = UnpublishedQuant::allocate(&rt, Arc::clone(&archive), tensor, 0, rows, cols)
            .unwrap()
            .unwrap();
        let pending = slot.submit(vec![blank]).unwrap();
        let completed = pending.complete().unwrap();
        assert_eq!(completed.bytes, bytes as u64);
        let mut ready = completed.matrices;
        let slot = completed.slot;
        let original = ready[0].device_address();
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i * 13 % 71) as f32 - 35.) / 256.)
            .collect();
        let got = ready[0].matvec(&x).unwrap();
        let mut copied = vec![0u8; bytes];
        assert!(rt.copy_device_to_host(
            copied.as_mut_ptr().cast(),
            ready[0].device.as_ref().unwrap().as_device_ptr(),
            bytes
        ));
        assert_eq!(copied, ready[0].host_payload());
        let cpu = CudaDeviceQuantMatrix::from_archive_range(
            None,
            Arc::clone(&archive),
            tensor,
            0,
            rows,
            cols,
            true,
        )
        .unwrap();
        let expected = crate::ggml::QuantMatvecKernel::cpu_parallel()
            .matvec_payload(tensor.ggml_type, cpu.host_payload(), &x, cols, rows)
            .unwrap();
        assert_eq!(got.len(), expected.len());
        for (a, b) in got.iter().zip(expected) {
            assert!((a - b).abs() <= 3e-4 * (1. + b.abs()), "{a}/{b}");
        }
        let mut recycled = UnpublishedQuant::recycle(ready.pop().unwrap()).unwrap();
        recycled.bind(Arc::clone(&archive), tensor, bytes).unwrap();
        let pending = slot.submit(vec![recycled]).unwrap();
        let completed = pending.complete().unwrap();
        assert_eq!(completed.matrices[0].device_address(), original);
        assert!(rt.copy_device_to_host(
            copied.as_mut_ptr().cast(),
            completed.matrices[0]
                .device
                .as_ref()
                .unwrap()
                .as_device_ptr(),
            bytes
        ));
        assert_eq!(
            copied.as_slice(),
            &archive.tensor_payload(tensor).unwrap()[bytes..2 * bytes]
        );
        let blank =
            UnpublishedQuant::recycle(completed.matrices.into_iter().next().unwrap()).unwrap();
        let pending = completed.slot.submit(vec![blank]).unwrap();
        drop(pending); // pending device/pinned/event owners drain before release
        drop(cpu);
        drop(ready);
        assert_eq!(
            rt.managed_memory_stats().unwrap().categories[device_memory::EXPERTS as usize],
            before.categories[device_memory::EXPERTS as usize]
        );
    }

    #[test]
    fn optional_async_mixed_q4_q6_refill_keeps_capacity_and_original_bytes() {
        if std::env::var("RBITNET_CUDA_ASYNC_TEST").as_deref() != Ok("1") {
            return;
        }
        let path = std::env::var("RBITNET_TEST_MIXED_GGUF")
            .expect("explicit mixed quant fixture required");
        let archive =
            Arc::new(crate::gguf::GgufArchive::mmap_path(std::path::Path::new(&path)).unwrap());
        let select = |ty| {
            archive
                .tensors
                .iter()
                .find(|t| t.name.ends_with("ffn_down_exps.weight") && t.ggml_type == ty)
                .unwrap()
        };
        let small = select(12);
        let large = select(14);
        assert_eq!(&small.dimensions[..2], &large.dimensions[..2]);
        let rows = large.dimensions[1] as usize;
        let cols = large.dimensions[0] as usize;
        let capacity = crate::ggml::ggml_row_size(large.ggml_type, cols as u64).unwrap() * rows;
        let rt = CudaRuntime::try_load().unwrap();
        let before = rt.managed_memory_stats().unwrap().categories[4];
        let channel = CopyStream::new(Arc::clone(&rt)).unwrap().unwrap();
        let mut slot = CopySlot::new(channel, capacity).unwrap();
        let mut blank = UnpublishedQuant::allocate(&rt, Arc::clone(&archive), large, 0, rows, cols)
            .unwrap()
            .unwrap();
        let address = blank.matrix.device_address();
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i * 13 % 71) as f32 - 35.) / 256.)
            .collect();
        for tensor in [small, large, small, large] {
            let bytes = crate::ggml::ggml_row_size(tensor.ggml_type, cols as u64).unwrap() * rows;
            blank.bind(Arc::clone(&archive), tensor, bytes).unwrap();
            let completed = slot.submit(vec![blank]).unwrap().complete().unwrap();
            assert_eq!(completed.bytes, bytes as u64);
            let mut matrices = completed.matrices;
            slot = completed.slot;
            let ready = matrices.pop().unwrap();
            assert_eq!(ready.device_address(), address);
            assert_eq!(ready.bytes(), bytes);
            assert_eq!(ready.ggml_type(), tensor.ggml_type);
            assert_eq!(
                rt.managed_memory_stats().unwrap().categories[4] - before,
                capacity as u64
            );
            let mut copied = vec![0; bytes];
            assert!(rt.copy_device_to_host(
                copied.as_mut_ptr().cast(),
                ready.device.as_ref().unwrap().as_device_ptr(),
                bytes
            ));
            assert_eq!(
                copied.as_slice(),
                &archive.tensor_payload(tensor).unwrap()[bytes..2 * bytes]
            );
            let actual = ready.matvec(&x).unwrap();
            let expected = crate::ggml::QuantMatvecKernel::cpu_parallel()
                .matvec_payload(tensor.ggml_type, ready.host_payload(), &x, cols, rows)
                .unwrap();
            for (a, b) in actual.into_iter().zip(expected) {
                assert!(
                    (a - b).abs() <= 3e-4 * (1. + b.abs()),
                    "mixed quant {a}/{b}"
                );
            }
            blank = UnpublishedQuant::recycle(ready).unwrap();
        }
        drop(blank);
        drop(slot);
        assert_eq!(rt.managed_memory_stats().unwrap().categories[4], before);
        println!("ASYNC_MIXED Q4/Q6/Q4/Q6 exact bytes, unchanged allocation, actual GPU matvec and physical ledger passed");
    }
}
