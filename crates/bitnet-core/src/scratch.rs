//! Runtime scratch buffers reused across token steps.

#[derive(Debug, Default)]
pub struct ScratchArena {
    pool: Vec<Vec<f32>>,
}

impl ScratchArena {
    pub fn take(&mut self, len: usize) -> Vec<f32> {
        if let Some(pos) = self.pool.iter().position(|v| v.capacity() >= len) {
            let mut v = self.pool.swap_remove(pos);
            v.resize(len, 0.0);
            v.fill(0.0);
            crate::perf::record_scratch_reuse_hit();
            v
        } else {
            crate::perf::record_scratch_alloc(len.saturating_mul(std::mem::size_of::<f32>()));
            vec![0.0; len]
        }
    }

    pub fn recycle(&mut self, mut v: Vec<f32>) {
        v.clear();
        self.pool.push(v);
    }

    pub fn clear(&mut self) {
        self.pool.clear();
    }
}
