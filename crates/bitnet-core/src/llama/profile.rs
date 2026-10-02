//! Opt-in wall timings on the calling thread, for sequential forward diagnostics.
//! Stages are disjoint; model loading, sampling and HTTP are measured separately.

use std::{cell::RefCell, collections::BTreeMap, time::Instant};

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct StageTiming {
    pub calls: u64,
    pub elapsed_ns: u64,
}

thread_local! {
    static STAGES: RefCell<BTreeMap<&'static str, StageTiming>> = RefCell::default();
}

pub fn take() -> BTreeMap<&'static str, StageTiming> {
    STAGES.with(|stages| std::mem::take(&mut *stages.borrow_mut()))
}

pub(crate) struct Span {
    stage: &'static str,
    start: Instant,
}

impl Span {
    pub(crate) fn new(stage: &'static str) -> Self {
        Self {
            stage,
            start: Instant::now(),
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let ns = self.start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        STAGES.with(|stages| {
            let mut stages = stages.borrow_mut();
            let timing = stages.entry(self.stage).or_default();
            timing.calls += 1;
            timing.elapsed_ns += ns;
        });
    }
}
