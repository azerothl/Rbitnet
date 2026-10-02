use std::sync::atomic::{AtomicBool, Ordering};

static INFERENCE_CANCELLED: AtomicBool = AtomicBool::new(false);

pub fn clear_inference_cancel() {
    INFERENCE_CANCELLED.store(false, Ordering::SeqCst);
}

pub fn request_inference_cancel() {
    INFERENCE_CANCELLED.store(true, Ordering::SeqCst);
}

pub fn inference_cancelled() -> bool {
    INFERENCE_CANCELLED.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_flag_roundtrip() {
        clear_inference_cancel();
        assert!(!inference_cancelled());
        request_inference_cancel();
        assert!(inference_cancelled());
        clear_inference_cancel();
        assert!(!inference_cancelled());
    }
}
