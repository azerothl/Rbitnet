//! FlashInfer / paged-attention FFI spike (feature `experimental-flashinfer` — not linked by default).

/// Returns whether the build would attempt FlashInfer when the feature is enabled.
pub fn flashinfer_available() -> bool {
    cfg!(feature = "experimental-flashinfer")
}

/// Documented entry for paginated attention on SM90+; native path remains default.
pub fn plan_paged_attention(
    _num_heads: u32,
    _page_size: u32,
    _num_pages: u32,
) -> Result<(), crate::error::BitNetError> {
    if !flashinfer_available() {
        return Err(crate::error::BitNetError::Inference(
            "FlashInfer spike: rebuild with --features experimental-flashinfer (not wired yet)".into(),
        ));
    }
    Err(crate::error::BitNetError::Inference(
        "FlashInfer FFI symbols not linked in this build".into(),
    ))
}
