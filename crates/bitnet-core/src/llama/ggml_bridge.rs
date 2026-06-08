//! Optional ggml-backed matmul (spike / hook only).
//!
//! Set `RBITNET_LLAMA_MATMUL=ggml` to request delegation. A real bridge library is not shipped
//! yet; with `--features experimental-ggml-kernels` we only change logging — kernels still run
//! in Rust until a `rbitnet_ggml_bridge` shared library is provided.

use std::sync::Once;

static WARN_ONCE: Once = Once::new();

#[inline]
pub fn ggml_mul_mat_requested() -> bool {
    std::env::var("RBITNET_LLAMA_MATMUL")
        .map(|v| v.eq_ignore_ascii_case("ggml"))
        .unwrap_or(false)
}

/// Call once per process when `RBITNET_LLAMA_MATMUL=ggml` is set.
pub fn warn_if_ggml_env_without_bridge() {
    if !ggml_mul_mat_requested() {
        return;
    }
    WARN_ONCE.call_once(|| {
        #[cfg(feature = "experimental-ggml-kernels")]
        {
            tracing::warn!(
                "RBITNET_LLAMA_MATMUL=ggml: experimental feature enabled; no ggml bridge DLL is bundled yet — using Rust kernels. \
                 See docs/GPU_NATIVE_ROADMAP.md / future release notes."
            );
        }
        #[cfg(not(feature = "experimental-ggml-kernels"))]
        {
            tracing::warn!(
                "RBITNET_LLAMA_MATMUL=ggml ignored: rebuild bitnet-core with `--features experimental-ggml-kernels` for the hook build (still Rust kernels until a bridge is added)."
            );
        }
    });
}
