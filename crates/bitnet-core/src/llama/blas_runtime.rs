//! Optional OpenBLAS `cblas_sgemv` via dynamic load (`RBITNET_BLAS=1`).
//! Install a system OpenBLAS build (DLL / `.so` / `.dylib`) exposing `cblas_sgemv`.

use std::sync::OnceLock;

use libloading::Library;

use crate::error::{BitNetError, Result};

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

type CblasSgemvFn = unsafe extern "C" fn(
    layout: i32,
    trans: i32,
    m: i32,
    n: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    x: *const f32,
    incx: i32,
    beta: f32,
    y: *mut f32,
    incy: i32,
);

struct BlasLib {
    _lib: Library,
    sgemv: CblasSgemvFn,
}

static BLAS: OnceLock<Option<BlasLib>> = OnceLock::new();

fn load_blas() -> Option<BlasLib> {
    let candidates: &[&str] = if cfg!(target_os = "windows") {
        &["libopenblas.dll", "openblas.dll"]
    } else if cfg!(target_os = "macos") {
        &[
            "libopenblas.dylib",
            "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
            "/usr/local/opt/openblas/lib/libopenblas.dylib",
        ]
    } else {
        &["libopenblas.so.0", "libopenblas.so"]
    };
    for path in candidates {
        let Ok(lib) = (unsafe { Library::new(path) }) else {
            continue;
        };
        let Ok(sym) = (unsafe { lib.get::<CblasSgemvFn>(b"cblas_sgemv\0") }) else {
            continue;
        };
        let sgemv = *sym;
        return Some(BlasLib { _lib: lib, sgemv });
    }
    None
}

fn blas_lib() -> Option<&'static BlasLib> {
    if !blas_attention_enabled() {
        return None;
    }
    BLAS.get_or_init(|| {
        let b = load_blas();
        if b.is_none() {
            tracing::warn!(
                "RBITNET_BLAS is enabled but OpenBLAS could not be loaded (install OpenBLAS and ensure the DLL/.so is on PATH / LD_LIBRARY_PATH). \
                 Falling back to scalar GEMV. See docs/USAGE.md."
            );
        }
        b
    })
    .as_ref()
}

/// True when `RBITNET_BLAS` requests acceleration **and** `cblas_sgemv` loaded successfully.
#[inline]
pub fn blas_ready() -> bool {
    blas_lib().is_some()
}

#[inline]
pub fn blas_attention_enabled() -> bool {
    matches!(
        std::env::var("RBITNET_BLAS").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

/// `y = alpha * A * x + beta * y` with `A` row-major `m × n`, `lda >= n`, `len(a) >= m * lda`.
pub fn sgemv_row_major_notrans(
    a: &[f32],
    m: usize,
    n: usize,
    lda: usize,
    alpha: f32,
    x: &[f32],
    y: &mut [f32],
    beta: f32,
) -> Result<()> {
    let Some(lib) = blas_lib() else {
        return Err(BitNetError::Inference(
            "OpenBLAS cblas_sgemv unavailable (set RBITNET_BLAS=1 and install libopenblas). See docs/USAGE.md"
                .into(),
        ));
    };
    if x.len() != n {
        return Err(BitNetError::Inference("blas sgemv: x len".into()));
    }
    if y.len() != m {
        return Err(BitNetError::Inference("blas sgemv: y len".into()));
    }
    if a.len() < m.checked_mul(lda).unwrap_or(0) {
        return Err(BitNetError::Inference("blas sgemv: A buffer too small".into()));
    }
    unsafe {
        (lib.sgemv)(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            alpha,
            a.as_ptr(),
            lda as i32,
            x.as_ptr(),
            1,
            beta,
            y.as_mut_ptr(),
            1,
        );
    }
    Ok(())
}
