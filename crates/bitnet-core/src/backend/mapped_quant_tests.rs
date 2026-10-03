//! Draft fixtures: immutable mmap lifetime and selected row offsets.
//! Not compiled or exercised until isolated MLA timings have finished.
use super::*;
use std::path::{Path, PathBuf};

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn fixture() -> (Temporary, Arc<crate::gguf::GgufArchive>) {
    static ID: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "rbitnet-mapped-{}-{}.gguf",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let name = b"expert.bank";
    let mut data = Vec::new();
    data.extend_from_slice(&0x46554747u32.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&3u32.to_le_bytes());
    for dimension in [32u64, 5, 3] {
        data.extend_from_slice(&dimension.to_le_bytes());
    }
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.resize(data.len().div_ceil(32) * 32, 0);
    for i in 0..480 {
        data.extend_from_slice(&(((i * 19 % 101) as f32 - 50.) / 32.).to_le_bytes());
    }
    std::fs::write(&path, data).unwrap();
    let archive = Arc::new(crate::gguf::GgufArchive::mmap_path(Path::new(&path)).unwrap());
    (Temporary(path), archive)
}
fn input() -> Vec<f32> {
    (0..32).map(|i| ((i * 7 % 13) as f32 - 6.) / 16.).collect()
}
fn golden(payload: &[u8], x: &[f32]) -> Vec<f32> {
    payload
        .chunks_exact(x.len() * 4)
        .map(|row| {
            row.chunks_exact(4)
                .zip(x)
                .map(|(b, &x)| f32::from_le_bytes(b.try_into().unwrap()) as f64 * x as f64)
                .sum::<f64>() as f32
        })
        .collect()
}
#[test]
fn mapped_quant_spans_keep_archive_alive_without_vec_mirrors() {
    let (_temporary, archive) = fixture();
    let tensor = archive.tensors[0].clone();
    let original = archive.tensor_payload(&tensor).unwrap();
    let expected_pointer = unsafe { original.as_ptr().add(5 * 32 * 4) };
    let expected = golden(&original[5 * 32 * 4..10 * 32 * 4], &input());
    let weak = Arc::downgrade(&archive);
    let matrix = CudaDeviceQuantMatrix::from_archive_range(
        None,
        Arc::clone(&archive),
        &tensor,
        5 * 32 * 4,
        5,
        32,
        true,
    )
    .unwrap();
    assert_eq!(matrix.host_payload().as_ptr(), expected_pointer);
    assert_eq!(matrix.bytes(), 5 * 32 * 4);
    assert!(matches!(
        matrix.host.as_ref(),
        QuantHostBacking::Mapped { .. }
    ));
    let clone = matrix.clone();
    drop(archive);
    assert!(weak.upgrade().is_some());
    let actual: Vec<_> = matrix
        .host_payload()
        .chunks_exact(128)
        .map(|row| crate::ggml::dot_row(0, row, &input()).unwrap())
        .collect();
    assert_eq!(actual, expected);
    drop(matrix);
    assert_eq!(clone.host_payload().as_ptr(), expected_pointer);
    drop(clone);
    assert!(weak.upgrade().is_none());
}
#[test]
fn mapped_quant_rejects_invalid_spans_and_row_geometry() {
    let (_temporary, archive) = fixture();
    let tensor = archive.tensors[0].clone();
    for (start, rows, cols) in [
        (1, 5, 32),
        (0, 5, 31),
        (0, 0, 32),
        (14 * 128, 2, 32),
        (usize::MAX, 5, 32),
        (0, usize::MAX, 32),
    ] {
        assert!(CudaDeviceQuantMatrix::from_archive_range(
            None,
            Arc::clone(&archive),
            &tensor,
            start,
            rows,
            cols,
            true
        )
        .is_err());
    }
    let matrix = CudaDeviceQuantMatrix::from_payload(None, 0, vec![0; 128], 1, 32).unwrap();
    assert!(matches!(matrix.host.as_ref(), QuantHostBacking::Owned(_)));
}
#[test]
fn optional_mapped_quant_gpu_rows_refill_and_leases() {
    if std::env::var("RBITNET_CUDA_QUANT_SMOKE").as_deref() != Ok("1") {
        return;
    }
    let (_temporary, archive) = fixture();
    let tensor = archive.tensors[0].clone();
    let rt = CudaRuntime::try_load().expect("CUDA required");
    let before = rt.managed_memory_stats().unwrap();
    let mut matrix = CudaDeviceQuantMatrix::from_archive_range(
        Some(&rt),
        Arc::clone(&archive),
        &tensor,
        0,
        5,
        32,
        true,
    )
    .unwrap();
    assert!(matrix.is_device_resident());
    let address = matrix.device_address();
    let held = matrix.clone();
    assert!(!matrix
        .refill_archive_range(Arc::clone(&archive), &tensor, 5 * 128)
        .unwrap());
    drop(held);
    assert!(matrix
        .refill_archive_range(Arc::clone(&archive), &tensor, 10 * 128)
        .unwrap());
    assert_eq!(matrix.device_address(), address);
    assert_eq!(matrix.host_payload().as_ptr(), unsafe {
        archive
            .tensor_payload(&tensor)
            .unwrap()
            .as_ptr()
            .add(10 * 128)
    });
    let x = input();
    let expected = golden(
        &archive.tensor_payload(&tensor).unwrap()[11 * 128..14 * 128],
        &x,
    );
    let actual = matrix.matvec_rows(&x, 1, 3).unwrap();
    for (&a, &b) in actual.iter().zip(&expected) {
        assert!((a - b).abs() < 1e-5);
    }
    assert!(matrix.matvec_rows(&x, 4, 2).is_err());
    assert!(matrix
        .refill_archive_range(Arc::clone(&archive), &tensor, 14 * 128)
        .is_err());
    let mut wrong = tensor.clone();
    wrong.ggml_type = 8;
    assert!(matrix
        .refill_archive_range(Arc::clone(&archive), &wrong, 0)
        .is_err());
    drop(matrix);
    let after = rt.managed_memory_stats().unwrap();
    assert_eq!(
        after.categories[device_memory::EXPERTS as usize],
        before.categories[device_memory::EXPERTS as usize]
    );
}
