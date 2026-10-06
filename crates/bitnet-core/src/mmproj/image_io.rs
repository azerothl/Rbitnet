//! Decode JPEG/PNG (raw bytes or data-URL) into CLIP-normalized CHW f32.

use base64::Engine as _;
use image::imageops::FilterType;
use image::DynamicImage;

use crate::error::{BitNetError, Result};

use super::config::MmprojConfig;

/// RGB image resized to `image_size×image_size` and CLIP-normalized, CHW planar.
#[derive(Debug, Clone)]
pub struct NormalizedImage {
    pub width: usize,
    pub height: usize,
    /// Length `3 * height * width`, channel-major (R plane, G plane, B plane).
    pub chw: Vec<f32>,
}

/// Decode image bytes or a `data:image/...;base64,...` URL into a normalized CHW tensor.
pub fn decode_and_normalize(bytes_or_data_url: &[u8], cfg: &MmprojConfig) -> Result<NormalizedImage> {
    let raw = decode_bytes_or_data_url(bytes_or_data_url)?;
    let img = image::load_from_memory(&raw).map_err(|e| {
        BitNetError::Inference(format!("failed to decode image (jpeg/png): {e}"))
    })?;
    normalize_dynamic_image(img, cfg)
}

/// Strip a data-URL prefix and base64-decode, or return the bytes unchanged.
pub fn decode_bytes_or_data_url(input: &[u8]) -> Result<Vec<u8>> {
    if let Some(rest) = strip_data_url_prefix(input) {
        let cleaned: Vec<u8> = rest
            .iter()
            .copied()
            .filter(|b| !b.is_ascii_whitespace())
            .collect();
        return base64::engine::general_purpose::STANDARD
            .decode(&cleaned)
            .map_err(|e| BitNetError::Inference(format!("invalid image data-URL base64: {e}")));
    }
    Ok(input.to_vec())
}

fn strip_data_url_prefix(input: &[u8]) -> Option<&[u8]> {
    const PREFIX: &[u8] = b"data:";
    if input.len() < PREFIX.len() || !input[..PREFIX.len()].eq_ignore_ascii_case(PREFIX) {
        return None;
    }
    let comma = input.iter().position(|&b| b == b',')?;
    let header = &input[PREFIX.len()..comma];
    // Require `;base64` in the media-type header (case-insensitive).
    let header_l = header.to_ascii_lowercase();
    if !header_l.windows(b";base64".len()).any(|w| w == b";base64") {
        return None;
    }
    Some(&input[comma + 1..])
}

fn normalize_dynamic_image(img: DynamicImage, cfg: &MmprojConfig) -> Result<NormalizedImage> {
    let size = cfg.image_size as u32;
    if size == 0 {
        return Err(BitNetError::Inference("mmproj image_size is zero".into()));
    }
    // LLaVA 1.5 / CLIP: square resize (bicubic-class filter).
    let rgb = img
        .resize_exact(size, size, FilterType::CatmullRom)
        .to_rgb8();
    let w = size as usize;
    let h = size as usize;
    let mut chw = vec![0.0f32; 3 * h * w];
    let mean = cfg.image_mean;
    let std = cfg.image_std;
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x as u32, y as u32);
            let idx = y * w + x;
            for c in 0..3 {
                let v = f32::from(p[c]) / 255.0;
                let denom = if std[c].abs() < 1e-12 { 1.0 } else { std[c] };
                chw[c * h * w + idx] = (v - mean[c]) / denom;
            }
        }
    }
    Ok(NormalizedImage {
        width: w,
        height: h,
        chw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmproj::config::{CLIP_IMAGE_MEAN, CLIP_IMAGE_STD, MmprojConfig, VitFfnOp};

    fn tiny_cfg(size: usize) -> MmprojConfig {
        MmprojConfig {
            image_size: size,
            patch_size: 14,
            n_embd: 64,
            n_layer: 2,
            n_head: 4,
            n_ff: 128,
            projection_dim: 64,
            layer_norm_eps: 1e-5,
            image_mean: CLIP_IMAGE_MEAN,
            image_std: CLIP_IMAGE_STD,
            ffn_op: VitFfnOp::GeluQuick,
            has_llava_projector: true,
            projector_type: None,
        }
    }

    /// Minimal 2×2 RGB PNG.
    fn tiny_png_bytes() -> Vec<u8> {
        // 2x2 solid-ish PNG generated via the `image` crate at runtime.
        let mut img = image::RgbImage::new(2, 2);
        img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        img.put_pixel(1, 0, image::Rgb([0, 255, 0]));
        img.put_pixel(0, 1, image::Rgb([0, 0, 255]));
        img.put_pixel(1, 1, image::Rgb([255, 255, 255]));
        let mut buf = Vec::new();
        let enc = image::codecs::png::PngEncoder::new(&mut buf);
        use image::ImageEncoder;
        enc.write_image(img.as_raw(), 2, 2, image::ExtendedColorType::Rgb8)
            .unwrap();
        buf
    }

    #[test]
    fn decode_png_bytes_normalizes_chw() {
        let cfg = tiny_cfg(28);
        let out = decode_and_normalize(&tiny_png_bytes(), &cfg).unwrap();
        assert_eq!(out.width, 28);
        assert_eq!(out.height, 28);
        assert_eq!(out.chw.len(), 3 * 28 * 28);
        // Normalized values should be finite and typically within a few units of zero.
        assert!(out.chw.iter().all(|v| v.is_finite()));
        let max_abs = out.chw.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs < 10.0, "unexpectedly large normalized value {max_abs}");
    }

    #[test]
    fn data_url_base64_roundtrip() {
        let png = tiny_png_bytes();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
        let url = format!("data:image/png;base64,{b64}");
        let cfg = tiny_cfg(14);
        let out = decode_and_normalize(url.as_bytes(), &cfg).unwrap();
        assert_eq!(out.chw.len(), 3 * 14 * 14);
    }
}
