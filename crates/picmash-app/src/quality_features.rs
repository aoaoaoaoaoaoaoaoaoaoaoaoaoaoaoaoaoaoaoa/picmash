use anyhow::Context;
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::{identity::canonical_embedding_image, model::AssetId};

pub const QUALITY_FEATURE_REVISION: &str = "quality_features_v1";
pub const TECHNICAL_PRIOR_ARTIFACT_KEY: &str = "linear_technical_prior_head_v1";
pub const TECHNICAL_DESCRIPTOR_DIM: usize = 7;
pub const VIBE_DESCRIPTOR_DIM: usize = 12;
const TECH_SIDE: u32 = 160;
const VIBE_SIDE: u32 = 128;
const WAVELET_LEVELS: usize = 3;
const TECHNICAL_PRIOR_RIDGE: f32 = 0.15;
const TECHNICAL_PRIOR_VARIANCE_FLOOR: f32 = 0.45;
const TECHNICAL_PRIOR_VARIANCE_CEILING: f32 = 1.4;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AssetQualityFeatures {
    pub technical: [f32; TECHNICAL_DESCRIPTOR_DIM],
    pub vibe: [f32; VIBE_DESCRIPTOR_DIM],
}

impl AssetQualityFeatures {
    #[must_use]
    pub const fn neutral() -> Self {
        Self {
            technical: [0.0; TECHNICAL_DESCRIPTOR_DIM],
            vibe: [0.0; VIBE_DESCRIPTOR_DIM],
        }
    }
}

#[derive(Debug, Clone)]
pub struct StoredAssetQualityFeatures {
    pub asset_id: AssetId,
    pub extractor_revision: String,
    pub features: AssetQualityFeatures,
    pub updated_at: OffsetDateTime,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LinearTechnicalPriorHead {
    pub weights: [f32; TECHNICAL_DESCRIPTOR_DIM],
    pub bias: f32,
    pub residual_variance: f32,
}

impl Default for LinearTechnicalPriorHead {
    fn default() -> Self {
        Self {
            weights: [0.32, 0.18, 0.95, -0.22, -0.28, -0.42, -0.42],
            bias: -0.95,
            residual_variance: 0.65,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TechnicalPriorSample {
    pub descriptor: [f32; TECHNICAL_DESCRIPTOR_DIM],
    pub target_mean: f32,
    pub target_variance: f32,
}

pub fn extract_asset_quality_features(bytes: &[u8]) -> anyhow::Result<AssetQualityFeatures> {
    let image = canonical_embedding_image(bytes).context("decoding asset for quality features")?;
    Ok(AssetQualityFeatures {
        technical: technical_descriptor(&image),
        vibe: vibe_descriptor(&image),
    })
}

#[must_use]
pub fn technical_prior_mean(descriptor: &[f32; TECHNICAL_DESCRIPTOR_DIM]) -> f32 {
    technical_prior_mean_with_head(&LinearTechnicalPriorHead::default(), descriptor)
}

#[must_use]
pub fn technical_prior_variance(descriptor: &[f32; TECHNICAL_DESCRIPTOR_DIM]) -> f32 {
    technical_prior_variance_with_head(&LinearTechnicalPriorHead::default(), descriptor)
}

#[must_use]
pub fn technical_prior_mean_with_head(
    head: &LinearTechnicalPriorHead,
    descriptor: &[f32; TECHNICAL_DESCRIPTOR_DIM],
) -> f32 {
    let raw = head.bias
        + head
            .weights
            .iter()
            .zip(descriptor.iter())
            .map(|(weight, value)| weight * value)
            .sum::<f32>();
    2.2 * raw.tanh()
}

#[must_use]
pub fn technical_prior_variance_with_head(
    head: &LinearTechnicalPriorHead,
    descriptor: &[f32; TECHNICAL_DESCRIPTOR_DIM],
) -> f32 {
    let artifact_load = descriptor[3] + descriptor[4] + descriptor[5] + descriptor[6];
    (head.residual_variance + 0.18 * artifact_load).clamp(
        TECHNICAL_PRIOR_VARIANCE_FLOOR,
        TECHNICAL_PRIOR_VARIANCE_CEILING,
    )
}

#[must_use]
pub fn fit_linear_technical_prior_head(
    samples: &[TechnicalPriorSample],
) -> Option<LinearTechnicalPriorHead> {
    if samples.len() < TECHNICAL_DESCRIPTOR_DIM + 1 {
        return None;
    }
    let rows = samples.len();
    let cols = TECHNICAL_DESCRIPTOR_DIM + 1;
    let mut x = DMatrix::<f32>::zeros(rows, cols);
    let mut y = DVector::<f32>::zeros(rows);
    let mut w = DVector::<f32>::zeros(rows);
    for (row, sample) in samples.iter().enumerate() {
        x[(row, 0)] = 1.0;
        for (index, value) in sample.descriptor.iter().copied().enumerate() {
            x[(row, index + 1)] = value;
        }
        y[row] = inverse_tanh_scaled(sample.target_mean);
        w[row] = 1.0 / sample.target_variance.max(1e-3);
    }
    let w_diag = DMatrix::<f32>::from_diagonal(&w);
    let xt = x.transpose();
    let normal = &xt * &w_diag * &x + DMatrix::<f32>::identity(cols, cols) * TECHNICAL_PRIOR_RIDGE;
    let rhs = &xt * &w_diag * &y;
    let solution = normal.lu().solve(&rhs)?;
    let mut weights = [0.0; TECHNICAL_DESCRIPTOR_DIM];
    for index in 0..TECHNICAL_DESCRIPTOR_DIM {
        weights[index] = solution[index + 1];
    }
    let fitted = LinearTechnicalPriorHead {
        weights,
        bias: solution[0],
        residual_variance: regression_residual_variance(samples, solution.as_slice()),
    };
    Some(fitted)
}

fn inverse_tanh_scaled(value: f32) -> f32 {
    let clamped = (value / 2.2).clamp(-0.999, 0.999);
    0.5 * ((1.0 + clamped) / (1.0 - clamped)).ln()
}

fn regression_residual_variance(samples: &[TechnicalPriorSample], solution: &[f32]) -> f32 {
    let (weighted_sum, total_weight) = samples.iter().fold((0.0, 0.0), |(sum, total), sample| {
        let predicted = solution[0]
            + sample
                .descriptor
                .iter()
                .zip(solution[1..].iter())
                .map(|(value, weight)| value * weight)
                .sum::<f32>();
        let residual = inverse_tanh_scaled(sample.target_mean) - predicted;
        let weight = 1.0 / sample.target_variance.max(1e-3);
        (sum + weight * residual * residual, total + weight)
    });
    (weighted_sum / total_weight.max(1e-3)).clamp(
        TECHNICAL_PRIOR_VARIANCE_FLOOR,
        TECHNICAL_PRIOR_VARIANCE_CEILING,
    )
}

fn technical_descriptor(image: &DynamicImage) -> [f32; TECHNICAL_DESCRIPTOR_DIM] {
    let (width, height) = image.dimensions();
    let tech_gray = image
        .resize_exact(TECH_SIDE, TECH_SIDE, FilterType::Triangle)
        .to_luma8();
    let plane = tech_gray
        .pixels()
        .map(|pixel| f32::from(pixel.0[0]) / 255.0)
        .collect::<Vec<_>>();
    let sharpness = mean_abs_laplacian(&plane, TECH_SIDE as usize, TECH_SIDE as usize);
    let wavelet = haar_wavelet_band_energies(&plane, TECH_SIDE as usize, TECH_SIDE as usize);
    let noise = wavelet[2];
    let blockiness = jpeg_blockiness(&plane, TECH_SIDE as usize, TECH_SIDE as usize);
    let (dark_clip, bright_clip) = clipping_fractions(&plane);
    [
        (((width as f32 * height as f32) / 1_000_000.0).max(1e-6)).ln(),
        (width.min(height) as f32).max(1.0).ln(),
        sharpness,
        noise,
        blockiness,
        dark_clip,
        bright_clip,
    ]
}

fn vibe_descriptor(image: &DynamicImage) -> [f32; VIBE_DESCRIPTOR_DIM] {
    let rgb = image
        .resize_exact(VIBE_SIDE, VIBE_SIDE, FilterType::Triangle)
        .to_rgb8();
    let mut luma = Vec::with_capacity((VIBE_SIDE * VIBE_SIDE) as usize);
    let mut opp_u = Vec::with_capacity((VIBE_SIDE * VIBE_SIDE) as usize);
    let mut opp_v = Vec::with_capacity((VIBE_SIDE * VIBE_SIDE) as usize);
    for pixel in rgb.pixels() {
        let r = f32::from(pixel.0[0]) / 255.0;
        let g = f32::from(pixel.0[1]) / 255.0;
        let b = f32::from(pixel.0[2]) / 255.0;
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        luma.push(y);
        opp_u.push(r - g);
        opp_v.push(0.5 * (r + g) - b);
    }
    let wavelet = haar_wavelet_band_energies(&luma, VIBE_SIDE as usize, VIBE_SIDE as usize);
    let (_, l_std) = mean_std(&luma);
    let (u_mean, u_std) = mean_std(&opp_u);
    let (v_mean, v_std) = mean_std(&opp_v);
    [
        wavelet[0],
        wavelet[1],
        wavelet[2],
        wavelet[3],
        wavelet[4],
        wavelet[5],
        wavelet[6],
        wavelet[7],
        wavelet[8],
        l_std,
        u_mean + u_std,
        v_mean + v_std,
    ]
}

fn mean_abs_laplacian(samples: &[f32], width: usize, height: usize) -> f32 {
    if width < 3 || height < 3 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let center = samples[y * width + x];
            let lap = 4.0 * center
                - samples[(y - 1) * width + x]
                - samples[(y + 1) * width + x]
                - samples[y * width + (x - 1)]
                - samples[y * width + (x + 1)];
            total += lap.abs();
            count += 1;
        }
    }
    (total / count.max(1) as f32).sqrt()
}

fn jpeg_blockiness(samples: &[f32], width: usize, height: usize) -> f32 {
    if width < 16 || height < 16 {
        return 0.0;
    }
    let mut boundary = 0.0;
    let mut interior = 0.0;
    let mut boundary_count = 0usize;
    let mut interior_count = 0usize;
    for y in 0..height {
        for x in 1..width {
            let diff = (samples[y * width + x] - samples[y * width + x - 1]).abs();
            if x % 8 == 0 {
                boundary += diff;
                boundary_count += 1;
            } else if x % 8 == 4 {
                interior += diff;
                interior_count += 1;
            }
        }
    }
    for y in 1..height {
        for x in 0..width {
            let diff = (samples[y * width + x] - samples[(y - 1) * width + x]).abs();
            if y % 8 == 0 {
                boundary += diff;
                boundary_count += 1;
            } else if y % 8 == 4 {
                interior += diff;
                interior_count += 1;
            }
        }
    }
    let boundary_mean = boundary / boundary_count.max(1) as f32;
    let interior_mean = interior / interior_count.max(1) as f32;
    (boundary_mean - interior_mean).max(0.0)
}

fn clipping_fractions(samples: &[f32]) -> (f32, f32) {
    let mut dark = 0usize;
    let mut bright = 0usize;
    for &sample in samples {
        if sample <= 0.02 {
            dark += 1;
        }
        if sample >= 0.98 {
            bright += 1;
        }
    }
    let inv = 1.0 / samples.len().max(1) as f32;
    (dark as f32 * inv, bright as f32 * inv)
}

fn mean_std(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    let variance = samples
        .iter()
        .map(|sample| {
            let delta = *sample - mean;
            delta * delta
        })
        .sum::<f32>()
        / samples.len() as f32;
    (mean, variance.sqrt())
}

fn haar_wavelet_band_energies(samples: &[f32], width: usize, height: usize) -> [f32; 9] {
    let mut out = [0.0; 9];
    let mut current = samples.to_vec();
    let mut current_width = width;
    let mut current_height = height;
    for level in 0..WAVELET_LEVELS {
        if current_width < 2 || current_height < 2 {
            break;
        }
        let next_width = current_width / 2;
        let next_height = current_height / 2;
        let mut low = vec![0.0; next_width * next_height];
        let mut horizontal = 0.0;
        let mut vertical = 0.0;
        let mut diagonal = 0.0;
        let mut count = 0usize;
        for y in 0..next_height {
            for x in 0..next_width {
                let base = 2 * y * current_width + 2 * x;
                let a = current[base];
                let b = current[base + 1];
                let c = current[base + current_width];
                let d = current[base + current_width + 1];
                let ll = (a + b + c + d) * 0.25;
                let hh = (a - b - c + d) * 0.25;
                let hv = (a + b - c - d) * 0.25;
                let vh = (a - b + c - d) * 0.25;
                low[y * next_width + x] = ll;
                horizontal += hv.abs();
                vertical += vh.abs();
                diagonal += hh.abs();
                count += 1;
            }
        }
        let inv = 1.0 / count.max(1) as f32;
        let offset = level * 3;
        out[offset] = horizontal * inv;
        out[offset + 1] = vertical * inv;
        out[offset + 2] = diagonal * inv;
        current = low;
        current_width = next_width;
        current_height = next_height;
    }
    out
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgb, RgbImage};

    use super::{
        LinearTechnicalPriorHead, TECHNICAL_DESCRIPTOR_DIM, TECHNICAL_PRIOR_VARIANCE_CEILING,
        TECHNICAL_PRIOR_VARIANCE_FLOOR, TechnicalPriorSample, VIBE_DESCRIPTOR_DIM,
        clipping_fractions, fit_linear_technical_prior_head, haar_wavelet_band_energies,
        mean_abs_laplacian, technical_descriptor, technical_prior_mean_with_head,
        technical_prior_variance_with_head, vibe_descriptor,
    };

    #[test]
    fn wavelet_descriptor_has_expected_dim() {
        let plane = vec![0.5; 128 * 128];
        let energies = haar_wavelet_band_energies(&plane, 128, 128);
        assert_eq!(energies.len(), 9);
    }

    #[test]
    fn technical_and_vibe_extractors_are_stable() {
        let mut image = RgbImage::new(192, 128);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            let tone = ((x + y) % 255) as u8;
            *pixel = Rgb([tone, tone.saturating_add(16), 255 - tone]);
        }
        let dynamic = DynamicImage::ImageRgb8(image);
        let technical = technical_descriptor(&dynamic);
        let vibe = vibe_descriptor(&dynamic);
        assert_eq!(technical.len(), TECHNICAL_DESCRIPTOR_DIM);
        assert_eq!(vibe.len(), VIBE_DESCRIPTOR_DIM);
        assert!(technical.iter().all(|value| value.is_finite()));
        assert!(vibe.iter().all(|value| value.is_finite()));
        assert!(mean_abs_laplacian(&vec![0.5; 64], 8, 8).is_finite());
        assert!(clipping_fractions(&[0.0, 0.5, 1.0]).0 > 0.0);
    }

    #[test]
    fn linear_technical_prior_head_recovers_synthetic_map() {
        let truth = LinearTechnicalPriorHead {
            weights: [0.24, -0.17, 0.81, -0.33, -0.28, -0.19, -0.11],
            bias: -0.42,
            residual_variance: 0.58,
        };
        let samples = (0..24)
            .map(|index| {
                let axis = index as f32;
                let descriptor = [
                    (0.11 * axis).sin(),
                    (0.07 * axis).cos(),
                    0.15 * axis - 1.2,
                    ((index % 5) as f32) * 0.18,
                    ((index % 7) as f32) * 0.12,
                    ((index % 3) as f32) * 0.09,
                    ((index % 4) as f32) * 0.06,
                ];
                TechnicalPriorSample {
                    descriptor,
                    target_mean: technical_prior_mean_with_head(&truth, &descriptor),
                    target_variance: technical_prior_variance_with_head(&truth, &descriptor),
                }
            })
            .collect::<Vec<_>>();
        let fitted = fit_linear_technical_prior_head(&samples).expect("fit synthetic head");
        let holdout = [
            [0.31, -0.22, -0.8, 0.36, 0.24, 0.09, 0.18],
            [-0.48, 0.41, 1.1, 0.18, 0.12, 0.0, 0.06],
            [0.19, 0.77, 0.3, 0.54, 0.24, 0.18, 0.12],
        ];
        for descriptor in holdout {
            let predicted = technical_prior_mean_with_head(&fitted, &descriptor);
            let expected = technical_prior_mean_with_head(&truth, &descriptor);
            assert!(
                (predicted - expected).abs() < 0.3,
                "expected recovered prior mean {predicted} ~= {expected}"
            );
        }
        assert!(
            fitted.residual_variance.is_finite()
                && fitted.residual_variance >= TECHNICAL_PRIOR_VARIANCE_FLOOR
                && fitted.residual_variance <= TECHNICAL_PRIOR_VARIANCE_CEILING
        );
    }
}
