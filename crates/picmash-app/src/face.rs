//! Face detection via SCRFD and alignment for `FaceMash` embedding.

use anyhow::Context;
use image::{DynamicImage, GenericImageView, Rgb, RgbImage, imageops::FilterType};
use imageproc::geometric_transformations::{Interpolation, Projection, warp_into};
use nalgebra::{SMatrix, SVector};
use ort::value::Tensor;

pub const SCRFD_INPUT_SIDE: u32 = 640;
const SCRFD_CONFIDENCE_THRESHOLD: f32 = 0.5;
const NMS_IOU_THRESHOLD: f32 = 0.4;
pub const ALIGNED_SIDE: u32 = 112;
pub const DISPLAY_FACE_SIDE: u32 = 224;

/// Canonical `ArcFace` alignment template for 112×112 crops.
const ALIGNMENT_TEMPLATE: [(f32, f32); 5] = [
    (38.2946, 51.6963), // left eye
    (73.5318, 51.5014), // right eye
    (56.0252, 71.7366), // nose tip
    (41.5493, 92.3655), // left mouth corner
    (70.7299, 92.2041), // right mouth corner
];

#[derive(Debug, Clone)]
pub struct FaceBbox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, Clone)]
pub struct FaceLandmarks(pub [(f32, f32); 5]);

impl FaceLandmarks {
    pub fn to_flat_bytes(&self) -> Vec<u8> {
        self.0
            .iter()
            .flat_map(|&(x, y)| {
                let mut buf = Vec::with_capacity(8);
                buf.extend_from_slice(&x.to_le_bytes());
                buf.extend_from_slice(&y.to_le_bytes());
                buf
            })
            .collect()
    }

    pub fn from_flat_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 40 {
            return None;
        }
        let mut landmarks = [(0.0f32, 0.0f32); 5];
        for (i, chunk) in bytes.chunks_exact(8).enumerate() {
            landmarks[i] = (
                f32::from_le_bytes(chunk[..4].try_into().ok()?),
                f32::from_le_bytes(chunk[4..8].try_into().ok()?),
            );
        }
        Some(Self(landmarks))
    }

    pub fn canonical_alignment_order(&self) -> [(f32, f32); 5] {
        canonicalize_alignment_landmarks(&self.0)
    }
}

#[derive(Debug, Clone)]
pub struct DetectedFace {
    pub bbox: FaceBbox,
    pub landmarks: FaceLandmarks,
    pub confidence: f32,
}

pub struct AlignedFace {
    pub crop: DynamicImage,
    pub source: DetectedFace,
}

#[derive(Debug, Clone, Copy)]
pub struct FaceCropSpec {
    pub side: u32,
    pub context_scale: f32,
}

pub const EMBEDDING_FACE_CROP: FaceCropSpec = FaceCropSpec {
    side: ALIGNED_SIDE,
    context_scale: 1.0,
};

pub const DISPLAY_FACE_CROP: FaceCropSpec = FaceCropSpec {
    side: DISPLAY_FACE_SIDE,
    context_scale: 2.0,
};

/// Parsed SCRFD outputs for a single stride level.
pub struct ScrfdStrideOutput {
    pub stride: u32,
    pub num_anchors: usize,
    pub score_classes: usize,
    pub scores: Vec<f32>,
    pub bboxes: Vec<f32>,
    pub kps: Vec<f32>,
}

/// Preprocess an image for SCRFD: resize-to-fit into 640×640, place the
/// resized image at the top-left origin, and normalize with
/// `(pixel − 127.5) / 128.0`.
///
/// Returns `(tensor, scale, pad_x, pad_y)` for coordinate recovery.
pub fn scrfd_input_tensor(image: &DynamicImage) -> anyhow::Result<(Tensor<f32>, f32, f32, f32)> {
    let (w, h) = image.dimensions();
    let scale = (SCRFD_INPUT_SIDE as f32 / w as f32).min(SCRFD_INPUT_SIDE as f32 / h as f32);
    let new_w = (w as f32 * scale).round() as u32;
    let new_h = (h as f32 * scale).round() as u32;

    let resized = image
        .resize_exact(new_w, new_h, FilterType::CatmullRom)
        .to_rgb8();
    let mut canvas = RgbImage::from_pixel(SCRFD_INPUT_SIDE, SCRFD_INPUT_SIDE, Rgb([0, 0, 0]));
    image::imageops::overlay(&mut canvas, &resized, 0, 0);

    let side = SCRFD_INPUT_SIDE as usize;
    let mut tensor_data = Vec::with_capacity(3 * side * side);
    for channel in 0..3usize {
        for pixel in canvas.pixels() {
            tensor_data.push((f32::from(pixel[channel]) - 127.5) / 128.0);
        }
    }

    let tensor = Tensor::from_array(([1, 3, side, side], tensor_data))
        .context("building SCRFD input tensor")?;
    Ok((tensor, scale, 0.0, 0.0))
}

/// Decode raw SCRFD stride-level outputs into face detections,
/// mapping coordinates back to the original image space.
pub fn decode_scrfd_outputs(
    strides: &[ScrfdStrideOutput],
    scale: f32,
    pad_x: f32,
    pad_y: f32,
) -> Vec<DetectedFace> {
    let mut faces = Vec::new();
    for stride_out in strides {
        let feat_h = SCRFD_INPUT_SIDE / stride_out.stride;
        let feat_w = SCRFD_INPUT_SIDE / stride_out.stride;
        let locations = (feat_h * feat_w) as usize;
        let anchors_per_loc = stride_out.num_anchors / locations.max(1);

        for anchor_idx in 0..stride_out.num_anchors {
            let score = if stride_out.score_classes == 1 {
                stride_out.scores[anchor_idx]
            } else {
                stride_out.scores[anchor_idx * stride_out.score_classes + 1]
            };
            if score < SCRFD_CONFIDENCE_THRESHOLD {
                continue;
            }

            let loc_idx = anchor_idx / anchors_per_loc.max(1);
            let col = loc_idx % feat_w as usize;
            let row = loc_idx / feat_w as usize;
            let cx = col as f32 * stride_out.stride as f32;
            let cy = row as f32 * stride_out.stride as f32;

            let bi = anchor_idx * 4;
            let stride = stride_out.stride as f32;
            let x1 = (cx - stride_out.bboxes[bi] * stride - pad_x) / scale;
            let y1 = (cy - stride_out.bboxes[bi + 1] * stride - pad_y) / scale;
            let x2 = (cx + stride_out.bboxes[bi + 2] * stride - pad_x) / scale;
            let y2 = (cy + stride_out.bboxes[bi + 3] * stride - pad_y) / scale;

            let ki = anchor_idx * 10;
            let mut landmarks = [(0.0f32, 0.0f32); 5];
            for (point_index, landmark) in landmarks.iter_mut().enumerate() {
                *landmark = (
                    (cx + stride_out.kps[ki + point_index * 2] * stride - pad_x) / scale,
                    (cy + stride_out.kps[ki + point_index * 2 + 1] * stride - pad_y) / scale,
                );
            }

            faces.push(DetectedFace {
                bbox: FaceBbox {
                    x: x1,
                    y: y1,
                    w: x2 - x1,
                    h: y2 - y1,
                },
                landmarks: FaceLandmarks(landmarks),
                confidence: score,
            });
        }
    }

    nms(&mut faces);
    faces
}

/// Tight face crop used for face embeddings.
pub fn align_face_for_embedding(image: &DynamicImage, face: &DetectedFace) -> AlignedFace {
    align_face_with_spec(image, face, EMBEDDING_FACE_CROP)
}

/// Looser portrait crop used for facemash display.
pub fn align_face_for_display(image: &DynamicImage, face: &DetectedFace) -> AlignedFace {
    align_face_with_spec(image, face, DISPLAY_FACE_CROP)
}

fn align_face_with_spec(
    image: &DynamicImage,
    face: &DetectedFace,
    spec: FaceCropSpec,
) -> AlignedFace {
    let src_landmarks = face.landmarks.canonical_alignment_order();
    let fitted_spec = fit_face_crop_spec(&src_landmarks, image.dimensions(), spec);
    let affine =
        estimate_similarity_transform(&src_landmarks, &scaled_alignment_template(fitted_spec));
    let crop = warp_affine(image, &affine, spec.side, spec.side);
    AlignedFace {
        crop: DynamicImage::ImageRgb8(crop),
        source: face.clone(),
    }
}

// ─── internals ───────────────────────────────────────────────

fn nms(faces: &mut Vec<DetectedFace>) {
    faces.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut suppressed = vec![false; faces.len()];
    for i in 0..faces.len() {
        if suppressed[i] {
            continue;
        }
        for j in (i + 1)..faces.len() {
            if !suppressed[j] && iou(&faces[i].bbox, &faces[j].bbox) > NMS_IOU_THRESHOLD {
                suppressed[j] = true;
            }
        }
    }
    let mut idx = 0;
    faces.retain(|_| {
        let keep = !suppressed[idx];
        idx += 1;
        keep
    });
}

fn iou(a: &FaceBbox, b: &FaceBbox) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let union = a.w * a.h + b.w * b.h - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn canonicalize_alignment_landmarks(landmarks: &[(f32, f32); 5]) -> [(f32, f32); 5] {
    let [eye_a, eye_b, nose, mouth_a, mouth_b] = *landmarks;
    let (left_eye, right_eye) = sort_screen_left_to_right(eye_a, eye_b);
    let (left_mouth, right_mouth) = sort_screen_left_to_right(mouth_a, mouth_b);
    [left_eye, right_eye, nose, left_mouth, right_mouth]
}

fn sort_screen_left_to_right(a: (f32, f32), b: (f32, f32)) -> ((f32, f32), (f32, f32)) {
    if a.0 <= b.0 { (a, b) } else { (b, a) }
}

fn scaled_alignment_template(spec: FaceCropSpec) -> [(f32, f32); 5] {
    let base_span = ALIGNED_SIDE as f32 - 1.0;
    let target_span = spec.side as f32 - 1.0;
    let side_scale = target_span / base_span;
    let center = target_span * 0.5;
    ALIGNMENT_TEMPLATE.map(|(x, y)| {
        let scaled_x = x * side_scale;
        let scaled_y = y * side_scale;
        (
            center + (scaled_x - center) / spec.context_scale,
            center + (scaled_y - center) / spec.context_scale,
        )
    })
}

fn fit_face_crop_spec(
    src_landmarks: &[(f32, f32); 5],
    image_dims: (u32, u32),
    spec: FaceCropSpec,
) -> FaceCropSpec {
    if spec.context_scale <= 1.0 {
        return spec;
    }

    let fits = |context_scale: f32| {
        let affine = estimate_similarity_transform(
            src_landmarks,
            &scaled_alignment_template(FaceCropSpec {
                context_scale,
                ..spec
            }),
        );
        crop_fits_inside_source(&affine, spec.side, spec.side, image_dims)
    };

    if fits(spec.context_scale) {
        return spec;
    }
    if !fits(1.0) {
        return FaceCropSpec {
            context_scale: 1.0,
            ..spec
        };
    }

    let mut low = 1.0f32;
    let mut high = spec.context_scale;
    for _ in 0..16 {
        let mid = (low + high) * 0.5;
        if fits(mid) {
            low = mid;
        } else {
            high = mid;
        }
    }
    FaceCropSpec {
        context_scale: low,
        ..spec
    }
}

fn crop_fits_inside_source(fwd: &[f32; 6], out_w: u32, out_h: u32, image_dims: (u32, u32)) -> bool {
    let Some(inv) = invert_affine(fwd) else {
        return false;
    };
    let max_x = image_dims.0 as f32 - 1.0;
    let max_y = image_dims.1 as f32 - 1.0;
    [
        (0.0, 0.0),
        (out_w.saturating_sub(1) as f32, 0.0),
        (0.0, out_h.saturating_sub(1) as f32),
        (
            out_w.saturating_sub(1) as f32,
            out_h.saturating_sub(1) as f32,
        ),
    ]
    .into_iter()
    .all(|(x, y)| {
        let (sx, sy) = apply_affine(&inv, x, y);
        sx >= 0.0 && sy >= 0.0 && sx <= max_x && sy <= max_y
    })
}

/// Estimate a 2D similarity transform (rotation + uniform scale + translation)
/// from 5 source landmarks to 5 target landmarks via least-squares.
///
/// Returns affine matrix as `[a, -b, tx, b, a, ty]` where the forward
/// map is `dst = [[a, -b], [b, a]] · src + [tx, ty]`.
fn estimate_similarity_transform(src: &[(f32, f32); 5], dst: &[(f32, f32); 5]) -> [f32; 6] {
    // System: dst_x = a·src_x − b·src_y + tx
    //         dst_y = b·src_x + a·src_y + ty
    // → 10 equations, 4 unknowns. Solve directly with QR instead of
    // normal-equation inversion; the landmark coordinates are large enough
    // that the naive `(AᵀA)⁻¹Aᵀb` path becomes numerically brittle.
    let mut design = SMatrix::<f32, 10, 4>::zeros();
    let mut rhs = SVector::<f32, 10>::zeros();
    for (i, ((sx, sy), (dx, dy))) in src.iter().zip(dst.iter()).enumerate() {
        let row_x = i * 2;
        let row_y = row_x + 1;
        design[(row_x, 0)] = *sx;
        design[(row_x, 1)] = -*sy;
        design[(row_x, 2)] = 1.0;
        rhs[row_x] = *dx;

        design[(row_y, 0)] = *sy;
        design[(row_y, 1)] = *sx;
        design[(row_y, 3)] = 1.0;
        rhs[row_y] = *dy;
    }

    let sol = design
        .svd(true, true)
        .solve(&rhs, 1e-6)
        .unwrap_or_else(|_| SVector::<f32, 4>::from([1.0, 0.0, 0.0, 0.0]));

    let (a, b) = (sol[0], sol[1]);
    let (tx, ty) = (sol[2], sol[3]);
    [a, -b, tx, b, a, ty]
}

fn invert_affine(fwd: &[f32; 6]) -> Option<[f32; 6]> {
    let det = fwd[0] * fwd[4] - fwd[1] * fwd[3];
    if det.abs() < 1e-10 {
        return None;
    }
    let inv_det = 1.0 / det;
    let ia = fwd[4] * inv_det;
    let ib = -fwd[1] * inv_det;
    let ic = -fwd[3] * inv_det;
    let id = fwd[0] * inv_det;
    let itx = -(ia * fwd[2] + ib * fwd[5]);
    let ity = -(ic * fwd[2] + id * fwd[5]);
    Some([ia, ib, itx, ic, id, ity])
}

fn apply_affine(affine: &[f32; 6], x: f32, y: f32) -> (f32, f32) {
    (
        affine[0] * x + affine[1] * y + affine[2],
        affine[3] * x + affine[4] * y + affine[5],
    )
}

/// Backward-map warp of `image` through the given 2×3 forward affine,
/// producing an `out_w × out_h` crop with bilinear interpolation.
fn warp_affine(image: &DynamicImage, fwd: &[f32; 6], out_w: u32, out_h: u32) -> RgbImage {
    let rgba = image.to_rgba8();
    let mut output = image::RgbaImage::from_pixel(out_w, out_h, image::Rgba([0, 0, 0, 255]));
    let Some(projection) = Projection::from_matrix([
        fwd[0], fwd[1], fwd[2], fwd[3], fwd[4], fwd[5], 0.0, 0.0, 1.0,
    ]) else {
        return DynamicImage::ImageRgba8(output).to_rgb8();
    };
    warp_into(
        &rgba,
        &projection,
        Interpolation::Bilinear,
        image::Rgba([0, 0, 0, 255]),
        &mut output,
    );
    DynamicImage::ImageRgba8(output).to_rgb8()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_transform_identity_for_aligned_points() {
        let pts: [(f32, f32); 5] = [
            (10.0, 20.0),
            (30.0, 20.0),
            (20.0, 35.0),
            (12.0, 45.0),
            (28.0, 45.0),
        ];
        let affine = estimate_similarity_transform(&pts, &pts);
        // a ≈ 1, b ≈ 0, tx ≈ 0, ty ≈ 0
        assert!((affine[0] - 1.0).abs() < 1e-3, "a={}", affine[0]);
        assert!(affine[1].abs() < 1e-3, "-b={}", affine[1]);
        assert!(affine[2].abs() < 1e-3, "tx={}", affine[2]);
        assert!(affine[3].abs() < 1e-3, "b={}", affine[3]);
        assert!((affine[4] - 1.0).abs() < 1e-3, "a={}", affine[4]);
        assert!(affine[5].abs() < 1e-3, "ty={}", affine[5]);
    }

    #[test]
    fn similarity_transform_remains_stable_for_large_translated_points() {
        let target = scaled_alignment_template(DISPLAY_FACE_CROP);
        let angle = 0.42f32;
        let scale = 1.31f32;
        let (sin, cos) = angle.sin_cos();
        let src = target.map(|(x, y)| {
            (
                scale * (cos * x - sin * y) + 2400.0,
                scale * (sin * x + cos * y) + 180.0,
            )
        });
        let affine = estimate_similarity_transform(&src, &target);
        for (source, expected) in src.into_iter().zip(target) {
            let mapped = apply_affine(&affine, source.0, source.1);
            let error = (mapped.0 - expected.0).hypot(mapped.1 - expected.1);
            assert!(
                error < 0.25,
                "mapped={mapped:?} expected={expected:?} error={error}"
            );
        }
    }

    #[test]
    fn nms_suppresses_overlapping() {
        let mut faces = vec![
            DetectedFace {
                bbox: FaceBbox {
                    x: 0.0,
                    y: 0.0,
                    w: 10.0,
                    h: 10.0,
                },
                landmarks: FaceLandmarks([(0.0, 0.0); 5]),
                confidence: 0.9,
            },
            DetectedFace {
                bbox: FaceBbox {
                    x: 1.0,
                    y: 1.0,
                    w: 10.0,
                    h: 10.0,
                },
                landmarks: FaceLandmarks([(0.0, 0.0); 5]),
                confidence: 0.8,
            },
            DetectedFace {
                bbox: FaceBbox {
                    x: 50.0,
                    y: 50.0,
                    w: 10.0,
                    h: 10.0,
                },
                landmarks: FaceLandmarks([(0.0, 0.0); 5]),
                confidence: 0.7,
            },
        ];
        nms(&mut faces);
        assert_eq!(faces.len(), 2); // overlapping pair collapsed, distant kept
        assert!((faces[0].confidence - 0.9).abs() < 1e-6);
        assert!((faces[1].confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn landmarks_round_trip_bytes() {
        let lm = FaceLandmarks([(1.5, 2.5), (3.0, 4.0), (5.5, 6.5), (7.0, 8.0), (9.5, 10.5)]);
        let bytes = lm.to_flat_bytes();
        let recovered = FaceLandmarks::from_flat_bytes(&bytes).unwrap();
        for i in 0..5 {
            assert_eq!(lm.0[i], recovered.0[i]);
        }
    }

    #[test]
    fn alignment_landmarks_are_canonicalized_left_to_right() {
        let landmarks = [
            (90.0, 20.0),
            (10.0, 18.0),
            (50.0, 45.0),
            (80.0, 78.0),
            (20.0, 80.0),
        ];
        let normalized = canonicalize_alignment_landmarks(&landmarks);
        assert_eq!(normalized[0], (10.0, 18.0));
        assert_eq!(normalized[1], (90.0, 20.0));
        assert_eq!(normalized[2], (50.0, 45.0));
        assert_eq!(normalized[3], (20.0, 80.0));
        assert_eq!(normalized[4], (80.0, 78.0));
    }

    #[test]
    fn scrfd_input_tensor_pads_at_origin() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(320, 160, Rgb([8, 16, 32])));
        let (_, scale, pad_x, pad_y) = scrfd_input_tensor(&image).unwrap();
        assert!((scale - 2.0).abs() < 1e-6, "scale={scale}");
        assert_eq!(pad_x, 0.0);
        assert_eq!(pad_y, 0.0);
    }

    #[test]
    fn scrfd_decode_scales_bbox_and_landmark_deltas_by_stride() {
        let faces = decode_scrfd_outputs(
            &[ScrfdStrideOutput {
                stride: 8,
                num_anchors: (SCRFD_INPUT_SIDE / 8 * SCRFD_INPUT_SIDE / 8) as usize,
                score_classes: 1,
                scores: {
                    let mut scores =
                        vec![0.0; (SCRFD_INPUT_SIDE / 8 * SCRFD_INPUT_SIDE / 8) as usize];
                    scores[0] = 0.9;
                    scores
                },
                bboxes: {
                    let mut bboxes =
                        vec![0.0; (SCRFD_INPUT_SIDE / 8 * SCRFD_INPUT_SIDE / 8 * 4) as usize];
                    bboxes[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
                    bboxes
                },
                kps: {
                    let mut kps =
                        vec![0.0; (SCRFD_INPUT_SIDE / 8 * SCRFD_INPUT_SIDE / 8 * 10) as usize];
                    kps[..10].copy_from_slice(&[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]);
                    kps
                },
            }],
            1.0,
            0.0,
            0.0,
        );
        assert_eq!(faces.len(), 1);
        let face = &faces[0];
        assert!((face.bbox.x + 8.0).abs() < 1e-6, "x={}", face.bbox.x);
        assert!((face.bbox.y + 16.0).abs() < 1e-6, "y={}", face.bbox.y);
        assert!((face.bbox.w - 32.0).abs() < 1e-6, "w={}", face.bbox.w);
        assert!((face.bbox.h - 48.0).abs() < 1e-6, "h={}", face.bbox.h);
        assert_eq!(face.landmarks.0[0], (4.0, 8.0));
        assert_eq!(face.landmarks.0[4], (36.0, 40.0));
    }

    #[test]
    fn display_template_halves_face_span_inside_crop() {
        let embedding = scaled_alignment_template(EMBEDDING_FACE_CROP);
        let display = scaled_alignment_template(DISPLAY_FACE_CROP);
        let embedding_eye_span = embedding[1].0 - embedding[0].0;
        let display_eye_span = display[1].0 - display[0].0;
        assert!((display_eye_span - embedding_eye_span).abs() < 0.6);
        let embedding_face_fraction = embedding_eye_span / (EMBEDDING_FACE_CROP.side as f32 - 1.0);
        let display_face_fraction = display_eye_span / (DISPLAY_FACE_CROP.side as f32 - 1.0);
        assert!(
            (display_face_fraction * 2.0 - embedding_face_fraction).abs() < 0.02,
            "embed={embedding_face_fraction} display={display_face_fraction}"
        );
    }

    #[test]
    fn display_crop_context_shrinks_near_source_edge() {
        let edge_biased = scaled_alignment_template(FaceCropSpec {
            side: DISPLAY_FACE_SIDE,
            context_scale: 1.0,
        })
        .map(|(x, y)| (x + 40.0, y));
        let fitted = fit_face_crop_spec(&edge_biased, (320, DISPLAY_FACE_SIDE), DISPLAY_FACE_CROP);
        assert!(fitted.context_scale < DISPLAY_FACE_CROP.context_scale);
        assert!(fitted.context_scale >= 1.0);
    }
}
