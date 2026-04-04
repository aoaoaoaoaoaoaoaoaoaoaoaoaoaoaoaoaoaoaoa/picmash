use std::{fs, path::Path};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::{
    asset_domain::AssetDomainLabel,
    model::{ExploreMapMode, SIMILARITY_DIM, SimilarityChoice},
};

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct TriadHandleDto {
    pub asset_a_id: String,
    pub asset_b_id: String,
    pub asset_c_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct PosteriorSummaryDto {
    pub mean: f32,
    pub sigma: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct AssetQualitySummaryDto {
    pub asset: PosteriorSummaryDto,
    pub baseline: PosteriorSummaryDto,
    pub semantic: Option<PosteriorSummaryDto>,
    pub vibe: Option<PosteriorSummaryDto>,
    pub technical: Option<PosteriorSummaryDto>,
    pub face: Option<PosteriorSummaryDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct AssetDomainSummaryDto {
    pub manual_label: Option<AssetDomainLabel>,
    pub predicted_label: Option<AssetDomainLabel>,
    pub predicted_percent: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct ExplorePointDto {
    pub asset_id: String,
    pub name: String,
    pub thumb_src: String,
    pub plot_x: f32,
    pub plot_y: f32,
    pub latent: [f32; SIMILARITY_DIM],
    pub compare_count: u32,
    pub win_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct FocusAssetDto {
    pub asset_id: String,
    pub name: String,
    pub thumb_src: String,
    pub preview_src: String,
    pub full_src: String,
    pub domain: AssetDomainSummaryDto,
    pub hearted: bool,
    pub compare_count: u32,
    pub win_count: u32,
    pub global_score: f32,
    pub quality: AssetQualitySummaryDto,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct ExploreNeighborDto {
    pub asset: FocusAssetDto,
    pub distance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct ExploreSelectionDto {
    pub focus: FocusAssetDto,
    pub neighbors: Vec<ExploreNeighborDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct TriadAssetDto {
    pub asset_id: String,
    pub name: String,
    pub full_src: String,
    pub domain: AssetDomainSummaryDto,
    pub hearted: bool,
    pub compare_count: u32,
    pub win_count: u32,
    pub quality: AssetQualitySummaryDto,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct TriadBootstrapDto {
    pub triad: Option<TriadHandleDto>,
    pub assets: Vec<TriadAssetDto>,
    pub empty_note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct ExploreBootstrapDto {
    pub mode: ExploreMapMode,
    pub triad: Option<TriadHandleDto>,
    pub focus_id: Option<String>,
    pub points: Vec<ExplorePointDto>,
    pub selection: Option<ExploreSelectionDto>,
    pub empty_note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct ExploreSelectionResponseDto {
    pub mode: ExploreMapMode,
    pub focus_id: String,
    pub selection: ExploreSelectionDto,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct TriadTrainRequestDto {
    pub triad: TriadHandleDto,
    pub choice: SimilarityChoice,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct RotateAssetRequestDto {
    pub asset_id: String,
    pub direction: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct HideAssetRequestDto {
    pub asset_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct HeartAssetRequestDto {
    pub asset_id: String,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct AssetDomainRequestDto {
    pub asset_id: String,
    pub label: AssetDomainLabel,
}

pub fn export_types(out_dir: &Path) -> anyhow::Result<()> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("creating TS contract directory {}", out_dir.display()))?;
    AssetDomainLabel::export_all_to(out_dir)?;
    TriadHandleDto::export_all_to(out_dir)?;
    PosteriorSummaryDto::export_all_to(out_dir)?;
    AssetQualitySummaryDto::export_all_to(out_dir)?;
    AssetDomainSummaryDto::export_all_to(out_dir)?;
    ExplorePointDto::export_all_to(out_dir)?;
    FocusAssetDto::export_all_to(out_dir)?;
    ExploreNeighborDto::export_all_to(out_dir)?;
    ExploreSelectionDto::export_all_to(out_dir)?;
    TriadAssetDto::export_all_to(out_dir)?;
    TriadBootstrapDto::export_all_to(out_dir)?;
    ExploreBootstrapDto::export_all_to(out_dir)?;
    ExploreSelectionResponseDto::export_all_to(out_dir)?;
    TriadTrainRequestDto::export_all_to(out_dir)?;
    RotateAssetRequestDto::export_all_to(out_dir)?;
    HideAssetRequestDto::export_all_to(out_dir)?;
    HeartAssetRequestDto::export_all_to(out_dir)?;
    AssetDomainRequestDto::export_all_to(out_dir)?;
    fs::write(
        out_dir.join("ExploreMapMode.ts"),
        "// This file was generated by picmash. Do not edit.\n\nexport type ExploreMapMode = \"raw\" | \"learned\";\n",
    )
    .with_context(|| format!("writing {}", out_dir.join("ExploreMapMode.ts").display()))?;
    fs::write(
        out_dir.join("SimilarityChoice.ts"),
        "// This file was generated by picmash. Do not edit.\n\nexport type SimilarityChoice = \"ab\" | \"ac\" | \"bc\";\n",
    )
    .with_context(|| format!("writing {}", out_dir.join("SimilarityChoice.ts").display()))?;
    Ok(())
}
