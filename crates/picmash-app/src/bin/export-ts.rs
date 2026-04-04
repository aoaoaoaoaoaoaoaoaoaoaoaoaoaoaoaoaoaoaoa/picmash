#![allow(unused_crate_dependencies)]

use std::{env, path::PathBuf};

use anyhow::Context;
use picmash_app::api;

fn main() -> anyhow::Result<()> {
    let out_dir = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: cargo run -p picmash-app --bin export-ts -- <out_dir>")?;
    api::export_types(&out_dir)?;
    Ok(())
}
