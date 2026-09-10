//! Picmash's native Poolrooms graft.

mod app;
mod application_paths;
use application_paths::PicmashPaths as _;
mod commands;
mod configuration;
mod host;
mod remote;
mod viewer;
mod witness;
mod worker;

/// Run the native application.
pub fn run() -> anyhow::Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let collection = match arguments.next() {
        Some(argument) if argument == "--help" || argument == "-h" => {
            println!(
                "usage: picmash [COLLECTION]\n       picmash --import-legacy DATABASE\n\nOpen or restore a local image collection."
            );
            return Ok(());
        }
        Some(argument) if argument == "--version" || argument == "-V" => {
            println!("picmash {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        Some(argument) if argument == "--import-legacy" => {
            let Some(source) = arguments.next() else {
                anyhow::bail!("usage: picmash --import-legacy DATABASE");
            };
            if arguments.next().is_some() {
                anyhow::bail!("usage: picmash --import-legacy DATABASE");
            }
            let paths = application_paths::claim()?;
            let engine = picmash_engine::Engine::open(paths.database_path())?;
            let report = engine.import_legacy(source)?;
            println!(
                "imported {} assets and {} observations ({} ambiguous){}",
                report.imported_assets,
                report.imported_observations,
                report.ambiguous_observations,
                if report.already_imported {
                    " · already imported"
                } else {
                    ""
                }
            );
            return Ok(());
        }
        Some(argument) => Some(std::path::PathBuf::from(argument)),
        None => None,
    };
    if arguments.next().is_some() {
        anyhow::bail!("usage: picmash [COLLECTION]");
    }
    let ctx = egui::Context::default();
    brass_poolrooms::chrome::install(&ctx);
    host::run(eternalist_apps::Ingress::Desktop, ctx, collection)
}
