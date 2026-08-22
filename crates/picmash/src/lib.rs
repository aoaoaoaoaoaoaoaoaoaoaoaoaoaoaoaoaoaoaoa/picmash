//! Picmash's native Poolrooms graft.

mod app;
mod commands;
mod configuration;
mod host;
mod witness;
mod worker;
mod xdg;

/// Run the native application.
pub fn run() -> anyhow::Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let collection = match arguments.next() {
        Some(argument) if argument == "--help" || argument == "-h" => {
            println!("usage: picmash [COLLECTION]\n\nOpen or restore a local image collection.");
            return Ok(());
        }
        Some(argument) if argument == "--version" || argument == "-V" => {
            println!("picmash {}", env!("CARGO_PKG_VERSION"));
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
    host::run(ctx, collection)
}
