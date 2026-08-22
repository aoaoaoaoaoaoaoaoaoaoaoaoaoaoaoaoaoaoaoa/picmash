#![expect(
    unused_crate_dependencies,
    reason = "the binary delegates native product construction to the package library"
)]

fn main() -> anyhow::Result<()> {
    picmash::run()
}
