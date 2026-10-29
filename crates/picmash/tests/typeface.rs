#![expect(
    unused_crate_dependencies,
    reason = "this boundary gate audits authored UI and the resolved font feature graph"
)]

#[test]
fn authored_ui_has_no_font_fallback() -> Result<(), Box<dyn std::error::Error>> {
    brass_poolrooms::chrome::glyph_audit::check_tree(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
    )
}
