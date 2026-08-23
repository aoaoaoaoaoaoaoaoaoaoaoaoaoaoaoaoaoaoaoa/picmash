use picmash_contract::Target;

#[cfg(feature = "egui-test")]
pub fn response(ui: &egui::Ui, target: Target, response: &egui::Response) {
    egui_tester_witness::egui::record_response(ui, target.to_string(), response);
}

#[cfg(not(feature = "egui-test"))]
pub fn response(_ui: &egui::Ui, _target: Target, _response: &egui::Response) {}

#[cfg(feature = "egui-test")]
pub fn rect(ctx: &egui::Context, target: Target, rect: egui::Rect) {
    egui_tester_witness::egui::record_rect(ctx, target.to_string(), rect);
}

#[cfg(not(feature = "egui-test"))]
pub fn rect(_ctx: &egui::Context, _target: Target, _rect: egui::Rect) {}
