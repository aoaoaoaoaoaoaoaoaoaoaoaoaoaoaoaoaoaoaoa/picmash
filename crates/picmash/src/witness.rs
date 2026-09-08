use picmash_contract::Target;

pub fn response(ui: &egui::Ui, target: Target, response: &egui::Response) {
    eternalist_apps::witness::response(ui, target, response);
}

pub fn rect(ctx: &egui::Context, target: Target, rect: egui::Rect) {
    eternalist_apps::witness::rect(ctx, target, rect);
}
