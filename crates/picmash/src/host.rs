use anyhow::Result;
use eternalist_apps::{NativeApp, ProductIdentity, WindowSpec};
use std::{path::PathBuf, time::Instant};

use crate::{app::Picmash, application_paths::PRODUCT};

pub fn run(ctx: egui::Context, collection: Option<PathBuf>) -> Result<()> {
    eternalist_apps::run_with(ctx, move |ctx| Picmash::open(ctx, collection))
}

impl NativeApp for Picmash {
    const PRODUCT: ProductIdentity = PRODUCT;
    const RELEASE: &'static str = env!("CARGO_PKG_VERSION");
    const WINDOW: WindowSpec = WindowSpec::new("picmash", [1_420.0, 900.0]);

    fn draw(&mut self, ui: &mut egui::Ui) {
        self.pulse(ui);
    }

    fn service_deadline(&self, _now: Instant) -> Option<Instant> {
        self.configuration_deadline()
    }

    fn service_deadline_reached(&mut self, now: Instant) -> bool {
        self.service_configuration(now)
    }

    fn water(
        &mut self,
        ctx: &egui::Context,
        pixels_per_point: f32,
        tooltip_rects: &[egui::Rect],
    ) -> brass_poolrooms::water::Frame {
        self.water_frame(ctx, pixels_per_point, tooltip_rects)
    }

    #[cfg(feature = "egui-test")]
    type Observation = crate::app::Observation;

    #[cfg(feature = "egui-test")]
    fn observe(&self, text_edit_focused: bool) -> Self::Observation {
        self.observe(text_edit_focused)
    }
}
