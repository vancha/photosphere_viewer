// A sanity-check harness for `projection::render_view`, nothing more: load a
// source photo, render one view of it with a chosen yaw/pitch, save the
// result so it can be looked at and judged "does this look like a real photo
// facing that direction, or garbage?"
use photosphere_viewer::projection::{render_view, View};
use std::env;

fn main() {
    let mut args = env::args().skip(1);
    let source_path = args.next().unwrap_or_else(|| "test_pattern.png".to_string());
    let yaw_deg: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let pitch_deg: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let out_path = args.next().unwrap_or_else(|| "view.png".to_string());

    let source = image::open(&source_path)
        .unwrap_or_else(|e| panic!("couldn't open {source_path}: {e}"))
        .to_rgb8();

    let view = View {
        yaw: yaw_deg.to_radians(),
        pitch: pitch_deg.to_radians(),
        fov_x: 90f64.to_radians(),
    };

    let start = std::time::Instant::now();
    let result = render_view(&source, &view, 1280, 800);
    println!("render_view took {:?}", start.elapsed());
    result.save(&out_path).expect("failed to save view");
    println!("wrote {out_path} (yaw={yaw_deg}°, pitch={pitch_deg}°)");
}
