// Builds a synthetic equirectangular test image: white dots, evenly spread
// over an actual sphere (not evenly spread over the distorted 2D image), on a
// black background. "Evenly spread over the sphere" matters — if dots were
// instead placed at evenly-spaced pixel coordinates, they'd visibly bunch up
// near the poles (equirectangular's own distortion), which would make this a
// test of that distortion rather than a test of whether the reprojection
// correctly undoes it.
use image::{Rgb, RgbImage};
use std::f64::consts::PI;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 512;
const DOT_COUNT: usize = 200;
/// Angular radius of each dot, in radians — a size on the *sphere*, not a
/// pixel count, so every dot ends up the same true size no matter where it
/// lands, poles included.
const DOT_RADIUS: f64 = 3.5 * PI / 180.0;

fn main() {
    let dots = fibonacci_sphere(DOT_COUNT);
    let mut img = RgbImage::new(WIDTH, HEIGHT);

    for y in 0..HEIGHT {
        let lat_deg = 90.0 - (y as f64 / HEIGHT as f64) * 180.0;
        let lat = lat_deg.to_radians();

        for x in 0..WIDTH {
            let lon_deg = (x as f64 / WIDTH as f64) * 360.0 - 180.0;
            let lon = lon_deg.to_radians();

            // This pixel's own direction, in the same (forward, right, up)
            // convention `projection.rs` uses.
            let pixel_dir = (lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());

            // Lit up only if within DOT_RADIUS (measured as a true angle
            // between directions, via the dot product) of some dot's center.
            let lit = dots.iter().any(|&dot| angle_between(pixel_dir, dot) < DOT_RADIUS);

            img.put_pixel(x, y, Rgb(if lit { [255, 255, 255] } else { [0, 0, 0] }));
        }
    }

    img.save("test_pattern.png").expect("failed to save test pattern");
    println!("wrote test_pattern.png ({WIDTH}x{HEIGHT}, {DOT_COUNT} dots)");
}

/// `count` points spread as evenly as possible over a unit sphere, via a
/// Fibonacci spiral — the standard trick for this, since naively stepping
/// latitude and longitude in fixed increments clusters points at the poles
/// exactly the way we're trying to avoid.
fn fibonacci_sphere(count: usize) -> Vec<(f64, f64, f64)> {
    let golden_angle = PI * (3.0 - 5f64.sqrt());
    (0..count)
        .map(|i| {
            // `up` steps evenly from +1 down to -1 across all the points.
            let up = 1.0 - (i as f64 / (count - 1) as f64) * 2.0;
            let ring_radius = (1.0 - up * up).max(0.0).sqrt();
            let angle = golden_angle * i as f64;
            (angle.cos() * ring_radius, angle.sin() * ring_radius, up)
        })
        .collect()
}

fn angle_between(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let dot = a.0 * b.0 + a.1 * b.1 + a.2 * b.2;
    dot.clamp(-1.0, 1.0).acos()
}
