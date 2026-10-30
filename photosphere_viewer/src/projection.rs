//! The one hard idea this whole project rests on: for every pixel of a
//! normal-looking (perspective) output image, figure out which pixel of the
//! equirectangular source photo it should show. No 3D scene, no GPU — just a
//! question answered once per pixel: "which direction am I looking, and what
//! color is that direction in the original photo?"

use image::{Rgb, RgbImage};
use rayon::prelude::*;

/// Where the virtual camera is looking, and how wide its view is.
pub struct View {
    /// Left/right look direction, in radians. 0 faces the source image's
    /// horizontal center; positive turns right.
    pub yaw: f64,
    /// Up/down look direction, in radians. 0 is level; positive looks up.
    pub pitch: f64,
    /// Horizontal field of view, in radians (a typical camera lens is
    /// roughly 1.0-1.4; wider feels more "fisheye", narrower more "zoomed in").
    pub fov_x: f64,
}

/// Renders what `view` sees of `source`, into an image of `out_width` x
/// `out_height`.
///
/// Two things make this fast enough to run on every drag update instead of
/// just at rest: `pitch`/`yaw`'s `sin`/`cos` depend only on *which direction
/// the camera is looking*, not on which pixel is being computed, so they're
/// worked out once up front instead of redone (identically) a million times
/// over; and every pixel is fully independent of every other one — nothing
/// about computing pixel (5, 5) needs to know anything about pixel (6, 5) —
/// so `rayon` spreads the work across every CPU core at once, one row per
/// chunk (not one pixel per chunk — handing rayon a million one-pixel chunks
/// would mean a million tiny units of work to schedule, and that bookkeeping
/// overhead eats a real chunk of the parallelism's benefit).
pub fn render_view(source: &RgbImage, view: &View, out_width: u32, out_height: u32) -> RgbImage {
    let tan_half_fov_x = (view.fov_x / 2.0).tan();
    let tan_half_fov_y = tan_half_fov_x * out_height as f64 / out_width as f64;
    let (pitch_sin, pitch_cos) = view.pitch.sin_cos();
    let (yaw_sin, yaw_cos) = view.yaw.sin_cos();

    let mut pixels = vec![0u8; out_width as usize * out_height as usize * 3];

    pixels
        .par_chunks_mut(out_width as usize * 3)
        .enumerate()
        .for_each(|(py, row)| {
            let py = py as u32;
            for (px, out_pixel) in row.chunks_mut(3).enumerate() {
                let px = px as u32;

                // Step 1: where does this output pixel sit on an imaginary
                // flat "window" one unit in front of the camera, in
                // camera-local coordinates? (Same idea as a camera's film
                // plane.)
                let window_x =
                    (2.0 * (px as f64 + 0.5) / out_width as f64 - 1.0) * tan_half_fov_x;
                let window_y =
                    (1.0 - 2.0 * (py as f64 + 0.5) / out_height as f64) * tan_half_fov_y;

                // Step 2: turn that into a 3D ray direction, as if the camera
                // faced the photo's horizontal/vertical center with no yaw
                // or pitch applied yet. `forward` is fixed at 1 (the film
                // plane is one unit away); `right`/`up` come straight from
                // the window position computed above.
                let (forward, right, up) = (1.0, window_x, window_y);

                // Step 3: tilt that ray up/down by `pitch` (a plain 2D
                // rotation of the forward/up pair — pitch never touches
                // `right`).
                let (forward, up) = (
                    forward * pitch_cos - up * pitch_sin,
                    forward * pitch_sin + up * pitch_cos,
                );

                // Step 4: turn the ray left/right by `yaw` (a 2D rotation of
                // the forward/right pair — yaw never touches `up`).
                let (forward, right) = (
                    forward * yaw_cos - right * yaw_sin,
                    forward * yaw_sin + right * yaw_cos,
                );

                // Step 5: that (forward, right, up) triple is a real 3D
                // direction in space now. Read it back as a longitude/
                // latitude pair, exactly the inverse of how an
                // equirectangular photo was laid out in the first place.
                let longitude = right.atan2(forward); // -π..π, 0 = straight ahead
                let latitude =
                    (up / (forward * forward + right * right + up * up).sqrt()).asin(); // -π/2..π/2

                // Step 6: longitude/latitude back to a pixel in the source photo.
                let src_x =
                    ((longitude / std::f64::consts::PI + 1.0) / 2.0) * source.width() as f64;
                let src_y = (0.5 - latitude / std::f64::consts::PI) * source.height() as f64;

                let color = sample_nearest(source, src_x, src_y);
                out_pixel.copy_from_slice(&color.0);
            }
        });

    RgbImage::from_raw(out_width, out_height, pixels)
        .expect("pixel buffer is exactly out_width * out_height * 3 bytes")
}

/// Nearest-neighbor lookup, clamped to the image's edges. Good enough for a
/// first version — a blurry/blocky pixel here and there is a fine trade for
/// "the math works and reads clearly."
fn sample_nearest(source: &RgbImage, x: f64, y: f64) -> Rgb<u8> {
    let x = (x.round() as i64).clamp(0, source.width() as i64 - 1) as u32;
    let y = (y.round() as i64).clamp(0, source.height() as i64 - 1) as u32;
    *source.get_pixel(x, y)
}
