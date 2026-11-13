// Per-pixel photosphere mapping.
//
// The vertex shader only places a unit sphere around the camera and passes
// each vertex's position along. The fragment shader then turns the *pixel's
// own* viewing direction into longitude/latitude and samples the photo there.
//
// An earlier version stored a photo coordinate in each mesh vertex and let the
// GPU interpolate it. That is only approximately right between vertices, and
// badly wrong at the poles, where every triangle of the top and bottom rows
// collapses to a point but its photo coordinates span a wide strip (visible
// pinching). Computing the mapping per pixel is exact everywhere, at the cost
// of one atan2 and one asin per pixel, which is nothing for a GPU.
//
// Switching photos crossfades `previous_texture` into `current_texture`
// rather than cutting instantly — both are always bound (when there's no
// real previous photo yet, both slots just point at the same texture, which
// blends into itself at any blend value with no visible effect).

struct Uniforms {
    view_proj: mat4x4<f32>,
    // Only .x is used (0.0 = fully previous, 1.0 = fully current); the rest
    // is padding to satisfy uniform buffer alignment rules.
    blend: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var previous_texture: texture_2d<f32>;
@group(0) @binding(2) var current_texture: texture_2d<f32>;
@group(0) @binding(3) var source_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Direction from the camera (at the sphere's center) to this point.
    @location(0) direction: vec3<f32>,
}

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.direction = in.position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Interpolating positions across a flat triangle shortens them a little,
    // so renormalize to get the exact direction.
    let d = normalize(in.direction);

    // Mesh convention: position = (right, up, -forward).
    let longitude = atan2(d.x, -d.z);   // -PI .. PI, 0 = straight ahead
    let latitude = asin(clamp(d.y, -1.0, 1.0)); // -PI/2 .. PI/2

    let uv = vec2<f32>(
        (longitude + PI) / TAU,
        0.5 - latitude / PI,
    );

    let previous_color = textureSample(previous_texture, source_sampler, uv);
    let current_color = textureSample(current_texture, source_sampler, uv);
    return mix(previous_color, current_color, uniforms.blend.x);
}
