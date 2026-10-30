// The standard way every real photosphere viewer does this (browser-based
// ones included): wrap the photo around the inside of an actual sphere mesh,
// and let the GPU's ordinary, decades-optimized triangle pipeline do the
// work — transform a few thousand vertices, rasterize, interpolate texture
// coordinates. No per-pixel trigonometry at all; the "which way am I
// looking" math (visible in how `gpu.rs` builds the sphere's vertices) only
// ever runs once, when the mesh is built, not every fragment of every frame.

struct Uniforms {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var source_texture: texture_2d<f32>;
@group(0) @binding(2) var source_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(source_texture, source_sampler, in.uv);
}
