//! An iced widget for drag-to-look-around viewing of equirectangular
//! (360-degree) photos. Does what every real photosphere viewer actually
//! does: build an actual sphere mesh, wrap the photo around its inside as a
//! texture, and let the GPU's ordinary triangle-rasterization pipeline — the
//! thing GPUs have been brutally optimized for since forever — handle the
//! rest. The "which direction is this part of the photo" math only runs
//! once, while building the mesh; every frame after that is just a standard
//! matrix transform and a texture sample, both about as cheap as GPU work
//! gets.
//!
//! ```ignore
//! iced_photosphere::photosphere(&source)
//!     .width(Length::Fill)
//!     .height(Length::Fill)
//! ```
use iced::widget::shader;
use iced::{Element, Event, Length, Point, Rectangle, mouse};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Radians of yaw/pitch per pixel dragged — same feel setting as the CPU
/// viewer.
const DRAG_SENSITIVITY: f32 = 0.005;

/// How finely subdivided the sphere mesh is. Cheap to raise — a GPU chews
/// through a few thousand vertices without noticing — but this is already
/// smooth enough that a photo's own resolution is the limiting factor, not
/// the mesh.
const LAT_SEGMENTS: u32 = 64;
const LON_SEGMENTS: u32 = 128;

/// A drag-to-look-around photosphere viewer, usable like any other iced
/// widget — everything below this (the shader `Program`, the wgpu pipeline,
/// the sphere mesh) is a private implementation detail, not part of the
/// public API.
pub struct Photosphere<'a> {
    source: &'a Arc<image::RgbaImage>,
    width: Length,
    height: Length,
}

/// Creates a new [`Photosphere`] widget showing `source`, an equirectangular
/// photo. Drag with the left mouse button to look around.
pub fn photosphere(source: &Arc<image::RgbaImage>) -> Photosphere<'_> {
    Photosphere {
        source,
        width: Length::Fill,
        height: Length::Fill,
    }
}

impl<'a> Photosphere<'a> {
    /// Sets the width of the widget.
    pub fn width(mut self, width: impl Into<Length>) -> Self {
        self.width = width.into();
        self
    }

    /// Sets the height of the widget.
    pub fn height(mut self, height: impl Into<Length>) -> Self {
        self.height = height.into();
        self
    }
}

impl<'a, Message> From<Photosphere<'a>> for Element<'a, Message>
where
    Message: 'a,
{
    fn from(widget: Photosphere<'a>) -> Self {
        shader::Shader::new(PhotosphereProgram {
            source: Arc::clone(widget.source),
        })
        .width(widget.width)
        .height(widget.height)
        .into()
    }
}

struct PhotosphereProgram {
    source: Arc<image::RgbaImage>,
}

#[derive(Default)]
struct ProgramState {
    yaw: f32,
    pitch: f32,
    drag_origin: Option<(Point, f32, f32)>,
}

impl<Message> shader::Program<Message> for PhotosphereProgram {
    type State = ProgramState;
    type Primitive = Primitive;

    fn update(
        &self,
        state: &mut Self::State,
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<shader::Action<Message>> {
        let Event::Mouse(mouse_event) = event else {
            return None;
        };

        match mouse_event {
            mouse::Event::ButtonPressed(mouse::Button::Left) => {
                let position = cursor.position_in(bounds)?;
                state.drag_origin = Some((position, state.yaw, state.pitch));
                None
            }
            mouse::Event::CursorMoved { .. } => {
                let (start, start_yaw, start_pitch) = state.drag_origin?;
                let position = cursor.position_in(bounds)?;
                let dx = position.x - start.x;
                let dy = position.y - start.y;

                // Dragging right turns the view left — same convention as
                // the CPU viewer, so the two feel identical to use.
                state.yaw = start_yaw - dx * DRAG_SENSITIVITY;
                state.pitch = (start_pitch + dy * DRAG_SENSITIVITY)
                    .clamp(-85f32.to_radians(), 85f32.to_radians());

                Some(shader::Action::request_redraw())
            }
            mouse::Event::ButtonReleased(mouse::Button::Left) => {
                state.drag_origin = None;
                None
            }
            _ => None,
        }
    }

    fn draw(&self, state: &Self::State, _cursor: mouse::Cursor, bounds: Rectangle) -> Self::Primitive {
        Primitive {
            source: Arc::clone(&self.source),
            yaw: state.yaw,
            pitch: state.pitch,
            aspect: bounds.width / bounds.height,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
}

/// One frame's worth of what the shader needs: the photo (shared, not
/// copied, via `Arc`) and the current view. Rebuilt every `draw` call —
/// cheap, since it's a handful of numbers plus a reference-counted pointer.
#[derive(Debug)]
struct Primitive {
    source: Arc<image::RgbaImage>,
    yaw: f32,
    pitch: f32,
    aspect: f32,
}

impl shader::Primitive for Primitive {
    type Pipeline = Pipeline;

    fn prepare(
        &self,
        pipeline: &mut Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _bounds: &Rectangle,
        _viewport: &shader::Viewport,
    ) {
        pipeline.ensure_texture(device, queue, &self.source);

        // Same 90-degree horizontal field of view the CPU viewer uses,
        // converted to the vertical FOV glam's projection matrix wants.
        let tan_half_fov_x = (90f32.to_radians() / 2.0).tan();
        let tan_half_fov_y = tan_half_fov_x / self.aspect;
        let fov_y = 2.0 * tan_half_fov_y.atan();

        let projection = glam::Mat4::perspective_rh(fov_y, self.aspect, 0.01, 10.0);
        // No translation anywhere — the camera always sits at the exact
        // center of the sphere, so "the view" is purely a rotation.
        let view = glam::Mat4::from_rotation_x(self.pitch) * glam::Mat4::from_rotation_y(-self.yaw);

        let uniforms = Uniforms {
            view_proj: (projection * view).to_cols_array_2d(),
        };
        queue.write_buffer(&pipeline.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn render(
        &self,
        pipeline: &Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        let Some(texture_state) = &pipeline.texture_state else {
            // `prepare` always runs before `render`, so this never actually
            // happens — but there's nothing sane to draw without a texture.
            return;
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("photosphere render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_scissor_rect(
            clip_bounds.x,
            clip_bounds.y,
            clip_bounds.width,
            clip_bounds.height,
        );
        pass.set_pipeline(&pipeline.render_pipeline);
        pass.set_bind_group(0, &texture_state.bind_group, &[]);
        pass.set_vertex_buffer(0, pipeline.vertex_buffer.slice(..));
        pass.set_index_buffer(pipeline.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..pipeline.index_count, 0, 0..1);
    }
}

/// Everything GPU-side that's shared across every frame: the compiled
/// shader/render pipeline and the sphere mesh (both built once, in `new` —
/// the mesh never changes no matter which photo is loaded), and whichever
/// photo is currently uploaded as a texture (built lazily, the first time a
/// `Primitive` actually needs it — `new` has no way to know the photo yet).
struct Pipeline {
    render_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    texture_state: Option<TextureState>,
}

struct TextureState {
    /// Kept only so `Arc::ptr_eq` can tell "this is the same photo as last
    /// frame" from "a new one was loaded" — re-uploading an unchanged photo
    /// to the GPU every single frame would be pure waste.
    source: Arc<image::RgbaImage>,
    bind_group: wgpu::BindGroup,
    _texture: wgpu::Texture,
    _view: wgpu::TextureView,
}

impl Pipeline {
    fn ensure_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &Arc<image::RgbaImage>,
    ) {
        if let Some(state) = &self.texture_state {
            if Arc::ptr_eq(&state.source, source) {
                return;
            }
        }

        let size = wgpu::Extent3d {
            width: source.width(),
            height: source.height(),
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("photosphere texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            source.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * source.width()),
                rows_per_image: Some(source.height()),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("photosphere bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        self.texture_state = Some(TextureState {
            source: Arc::clone(source),
            bind_group,
            _texture: texture,
            _view: view,
        });
    }
}

/// A UV-sphere: rows of latitude from the north pole (`i = 0`) to the south
/// pole (`i = LAT_SEGMENTS`), each split into columns of longitude. Each
/// vertex's position and its photo-UV coordinate come from the exact same
/// longitude/latitude math `projection.rs` uses — laid out once, here,
/// rather than recomputed every pixel of every frame the way the first GPU
/// attempt did it.
fn build_sphere_mesh() -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();

    for i in 0..=LAT_SEGMENTS {
        let v = i as f32 / LAT_SEGMENTS as f32;
        let latitude = std::f32::consts::FRAC_PI_2 - v * std::f32::consts::PI;

        for j in 0..=LON_SEGMENTS {
            let u = j as f32 / LON_SEGMENTS as f32;
            let longitude = u * std::f32::consts::TAU - std::f32::consts::PI;

            let forward = latitude.cos() * longitude.cos();
            let right = latitude.cos() * longitude.sin();
            let up = latitude.sin();

            // (forward, right, up) -> glam's right-handed (x = right,
            // y = up, z = -forward), done once here instead of needing any
            // axis-convention juggling in the shader.
            vertices.push(Vertex {
                position: [right, up, -forward],
                uv: [u, v],
            });
        }
    }

    let mut indices = Vec::new();
    let row_len = LON_SEGMENTS + 1;
    for i in 0..LAT_SEGMENTS {
        for j in 0..LON_SEGMENTS {
            let a = i * row_len + j;
            let b = a + 1;
            let c = a + row_len;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    (vertices, indices)
}

impl shader::Pipeline for Pipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("photosphere shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("photosphere bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("photosphere pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as u64,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        };

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("photosphere render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[vertex_buffer_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("photosphere uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("photosphere sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let (vertices, indices) = build_sphere_mesh();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("photosphere vertex buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("photosphere index buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Pipeline {
            render_pipeline,
            bind_group_layout,
            uniform_buffer,
            sampler,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            texture_state: None,
        }
    }
}
