//! An iced widget for drag-to-look-around viewing of equirectangular
//! (360-degree) photos. Does what every real photosphere viewer actually
//! does: build an actual sphere mesh, wrap the photo around its inside as a
//! texture, and let the GPU's ordinary triangle-rasterization pipeline — the
//! thing GPUs have been brutally optimized for since forever — handle the
//! rest. The fragment shader works out, per pixel, which part of the photo
//! lies in that direction (longitude/latitude), so the mapping is exact
//! everywhere, including the poles, instead of being interpolated between
//! mesh vertices.
//!
//! The widget never keeps the full decoded photo around on the CPU side.
//! [`load`] hands it off and returns a cheap [`PhotoId`]; the pixel data
//! lives only long enough to make it onto the GPU as a texture; the id is
//! all that's needed for every frame after that.
//!
//! ```ignore
//! let id = iced_photosphere::load(source); // source: Arc<image::RgbaImage>
//! iced_photosphere::photosphere(id)
//!     .width(Length::Fill)
//!     .height(Length::Fill)
//! ```
use cosmic::iced::animation::Animation;
use cosmic::iced::wgpu::{self, util::DeviceExt};
use cosmic::iced::widget::shader;
use cosmic::iced::time::{Duration, Instant};
use cosmic::iced::{Element, Event, Length, Point, Rectangle, mouse, window};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Radians of yaw/pitch per pixel dragged at the default field of view.
const DRAG_SENSITIVITY: f32 = 0.005;

/// Horizontal field of view, in degrees. Scrolling zooms between these.
const DEFAULT_FOV: f32 = 90.0;
const MIN_FOV: f32 = 30.0;
const MAX_FOV: f32 = 90.0;

/// Degrees of field of view per scrolled line (or per 10 scrolled pixels on
/// touchpads, which report pixels).
const ZOOM_STEP: f32 = 4.0;

/// Inertia: after letting go mid-drag, the view keeps gliding and slows down
/// exponentially. Velocity is multiplied by `exp(-INERTIA_DECAY * seconds)`,
/// so higher means a shorter glide. The glide stops below `INERTIA_MIN_SPEED`.
const INERTIA_DECAY: f32 = 4.0;
const INERTIA_MIN_SPEED: f32 = 0.02; // radians per second
/// A release counts as a "flick" only if the cursor moved this recently.
const FLICK_WINDOW: Duration = Duration::from_millis(80);

/// How finely subdivided the sphere mesh is. Cheap to raise — a GPU chews
/// through a few thousand vertices without noticing — but this is already
/// smooth enough that a photo's own resolution is the limiting factor, not
/// the mesh.
const LAT_SEGMENTS: u32 = 64;
const LON_SEGMENTS: u32 = 128;

/// How long switching photos takes to crossfade from the old one to the new
/// one, instead of cutting instantly.
const TRANSITION_DURATION: Duration = Duration::from_millis(300);

/// Identifies a photo handed to [`load`], without carrying its pixel data
/// around. Cheap to copy and to compare — just a counter value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhotoId(u64);

/// Holds a loaded photo just long enough for the GPU pipeline to pick it up
/// and upload it as a texture. One slot is enough — this app only ever shows
/// one photo at a time, and loading a new one before the previous upload
/// happened simply supersedes it (which is correct: nothing would ever have
/// asked to render the superseded one anyway).
static STAGED: Mutex<Option<(u64, Arc<image::RgbaImage>)>> = Mutex::new(None);
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// Hands a decoded photo off to the widget and returns a [`PhotoId`] for it.
/// The caller can drop its own `Arc` right after this returns — from this
/// point on, the id is all that's needed to keep showing this photo; the
/// pixel data itself is freed as soon as [`photosphere`] has displayed it for
/// one frame and the GPU has its own copy.
pub fn load(image: Arc<image::RgbaImage>) -> PhotoId {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    *STAGED.lock().unwrap() = Some((id, image));
    PhotoId(id)
}

/// A drag-to-look-around photosphere viewer, usable like any other iced
/// widget — everything below this (the shader `Program`, the wgpu pipeline,
/// the sphere mesh) is a private implementation detail, not part of the
/// public API.
pub struct Photosphere {
    id: PhotoId,
    width: Length,
    height: Length,
}

/// Creates a new [`Photosphere`] widget showing the photo identified by
/// `id` (from [`load`]). Drag with the left mouse button to look around
/// (release mid-drag to let the view glide to a stop) and scroll to zoom.
pub fn photosphere(id: PhotoId) -> Photosphere {
    Photosphere {
        id,
        width: Length::Fill,
        height: Length::Fill,
    }
}

impl Photosphere {
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

impl<'a, Message> From<Photosphere> for Element<'a, Message, cosmic::Theme, cosmic::Renderer>
where
    Message: 'a,
{
    fn from(widget: Photosphere) -> Self {
        shader::Shader::new(PhotosphereProgram { id: widget.id })
            .width(widget.width)
            .height(widget.height)
            .into()
    }
}

struct PhotosphereProgram {
    id: PhotoId,
}

struct ProgramState {
    yaw: f32,
    pitch: f32,
    fov: f32,
    drag_origin: Option<(Point, f32, f32)>,
    /// Smoothed look velocity (yaw, pitch) in radians/second while dragging.
    velocity: (f32, f32),
    last_move: Option<(Instant, f32, f32)>,
    /// Some while the view is gliding after a flick.
    gliding: bool,
    last_tick: Option<Instant>,
    /// The id `draw` used last frame — comparing against it is how a newly
    /// navigated-to photo gets noticed from inside `update`, which is the
    /// only place able to request the extra redraws a crossfade needs.
    last_seen_id: Option<PhotoId>,
    /// Some until this instant while a crossfade is in progress, so `update`
    /// knows to keep requesting redraws even when nothing else (dragging,
    /// gliding) already would.
    transitioning_until: Option<Instant>,
}

impl Default for ProgramState {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            fov: DEFAULT_FOV,
            drag_origin: None,
            velocity: (0.0, 0.0),
            last_move: None,
            gliding: false,
            last_tick: None,
            last_seen_id: None,
            transitioning_until: None,
        }
    }
}

impl ProgramState {
    /// Drag distance to rotation: scaled by zoom so the photo roughly
    /// follows the cursor whether zoomed in or out.
    fn sensitivity(&self) -> f32 {
        DRAG_SENSITIVITY * (self.fov / DEFAULT_FOV)
    }

    fn set_pitch(&mut self, pitch: f32) {
        let limit = 85f32.to_radians();
        self.pitch = pitch.clamp(-limit, limit);
    }
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
        // Gliding is driven by redraw events: each frame advances the view
        // by velocity * elapsed time and asks for the next frame. A
        // crossfade is driven the same way — `draw` always carries whatever
        // id was passed to this frame's `photosphere(id)` call, so noticing
        // it changed is how a newly navigated-to photo gets caught here.
        if let Event::Window(window::Event::RedrawRequested(now)) = event {
            if state.last_seen_id != Some(self.id) {
                state.last_seen_id = Some(self.id);
                state.transitioning_until = Some(*now + TRANSITION_DURATION);
            }
            let transitioning = state.transitioning_until.is_some_and(|until| *now < until);
            if !transitioning {
                state.transitioning_until = None;
            }

            if !state.gliding {
                return transitioning.then(shader::Action::request_redraw);
            }
            let Some(previous) = state.last_tick.replace(*now) else {
                return Some(shader::Action::request_redraw());
            };
            let dt = now.saturating_duration_since(previous).as_secs_f32();

            let decay = (-INERTIA_DECAY * dt).exp();
            state.velocity.0 *= decay;
            state.velocity.1 *= decay;

            state.yaw += state.velocity.0 * dt;
            let before = state.pitch;
            state.set_pitch(state.pitch + state.velocity.1 * dt);
            if state.pitch != before + state.velocity.1 * dt {
                // Hit the top or bottom limit: stop pushing against it.
                state.velocity.1 = 0.0;
            }

            if state.velocity.0.hypot(state.velocity.1) < INERTIA_MIN_SPEED {
                state.gliding = false;
                state.last_tick = None;
                return transitioning.then(shader::Action::request_redraw);
            }
            return Some(shader::Action::request_redraw());
        }

        let Event::Mouse(mouse_event) = event else {
            return None;
        };

        match mouse_event {
            mouse::Event::ButtonPressed(mouse::Button::Left) => {
                let position = cursor.position_in(bounds)?;
                // Grabbing the photo stops any glide immediately.
                state.gliding = false;
                state.last_tick = None;
                state.velocity = (0.0, 0.0);
                state.last_move = Some((Instant::now(), state.yaw, state.pitch));
                state.drag_origin = Some((position, state.yaw, state.pitch));
                None
            }
            mouse::Event::CursorMoved { .. } => {
                let (start, start_yaw, start_pitch) = state.drag_origin?;
                let position = cursor.position_in(bounds)?;
                let dx = position.x - start.x;
                let dy = position.y - start.y;
                let sensitivity = state.sensitivity();

                // Dragging right turns the view right, matching SphereView's
                // feel (the view follows the drag, rather than the photo
                // being grabbed and dragged like a ball).
                state.yaw = start_yaw + dx * sensitivity;
                state.set_pitch(start_pitch - dy * sensitivity);

                // Track how fast the view is moving, smoothed so one jittery
                // sample can't launch the glide.
                let now = Instant::now();
                if let Some((then, last_yaw, last_pitch)) = state.last_move {
                    let dt = now.saturating_duration_since(then).as_secs_f32().max(0.001);
                    let instant = ((state.yaw - last_yaw) / dt, (state.pitch - last_pitch) / dt);
                    state.velocity.0 = 0.5 * state.velocity.0 + 0.5 * instant.0;
                    state.velocity.1 = 0.5 * state.velocity.1 + 0.5 * instant.1;
                }
                state.last_move = Some((now, state.yaw, state.pitch));

                Some(shader::Action::request_redraw())
            }
            mouse::Event::ButtonReleased(mouse::Button::Left) => {
                state.drag_origin = None;

                // Only a release right after movement is a flick; holding
                // still before letting go should leave the view where it is.
                let flicked = state
                    .last_move
                    .is_some_and(|(then, ..)| then.elapsed() < FLICK_WINDOW);
                state.last_move = None;

                if flicked && state.velocity.0.hypot(state.velocity.1) > INERTIA_MIN_SPEED {
                    state.gliding = true;
                    state.last_tick = None;
                    Some(shader::Action::request_redraw())
                } else {
                    state.velocity = (0.0, 0.0);
                    None
                }
            }
            mouse::Event::WheelScrolled { delta } => {
                cursor.position_in(bounds)?;
                let lines = match delta {
                    mouse::ScrollDelta::Lines { y, .. } => *y,
                    mouse::ScrollDelta::Pixels { y, .. } => *y / 10.0,
                };
                // Scrolling up zooms in, i.e. narrows the field of view.
                state.fov = (state.fov - lines * ZOOM_STEP).clamp(MIN_FOV, MAX_FOV);
                Some(shader::Action::request_redraw().and_capture())
            }
            _ => None,
        }
    }

    fn draw(&self, state: &Self::State, _cursor: mouse::Cursor, bounds: Rectangle) -> Self::Primitive {
        Primitive {
            id: self.id,
            yaw: state.yaw,
            pitch: state.pitch,
            fov: state.fov,
            aspect: bounds.width / bounds.height,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    /// Only `[0]` is used; the rest pads the struct to a 16-byte multiple,
    /// which uniform buffers require.
    blend: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
}

/// One frame's worth of what the shader needs: which photo (by id, not the
/// pixel data itself) and the current view. Rebuilt every `draw` call —
/// cheap, since it's just a handful of numbers.
#[derive(Debug)]
struct Primitive {
    id: PhotoId,
    yaw: f32,
    pitch: f32,
    fov: f32,
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
        pipeline.ensure_texture(device, queue, self.id);

        let now = Instant::now();
        let blend = pipeline.transition.interpolate_with(|value| value, now);
        if !pipeline.transition.is_animating(now) && pipeline.previous_texture.is_some() {
            // The crossfade has finished: drop the old texture (freeing its
            // GPU memory) and point both bind-group slots at `current` alone.
            pipeline.previous_texture = None;
            pipeline.rebuild_bind_group(device);
        }

        // Horizontal field of view (changed by scrolling), converted to the
        // vertical FOV glam's projection matrix wants.
        let tan_half_fov_x = (self.fov.to_radians() / 2.0).tan();
        let tan_half_fov_y = tan_half_fov_x / self.aspect;
        let fov_y = 2.0 * tan_half_fov_y.atan();

        let projection = glam::Mat4::perspective_rh(fov_y, self.aspect, 0.01, 10.0);
        // No translation anywhere — the camera always sits at the exact
        // center of the sphere, so "the view" is purely a rotation.
        let view = glam::Mat4::from_rotation_x(self.pitch) * glam::Mat4::from_rotation_y(-self.yaw);

        let uniforms = Uniforms {
            view_proj: (projection * view).to_cols_array_2d(),
            blend: [blend, 0.0, 0.0, 0.0],
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
        let Some(bind_group) = &pipeline.bind_group else {
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
            multiview_mask: None,
        });

        pass.set_scissor_rect(
            clip_bounds.x,
            clip_bounds.y,
            clip_bounds.width,
            clip_bounds.height,
        );
        pass.set_pipeline(&pipeline.render_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
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
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    current_texture: Option<TextureState>,
    /// The photo being crossfaded *from*. `None` once there's nothing to
    /// fade from (no photo shown yet, or the crossfade already finished).
    previous_texture: Option<TextureState>,
    /// 0.0 = show `previous_texture`, 1.0 = show `current_texture`.
    transition: Animation<f32>,
}

struct TextureState {
    /// Which photo this is — just the id, not the pixel data, so this
    /// doesn't keep a second full-resolution copy of the photo resident in
    /// RAM alongside the one the GPU already has.
    id: PhotoId,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Pipeline {
    fn ensure_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, id: PhotoId) {
        if let Some(state) = &self.current_texture {
            if state.id == id {
                return;
            }
        }

        // Take (not clone) the staged photo: once it's uploaded below, the
        // only `Arc<image::RgbaImage>` in the whole program goes out of
        // scope at the end of this function and its ~100+ MB buffer is
        // freed — the GPU texture is the only copy left after that.
        let source = {
            let mut staged = STAGED.lock().unwrap();
            match staged.take() {
                Some((staged_id, image)) if staged_id == id.0 => image,
                other => {
                    // Wrong or missing id — e.g. the GPU device was lost and
                    // needs a fresh upload, but the original pixel data was
                    // already freed. Nothing sane to do but leave the old
                    // texture (or none) in place; put back whatever we took.
                    *staged = other;
                    return;
                }
            }
        };

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
            // Plain Unorm, not UnormSrgb: libcosmic forces iced's "web-colors"
            // feature on, which disables gamma correction and makes the
            // compositor pick a non-sRGB render surface. An UnormSrgb texture
            // would get linearized on every sample, then written straight to
            // that non-sRGB target with no re-encode step, making the whole
            // image noticeably darker than the source photo.
            format: wgpu::TextureFormat::Rgba8Unorm,
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
        // `source` drops at the end of this block — its pixel buffer is
        // freed, since the texture upload above was the only thing that
        // still needed it.

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let new_texture = TextureState {
            id,
            _texture: texture,
            view,
        };

        if self.current_texture.is_some() {
            // A real crossfade: there was already a photo showing, so fade
            // from it into the new one instead of cutting.
            self.previous_texture = self.current_texture.take();
            self.transition = Animation::new(0.0_f32).duration(TRANSITION_DURATION);
            self.transition.go_mut(1.0, Instant::now());
        }
        self.current_texture = Some(new_texture);

        self.rebuild_bind_group(device);
    }

    /// Rebuilds the bind group from whatever `current_texture`/
    /// `previous_texture` currently are. When there's no real previous photo
    /// (nothing shown yet, or a crossfade just finished), both texture slots
    /// point at `current_texture` — blending a texture into itself at any
    /// ratio is still just that texture, so this is always safe.
    fn rebuild_bind_group(&mut self, device: &wgpu::Device) {
        let Some(current) = &self.current_texture else {
            return;
        };
        let previous = self.previous_texture.as_ref().unwrap_or(current);

        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("photosphere bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&previous.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&current.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        }));
    }
}

/// A UV-sphere of unit-length positions: rows of latitude from the north pole
/// (`i = 0`) to the south pole (`i = LAT_SEGMENTS`), each split into columns of
/// longitude. It only supplies geometry (and a direction per vertex); the photo
/// coordinate is computed per pixel in the fragment shader, which keeps it
/// exact even at the poles.
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
            // y = up, z = -forward). The fragment shader inverts this.
            vertices.push(Vertex {
                position: [right, up, -forward],
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

        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("photosphere bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        // Also FRAGMENT now: the fragment shader reads the
                        // uniform's blend factor, not just view_proj.
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    texture_entry(1), // previous_texture
                    texture_entry(2), // current_texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("photosphere pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
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
            multiview_mask: None,
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
            bind_group: None,
            uniform_buffer,
            sampler,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            current_texture: None,
            previous_texture: None,
            // Settled at 1.0 ("fully current") by default — there's nothing
            // to fade from until a second photo is ever loaded.
            transition: Animation::new(1.0_f32),
        }
    }
}
