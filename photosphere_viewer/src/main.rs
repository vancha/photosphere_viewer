// Two screens: `Picker` (no photo loaded yet — just an "Open" button) and
// `Viewer` (drag-to-look-around, same as before). `Message::Viewer(..)` wraps
// everything the viewer itself emits, so the top level only needs to know
// about loading a file; everything about *looking at* one stays exactly as
// it was, just renamed from `App`/`Message` to `Viewer`/`ViewerMessage`.
use iced::widget::image::Handle;
use iced::widget::{button, column, container, mouse_area, text};
use iced::{Element, Length, Point, Size, Subscription, Task};
use photosphere_viewer::projection::{View, render_view};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

const VIEW_WIDTH: u32 = 1024;
const VIEW_HEIGHT: u32 = 768;
/// Radians of yaw/pitch per pixel dragged. Purely a feel setting — raise it
/// for a more sensitive drag, lower it for a slower one.
const DRAG_SENSITIVITY: f64 = 0.005;
/// How often to check "has resizing stopped?" — also the longest a settled
/// resize can take to actually trigger a downscale.
const SETTLE_CHECK_INTERVAL: Duration = Duration::from_millis(300);

enum App {
    /// No photo loaded yet. `error` holds a message from the last failed
    /// load attempt, if any, so the picker screen can show why.
    Picker { error: Option<String> },
    Viewer(Viewer),
}

#[derive(Debug, Clone)]
enum Message {
    OpenFileDialog,
    FileSelected(Option<PathBuf>),
    ImageLoaded(Result<Arc<image::RgbImage>, String>),
    CloseRequested,
    Viewer(ViewerMessage),
}

fn update(app: &mut App, message: Message) -> Task<Message> {
    match message {
        // Hard-exit rather than letting the default shutdown run: tokio's
        // runtime waits for every `spawn_blocking` task to actually finish
        // before it'll let the process die, and a render or downscale still
        // in flight (very possible right after dragging or resizing — the
        // moment you'd naturally go to close the window) can make that wait
        // genuinely noticeable. There's nothing here worth waiting to clean
        // up, so just leave immediately instead.
        Message::CloseRequested => std::process::exit(0),
        Message::OpenFileDialog => Task::perform(pick_file(), Message::FileSelected),
        Message::FileSelected(None) => Task::none(),
        Message::FileSelected(Some(path)) => {
            Task::perform(load_image_async(path), Message::ImageLoaded)
        }
        Message::ImageLoaded(Ok(source)) => {
            // Carry over the known window size, if any, so the new photo's
            // very first render already targets the right resolution instead
            // of falling back to a fixed default until the next resize event.
            let window_size = match app {
                App::Viewer(viewer) => viewer.window_size,
                App::Picker { .. } => Size::ZERO,
            };
            *app = App::Viewer(Viewer::new(source, window_size));
            Task::none()
        }
        Message::ImageLoaded(Err(error)) => {
            *app = App::Picker { error: Some(error) };
            Task::none()
        }
        Message::Viewer(message) => {
            let App::Viewer(viewer) = app else {
                return Task::none();
            };
            viewer.update(message).map(Message::Viewer)
        }
    }
}

fn view(app: &App) -> Element<'_, Message> {
    match app {
        App::Picker { error } => {
            let mut content = column![
                text("No photo loaded"),
                button("Open Image...").on_press(Message::OpenFileDialog),
            ]
            .spacing(12)
            .align_x(iced::Alignment::Center);

            if let Some(error) = error {
                content = content.push(text(error.clone()));
            }

            container(content)
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x(Length::Fill)
                .center_y(Length::Fill)
                .into()
        }
        App::Viewer(viewer) => column![
            button("Open Image...").on_press(Message::OpenFileDialog),
            viewer.view().map(Message::Viewer),
        ]
        .into(),
    }
}

fn subscription(app: &App) -> Subscription<Message> {
    let close = iced::window::close_events().map(|_id| Message::CloseRequested);
    let screen = match app {
        App::Picker { .. } => Subscription::none(),
        App::Viewer(viewer) => viewer.subscription().map(Message::Viewer),
    };
    Subscription::batch([close, screen])
}

async fn pick_file() -> Option<PathBuf> {
    rfd::AsyncFileDialog::new()
        .add_filter("Images", &["jpg", "jpeg", "png"])
        .pick_file()
        .await
        .map(|handle| handle.path().to_path_buf())
}

/// Loads and decodes a photo off the async executor's own worker threads —
/// decoding a large JPEG is real, synchronous CPU work, so this follows the
/// same `spawn_blocking` rule as `render_async`/`downscale_async` below.
async fn load_image_async(path: PathBuf) -> Result<Arc<image::RgbImage>, String> {
    let result = tokio::task::spawn_blocking(move || {
        image::open(&path)
            .map(|image| Arc::new(image.to_rgb8()))
            .map_err(|error| format!("Couldn't open {}: {error}", path.display()))
    })
    .await;

    match result {
        Ok(loaded) => loaded,
        Err(error) => Err(format!("Loading task panicked: {error}")),
    }
}

// --- Viewer: drag-to-look-around on top of `projection::render_view`,
// updating live while dragging rather than only on release. The one rule
// that makes that safe: never more than one render in flight at a time. If
// the cursor moves again before the current render finishes, nothing new is
// started yet — the moment it *does* finish, it immediately re-checks where
// the cursor ended up and chases that instead. That naturally throttles the
// rate to "as fast as this machine can actually render," and since results
// can only ever arrive one at a time, a newer frame can never get clobbered
// by an older one finishing late. The exact same chase-the-latest-target
// rule also guards the window-resize-triggered source downscale below.

struct Viewer {
    /// The original, full-resolution photo — never touched after loading, so
    /// repeated downscales (as the window keeps changing size) always start
    /// fresh from full quality instead of compounding on an already-shrunk copy.
    master_source: Arc<image::RgbImage>,
    /// What `render_view` actually samples from right now — `master_source`
    /// at first, swapped for a downscaled copy once one finishes computing.
    source: Arc<image::RgbImage>,
    /// The window's current size, as last reported by `window::resize_events`.
    window_size: Size,
    /// What `window_size` was the last time the settle-check ran — used to
    /// tell "still resizing" (it changed again) from "settled" (it didn't).
    last_checked_size: Size,
    /// The size we most recently started (or finished) downscaling for, so
    /// settling on the same size twice in a row doesn't redo the work.
    downscaled_for: Size,
    /// Whether a downscale `Task` is currently in flight — the same
    /// never-more-than-one-at-a-time rule `rendering` enforces for renders,
    /// and for the same reason: without it, settling on several sizes in a
    /// row (common during a multi-step resize) fires one downscale per
    /// settle, and since they're not guaranteed to finish in order, a newer,
    /// better result could get silently overwritten by an older one that
    /// just happened to take longer.
    downscaling: bool,
    yaw: f64,
    pitch: f64,
    /// What's currently on screen. Only replaced once a re-render finishes.
    handle: Handle,
    /// The latest cursor position `mouse_area` has reported, tracked
    /// continuously so that whenever a drag *ends*, we know where it ended —
    /// `on_release` itself doesn't carry a position.
    last_cursor: Point,
    /// Set on press, cleared on release: where the drag started, and what
    /// `yaw`/`pitch` were at that moment, so the total drag distance can be
    /// turned into an absolute new angle rather than an accumulating one.
    drag_origin: Option<(Point, f64, f64)>,
    /// Whether a render `Task` is currently in flight.
    rendering: bool,
    /// The `(yaw, pitch)` the most recently *started* render was asked for —
    /// compared against the live `yaw`/`pitch` to tell "nothing's changed
    /// since we last asked" apart from "the cursor moved again, go again".
    last_requested: (f64, f64),
}

#[derive(Debug, Clone)]
enum ViewerMessage {
    DragStarted,
    CursorMoved(Point),
    DragEnded,
    ViewRendered(Handle),
    WindowResized(Size),
    SettleCheckTicked,
    SourceDownscaled(Arc<image::RgbImage>),
}

impl Viewer {
    fn new(source: Arc<image::RgbImage>, window_size: Size) -> Self {
        let (width, height) = render_dimensions(window_size);
        let view = View {
            yaw: 0.0,
            pitch: 0.0,
            fov_x: 90f64.to_radians(),
        };
        let rendered = render_view(&source, &view, width, height);
        let handle =
            Handle::from_rgba(rendered.width(), rendered.height(), to_rgba_bytes(&rendered));

        Viewer {
            master_source: Arc::clone(&source),
            source,
            window_size,
            last_checked_size: window_size,
            downscaled_for: Size::ZERO,
            downscaling: false,
            yaw: 0.0,
            pitch: 0.0,
            handle,
            last_cursor: Point::ORIGIN,
            drag_origin: None,
            rendering: false,
            last_requested: (0.0, 0.0),
        }
    }

    fn update(&mut self, message: ViewerMessage) -> Task<ViewerMessage> {
        match message {
            ViewerMessage::DragStarted => {
                self.drag_origin = Some((self.last_cursor, self.yaw, self.pitch));
                Task::none()
            }
            ViewerMessage::CursorMoved(position) => {
                self.last_cursor = position;
                let Some((start_cursor, start_yaw, start_pitch)) = self.drag_origin else {
                    return Task::none();
                };
                let dx = position.x - start_cursor.x;
                let dy = position.y - start_cursor.y;

                // Dragging right turns the view left, the way dragging a
                // photo itself (rather than steering a camera) usually feels.
                self.yaw = start_yaw - f64::from(dx) * DRAG_SENSITIVITY;
                // Stay shy of the true poles — beyond there, "left/right"
                // stops meaning anything sensible to drag toward.
                self.pitch = (start_pitch + f64::from(dy) * DRAG_SENSITIVITY)
                    .clamp(-85f64.to_radians(), 85f64.to_radians());

                self.request_render()
            }
            ViewerMessage::DragEnded => {
                self.drag_origin = None;
                Task::none()
            }
            ViewerMessage::ViewRendered(handle) => {
                self.handle = handle;
                self.rendering = false;
                // If the cursor kept moving while this render was in flight,
                // `yaw`/`pitch` have already moved past what we just
                // rendered — chase it. A no-op if nothing changed meanwhile.
                self.request_render()
            }
            ViewerMessage::WindowResized(size) => {
                self.window_size = size;
                Task::none()
            }
            ViewerMessage::SettleCheckTicked => {
                // "Settled" means: unchanged since the last time we checked.
                let settled = self.window_size == self.last_checked_size;
                self.last_checked_size = self.window_size;
                if settled {
                    self.request_downscale()
                } else {
                    Task::none()
                }
            }
            ViewerMessage::SourceDownscaled(source) => {
                self.source = source;
                self.downscaling = false;
                // The view on screen was rendered from the old source —
                // refresh it. `last_requested` is set to a value `yaw`/
                // `pitch` can never actually equal, so `request_render` can't
                // mistake this for "we already rendered this exact view" and
                // skip it.
                self.last_requested = (f64::NAN, f64::NAN);
                // Also chase: if the window settled somewhere *else* while
                // this downscale was running (skipped while `downscaling`
                // was true), go again now, toward wherever it actually is.
                Task::batch([self.request_render(), self.request_downscale()])
            }
        }
    }

    /// Starts a render toward the current `yaw`/`pitch` — unless one's
    /// already running, or we'd just be asking for the exact view we last
    /// asked for, in which case this does nothing.
    fn request_render(&mut self) -> Task<ViewerMessage> {
        if self.rendering || (self.yaw, self.pitch) == self.last_requested {
            return Task::none();
        }
        self.rendering = true;
        self.last_requested = (self.yaw, self.pitch);
        let source = Arc::clone(&self.source);
        let (width, height) = render_dimensions(self.window_size);
        Task::perform(
            render_async(source, self.yaw, self.pitch, width, height),
            ViewerMessage::ViewRendered,
        )
    }

    /// Starts a downscale toward the current `window_size` — unless one's
    /// already running, or we'd just be asking for a size we already
    /// (started) downscaling for.
    fn request_downscale(&mut self) -> Task<ViewerMessage> {
        if self.downscaling || self.window_size == self.downscaled_for {
            return Task::none();
        }
        self.downscaling = true;
        self.downscaled_for = self.window_size;
        Task::perform(
            downscale_async(Arc::clone(&self.master_source), self.window_size),
            ViewerMessage::SourceDownscaled,
        )
    }

    fn view(&self) -> Element<'_, ViewerMessage> {
        mouse_area(iced::widget::image(self.handle.clone()))
            .on_press(ViewerMessage::DragStarted)
            .on_release(ViewerMessage::DragEnded)
            .on_move(ViewerMessage::CursorMoved)
            .into()
    }

    /// Window resizes fire rapidly while actively dragging a window edge,
    /// far too often to downscale on every one — `resize_events` just
    /// records the latest size. `SettleCheckTicked`, a plain repeating
    /// timer, is what actually decides "resizing has stopped" (unchanged
    /// since the last tick) and triggers the one-time downscale.
    fn subscription(&self) -> Subscription<ViewerMessage> {
        Subscription::batch([
            iced::window::resize_events().map(|(_id, size)| ViewerMessage::WindowResized(size)),
            iced::time::every(SETTLE_CHECK_INTERVAL).map(|_| ViewerMessage::SettleCheckTicked),
        ])
    }
}

/// What to actually render at: the real window size once known, so no pixel
/// is ever computed just to be thrown away scaling it back down to fit —
/// falling back to a fixed default before the first resize event arrives,
/// since the true size isn't known yet at that point.
fn render_dimensions(window_size: Size) -> (u32, u32) {
    if window_size.width < 1.0 || window_size.height < 1.0 {
        return (VIEW_WIDTH, VIEW_HEIGHT);
    }
    (window_size.width.round() as u32, window_size.height.round() as u32)
}

/// Does the actual (CPU-bound, not-instant) reprojection work. Run via
/// `tokio::task::spawn_blocking` rather than directly as an `async fn` body —
/// this function never actually awaits anything, it's synchronous CPU work
/// start to finish, and handing that straight to the async executor would
/// occupy one of its worker threads for the whole duration instead of the
/// dedicated thread pool blocking work is supposed to run on. Do that enough
/// times in a row (a long dragging session, say) and the async side — which
/// is also how other messages, like a window-close click, get delivered —
/// can end up starved behind a growing backlog of blocked workers.
async fn render_async(
    source: Arc<image::RgbImage>,
    yaw: f64,
    pitch: f64,
    width: u32,
    height: u32,
) -> Handle {
    tokio::task::spawn_blocking(move || {
        let view = View {
            yaw,
            pitch,
            fov_x: 90f64.to_radians(),
        };
        let rendered = render_view(&source, &view, width, height);
        Handle::from_rgba(rendered.width(), rendered.height(), to_rgba_bytes(&rendered))
    })
    .await
    .expect("render task panicked")
}

/// Shrinks `master` toward roughly double the window's largest dimension —
/// generous enough that the view still has real detail to sample from, while
/// avoiding per-pixel lookups into a photo far bigger than anything currently
/// on screen could show. Never *upscales* past the original: if the window
/// is already bigger than that, there's nothing worth doing. Also run via
/// `spawn_blocking` — same reasoning as `render_async`.
async fn downscale_async(master: Arc<image::RgbImage>, window_size: Size) -> Arc<image::RgbImage> {
    tokio::task::spawn_blocking(move || {
        let target_width = ((window_size.width.max(window_size.height) * 2.0) as u32)
            .clamp(256, master.width());
        if target_width >= master.width() {
            return master;
        }
        // Equirectangular photos are always 2:1 — keep that ratio.
        let target_height = target_width / 2;

        let resized = image::imageops::resize(
            master.as_ref(),
            target_width,
            target_height,
            image::imageops::FilterType::Triangle,
        );
        Arc::new(resized)
    })
    .await
    .expect("downscale task panicked")
}

fn boot(initial_path: Option<PathBuf>) -> (App, Task<Message>) {
    let task = match initial_path {
        Some(path) => Task::perform(load_image_async(path), Message::ImageLoaded),
        None => Task::none(),
    };
    (App::Picker { error: None }, task)
}

fn main() -> iced::Result {
    let initial_path = std::env::args().nth(1).map(PathBuf::from);

    iced::application(move || boot(initial_path.clone()), update, view)
        .title("Photosphere Viewer")
        .subscription(subscription)
        .run()
}

/// `Handle::from_rgba` wants 4 bytes per pixel; `render_view` deals in plain
/// RGB, so this just pads an opaque alpha byte onto every pixel.
fn to_rgba_bytes(img: &image::RgbImage) -> Vec<u8> {
    let mut out = Vec::with_capacity((img.width() * img.height() * 4) as usize);
    for pixel in img.pixels() {
        out.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
    }
    out
}
