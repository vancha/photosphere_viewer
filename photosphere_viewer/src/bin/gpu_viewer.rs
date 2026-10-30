// Same Picker/Viewer structure as the main CPU binary (src/main.rs) — but
// notice how much smaller the "Viewer" side is here. The CPU version needed
// a whole `Viewer` struct tracking window size, a downscaled source, render-
// in-flight state, and a chase-the-latest-target dance. None of that exists
// here: all of it lives *inside* `iced_photosphere::photosphere`, so a
// loaded photo is all the state this binary needs to carry around at all.
use iced::widget::{button, column, container, text};
use iced::{Alignment, Element, Length, Subscription, Task, window};
use std::path::PathBuf;
use std::sync::Arc;

enum App {
    /// No photo loaded yet. `error` holds a message from the last failed
    /// load attempt, if any, so the picker screen can show why.
    Picker { error: Option<String> },
    Loading,
    Viewer(Arc<image::RgbaImage>),
}

#[derive(Debug, Clone)]
enum Message {
    OpenFileDialog,
    FileSelected(Option<PathBuf>),
    ImageLoaded(Result<Arc<image::RgbaImage>, String>),
    CloseRequested,
}

fn update(app: &mut App, message: Message) -> Task<Message> {
    match message {
        // Hard-exit rather than letting the default shutdown run: tokio's
        // runtime waits for every `spawn_blocking` task (like the image
        // decode below) to actually finish before it'll let the process
        // die. Nothing here is worth waiting to clean up.
        Message::CloseRequested => std::process::exit(0),
        Message::OpenFileDialog => Task::perform(pick_file(), Message::FileSelected),
        Message::FileSelected(None) => Task::none(),
        Message::FileSelected(Some(path)) => {
            *app = App::Loading;
            Task::perform(load_image_async(path), Message::ImageLoaded)
        }
        Message::ImageLoaded(Ok(source)) => {
            *app = App::Viewer(source);
            Task::none()
        }
        Message::ImageLoaded(Err(error)) => {
            *app = App::Picker { error: Some(error) };
            Task::none()
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
            .align_x(Alignment::Center);

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
        App::Loading => container(text("Loading..."))
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into(),
        App::Viewer(source) => column![
            button("Open Image...").on_press(Message::OpenFileDialog),
            iced_photosphere::photosphere(source)
                .width(Length::Fill)
                .height(Length::Fill),
        ]
        .into(),
    }
}

fn subscription(_app: &App) -> Subscription<Message> {
    window::close_events().map(|_id| Message::CloseRequested)
}

async fn pick_file() -> Option<PathBuf> {
    rfd::AsyncFileDialog::new()
        .add_filter("Images", &["jpg", "jpeg", "png"])
        .pick_file()
        .await
        .map(|handle| handle.path().to_path_buf())
}

/// Loads and decodes a photo off the async executor's own worker threads —
/// decoding a large JPEG is real, synchronous CPU work.
async fn load_image_async(path: PathBuf) -> Result<Arc<image::RgbaImage>, String> {
    let result = tokio::task::spawn_blocking(move || {
        let start = std::time::Instant::now();
        let decoded = image::open(&path)
            .map_err(|error| format!("Couldn't open {}: {error}", path.display()))?;
        eprintln!("[DEBUG] decode: {:?}", start.elapsed());

        let start = std::time::Instant::now();
        let rgba = decoded.to_rgba8();
        eprintln!(
            "[DEBUG] to_rgba8 ({}x{}): {:?}",
            rgba.width(),
            rgba.height(),
            start.elapsed()
        );

        Ok(Arc::new(rgba))
    })
    .await;

    match result {
        Ok(loaded) => loaded,
        Err(error) => Err(format!("Loading task panicked: {error}")),
    }
}

fn boot(initial_path: Option<PathBuf>) -> (App, Task<Message>) {
    match initial_path {
        Some(path) => (
            App::Loading,
            Task::perform(load_image_async(path), Message::ImageLoaded),
        ),
        None => (App::Picker { error: None }, Task::none()),
    }
}

fn main() -> iced::Result {
    let initial_path = std::env::args().nth(1).map(PathBuf::from);

    iced::application(move || boot(initial_path.clone()), update, view)
        .title("Photosphere Viewer (GPU)")
        .subscription(subscription)
        .run()
}
