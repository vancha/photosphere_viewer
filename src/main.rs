//! A 360-degree photo viewer for COSMIC.

use cosmic::app::context_drawer::{self, ContextDrawer};
use cosmic::app::{ApplicationExt, Core, Settings, Task};
use cosmic::dialog::file_chooser::{self, FileFilter};
use cosmic::iced::keyboard::Key;
use cosmic::iced::keyboard::key::Named;
use cosmic::iced::{Alignment, Event, Length, Size, Subscription, event, keyboard, window};
use cosmic::widget::about::About;
use cosmic::widget::menu::action::MenuAction;
use cosmic::widget::menu::key_bind::{KeyBind, Modifier};
use cosmic::widget::menu::{self, ItemHeight, ItemWidth};
use cosmic::widget::nav_bar;
use cosmic::widget::{self, RcElementWrapper, button};
use cosmic::{Element, executor};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use url::Url;

mod spinner;

const APP_ID: &str = "dev.vancha.PhotosphereViewer";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let initial_path = std::env::args().nth(1).map(PathBuf::from);
    let settings = Settings::default().size(Size::new(1100.0, 700.0));
    cosmic::app::run::<App>(settings, initial_path)?;
    Ok(())
}

enum State {
    Empty,
    Loading,
    // Just an id, not the decoded photo itself — `iced_photosphere::load`
    // holds the pixel data only until the GPU has uploaded it, then drops
    // it, so this doesn't keep a redundant full-resolution copy in RAM for
    // as long as the photo stays open.
    Viewer(iced_photosphere::PhotoId),
    Error(String),
}

#[derive(Clone, Debug)]
enum Message {
    OpenFile,
    Selected(Vec<Url>),
    Cancelled,
    DialogFailed(String),
    /// A decode (either the one the user is waiting on, or a speculative
    /// prefetch of a neighboring photo) finished for the photo at this
    /// index in `App::opened`.
    Decoded(usize, Result<Arc<image::RgbaImage>, String>),
    ToggleAbout,
    ToggleShortcuts,
    ToggleFullscreen,
    OpenUrl(String),
    Quit,
    PreviousImage,
    NextImage,
    /// Fires ~1s after a navigation shows the position indicator. Only acts
    /// if its generation still matches `App::position_indicator_generation`
    /// — otherwise a newer navigation already superseded it.
    HidePositionIndicator(u64),
}

/// Menu actions, mapped to a [`Message`] via [`MenuAction::message`] — the
/// same mapping a [`KeyBind`] looks up to fire a menu item's action from the
/// keyboard instead of a click.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Action {
    OpenFile,
    ToggleAbout,
    ToggleShortcuts,
    ToggleFullscreen,
    Quit,
    PreviousImage,
    NextImage,
}

impl MenuAction for Action {
    type Message = Message;

    fn message(&self) -> Self::Message {
        match self {
            Action::OpenFile => Message::OpenFile,
            Action::ToggleAbout => Message::ToggleAbout,
            Action::ToggleShortcuts => Message::ToggleShortcuts,
            Action::ToggleFullscreen => Message::ToggleFullscreen,
            Action::Quit => Message::Quit,
            Action::PreviousImage => Message::PreviousImage,
            Action::NextImage => Message::NextImage,
        }
    }
}

/// Which context drawer (the panel on the right, opened via the `View` menu)
/// is currently showing, if any.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ContextPage {
    About,
    Shortcuts,
}

struct App {
    core: Core,
    state: State,
    about: About,
    context_page: Option<ContextPage>,
    key_binds: HashMap<KeyBind, Action>,

    nav_model: nav_bar::Model,
    /// Every photo opened this session, in the order it was opened — a nav
    /// bar entry's `data::<usize>` is an index into this.
    opened: Vec<PathBuf>,
    /// `nav_model`'s item id for `opened[i]`, kept in lockstep with it.
    nav_ids: Vec<nav_bar::Id>,
    current_index: Option<usize>,
    /// Decoded pixel data, but only ever for the current photo and its
    /// immediate neighbors (`current_index` +/- 1) — a sliding window, not a
    /// full history. Letting every opened photo linger here would bring
    /// back exactly the per-photo RAM cost the GPU-upload design avoids.
    cache: HashMap<usize, Arc<image::RgbaImage>>,
    /// Indices currently being decoded, so a photo doesn't get a duplicate
    /// prefetch task queued behind the one already in flight for it.
    pending: HashSet<usize>,
    /// True while navigating to a photo that isn't cached yet and a photo is
    /// already on screen — shows the spinner as an overlay on top of it
    /// instead of replacing the view, so the current photo never disappears
    /// while the next one decodes.
    loading_next: bool,
    /// True right after navigating, while the "x / y" position indicator is
    /// showing as an overlay — auto-hidden ~1s later.
    show_position_indicator: bool,
    /// Bumped on every navigation; a delayed `HidePositionIndicator` only
    /// actually hides it if its generation still matches, so rapid
    /// navigation keeps the indicator visible instead of flickering.
    position_indicator_generation: u64,
    fullscreen: bool,
}

impl cosmic::Application for App {
    type Executor = executor::Default;
    type Flags = Option<PathBuf>;
    type Message = Message;
    const APP_ID: &'static str = APP_ID;

    fn core(&self) -> &Core {
        &self.core
    }

    fn core_mut(&mut self) -> &mut Core {
        &mut self.core
    }

    fn init(core: Core, initial_path: Self::Flags) -> (Self, Task<Self::Message>) {
        let about = About::default()
            .name("Photosphere Viewer")
            .icon(widget::icon::from_name(Self::APP_ID))
            .version(env!("CARGO_PKG_VERSION"))
            .comments("A 360-degree photo viewer for COSMIC.");

        let mut app = App {
            core,
            state: State::Empty,
            about,
            context_page: None,
            key_binds: key_binds(),
            nav_model: nav_bar::Model::default(),
            opened: Vec::new(),
            nav_ids: Vec::new(),
            current_index: None,
            cache: HashMap::new(),
            pending: HashSet::new(),
            loading_next: false,
            show_position_indicator: false,
            position_indicator_generation: 0,
            fullscreen: false,
        };
        let task = match initial_path {
            Some(path) => app.open_path(path),
            None => Task::none(),
        };
        (app, task)
    }

    fn header_start(&self) -> Vec<Element<'_, Self::Message>> {
        vec![menu_bar(&self.key_binds)]
    }

    fn header_end(&self) -> Vec<Element<'_, Self::Message>> {
        vec![button::suggested("Open Image...").on_press(Message::OpenFile).into()]
    }

    fn nav_model(&self) -> Option<&nav_bar::Model> {
        if self.opened.is_empty() {
            None
        } else {
            Some(&self.nav_model)
        }
    }

    fn on_nav_select(&mut self, id: nav_bar::Id) -> Task<Self::Message> {
        match self.nav_model.data::<usize>(id) {
            Some(&index) => self.navigate_to(index),
            None => Task::none(),
        }
    }

    fn subscription(&self) -> Subscription<Self::Message> {
        event::listen_with(|event, _status, _window_id| {
            let Event::Keyboard(keyboard::Event::KeyPressed {
                key,
                modifiers,
                physical_key,
                ..
            }) = event
            else {
                return None;
            };

            key_binds()
                .iter()
                .find(|(key_bind, _)| key_bind.matches(modifiers, &key, Some(&physical_key)))
                .map(|(_, action)| action.message())
        })
    }

    fn context_drawer(&self) -> Option<ContextDrawer<'_, Self::Message>> {
        match self.context_page? {
            ContextPage::About => Some(context_drawer::about(
                &self.about,
                |url| Message::OpenUrl(url.to_owned()),
                Message::ToggleAbout,
            )),
            ContextPage::Shortcuts => Some(
                context_drawer::context_drawer(
                    shortcuts_view(&self.key_binds),
                    Message::ToggleShortcuts,
                )
                .title("Keyboard Shortcuts"),
            ),
        }
    }

    fn update(&mut self, message: Self::Message) -> Task<Self::Message> {
        match message {
            Message::OpenFile => cosmic::task::future(async {
                let dialog = file_chooser::open::Dialog::new()
                    .title("Choose photospheres")
                    .filter(
                        FileFilter::new("Images")
                            .glob("*.jpg")
                            .glob("*.jpeg")
                            .glob("*.png"),
                    );
                match dialog.open_files().await {
                    Ok(response) => Message::Selected(response.urls().to_vec()),
                    Err(file_chooser::Error::Cancelled) => Message::Cancelled,
                    Err(error) => Message::DialogFailed(error.to_string()),
                }
            }),
            Message::Cancelled => Task::none(),
            Message::DialogFailed(error) => {
                self.state = State::Error(error);
                Task::none()
            }
            Message::Selected(urls) => {
                let mut tasks = Vec::new();
                for url in urls {
                    match url.to_file_path() {
                        Ok(path) => tasks.push(self.open_path(path)),
                        Err(()) => self.state = State::Error(format!("{url} is not a local file")),
                    }
                }
                Task::batch(tasks)
            }
            Message::Decoded(index, result) => {
                self.pending.remove(&index);

                let Some(current) = self.current_index else {
                    return Task::none();
                };
                if index.abs_diff(current) > 1 {
                    // The user has already moved past this photo's window
                    // by the time its decode finished — discard it rather
                    // than caching or displaying something stale.
                    return Task::none();
                }

                match result {
                    Ok(image) => {
                        self.cache.insert(index, Arc::clone(&image));
                        if index == current {
                            self.loading_next = false;
                            self.state = State::Viewer(iced_photosphere::load(image));
                        }
                    }
                    Err(error) => {
                        if index == current {
                            self.loading_next = false;
                            self.state = State::Error(error);
                        }
                    }
                }
                Task::none()
            }
            Message::ToggleAbout => {
                self.toggle_context(ContextPage::About);
                Task::none()
            }
            Message::ToggleShortcuts => {
                self.toggle_context(ContextPage::Shortcuts);
                Task::none()
            }
            Message::ToggleFullscreen => {
                // Deliberately `toggle_maximize`, not `window::set_mode`.
                // True fullscreen also hides the system panel, but exiting
                // it on a tiled window hits a cosmic-comp bug where the
                // restore doesn't reliably land (confirmed not fixable by
                // self-checking/retrying `window::mode` — that flag reports
                // correctly even when the actual bug, in geometry restore,
                // still happens). `toggle_maximize` sidesteps it entirely,
                // same as cosmic-term. Trade-off: panel stays visible.
                self.fullscreen = !self.fullscreen;
                self.core_mut().window.show_headerbar = !self.fullscreen;
                cosmic::command::toggle_maximize(self.core.main_window_id().unwrap())
            }
            Message::OpenUrl(url) => {
                if let Err(error) = open::that_detached(&url) {
                    eprintln!("Failed to open URL {url}: {error}");
                }
                Task::none()
            }
            Message::Quit => window::close(self.core.main_window_id().unwrap()),
            Message::PreviousImage => match self.current_index {
                Some(index) if index > 0 => self.navigate_to(index - 1),
                _ => Task::none(),
            },
            Message::NextImage => match self.current_index {
                Some(index) if index + 1 < self.opened.len() => self.navigate_to(index + 1),
                _ => Task::none(),
            },
            Message::HidePositionIndicator(generation) => {
                if generation == self.position_indicator_generation {
                    self.show_position_indicator = false;
                }
                Task::none()
            }
        }
    }

    fn view(&self) -> Element<'_, Self::Message> {
        match &self.state {
            State::Viewer(id) => {
                let sphere: Element<'_, Message> = iced_photosphere::photosphere(*id)
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .into();

                let mut layers = vec![sphere];
                if self.loading_next {
                    layers.push(centered(loading_indicator()));
                }
                if self.show_position_indicator {
                    if let Some(index) = self.current_index {
                        layers.push(position_indicator(index, self.opened.len()));
                    }
                }

                if layers.len() > 1 {
                    cosmic::iced::widget::stack(layers).into()
                } else {
                    layers.remove(0)
                }
            }
            State::Empty => centered(widget::text::title3("No photo loaded")),
            State::Loading => centered(loading_indicator()),
            State::Error(error) => centered(
                widget::column::with_capacity(2)
                    .push(widget::text::title3("Couldn't open that photo"))
                    .push(widget::text::body(error.as_str()))
                    .spacing(12)
                    .align_x(Alignment::Center),
            ),
        }
    }
}

impl App {
    /// Opens `path`, reusing its existing nav entry (and jumping straight to
    /// it) if it's already been opened this session, otherwise adding a new
    /// one at the end of the list.
    fn open_path(&mut self, path: PathBuf) -> Task<Message> {
        if let Some(index) = self.opened.iter().position(|opened| *opened == path) {
            return self.navigate_to(index);
        }

        let index = self.opened.len();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("Photo")
            .to_string();
        let nav_id = self.nav_model.insert().text(name).data(index).activate().id();

        self.opened.push(path);
        self.nav_ids.push(nav_id);

        self.navigate_to(index)
    }

    /// Shows photo `index`, from cache if it's already decoded (instant —
    /// this is the common case when moving to a photo that was already
    /// prefetched as a neighbor), or by decoding it otherwise. Either way,
    /// also keeps the cache trimmed to just `index` and its neighbors, and
    /// kicks off prefetching for whichever neighbors aren't cached yet.
    fn navigate_to(&mut self, index: usize) -> Task<Message> {
        self.current_index = Some(index);
        self.nav_model.activate(self.nav_ids[index]);
        self.cache.retain(|&cached, _| cached.abs_diff(index) <= 1);

        self.show_position_indicator = true;
        self.position_indicator_generation += 1;
        let generation = self.position_indicator_generation;

        let mut tasks = vec![cosmic::task::future(async move {
            tokio::time::sleep(Duration::from_secs(1)).await;
            Message::HidePositionIndicator(generation)
        })];

        if let Some(image) = self.cache.get(&index).cloned() {
            self.loading_next = false;
            self.state = State::Viewer(iced_photosphere::load(image));
        } else {
            // Keep whatever's already showing on screen rather than
            // replacing it with the full loading screen, unless there's
            // nothing to keep showing (first photo, or recovering from an
            // error) — only then does the loading screen take over fully.
            self.loading_next = true;
            if !matches!(self.state, State::Viewer(_)) {
                self.state = State::Loading;
            }
            tasks.push(self.decode(index));
        }

        for neighbor in [index.checked_sub(1), index.checked_add(1)].into_iter().flatten() {
            if neighbor < self.opened.len()
                && !self.cache.contains_key(&neighbor)
                && !self.pending.contains(&neighbor)
            {
                tasks.push(self.decode(neighbor));
            }
        }

        Task::batch(tasks)
    }

    /// Opens `page` in the context drawer, or closes the drawer if `page` is
    /// already the one showing.
    fn toggle_context(&mut self, page: ContextPage) {
        self.context_page = if self.context_page == Some(page) { None } else { Some(page) };
        self.set_show_context(self.context_page.is_some());
    }

    /// Kicks off decoding `opened[index]`, marking it pending so it isn't
    /// queued again if already in flight.
    fn decode(&mut self, index: usize) -> Task<Message> {
        self.pending.insert(index);
        let path = self.opened[index].clone();
        cosmic::task::future(async move { Message::Decoded(index, load_image(path).await) })
    }
}

fn menu_bar(key_binds: &HashMap<KeyBind, Action>) -> Element<'_, Message> {
    menu::bar(vec![
        menu::Tree::with_children(
            RcElementWrapper::new(Element::from(menu::root("File"))),
            menu::items(
                key_binds,
                vec![
                    menu::Item::Button(
                        "Open Image...",
                        Some(widget::icon::from_name("document-open-symbolic").into()),
                        Action::OpenFile,
                    ),
                    menu::Item::Divider,
                    menu::Item::Button(
                        "Quit",
                        Some(widget::icon::from_name("window-close-symbolic").into()),
                        Action::Quit,
                    ),
                ],
            ),
        ),
        menu::Tree::with_children(
            RcElementWrapper::new(Element::from(menu::root("View"))),
            menu::items(
                key_binds,
                vec![
                    menu::Item::Button("Toggle Fullscreen", None, Action::ToggleFullscreen),
                    menu::Item::Divider,
                    menu::Item::Button("Keyboard Shortcuts", None, Action::ToggleShortcuts),
                    menu::Item::Divider,
                    menu::Item::Button("About Photosphere Viewer", None, Action::ToggleAbout),
                ],
            ),
        ),
    ])
    .item_height(ItemHeight::Dynamic(40))
    .item_width(ItemWidth::Uniform(240))
    .spacing(4.0)
    .into()
}

fn key_binds() -> HashMap<KeyBind, Action> {
    let mut key_binds = HashMap::new();

    key_binds.insert(
        KeyBind {
            modifiers: vec![Modifier::Ctrl],
            key: Key::Character("o".into()),
        },
        Action::OpenFile,
    );
    key_binds.insert(
        KeyBind {
            modifiers: vec![Modifier::Ctrl],
            key: Key::Character("q".into()),
        },
        Action::Quit,
    );
    key_binds.insert(
        KeyBind {
            modifiers: vec![],
            key: Key::Named(Named::ArrowLeft),
        },
        Action::PreviousImage,
    );
    key_binds.insert(
        KeyBind {
            modifiers: vec![],
            key: Key::Named(Named::ArrowRight),
        },
        Action::NextImage,
    );
    key_binds.insert(
        KeyBind {
            modifiers: vec![],
            key: Key::Named(Named::F11),
        },
        Action::ToggleFullscreen,
    );

    key_binds
}

/// Builds the read-only "Action — Key" list shown in the "Keyboard
/// Shortcuts" context drawer, derived straight from `key_binds` so it can't
/// drift out of sync with what the keys actually do. Actions with no
/// keyboard binding (only reachable via the menu) are left out.
fn shortcuts_view<'a>(key_binds: &HashMap<KeyBind, Action>) -> Element<'a, Message> {
    let label = |action: Action| match action {
        Action::OpenFile => "Open Image",
        Action::PreviousImage => "Previous Image",
        Action::NextImage => "Next Image",
        Action::ToggleFullscreen => "Toggle Fullscreen",
        Action::Quit => "Quit",
        Action::ToggleAbout | Action::ToggleShortcuts => "",
    };

    [
        Action::OpenFile,
        Action::PreviousImage,
        Action::NextImage,
        Action::ToggleFullscreen,
        Action::Quit,
    ]
        .into_iter()
        .filter_map(|action| {
            let (bind, _) = key_binds.iter().find(|(_, bound)| **bound == action)?;
            Some(
                widget::settings::item::builder(label(action))
                    .control(widget::text::body(bind.to_string())),
            )
        })
        .fold(widget::settings::section(), |section, item| section.add(item))
        .into()
}

fn loading_indicator<'a>() -> Element<'a, Message> {
    widget::column::with_capacity(2)
        .push(spinner::spinner())
        .push(widget::text::title3("Loading..."))
        .spacing(12)
        .align_x(Alignment::Center)
        .into()
}

/// A small "x / y" pill, floating over the bottom-center of the viewport —
/// shows which photo you're on and how many are open, so it's obvious
/// there's something to navigate between even before anyone finds the
/// arrow keys.
fn position_indicator<'a>(current: usize, total: usize) -> Element<'a, Message> {
    let pill = widget::container(widget::text::body(format!("{} / {}", current + 1, total)))
        .padding([6, 16])
        .class(cosmic::theme::Container::Tooltip);

    widget::container(pill)
        .center_x(Length::Fill)
        .align_bottom(Length::Fill)
        .padding([0, 0, 24, 0])
        .into()
}

fn centered<'a>(content: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
    widget::container(content)
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x(Length::Fill)
        .center_y(Length::Fill)
        .into()
}

/// Decoding a big JPEG is synchronous CPU work, so keep it off the async
/// workers. (Always build with `--release`: debug decode is ~65x slower.)
async fn load_image(path: PathBuf) -> Result<Arc<image::RgbaImage>, String> {
    tokio::task::spawn_blocking(move || {
        image::open(&path)
            .map(|decoded| Arc::new(decoded.to_rgba8()))
            .map_err(|error| format!("{}: {error}", path.display()))
    })
    .await
    .map_err(|error| format!("loading task failed: {error}"))?
}
