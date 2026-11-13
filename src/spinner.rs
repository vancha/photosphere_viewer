//! A small "loading" indicator: a ring with a rotating arc. Self-animates by
//! requesting a redraw every frame while it's on screen (same pattern the
//! photosphere widget uses for its drag-release glide) and costs nothing
//! once it's gone.
use cosmic::iced::time::Instant;
use cosmic::iced::widget::canvas::{self, Canvas, Frame, Geometry, Path, Stroke};
use cosmic::iced::{Color, Event, Length, Radians, Rectangle, mouse, window};
use std::f32::consts::TAU;

const SIZE: f32 = 32.0;
const STROKE_WIDTH: f32 = 4.0;
const TURNS_PER_SECOND: f32 = 0.9;
/// How much of the ring the rotating arc covers.
const ARC_FRACTION: f32 = 0.75;

/// A small spinning loading indicator.
pub fn spinner<'a, Message: 'a>() -> cosmic::Element<'a, Message> {
    Canvas::new(Spinner)
        .width(Length::Fixed(SIZE))
        .height(Length::Fixed(SIZE))
        .into()
}

struct Spinner;

#[derive(Default)]
struct State {
    angle: f32,
    last_tick: Option<Instant>,
}

impl<Message> canvas::Program<Message, cosmic::Theme, cosmic::Renderer> for Spinner {
    type State = State;

    fn update(
        &self,
        state: &mut State,
        event: &Event,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Option<canvas::Action<Message>> {
        let Event::Window(window::Event::RedrawRequested(now)) = event else {
            return None;
        };

        if let Some(last) = state.last_tick {
            let dt = now.duration_since(last).as_secs_f32();
            state.angle = (state.angle + dt * TURNS_PER_SECOND * TAU) % TAU;
        }
        state.last_tick = Some(*now);

        Some(canvas::Action::request_redraw())
    }

    fn draw(
        &self,
        state: &State,
        renderer: &cosmic::Renderer,
        theme: &cosmic::Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry<cosmic::Renderer>> {
        let mut frame = Frame::new(renderer, bounds.size());
        let center = frame.center();
        let radius = bounds.size().width.min(bounds.size().height) / 2.0 - STROKE_WIDTH;

        let palette = theme.cosmic();
        let track = palette.bg_divider();
        let accent = palette.accent.base;

        frame.stroke(
            &Path::circle(center, radius),
            Stroke::default()
                .with_color(Color::from_rgba(track.red, track.green, track.blue, track.alpha))
                .with_width(STROKE_WIDTH),
        );

        let arc = Path::new(|builder| {
            builder.arc(canvas::path::Arc {
                center,
                radius,
                start_angle: Radians(state.angle),
                end_angle: Radians(state.angle + TAU * ARC_FRACTION),
            });
        });

        frame.stroke(
            &arc,
            Stroke::default()
                .with_color(Color::from_rgba(accent.red, accent.green, accent.blue, accent.alpha))
                .with_width(STROKE_WIDTH),
        );

        vec![frame.into_geometry()]
    }
}
