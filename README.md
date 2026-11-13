# Photosphere Viewer

A 360-degree photo viewer for COSMIC. Open an equirectangular panorama and
look around by dragging — the image is rendered as a textured sphere on the
GPU, with no CPU-side reprojection involved.

## Keyboard shortcuts

| Key            | Action             |
|----------------|--------------------|
| Ctrl+O         | Open image         |
| Ctrl+Q         | Quit               |
| ←              | Previous image     |
| →              | Next image         |
| F11            | Toggle fullscreen  |

(Also available from the app menu, including a "Keyboard Shortcuts" page
that lists these live from the actual bindings.)

## Running it

```
git clone https://github.com/vancha/photosphere_viewer.git
cd photosphere_viewer
cargo run --release
```

Optionally pass a starting image path:

```
cargo run --release -- /path/to/photo.jpg
```
