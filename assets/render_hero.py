"""Render the hero images used in the README. Re-run when the API changes."""

import math
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image
import pyvista as pv

from pyvista_manifold import level_set

pv.OFF_SCREEN = True
pv.global_theme.background = 'white'
pv.global_theme.window_size = (1100, 800)

ASSETS = Path(__file__).parent

# Poly Haven HDR (CC0) used for image-based lighting on every PBR render.
_HDRI_SLUG = 'kloofendal_48d_partly_cloudy'
_HDRI_RESOLUTION = '1k'  # IBL convolutions smooth the input -- 1k is plenty.
_HDRI_CACHE = Path.home() / '.cache' / 'pyvista-manifold' / 'hdri'

_ENV_TEXTURE: pv.Texture | None = None


def _hdri_path() -> Path:
    _HDRI_CACHE.mkdir(parents=True, exist_ok=True)
    target = _HDRI_CACHE / f'{_HDRI_SLUG}_{_HDRI_RESOLUTION}.hdr'
    if not target.exists():
        url = f'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/{_HDRI_RESOLUTION}/{target.name}'
        tmp = target.with_suffix('.hdr.part')
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    return target


def _env_texture() -> pv.Texture:
    global _ENV_TEXTURE
    if _ENV_TEXTURE is None:
        tex = pv.read_texture(str(_hdri_path()))
        tex.mipmap = True
        tex.interpolate = True
        _ENV_TEXTURE = tex
    return _ENV_TEXTURE


def _z_up_environment_rotation() -> np.ndarray:
    """Return the 3x3 rotation that maps Y-up IBL data to PyVista's Z-up world."""
    return pv.Transform().rotate_x(90).rotation_matrix


def _setup_pbr(pl: pv.Plotter) -> None:
    """Configure the active renderer for PBR image-based lighting (IBL only).

    Removes the default light kit so the only lighting comes from the HDR.
    Otherwise VTK's headlight and key/fill lights add bright spot reflections
    on top of the IBL response.
    """
    pl.remove_all_lights(only_active=True)
    pl.set_environment_texture(
        _env_texture(),
        is_srgb=False,
        rotation=_z_up_environment_rotation(),
        show_background=False,
    )


def _save_webp_animation(
    target: str | Path,
    frames: list[np.ndarray],
    *,
    duration: int | list[int],
) -> None:
    """Write RGBA frames to a lossless animated WebP."""
    pil_frames = [Image.fromarray(frame, mode='RGBA') for frame in frames]
    pil_frames[0].save(
        target,
        format='WEBP',
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
        lossless=True,
        quality=100,
        method=6,
    )


def _write_webp_animation(
    path: Path,
    frames: list[np.ndarray],
    *,
    duration: int | list[int],
) -> None:
    _save_webp_animation(path, frames, duration=duration)


def gyroid(x: float, y: float, z: float, scale: float = 4.0) -> float:
    """TPMS gyroid scalar field; the iso-surface at level 0 is the gyroid."""
    s = scale
    return -(
        math.sin(s * x) * math.cos(s * y)
        + math.sin(s * y) * math.cos(s * z)
        + math.sin(s * z) * math.cos(s * x)
    )


def render_bracket() -> None:
    """Render the brushed-aluminum bracket hero image."""
    plate = pv.Cube(x_length=8.0, y_length=5.0, z_length=0.6, center=(0, 0, 0.3))
    boss = pv.Cylinder(
        radius=1.6, height=1.6, direction=(0, 0, 1), center=(0, 0, 1.0), resolution=96
    )
    bracket = plate.manifold.union(boss)
    for cx, cy in [(-3.2, -1.7), (3.2, -1.7), (-3.2, 1.7), (3.2, 1.7)]:
        hole = pv.Cylinder(
            radius=0.4, height=2.0, direction=(0, 0, 1), center=(cx, cy, 0.5), resolution=48
        )
        bracket = bracket.manifold.difference(hole)
    bore = pv.Cylinder(
        radius=0.7, height=3.0, direction=(0, 0, 1), center=(0, 0, 1.0), resolution=96
    )
    bracket = bracket.manifold.difference(bore)

    pl = pv.Plotter()
    _setup_pbr(pl)
    pl.add_mesh(
        bracket,
        color='#c2c5c9',
        pbr=True,
        metallic=0.95,
        roughness=0.4,
        smooth_shading=True,
        split_sharp_edges=True,
    )
    pl.camera_position = [(11, -13, 8), (0, 0, 0.6), (0, 0, 1)]
    pl.screenshot(ASSETS / 'bracket.png')
    pl.close()


def _upright_cow() -> pv.PolyData:
    """Return the example cow centered, scaled, and rotated to stand upright (z-up)."""
    cow = pv.examples.download_cow()
    cow = cow.rotate_x(90)  # native data is y-up; PyVista convention is z-up
    return cow.translate(-np.array(cow.center)).scale(0.6)


def render_lattice_cow() -> None:
    """Render the cow-with-gyroid-infill hero image."""
    cow = _upright_cow()
    lattice = level_set(
        lambda x, y, z: gyroid(x, y, z, scale=6.0),
        bounds=(-5, -3, -3, 5, 3, 3),
        edge_length=0.04,
    )
    infilled = cow.manifold.intersection(lattice)

    pl = pv.Plotter()
    pl.enable_anti_aliasing('msaa')
    pl.add_mesh(
        infilled,
        color='salmon',
        smooth_shading=True,
        split_sharp_edges=True,
        specular=0.4,
        specular_power=18,
    )
    pl.add_mesh(cow, color='lightsteelblue', opacity=0.12)
    pl.camera_position = [(7.5, -8, 2.5), (0, 0, 0), (0, 0, 1)]
    pl.camera.zoom(1.2)
    pl.screenshot(ASSETS / 'lattice_cow.png')
    pl.close()


def render_topographic() -> None:
    """Render the stacked-horse-slices topographic hero image."""
    horse = pv.examples.download_horse()
    horse = horse.translate(-np.array(horse.center)).scale(50.0)
    z_min, z_max = horse.bounds.z_min, horse.bounds.z_max
    heights = np.linspace(z_min + 0.05, z_max - 0.05, 36)

    pl = pv.Plotter()
    for z in heights:
        c = horse.manifold.slice_z(z)
        if c.n_points == 0:
            continue
        c['height'] = np.full(c.n_points, z)
        pl.add_mesh(c, scalars='height', cmap='viridis', line_width=2.5, show_scalar_bar=False)
    pl.enable_anti_aliasing('msaa')
    pl.camera_position = [(7, -8, 4), (0, 0, 0.5), (0, 0, 1)]
    pl.screenshot(ASSETS / 'topographic.png')
    pl.close()


def render_fracture() -> None:
    """Render the Voronoi-style cube-fracture hero image."""
    rng = np.random.default_rng(42)
    pieces = [pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0)]
    for _ in range(8):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        offset = rng.uniform(-0.5, 0.5)
        new_pieces = []
        for p in pieces:
            a, b = p.manifold.split_by_plane(tuple(n), offset)
            if a.n_points:
                new_pieces.append(a)
            if b.n_points:
                new_pieces.append(b)
        pieces = new_pieces

    pl = pv.Plotter()
    for p in pieces:
        centroid = np.array(p.center)
        shifted = p.translate(0.15 * centroid / max(np.linalg.norm(centroid), 1e-6))
        color = rng.uniform(0.4, 0.95, size=3)
        pl.add_mesh(
            shifted, color=color, smooth_shading=True, split_sharp_edges=True, specular=0.3
        )
    pl.enable_anti_aliasing('msaa')
    pl.camera_position = [(5, -5, 3.5), (0, 0, 0), (0, 0, 1)]
    pl.screenshot(ASSETS / 'fracture.png')
    pl.close()


def _gyroid_blob() -> pv.PolyData:
    tpms = level_set(
        lambda x, y, z: gyroid(x, y, z, scale=2.0),
        bounds=(-3, -3, -3, 3, 3, 3),
        edge_length=0.04,
    )
    return tpms.manifold.intersection(
        pv.Sphere(radius=2.6, theta_resolution=64, phi_resolution=64),
    )


def render_gyroid() -> None:
    """Render the gold gyroid sphere hero image."""
    pl = pv.Plotter()
    _setup_pbr(pl)
    pl.add_mesh(
        _gyroid_blob(),
        color='#d8a23c',
        pbr=True,
        metallic=0.7,
        roughness=0.25,
        smooth_shading=True,
        split_sharp_edges=True,
    )
    pl.camera_position = 'iso'
    pl.screenshot(ASSETS / 'gyroid.png')
    pl.close()


def render_gyroid_animation() -> None:
    """Hero animation: gold gyroid blob on a closed orbit for seamless looping."""
    blob = _gyroid_blob()

    pl = pv.Plotter()
    pl.window_size = [560, 560]
    _setup_pbr(pl)
    pl.add_mesh(
        blob,
        color='#d8a23c',
        pbr=True,
        metallic=0.7,
        roughness=0.25,
        smooth_shading=True,
        split_sharp_edges=True,
    )
    pl.camera_position = 'iso'
    pl.camera.zoom(1.25)

    n_steps = 60
    fps = 22
    frame_duration = round(1000 / fps)
    frames: list[np.ndarray] = []
    pl.render()
    focal_point = np.array(pl.camera.focal_point)
    base_position = np.array(pl.camera.position) - focal_point
    base_up = np.array(pl.camera.up)
    for angle in np.linspace(0.0, 360.0, n_steps + 1):
        rotation = pv.Transform().rotate_z(angle).rotation_matrix
        pl.camera.position = tuple(focal_point + rotation @ base_position)
        pl.camera.up = tuple(rotation @ base_up)
        pl.render()
        frame = pl.screenshot(None, return_img=True, transparent_background=True)
        frames.append(frame)

    pl.close()
    durations = [frame_duration] * n_steps + [1]
    _write_webp_animation(ASSETS / 'gyroid.webp', frames, duration=durations)


def render_banner() -> None:
    """Wide hero banner: three CSG showcases as linked subplots.

    The gold gyroid sphere lives in the rotating animation at the top of the
    README, so this banner skips it.
    """
    # 1. Bracket built from primitives
    plate = pv.Cube(x_length=4.5, y_length=2.8, z_length=0.35, center=(0, 0, 0.18))
    boss = pv.Cylinder(
        radius=0.85, height=0.85, direction=(0, 0, 1), center=(0, 0, 0.55), resolution=96
    )
    bracket = plate.manifold.union(boss)
    for cx, cy in [(-1.7, -0.95), (1.7, -0.95), (-1.7, 0.95), (1.7, 0.95)]:
        h = pv.Cylinder(
            radius=0.22, height=1.5, direction=(0, 0, 1), center=(cx, cy, 0.3), resolution=48
        )
        bracket = bracket.manifold.difference(h)
    bracket = bracket.manifold.difference(
        pv.Cylinder(
            radius=0.38, height=2.0, direction=(0, 0, 1), center=(0, 0, 0.55), resolution=96
        ),
    )

    # 2. Lattice cow
    cow = _upright_cow()
    lattice = level_set(
        lambda x, y, z: gyroid(x, y, z, scale=6.0),
        bounds=(-5, -3, -3, 5, 3, 3),
        edge_length=0.04,
    )
    lattice_cow = cow.manifold.intersection(lattice)

    # 3. Voronoi-style fracture of a cube
    rng = np.random.default_rng(7)
    pieces = [pv.Cube(x_length=2.4, y_length=2.4, z_length=2.4)]
    for _ in range(7):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        offset = rng.uniform(-0.55, 0.55)
        new_pieces = []
        for p in pieces:
            a, b = p.manifold.split_by_plane(tuple(n), offset)
            if a.n_points:
                new_pieces.append(a)
            if b.n_points:
                new_pieces.append(b)
        pieces = new_pieces

    pl = pv.Plotter(shape=(1, 3), border=False)
    pl.window_size = [2100, 700]
    pl.set_background('white')

    pl.subplot(0, 0)
    _setup_pbr(pl)
    pl.add_mesh(
        bracket,
        color='#c2c5c9',
        pbr=True,
        metallic=0.95,
        roughness=0.4,
        smooth_shading=True,
        split_sharp_edges=True,
    )
    pl.camera_position = [(7, -8, 5), (0, 0, 0.4), (0, 0, 1)]

    pl.subplot(0, 1)
    pl.add_mesh(
        lattice_cow,
        color='salmon',
        smooth_shading=True,
        split_sharp_edges=True,
        specular=0.4,
        specular_power=18,
    )
    pl.add_mesh(cow, color='lightsteelblue', opacity=0.1)
    pl.camera_position = [(8, -9, 2.5), (0, 0, 0), (0, 0, 1)]
    pl.camera.zoom(1.05)

    pl.subplot(0, 2)
    for p in pieces:
        centroid = np.array(p.center)
        shifted = p.translate(0.22 * centroid / max(np.linalg.norm(centroid), 1e-6))
        color = rng.uniform(0.45, 0.92, size=3)
        pl.add_mesh(
            shifted, color=color, smooth_shading=True, split_sharp_edges=True, specular=0.35
        )
    pl.enable_anti_aliasing('msaa', multi_samples=8)
    pl.camera_position = [(5, -5, 4), (0, 0, 0), (0, 0, 1)]

    pl.screenshot(ASSETS / 'banner.png')
    pl.close()


if __name__ == '__main__':
    render_gyroid_animation()
    render_banner()
    render_bracket()
    render_gyroid()
    render_lattice_cow()
    render_topographic()
    render_fracture()
    print('rendered hero assets to', ASSETS)
