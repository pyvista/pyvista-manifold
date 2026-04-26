"""Image-regression tests for visual outputs of the accessor."""

import math

import pyvista as pv

import pyvista_manifold as pvm


def _plot(mesh: pv.PolyData) -> pv.Plotter:
    pl = pv.Plotter()
    pl.add_mesh(mesh, color='lightsteelblue')
    pl.camera_position = 'iso'
    return pl


def test_difference_render(verify_image_cache):
    cube = pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0)
    sphere = pv.Sphere(radius=1.2, theta_resolution=64, phi_resolution=64)
    pl = _plot(cube.manifold.difference(sphere))
    pl.show()


def test_union_render(verify_image_cache):
    a = pv.Cylinder(direction=(1, 0, 0), radius=0.4, height=2.0, resolution=64)
    b = pv.Cylinder(direction=(0, 1, 0), radius=0.4, height=2.0, resolution=64)
    c = pv.Cylinder(direction=(0, 0, 1), radius=0.4, height=2.0, resolution=64)
    pl = _plot(a.manifold.union(b).manifold.union(c))
    pl.show()


def test_intersection_render(verify_image_cache):
    a = pv.Cube(center=(0, 0, 0))
    b = pv.Sphere(radius=0.7, center=(0.2, 0.2, 0.2))
    pl = _plot(a.manifold.intersection(b))
    pl.show()


def test_hull_render(verify_image_cache):
    a = pv.Cube(center=(0, 0, 0))
    b = pv.Sphere(radius=0.4, center=(2.0, 0.0, 0.0))
    pl = _plot(a.manifold.hull_with(b))
    pl.show()


def test_torus_render(verify_image_cache):
    big = pv.Cylinder(radius=1.0, height=0.4, resolution=64)
    hole = pv.Cylinder(radius=0.5, height=2.0, resolution=64)
    pl = _plot(big.manifold.difference(hole))
    pl.show()


def test_level_set_gyroid(verify_image_cache):
    def gyroid(x: float, y: float, z: float) -> float:
        s = 2.0
        return -(
            math.sin(s * x) * math.cos(s * y)
            + math.sin(s * y) * math.cos(s * z)
            + math.sin(s * z) * math.cos(s * x)
        )

    iso = pvm.level_set(gyroid, bounds=(-2, -2, -2, 2, 2, 2), edge_length=0.15)
    pl = _plot(iso)
    pl.show()
