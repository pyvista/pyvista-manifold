"""Tests for PolyData / Manifold conversion."""

import manifold3d
import numpy as np
import pytest
import pyvista as pv

import pyvista_manifold as pvm


def test_to_manifold_sphere_round_trip():
    sphere = pv.Sphere()
    m = pvm.to_manifold(sphere)
    assert isinstance(m, manifold3d.Manifold)
    assert not m.is_empty()
    assert m.status() == manifold3d.Error.NoError
    back = pvm.from_manifold(m)
    assert isinstance(back, pv.PolyData)
    assert back.n_points > 0
    assert back.is_all_triangles
    np.testing.assert_allclose(back.volume, sphere.volume, rtol=0.05)


def test_to_manifold_triangulates_quads():
    cube = pv.Cube()
    assert not cube.is_all_triangles
    m = pvm.to_manifold(cube)
    assert m.status() == manifold3d.Error.NoError
    np.testing.assert_allclose(m.volume(), 1.0, rtol=1e-4)


def test_to_manifold_rejects_non_polydata():
    grid = pv.ImageData(dimensions=(5, 5, 5))
    with pytest.raises(TypeError, match=r'Expected pyvista\.PolyData'):
        pvm.to_manifold(grid)  # type: ignore[arg-type]  # runtime validation under test


def test_to_manifold_empty():
    empty = pv.PolyData()
    m = pvm.to_manifold(empty)
    assert m.is_empty()


def test_from_manifold_empty():
    poly = pvm.from_manifold(manifold3d.Manifold())
    assert isinstance(poly, pv.PolyData)
    assert poly.n_points == 0


OFFSET = 1e5


@pytest.mark.parametrize(
    ('op', 'shift'),
    [
        (lambda m: m.manifold.difference(pv.Cube(center=(OFFSET + 0.5, OFFSET, OFFSET))), 0.0),
        (lambda m: m.manifold.translate((0.1, 0.2, 0.3)), (0.1, 0.2, 0.3)),
        (lambda m: m.manifold.transform(np.c_[np.eye(3), [0.1, 0.2, 0.3]]), (0.1, 0.2, 0.3)),
        (lambda m: pvm.hull_points(m.points), 0.0),
        (lambda m: m.manifold.calculate_normals(), 0.0),
        (lambda m: m.manifold.calculate_curvature(), 0.0),
    ],
    ids=[
        'difference',
        'translate',
        'transform',
        'hull_points',
        'calculate_normals',
        'calculate_curvature',
    ],
)
def test_float64_coordinates_survive_far_from_origin(op, shift):
    sphere = pv.Sphere(radius=0.5)
    sphere.points = sphere.points.astype(np.float64) + OFFSET
    out = op(sphere)
    # Vertices away from the cut are input vertices, shifted.
    kept = out.points[out.points[:, 0] < OFFSET - 1e-3] - np.asarray(shift)
    nearest = np.linalg.norm(kept[:, None] - sphere.points[None], axis=-1).min(axis=1)
    assert kept.shape[0] > 100
    np.testing.assert_allclose(nearest, 0.0, atol=1e-9)
    assert out.points.dtype == np.float64


@pytest.mark.parametrize(
    ('make', 'axes', 'poly_axes'),
    [
        (lambda poly: pvm.extrude(poly, height=1.0), [0, 1], [0, 1]),
        (lambda poly: pvm.revolve(poly, segments=16), [2], [1]),  # the polygon's y becomes z
    ],
    ids=['extrude', 'revolve'],
)
def test_float64_polygons_survive_far_from_origin(make, axes, poly_axes):
    # A polygon far up y; extruding keeps its x and y, revolving about y keeps its y.
    poly = np.array([[1.1, 0.1], [1.9, 0.2], [1.8, 0.7], [1.3, 0.9]]) + np.array([0.0, OFFSET])
    out = make(poly)
    expected = poly[:, poly_axes]
    nearest = np.abs(out.points[:, None, axes] - expected[None]).max(axis=-1).min(axis=1)
    # Polygons pass through Clipper2's fixed-point grid; float32 misses by ~4e-3 here.
    np.testing.assert_allclose(nearest, 0.0, atol=1e-6)
    assert out.points.dtype == np.float64


def test_to_manifold_with_point_data():
    sphere = pv.Sphere()
    sphere.point_data['scalar'] = np.linspace(0.0, 1.0, sphere.n_points)
    m = pvm.to_manifold(sphere, point_data_keys=['scalar'])
    mesh = m.to_mesh()
    assert np.asarray(mesh.vert_properties).shape[1] == 4
    back = pvm.from_manifold(m, property_names=['scalar'])
    assert 'scalar' in back.point_data
