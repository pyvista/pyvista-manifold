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


def test_to_manifold_with_point_data():
    sphere = pv.Sphere()
    sphere.point_data['scalar'] = np.linspace(0.0, 1.0, sphere.n_points)
    m = pvm.to_manifold(sphere, point_data_keys=['scalar'])
    mesh = m.to_mesh()
    assert np.asarray(mesh.vert_properties).shape[1] == 4
    back = pvm.from_manifold(m, property_names=['scalar'])
    assert 'scalar' in back.point_data
