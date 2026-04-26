"""Tests for module-level factory functions."""

import math

import numpy as np
import pytest

import pyvista_manifold as pvm


def test_extrude_square():
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    box = pvm.extrude(poly, height=2.0)
    np.testing.assert_allclose(box.volume, 2.0, rtol=1e-4)


def test_revolve_rect():
    # Rectangle at x=[1,2], y=[0,1] revolved 360° gives an annulus solid
    poly = np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]])
    annulus = pvm.revolve(poly, segments=64)
    expected = math.pi * (2.0**2 - 1.0**2) * 1.0
    np.testing.assert_allclose(annulus.volume, expected, rtol=0.02)


def test_level_set_sphere():
    def field(x: float, y: float, z: float) -> float:
        return 1.0 - (x * x + y * y + z * z)

    iso = pvm.level_set(field, bounds=(-1.5, -1.5, -1.5, 1.5, 1.5, 1.5), edge_length=0.1)
    assert iso.n_points > 0
    np.testing.assert_allclose(iso.volume, (4 / 3) * math.pi, rtol=0.05)


def test_level_set_bounds_validation():
    with pytest.raises(ValueError, match='6 elements'):
        pvm.level_set(lambda x, y, z: 1.0, bounds=(0, 0, 0), edge_length=0.1)


def test_hull_points():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.1, 0.1, 0.1],  # strictly interior, should be discarded
        ],
    )
    hull = pvm.hull_points(pts)
    assert hull.n_points == 4
    np.testing.assert_allclose(hull.volume, 1 / 6, rtol=1e-4)


def test_hull_points_wrong_shape():
    with pytest.raises(ValueError, match=r'\(N, 3\)'):
        pvm.hull_points(np.zeros((5, 2)))
