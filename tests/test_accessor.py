"""Tests for the ManifoldAccessor exposed on PolyData."""

import math

import manifold3d
import numpy as np
import pytest
import pyvista as pv

import pyvista_manifold as pvm
import pyvista_manifold._accessor as accessor_module


def test_accessor_registered():
    assert hasattr(pv.PolyData, 'manifold')
    sphere = pv.Sphere()
    assert isinstance(sphere.manifold, pvm.ManifoldAccessor)
    assert sphere.manifold is sphere.manifold  # cached per instance


def test_accessor_to_manifold():
    m = pv.Sphere().manifold.to_manifold()
    assert isinstance(m, manifold3d.Manifold)


def test_accessor_from_manifold_classmethod_style():
    cube = pv.Cube().triangulate()
    m = cube.manifold.to_manifold()
    poly = pv.PolyData.manifold.from_manifold(m)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_points == m.num_vert()
    assert poly.n_cells == m.num_tri()


def test_accessor_from_manifold_instance_access():
    cube = pv.Cube().triangulate()
    m = cube.manifold.to_manifold()
    poly = cube.manifold.from_manifold(m)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_points == m.num_vert()


def test_accessor_from_manifold_property_names():
    cube = pv.Cube().triangulate()
    cube.point_data['scalar'] = np.arange(cube.n_points, dtype=float)
    m = cube.manifold.to_manifold(point_data_keys=['scalar'])
    poly = pv.PolyData.manifold.from_manifold(m, property_names=['scalar'])
    assert 'scalar' in poly.point_data
    assert 'property_0' not in poly.point_data


def test_accessor_from_manifold_empty():
    poly = pv.PolyData.manifold.from_manifold(manifold3d.Manifold())
    assert isinstance(poly, pv.PolyData)
    assert poly.n_points == 0


def test_accessor_to_manifold_uses_default_cache():
    sphere = pv.Sphere()
    first = sphere.manifold.to_manifold()
    second = sphere.manifold.to_manifold()
    assert first is second


def test_accessor_to_manifold_invalidates_cache_after_mesh_mutation():
    cube = pv.Cube()
    first = cube.manifold.to_manifold()
    cube.points[:, 0] *= 2.0
    second = cube.manifold.to_manifold()
    assert second is not first
    np.testing.assert_allclose(second.bounding_box()[0], -1.0, atol=1e-5)


def test_accessor_to_manifold_does_not_cache_point_data_queries():
    sphere = pv.Sphere()
    sphere.point_data['scalar'] = np.linspace(0.0, 1.0, sphere.n_points)
    first = sphere.manifold.to_manifold(point_data_keys=['scalar'])
    second = sphere.manifold.to_manifold(point_data_keys=['scalar'])
    assert first is not second


def test_difference_reuses_other_accessor_cache(monkeypatch):
    cube = pv.Cube()
    sphere = pv.Sphere(radius=0.5)
    _ = sphere.manifold.to_manifold()

    calls = 0
    original_to_manifold = accessor_module.to_manifold

    def _counting_to_manifold(
        mesh: pv.PolyData,
        *,
        point_data_keys: tuple[str, ...] | list[str] | None = None,
        clean: bool = True,
    ) -> manifold3d.Manifold:
        nonlocal calls
        calls += 1
        return original_to_manifold(mesh, point_data_keys=point_data_keys, clean=clean)

    monkeypatch.setattr(accessor_module, 'to_manifold', _counting_to_manifold)

    _ = cube.manifold.difference(sphere)

    assert calls == 1


def test_difference():
    cube = pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0)  # centered, 8 unit volume
    sphere = pv.Sphere(radius=0.9)  # fully inside the cube
    diff = cube.manifold.difference(sphere)
    assert diff.is_all_triangles
    np.testing.assert_allclose(
        diff.volume,
        8.0 - (4 / 3) * math.pi * 0.9**3,
        rtol=0.02,
    )


def test_union_overlapping_cubes():
    a = pv.Cube(center=(0, 0, 0))
    b = pv.Cube(center=(0.5, 0.5, 0.5))
    u = a.manifold.union(b)
    np.testing.assert_allclose(u.volume, 1.0 + 1.0 - 0.125, rtol=1e-4)


def test_intersection():
    a = pv.Cube(center=(0, 0, 0))
    b = pv.Cube(center=(0.5, 0.5, 0.5))
    inter = a.manifold.intersection(b)
    np.testing.assert_allclose(inter.volume, 0.125, rtol=1e-4)


def test_difference_accepts_manifold_directly():
    a = pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0)
    b_m = pvm.to_manifold(pv.Sphere(radius=0.7))
    diff = a.manifold.difference(b_m)
    assert diff.n_points > 0


def test_batch_boolean_subtract():
    a = pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0)
    b = pv.Sphere(radius=0.5, center=(-0.5, 0, 0))
    c = pv.Sphere(radius=0.5, center=(0.5, 0, 0))
    out = a.manifold.batch_boolean([b, c], op=pvm.OpType.Subtract)
    assert out.n_points > 0
    assert out.volume < a.volume


def test_hull_of_concave_shape():
    a = pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0)
    b = pv.Cube(center=(-0.5, -0.5, -0.5))
    bowl = a.manifold.difference(b)
    # Hull of the carved cube: 8.0 minus the corner tetrahedron of side 1.
    np.testing.assert_allclose(bowl.manifold.hull().volume, 8.0 - 1 / 6, rtol=1e-4)


def test_hull_with_others():
    a = pv.Cube()
    b = pv.Cube(center=(2.0, 0.0, 0.0))
    h = a.manifold.hull_with(b)
    np.testing.assert_allclose(h.volume, 3.0, rtol=1e-4)


def test_translate():
    cube = pv.Cube(center=(0, 0, 0))  # corners (-0.5, -0.5, -0.5)..(0.5, 0.5, 0.5)
    moved = cube.manifold.translate((10.0, 0.0, 0.0))
    np.testing.assert_allclose(moved.center, [10.0, 0.0, 0.0], atol=1e-5)


def test_rotate_90_about_z():
    cube = pv.Cube(x_length=1.0, y_length=2.0, z_length=3.0)
    rotated = cube.manifold.rotate((0.0, 0.0, 90.0))
    xmin, xmax, ymin, ymax, zmin, zmax = rotated.bounds
    np.testing.assert_allclose(xmax - xmin, 2.0, atol=1e-4)
    np.testing.assert_allclose(ymax - ymin, 1.0, atol=1e-4)
    np.testing.assert_allclose(zmax - zmin, 3.0, atol=1e-4)


def test_scale_uniform():
    cube = pv.Cube()
    s = cube.manifold.scale(2.0)
    np.testing.assert_allclose(s.volume, 8.0, rtol=1e-4)


def test_scale_anisotropic():
    cube = pv.Cube()
    s = cube.manifold.scale((2.0, 3.0, 4.0))
    np.testing.assert_allclose(s.volume, 24.0, rtol=1e-4)


def test_scale_rejects_wrong_length():
    with pytest.raises(ValueError, match='3 elements'):
        pv.Cube().manifold.scale((2.0, 3.0))


def test_mirror():
    cube = pv.Cube(center=(0.5, 0.5, 0.5))  # corners (0,0,0)..(1,1,1)
    m = cube.manifold.mirror((1.0, 0.0, 0.0))
    xmin, xmax, *_ = m.bounds
    np.testing.assert_allclose(xmin, -1.0, atol=1e-5)
    np.testing.assert_allclose(xmax, 0.0, atol=1e-5)


def test_transform_identity():
    eye = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    out = pv.Cube().manifold.transform(eye)
    np.testing.assert_allclose(out.volume, 1.0, rtol=1e-4)


def test_transform_wrong_shape():
    with pytest.raises(ValueError, match=r'\(3, 4\)'):
        pv.Cube().manifold.transform(np.eye(4))


def test_warp_translate_along_x():
    def shift(p):
        return (p[0] + 5.0, p[1], p[2])

    out = pv.Cube(center=(0, 0, 0)).manifold.warp(shift)
    np.testing.assert_allclose(out.center, [5.0, 0.0, 0.0], atol=1e-4)


def test_warp_batch():
    def shift(pts):
        out = np.asarray(pts).copy()
        out[:, 0] += 5.0
        return out

    out = pv.Cube(center=(0, 0, 0)).manifold.warp(shift, batch=True)
    np.testing.assert_allclose(out.center, [5.0, 0.0, 0.0], atol=1e-4)


def test_refine():
    cube = pv.Cube()
    out = cube.manifold.refine(2)
    assert out.n_points > cube.n_points


def test_refine_to_length_increases_resolution():
    sphere = pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8)
    finer = sphere.manifold.refine_to_length(0.2)
    assert finer.n_points >= sphere.n_points


def test_smooth_out_makes_sphere():
    cube = pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0)
    smoothed = cube.manifold.refine(8).manifold.smooth_out(min_sharp_angle=180.0)
    assert smoothed.n_points > cube.n_points


def test_split_by_plane():
    cube = pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0)  # centered, vol 8
    pos, neg = cube.manifold.split_by_plane((0.0, 0.0, 1.0))
    np.testing.assert_allclose(pos.volume, 4.0, rtol=1e-4)
    np.testing.assert_allclose(neg.volume, 4.0, rtol=1e-4)


def test_trim_by_plane():
    cube = pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0)
    half = cube.manifold.trim_by_plane((0.0, 0.0, 1.0))
    np.testing.assert_allclose(half.volume, 4.0, rtol=1e-4)


def test_split_by_solid():
    a = pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0)
    b = pv.Cube()
    inside, outside = a.manifold.split(b)
    np.testing.assert_allclose(inside.volume + outside.volume, a.volume, rtol=1e-4)


def test_decompose():
    a = pv.Cube()
    b = pv.Cube(center=(5.0, 0.0, 0.0))
    composed = a.manifold.compose_with(b)
    parts = composed.manifold.decompose()
    assert len(parts) == 2
    assert all(p.n_points > 0 for p in parts)


def test_minkowski_sum():
    a = pv.Cube()
    b = pv.Sphere(radius=0.2)
    out = a.manifold.minkowski_sum(b)
    assert out.volume > a.volume


def test_minkowski_difference():
    a = pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0)
    b = pv.Sphere(radius=0.2)
    out = a.manifold.minkowski_difference(b)
    assert out.volume < a.volume


def test_slice_z():
    sphere = pv.Sphere(radius=1.0)
    contour = sphere.manifold.slice_z(0.0)
    assert contour.n_points > 0
    assert contour.n_lines > 0
    np.testing.assert_allclose(contour.points[:, 2], 0.0, atol=1e-4)


def test_slice_z_above_top():
    out = pv.Sphere(radius=1.0).manifold.slice_z(2.0)
    assert out.n_points == 0


def test_project():
    sphere = pv.Sphere(radius=1.0, center=(0, 0, 5.0))
    proj = sphere.manifold.project()
    assert proj.n_points > 0
    np.testing.assert_allclose(proj.points[:, 2], 0.0, atol=1e-5)


def test_simplify_reduces_count():
    sphere = pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=64)
    simpler = sphere.manifold.simplify(0.05)
    assert simpler.n_points <= sphere.n_points


def test_property_volume():
    cube = pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0)
    np.testing.assert_allclose(cube.manifold.volume, 8.0, rtol=1e-4)


def test_property_surface_area():
    cube = pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0)
    np.testing.assert_allclose(cube.manifold.surface_area, 24.0, rtol=1e-4)


def test_property_genus_sphere():
    assert pv.Sphere().manifold.genus == 0


def test_property_genus_torus():
    big = pv.Cylinder(radius=1.0, height=0.4, resolution=64)
    hole = pv.Cylinder(radius=0.5, height=2.0, resolution=64)
    torus = big.manifold.difference(hole)
    assert torus.manifold.genus == 1


def test_property_bounds():
    cube = pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0)
    np.testing.assert_allclose(cube.manifold.bounds, (-1, 1, -1, 1, -1, 1), atol=1e-5)
    np.testing.assert_allclose(cube.manifold.bounds, cube.bounds, atol=1e-5)


def test_property_is_valid():
    assert pv.Cube().manifold.is_valid


def test_property_num_counts():
    cube = pv.Cube()
    assert cube.manifold.num_vert > 0
    assert cube.manifold.num_tri > 0
    assert cube.manifold.num_edge > 0


def test_property_status_enum():
    assert pv.Cube().manifold.status == manifold3d.Error.NoError


def test_min_gap():
    a = pv.Cube()  # centered at origin, extent [-0.5, 0.5]
    b = pv.Cube(center=(3.0, 0.0, 0.0))  # extent [2.5, 3.5] x [-.5,.5] x [-.5,.5]
    gap = a.manifold.min_gap(b, search_length=5.0)
    np.testing.assert_allclose(gap, 2.0, atol=0.01)


def test_chaining_with_core_filters():
    cube = pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0)
    sphere = pv.Sphere(radius=1.2)
    chained = cube.manifold.difference(sphere).clean().triangulate()
    assert isinstance(chained, pv.PolyData)
    assert chained.is_all_triangles


def test_compose_with_method():
    a = pv.Cube()
    b = pv.Cube(center=(5.0, 0.0, 0.0))
    composed = a.manifold.compose_with(b)
    np.testing.assert_allclose(composed.volume, 2.0, rtol=1e-4)


def test_calculate_normals_unit_sphere():
    sphere = pv.Sphere(theta_resolution=24, phi_resolution=24)
    out = sphere.manifold.calculate_normals()
    assert list(out.point_data.keys()) == ['Normals']
    normals = np.asarray(out.point_data['Normals'])
    assert normals.shape == (out.n_points, 3)
    # On a unit sphere centered at origin, normals point outward (= position).
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=0.05)


def test_calculate_normals_with_offset_index():
    sphere = pv.Sphere(theta_resolution=16, phi_resolution=16)
    a = sphere.manifold.calculate_normals()
    b = sphere.manifold.calculate_normals(normal_idx=2)
    # Padding offsets shouldn't change the visible result.
    assert list(b.point_data.keys()) == ['Normals']
    np.testing.assert_allclose(a.point_data['Normals'], b.point_data['Normals'])


def test_calculate_curvature_sphere():
    sphere = pv.Sphere(theta_resolution=32, phi_resolution=32, radius=1.0)
    out = sphere.manifold.calculate_curvature()
    assert set(out.point_data.keys()) == {'GaussianCurvature', 'MeanCurvature'}
    # Unit sphere: analytical Gaussian = 1/R^2 = 1; manifold reports
    # Mean = 2H = 2/R (twice the principal-mean). Both should be roughly
    # constant across the sphere.
    np.testing.assert_allclose(np.mean(out.point_data['GaussianCurvature']), 1.0, atol=0.05)
    np.testing.assert_allclose(np.mean(out.point_data['MeanCurvature']), 2.0, atol=0.05)


def test_set_tolerance_roundtrip():
    cube = pv.Cube()
    out = cube.manifold.set_tolerance(1e-3)
    assert out.manifold.tolerance > 0


def test_original_id_unset():
    assert pv.Cube().manifold.original_id == -1


def test_set_properties_appends_channel():
    # Callback writes 4 extra channels per vertex; xyz positions are kept
    # separately by manifold and not touched here.
    def f(_pos, _old):
        return (1.0, 2.0, 3.0, 7.0)

    out = pv.Cube().manifold.set_properties(4, f)
    assert {'property_0', 'property_1', 'property_2', 'property_3'} <= set(out.point_data.keys())
    np.testing.assert_allclose(out.point_data['property_0'], 1.0)
    np.testing.assert_allclose(out.point_data['property_3'], 7.0)


def test_project_silhouette():
    sphere = pv.Sphere(radius=1.0, center=(0, 0, 5))
    proj = sphere.manifold.project()
    assert proj.n_points > 0
    np.testing.assert_allclose(proj.points[:, 2], 0.0, atol=1e-5)
