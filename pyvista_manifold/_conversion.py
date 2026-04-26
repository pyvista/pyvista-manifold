"""Conversion between :class:`pyvista.PolyData` and :class:`manifold3d.Manifold`."""

from collections.abc import Sequence

import manifold3d
import numpy as np
import pyvista as pv


def to_manifold(
    mesh: pv.PolyData,
    *,
    point_data_keys: Sequence[str] | None = None,
    clean: bool = True,
) -> manifold3d.Manifold:
    """Convert a :class:`pyvista.PolyData` into a :class:`manifold3d.Manifold`.

    The input is triangulated, and by default also cleaned (duplicate
    vertices merged) so that PyVista primitives like :func:`pyvista.Cube`,
    :func:`pyvista.Cylinder`, and OBJ/STL loads with seam-duplicated
    vertices convert into a proper closed manifold solid. If the result
    still isn't manifold, the returned object will report a non-zero
    ``status()``.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Input surface mesh.
    point_data_keys : sequence of str, optional
        Names of ``point_data`` arrays on ``mesh`` to pack into the
        Manifold's per-vertex property channels (columns beyond xyz). Each
        array must be 1D or 2D with matching ``n_points``.
    clean : bool, default: True
        Run :meth:`pyvista.PolyData.clean` to merge duplicate vertices
        before triangulation. Disable when the input is known-clean and
        you want to preserve every vertex.

    Returns
    -------
    manifold3d.Manifold
        The Manifold representation of the input mesh.

    """
    if not isinstance(mesh, pv.PolyData):
        msg = f'Expected pyvista.PolyData, got {type(mesh).__name__}.'
        raise TypeError(msg)

    if mesh.n_points == 0:
        return manifold3d.Manifold()

    pd = mesh.clean() if clean else mesh
    if not pd.is_all_triangles:
        pd = pd.triangulate()
    vertices = np.ascontiguousarray(pd.points, dtype=np.float32)
    tri_verts = np.ascontiguousarray(pd.regular_faces, dtype=np.uint32)

    if point_data_keys:
        cols: list[np.ndarray] = [vertices]
        for key in point_data_keys:
            arr = np.asarray(pd.point_data[key], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim != 2:
                msg = f'point_data[{key!r}] must be 1D or 2D, got ndim={arr.ndim}.'
                raise ValueError(msg)
            if arr.shape[0] != vertices.shape[0]:
                msg = (
                    f'point_data[{key!r}] has {arr.shape[0]} rows but mesh has '
                    f'{vertices.shape[0]} points.'
                )
                raise ValueError(msg)
            cols.append(np.ascontiguousarray(arr, dtype=np.float32))
        vert_props = np.ascontiguousarray(np.hstack(cols), dtype=np.float32)
    else:
        vert_props = vertices

    mesh_data = manifold3d.Mesh(vert_properties=vert_props, tri_verts=tri_verts)
    return manifold3d.Manifold(mesh_data)


def from_manifold(
    manifold: manifold3d.Manifold,
    *,
    property_names: Sequence[str] | None = None,
) -> pv.PolyData:
    """Convert a :class:`manifold3d.Manifold` back to :class:`pyvista.PolyData`.

    Per-vertex property channels beyond xyz are unpacked into
    ``point_data`` arrays.

    Parameters
    ----------
    manifold : manifold3d.Manifold
        Source Manifold.
    property_names : sequence of str, optional
        Names to assign to each extra property column (in order). If fewer
        names are given than columns, remaining columns get default names
        (``property_<i>``); if more, extras are ignored.

    Returns
    -------
    pyvista.PolyData
        The PolyData representation. Empty if ``manifold.is_empty()``.

    """
    if manifold.is_empty():
        return pv.PolyData()

    mesh_data = manifold.to_mesh()
    vert_props = np.asarray(mesh_data.vert_properties)
    tri_verts = np.asarray(mesh_data.tri_verts, dtype=np.int_)

    vertices = np.ascontiguousarray(vert_props[:, :3], dtype=np.float64)

    n_tri = tri_verts.shape[0]
    faces = np.empty((n_tri, 4), dtype=np.int_)
    faces[:, 0] = 3
    faces[:, 1:] = tri_verts

    # PolyData stubs want literal ``dtype[int]`` which numpy ints don't satisfy;
    # the runtime accepts any int ndarray.
    poly = pv.PolyData(vertices, faces.ravel())  # type: ignore[arg-type]

    extra = vert_props[:, 3:]
    if extra.shape[1] > 0:
        names = list(property_names) if property_names else []
        for i in range(extra.shape[1]):
            name = names[i] if i < len(names) else f'property_{i}'
            poly.point_data[name] = np.ascontiguousarray(extra[:, i])

    return poly
