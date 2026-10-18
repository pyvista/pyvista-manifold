"""Conversion between :class:`pyvista.PolyData` and :class:`manifold3d.Manifold`."""

from collections.abc import Sequence

import manifold3d
import numpy as np
import numpy.typing as npt
import pyvista as pv

# Cell array ``_to_manifold`` adds to its own copy of a mesh to tag triangles.
_FACE_ID = '_pyvista_manifold_face_id'


def _as_contiguous_array(array: npt.ArrayLike, *, dtype: npt.DTypeLike) -> np.ndarray:
    """Return ``array`` as a contiguous NumPy array with the requested dtype."""
    contiguous_array = np.asarray(array, dtype=dtype)
    if contiguous_array.flags.c_contiguous:
        return contiguous_array
    return np.ascontiguousarray(contiguous_array)


def _polydata_from_mesh_data(mesh_data: manifold3d.Mesh64) -> tuple[pv.PolyData, np.ndarray]:
    """Build ``PolyData`` geometry from a Manifold mesh export."""
    vert_props = np.asarray(mesh_data.vert_properties)
    tri_verts = _as_contiguous_array(mesh_data.tri_verts, dtype=np.dtype(pv.ID_TYPE))
    points = np.array(vert_props[:, :3], copy=True)

    poly = pv.PolyData.from_regular_faces(points, tri_verts, deep=False)
    return poly, vert_props


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
    return _to_manifold(mesh, point_data_keys=point_data_keys, clean=clean)


def _to_manifold(
    mesh: pv.PolyData,
    *,
    point_data_keys: Sequence[str] | None = None,
    clean: bool = True,
    first_face_id: int | None = None,
) -> manifold3d.Manifold:
    """Convert ``mesh``; with ``first_face_id``, tag each triangle with its source cell.

    The tag is ``first_face_id`` plus the index of the input cell the triangle
    came from, and survives Manifold operations as the output ``face_id``.
    """
    if not isinstance(mesh, pv.PolyData):
        msg = f'Expected pyvista.PolyData, got {type(mesh).__name__}.'
        raise TypeError(msg)

    if mesh.n_points == 0:
        return manifold3d.Manifold()

    pd = mesh
    if first_face_id is not None:
        pd = mesh.copy(deep=False)
        pd.cell_data[_FACE_ID] = np.arange(first_face_id, first_face_id + mesh.n_cells)
    pd = pd.clean() if clean else pd
    if pd.n_verts or pd.n_lines:
        # Vertex and line cells (e.g. a triangle ``clean`` collapsed) come
        # first; drop them so the cell data stays aligned with the triangles.
        pd = pd.remove_cells(np.arange(pd.n_verts + pd.n_lines))
    if not pd.is_all_triangles:
        pd = pd.triangulate()
    vertices = _as_contiguous_array(pd.points, dtype=np.dtype(np.float64))
    tri_verts = _as_contiguous_array(pd.regular_faces, dtype=np.dtype(np.uint64))

    if point_data_keys:
        cols: list[np.ndarray] = [vertices]
        for key in point_data_keys:
            arr = _as_contiguous_array(pd.point_data[key], dtype=np.dtype(np.float64))
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
            cols.append(arr)
        vert_props = np.concatenate(cols, axis=1)
    else:
        vert_props = vertices

    face_id = None
    if first_face_id is not None:
        face_id = _as_contiguous_array(pd.cell_data[_FACE_ID], dtype=np.dtype(np.uint64))
    mesh_data = manifold3d.Mesh64(vert_properties=vert_props, tri_verts=tri_verts, face_id=face_id)
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

    poly, vert_props = _polydata_from_mesh_data(manifold.to_mesh64())

    extra = vert_props[:, 3:]
    if extra.shape[1] > 0:
        names = list(property_names) if property_names else []
        for i in range(extra.shape[1]):
            name = names[i] if i < len(names) else f'property_{i}'
            poly.point_data[name] = np.array(extra[:, i], copy=True)

    return poly
