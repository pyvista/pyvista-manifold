"""PyVista accessor that exposes :mod:`manifold3d` operations on PolyData.

Importing this module registers the ``manifold`` accessor namespace on
:class:`pyvista.PolyData` via PyVista's plugin entry point system; users
do not normally need to import it directly.
"""

import atexit
from collections.abc import Callable, Iterable, Sequence
import weakref

import manifold3d
import numpy as np
import pyvista as pv

from pyvista_manifold._conversion import _polydata_from_mesh_data, from_manifold, to_manifold

_LIVE_ACCESSORS: weakref.WeakSet = weakref.WeakSet()


def _coerce_manifold(obj: pv.PolyData | manifold3d.Manifold) -> manifold3d.Manifold:
    """Accept either a PolyData or an already-converted Manifold."""
    if isinstance(obj, manifold3d.Manifold):
        return obj
    if isinstance(obj, pv.PolyData):
        return obj.manifold.to_manifold()
    msg = f'Expected pyvista.PolyData or manifold3d.Manifold, got {type(obj).__name__}.'
    raise TypeError(msg)


def _cross_section_to_polydata(cs: manifold3d.CrossSection, *, z: float = 0.0) -> pv.PolyData:
    """Convert a 2D CrossSection to a PolyData of closed polylines at height ``z``."""
    if cs.is_empty():
        return pv.PolyData()
    contours = cs.to_polygons()
    points: list[np.ndarray] = []
    lines: list[list[int]] = []
    offset = 0
    for contour in contours:
        contour = np.asarray(contour, dtype=np.float64)
        n = contour.shape[0]
        if n < 2:
            continue
        pts3d = np.column_stack([contour, np.full(n, z)])
        points.append(pts3d)
        lines.append([*range(offset, offset + n), offset])
        offset += n
    if not points:
        return pv.PolyData()
    return pv.PolyData(np.vstack(points), lines=pv.CellArray.from_irregular_cells(lines))


@pv.register_dataset_accessor('manifold', pv.PolyData)
class ManifoldAccessor:
    """Manifold operations exposed as ``polydata.manifold.<method>(...)``.

    Every operation that returns geometry returns a fresh
    :class:`pyvista.PolyData`, never a :class:`~manifold3d.Manifold`. Use
    :meth:`to_manifold` to drop down to the raw Manifold object when
    needed. The default ``clean=True`` conversion is cached on the
    accessor instance and invalidated whenever the source dataset is
    modified.

    Examples
    --------
    >>> import pyvista as pv
    >>> import pyvista_manifold  # registers the accessor
    >>> a = pv.Cube()
    >>> b = pv.Sphere(radius=0.7).translate((0.5, 0.5, 0.5))
    >>> result = a.manifold.difference(b)

    """

    def __init__(self, mesh: pv.PolyData) -> None:
        self._mesh = mesh
        self._cached_manifold: manifold3d.Manifold | None = None
        self._cached_manifold_mtime: int | None = None
        _LIVE_ACCESSORS.add(self)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _mesh_mtime(self) -> int:
        """Return the dataset modification time used for cache invalidation."""
        # PyVista does not currently expose a public wrapper around VTK's
        # dataset modified time. Use it here so the cached Manifold stays in
        # sync with mutable ``PolyData`` instances.
        return int(self._mesh.GetMTime())

    def _default_manifold(self) -> manifold3d.Manifold:
        """Return the cached default conversion for this dataset."""
        current_mtime = self._mesh_mtime()
        if self._cached_manifold_mtime != current_mtime or self._cached_manifold is None:
            self._cached_manifold = to_manifold(self._mesh)
            self._cached_manifold_mtime = current_mtime
        return self._cached_manifold

    def _clear_cached_manifold(self) -> None:
        """Drop any cached Manifold wrapper held by this accessor."""
        self._cached_manifold = None
        self._cached_manifold_mtime = None

    def to_manifold(
        self,
        *,
        point_data_keys: Sequence[str] | None = None,
        clean: bool = True,
    ) -> manifold3d.Manifold:
        """Return the underlying mesh as a :class:`~manifold3d.Manifold`.

        Parameters
        ----------
        point_data_keys : sequence of str, optional
            See :func:`pyvista_manifold.to_manifold`.
        clean : bool, default: True
            See :func:`pyvista_manifold.to_manifold`.

        Returns
        -------
        manifold3d.Manifold
            Manifold representation of this mesh.

        """
        if point_data_keys is None and clean:
            return self._default_manifold()
        return to_manifold(self._mesh, point_data_keys=point_data_keys, clean=clean)

    @staticmethod
    def from_manifold(
        manifold: manifold3d.Manifold,
        *,
        property_names: Sequence[str] | None = None,
    ) -> pv.PolyData:
        """Build a :class:`pyvista.PolyData` from a :class:`~manifold3d.Manifold`.

        Mirrors :func:`pyvista_manifold.from_manifold`; exposed on the
        accessor so it is reachable as ``pv.PolyData.manifold.from_manifold(m)``
        without importing the module-level function.

        Parameters
        ----------
        manifold : manifold3d.Manifold
            Source Manifold.
        property_names : sequence of str, optional
            See :func:`pyvista_manifold.from_manifold`.

        Returns
        -------
        pyvista.PolyData
            The PolyData representation.

        """
        return from_manifold(manifold, property_names=property_names)

    # ------------------------------------------------------------------
    # Boolean operations
    # ------------------------------------------------------------------

    def union(self, other: pv.PolyData | manifold3d.Manifold) -> pv.PolyData:
        """Union with another solid.

        Parameters
        ----------
        other : pyvista.PolyData or manifold3d.Manifold
            Second operand.

        Returns
        -------
        pyvista.PolyData
            The union result.

        """
        return from_manifold(self.to_manifold() + _coerce_manifold(other))

    def difference(self, other: pv.PolyData | manifold3d.Manifold) -> pv.PolyData:
        """Subtract ``other`` from this mesh.

        Parameters
        ----------
        other : pyvista.PolyData or manifold3d.Manifold
            Second operand.

        Returns
        -------
        pyvista.PolyData
            The difference result.

        """
        return from_manifold(self.to_manifold() - _coerce_manifold(other))

    def intersection(self, other: pv.PolyData | manifold3d.Manifold) -> pv.PolyData:
        """Intersect this mesh with ``other``.

        Parameters
        ----------
        other : pyvista.PolyData or manifold3d.Manifold
            Second operand.

        Returns
        -------
        pyvista.PolyData
            The intersection result.

        """
        return from_manifold(self.to_manifold() ^ _coerce_manifold(other))

    def batch_boolean(
        self,
        others: Iterable[pv.PolyData | manifold3d.Manifold],
        op: manifold3d.OpType = manifold3d.OpType.Add,
    ) -> pv.PolyData:
        """Apply an n-ary Boolean against a batch of other meshes.

        Parameters
        ----------
        others : iterable of pyvista.PolyData or manifold3d.Manifold
            Additional operands; ``self`` is prepended.
        op : manifold3d.OpType, default: OpType.Add
            ``Add`` (union), ``Subtract`` (difference) or ``Intersect``.

        Returns
        -------
        pyvista.PolyData
            The Boolean result.

        """
        items = [self.to_manifold()] + [_coerce_manifold(o) for o in others]
        return from_manifold(manifold3d.Manifold.batch_boolean(items, op))

    # ------------------------------------------------------------------
    # Hull
    # ------------------------------------------------------------------

    def hull(self) -> pv.PolyData:
        """Convex hull of this mesh's vertices.

        Returns
        -------
        pyvista.PolyData
            The convex hull.

        """
        return from_manifold(self.to_manifold().hull())

    def hull_with(
        self,
        *others: pv.PolyData | manifold3d.Manifold,
    ) -> pv.PolyData:
        """Convex hull of this mesh's vertices combined with others.

        Parameters
        ----------
        *others : pyvista.PolyData or manifold3d.Manifold
            Extra meshes whose vertices contribute to the hull.

        Returns
        -------
        pyvista.PolyData
            The combined hull.

        """
        items = [self.to_manifold()] + [_coerce_manifold(o) for o in others]
        return from_manifold(manifold3d.Manifold.batch_hull(items))

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def translate(self, t: Sequence[float]) -> pv.PolyData:
        """Translate by a 3-vector.

        Parameters
        ----------
        t : sequence of float
            ``(tx, ty, tz)`` translation.

        Returns
        -------
        pyvista.PolyData
            The translated mesh.

        """
        return from_manifold(self.to_manifold().translate(tuple(t)))

    def rotate(self, r: Sequence[float]) -> pv.PolyData:
        """Rotate using XYZ Euler angles in degrees.

        Parameters
        ----------
        r : sequence of float
            ``(rx, ry, rz)`` Euler angles in degrees.

        Returns
        -------
        pyvista.PolyData
            The rotated mesh.

        """
        return from_manifold(self.to_manifold().rotate(tuple(r)))

    def scale(self, s: float | Sequence[float]) -> pv.PolyData:
        """Scale uniformly or per-axis.

        Parameters
        ----------
        s : float or sequence of float
            Scale factor; scalar for uniform, 3-vector for per-axis.

        Returns
        -------
        pyvista.PolyData
            The scaled mesh.

        """
        if isinstance(s, (int, float, np.integer, np.floating)):
            s_t: tuple[float, float, float] = (float(s), float(s), float(s))
        else:
            vals = [float(v) for v in s]
            if len(vals) != 3:
                msg = f'scale sequence must have 3 elements, got {len(vals)}.'
                raise ValueError(msg)
            s_t = (vals[0], vals[1], vals[2])
        return from_manifold(self.to_manifold().scale(s_t))

    def mirror(self, normal: Sequence[float]) -> pv.PolyData:
        """Mirror about a plane through the origin with the given normal.

        Parameters
        ----------
        normal : sequence of float
            Plane normal.

        Returns
        -------
        pyvista.PolyData
            The mirrored mesh.

        """
        return from_manifold(self.to_manifold().mirror(tuple(normal)))

    def transform(self, matrix: np.ndarray) -> pv.PolyData:
        """Apply a 3x4 column-major affine transform.

        Parameters
        ----------
        matrix : array-like
            ``(3, 4)`` column-major affine transform matrix.

        Returns
        -------
        pyvista.PolyData
            The transformed mesh.

        """
        m = np.ascontiguousarray(matrix, dtype=np.float32)
        if m.shape != (3, 4):
            msg = f'transform matrix must be (3, 4), got {m.shape}.'
            raise ValueError(msg)
        return from_manifold(self.to_manifold().transform(m))

    def warp(
        self,
        f: Callable[..., np.ndarray] | Callable[..., Sequence[float]],
        *,
        batch: bool = False,
    ) -> pv.PolyData:
        """Apply a per-vertex warp callback.

        Parameters
        ----------
        f : callable
            If ``batch=False``, called as ``f((x, y, z)) -> (x', y', z')``.
            If ``batch=True``, called as ``f(positions: ndarray (N, 3)) ->
            ndarray (N, 3)`` (much faster on large meshes).
        batch : bool, default: False
            Whether to dispatch the vectorized callback.

        Returns
        -------
        pyvista.PolyData
            The warped mesh.

        """
        m = self.to_manifold()
        warped = m.warp_batch(f) if batch else m.warp(f)
        return from_manifold(warped)

    # ------------------------------------------------------------------
    # Refinement / smoothing
    # ------------------------------------------------------------------

    def refine(self, n: int) -> pv.PolyData:
        """Subdivide every edge into ``n`` segments.

        Parameters
        ----------
        n : int
            Number of segments per original edge (``n >= 1``).

        Returns
        -------
        pyvista.PolyData
            The refined mesh.

        """
        return from_manifold(self.to_manifold().refine(n))

    def refine_to_length(self, length: float) -> pv.PolyData:
        """Adaptively refine until every edge is shorter than ``length``.

        Parameters
        ----------
        length : float
            Target maximum edge length.

        Returns
        -------
        pyvista.PolyData
            The refined mesh.

        """
        return from_manifold(self.to_manifold().refine_to_length(length))

    def refine_to_tolerance(self, tolerance: float) -> pv.PolyData:
        """Refine until geometric error is below ``tolerance``.

        Parameters
        ----------
        tolerance : float
            Target geometric tolerance.

        Returns
        -------
        pyvista.PolyData
            The refined mesh.

        """
        return from_manifold(self.to_manifold().refine_to_tolerance(tolerance))

    def smooth_out(
        self,
        *,
        min_sharp_angle: float = 60.0,
        min_smoothness: float = 0.0,
    ) -> pv.PolyData:
        """Smooth without explicit per-vertex normals.

        Parameters
        ----------
        min_sharp_angle : float, default: 60.0
            Minimum dihedral angle (degrees) treated as a sharp edge.
        min_smoothness : float, default: 0.0
            Minimum smoothness factor applied across the mesh.

        Returns
        -------
        pyvista.PolyData
            The smoothed mesh.

        """
        return from_manifold(
            self.to_manifold().smooth_out(
                min_sharp_angle=min_sharp_angle,
                min_smoothness=min_smoothness,
            ),
        )

    def smooth_by_normals(self, normal_idx: int) -> pv.PolyData:
        """Smooth using vertex normals stored at ``normal_idx``.

        Parameters
        ----------
        normal_idx : int
            Property channel index where x normals start (3 channels are read).

        Returns
        -------
        pyvista.PolyData
            The smoothed mesh.

        """
        return from_manifold(self.to_manifold().smooth_by_normals(normal_idx))

    def calculate_normals(
        self,
        normal_idx: int = 0,
        *,
        min_sharp_angle: float = 60.0,
    ) -> pv.PolyData:
        """Compute and store per-vertex normals as point-data ``Normals``.

        Parameters
        ----------
        normal_idx : int, default: 0
            Offset within Manifold's per-vertex property channels (beyond
            xyz) at which the 3-component normals are written. ``0`` writes
            them immediately after xyz; non-zero values leave that many
            zero-padded channels in front of the normals.
        min_sharp_angle : float, default: 60.0
            Dihedral angle threshold for a sharp edge.

        Returns
        -------
        pyvista.PolyData
            The mesh with a ``Normals`` (``N x 3``) point-data array.

        """
        m = self.to_manifold().calculate_normals(normal_idx, min_sharp_angle=min_sharp_angle)
        poly, vert_props = _polydata_from_mesh_data(m.to_mesh())
        start = 3 + normal_idx
        poly.point_data['Normals'] = np.ascontiguousarray(vert_props[:, start : start + 3])
        return poly

    def calculate_curvature(
        self,
        *,
        gaussian_idx: int = 0,
        mean_idx: int = 1,
    ) -> pv.PolyData:
        """Compute Gaussian and Mean curvature as point-data arrays.

        Parameters
        ----------
        gaussian_idx : int, default: 0
            Offset within Manifold's per-vertex property channels (beyond
            xyz) at which Gaussian curvature is written.
        mean_idx : int, default: 1
            Offset within Manifold's per-vertex property channels (beyond
            xyz) at which Mean curvature is written.

        Returns
        -------
        pyvista.PolyData
            The mesh with ``GaussianCurvature`` and ``MeanCurvature``
            point-data arrays.

        """
        m = self.to_manifold().calculate_curvature(gaussian_idx, mean_idx)
        poly, vert_props = _polydata_from_mesh_data(m.to_mesh())
        poly.point_data['GaussianCurvature'] = np.ascontiguousarray(
            vert_props[:, 3 + gaussian_idx]
        )
        poly.point_data['MeanCurvature'] = np.ascontiguousarray(vert_props[:, 3 + mean_idx])
        return poly

    # ------------------------------------------------------------------
    # Splits & decomposition
    # ------------------------------------------------------------------

    def split(
        self,
        cutter: pv.PolyData | manifold3d.Manifold,
    ) -> tuple[pv.PolyData, pv.PolyData]:
        """Cut by another solid; return ``(inside, outside)``.

        Parameters
        ----------
        cutter : pyvista.PolyData or manifold3d.Manifold
            The cutting solid.

        Returns
        -------
        inside : pyvista.PolyData
            Portion of ``self`` inside ``cutter``.
        outside : pyvista.PolyData
            Portion of ``self`` outside ``cutter``.

        """
        inside, outside = self.to_manifold().split(_coerce_manifold(cutter))
        return from_manifold(inside), from_manifold(outside)

    def split_by_plane(
        self,
        normal: Sequence[float],
        offset: float = 0.0,
    ) -> tuple[pv.PolyData, pv.PolyData]:
        """Cut by an infinite plane; return ``(positive, negative)`` halves.

        Parameters
        ----------
        normal : sequence of float
            Plane normal (need not be unit-length).
        offset : float, default: 0.0
            Signed distance of the plane from the origin along ``normal``.

        Returns
        -------
        positive : pyvista.PolyData
            Half on the side ``normal`` points toward.
        negative : pyvista.PolyData
            Other half.

        """
        pos, neg = self.to_manifold().split_by_plane(tuple(normal), offset)
        return from_manifold(pos), from_manifold(neg)

    def trim_by_plane(
        self,
        normal: Sequence[float],
        offset: float = 0.0,
    ) -> pv.PolyData:
        """Discard the half-space on the side opposite ``normal``.

        Parameters
        ----------
        normal : sequence of float
            Plane normal.
        offset : float, default: 0.0
            Signed distance from the origin.

        Returns
        -------
        pyvista.PolyData
            Trimmed mesh.

        """
        return from_manifold(self.to_manifold().trim_by_plane(tuple(normal), offset))

    def decompose(self) -> list[pv.PolyData]:
        """Split into disconnected components.

        Returns
        -------
        list of pyvista.PolyData
            One PolyData per connected component.

        """
        return [from_manifold(m) for m in self.to_manifold().decompose()]

    # ------------------------------------------------------------------
    # Minkowski
    # ------------------------------------------------------------------

    def minkowski_sum(self, other: pv.PolyData | manifold3d.Manifold) -> pv.PolyData:
        """Minkowski sum (``self`` ⊕ ``other``).

        Parameters
        ----------
        other : pyvista.PolyData or manifold3d.Manifold
            The structuring solid.

        Returns
        -------
        pyvista.PolyData
            The Minkowski sum.

        """
        return from_manifold(self.to_manifold().minkowski_sum(_coerce_manifold(other)))

    def minkowski_difference(
        self,
        other: pv.PolyData | manifold3d.Manifold,
    ) -> pv.PolyData:
        """Minkowski difference (``self`` ⊖ ``other``).

        Parameters
        ----------
        other : pyvista.PolyData or manifold3d.Manifold
            The structuring solid.

        Returns
        -------
        pyvista.PolyData
            The Minkowski difference.

        """
        return from_manifold(self.to_manifold().minkowski_difference(_coerce_manifold(other)))

    # ------------------------------------------------------------------
    # 3D → 2D
    # ------------------------------------------------------------------

    def slice_z(self, z: float = 0.0) -> pv.PolyData:
        """Slice by a horizontal plane at height ``z``; return polylines.

        Parameters
        ----------
        z : float, default: 0.0
            Plane height.

        Returns
        -------
        pyvista.PolyData
            Closed polylines at ``z`` describing the cross section.

        """
        cs = self.to_manifold().slice(z)
        return _cross_section_to_polydata(cs, z=z)

    def project(self) -> pv.PolyData:
        """Orthographic projection of the silhouette onto the XY plane.

        Returns
        -------
        pyvista.PolyData
            Closed polylines at ``z = 0`` describing the silhouette.

        """
        cs = self.to_manifold().project()
        return _cross_section_to_polydata(cs, z=0.0)

    # ------------------------------------------------------------------
    # Tolerance / simplification / property channels
    # ------------------------------------------------------------------

    def simplify(self, tolerance: float) -> pv.PolyData:
        """Coarsen the mesh while keeping geometry within ``tolerance``.

        Parameters
        ----------
        tolerance : float
            Maximum allowed deviation.

        Returns
        -------
        pyvista.PolyData
            Simplified mesh.

        """
        return from_manifold(self.to_manifold().simplify(tolerance))

    def set_tolerance(self, tolerance: float) -> pv.PolyData:
        """Set the mesh's numerical tolerance, returning a new PolyData.

        Parameters
        ----------
        tolerance : float
            New tolerance value.

        Returns
        -------
        pyvista.PolyData
            Mesh with updated tolerance.

        """
        return from_manifold(self.to_manifold().set_tolerance(tolerance))

    def set_properties(
        self,
        new_num_prop: int,
        f: Callable[..., Sequence[float]],
    ) -> pv.PolyData:
        """Rewrite per-vertex property channels via a callback.

        Parameters
        ----------
        new_num_prop : int
            Total number of property channels in the result (``>= 3``).
        f : callable
            ``f((x, y, z), old_props) -> new_props`` invoked per vertex.

        Returns
        -------
        pyvista.PolyData
            Mesh with the rewritten property channels exposed as
            ``point_data['property_0']``, ``property_1``, etc.

        """
        return from_manifold(self.to_manifold().set_properties(new_num_prop, f))

    def as_original(self) -> pv.PolyData:
        """Mark the result as a fresh original (assigns a new original ID).

        Returns
        -------
        pyvista.PolyData
            Mesh registered as an original within Manifold's tracking.

        """
        return from_manifold(self.to_manifold().as_original())

    # ------------------------------------------------------------------
    # Properties / queries
    # ------------------------------------------------------------------

    @property
    def volume(self) -> float:
        """Signed volume of the solid (negative if inverted).

        Returns
        -------
        float
            Volume in cubed units.

        """
        return float(self.to_manifold().volume())

    @property
    def surface_area(self) -> float:
        """Total surface area.

        Returns
        -------
        float
            Surface area in squared units.

        """
        return float(self.to_manifold().surface_area())

    @property
    def genus(self) -> int:
        """Topological genus of the surface (number of handles).

        Returns
        -------
        int
            Genus.

        """
        return int(self.to_manifold().genus())

    @property
    def num_vert(self) -> int:
        """Vertex count after Manifold reconstruction.

        Returns
        -------
        int
            Vertex count.

        """
        return int(self.to_manifold().num_vert())

    @property
    def num_edge(self) -> int:
        """Edge count after Manifold reconstruction.

        Returns
        -------
        int
            Edge count.

        """
        return int(self.to_manifold().num_edge())

    @property
    def num_tri(self) -> int:
        """Triangle count after Manifold reconstruction.

        Returns
        -------
        int
            Triangle count.

        """
        return int(self.to_manifold().num_tri())

    @property
    def is_empty(self) -> bool:
        """Whether the manifold representation has any geometry.

        Returns
        -------
        bool
            ``True`` if empty.

        """
        return bool(self.to_manifold().is_empty())

    @property
    def is_valid(self) -> bool:
        """Whether the input is a valid (closed, manifold) solid.

        Returns
        -------
        bool
            ``True`` when ``status`` is :attr:`~manifold3d.Error.NoError`.

        """
        return self.status == manifold3d.Error.NoError

    @property
    def status(self) -> manifold3d.Error:
        """Manifold error status for this mesh.

        Returns
        -------
        manifold3d.Error
            ``NoError`` for a valid solid; any other value indicates a
            specific construction issue.

        """
        return self.to_manifold().status()

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """Axis-aligned bounding box in PyVista order.

        Returns
        -------
        tuple of float
            ``(xmin, xmax, ymin, ymax, zmin, zmax)``. Matches the order
            returned by :attr:`pyvista.DataSet.bounds`.

        """
        xmin, ymin, zmin, xmax, ymax, zmax = self.to_manifold().bounding_box()
        return (xmin, xmax, ymin, ymax, zmin, zmax)

    @property
    def original_id(self) -> int:
        """Original Manifold ID, or ``-1`` if not an original.

        Returns
        -------
        int
            Original ID.

        """
        return int(self.to_manifold().original_id())

    @property
    def tolerance(self) -> float:
        """Numerical tolerance used by Manifold.

        Returns
        -------
        float
            The tolerance value.

        """
        return float(self.to_manifold().get_tolerance())

    def min_gap(
        self,
        other: pv.PolyData | manifold3d.Manifold,
        search_length: float,
    ) -> float:
        """Minimum gap between this and another solid.

        Parameters
        ----------
        other : pyvista.PolyData or manifold3d.Manifold
            Other solid.
        search_length : float
            Upper bound on gap distance to search; values beyond this are
            reported as ``search_length``.

        Returns
        -------
        float
            Closest distance, clamped at ``search_length``.

        """
        return float(self.to_manifold().min_gap(_coerce_manifold(other), search_length))

    # ------------------------------------------------------------------
    # Batch helpers (exposed here for fluent chaining)
    # ------------------------------------------------------------------

    def compose_with(
        self,
        *others: pv.PolyData | manifold3d.Manifold,
    ) -> pv.PolyData:
        """Disjointly compose with other meshes (no Boolean).

        Parameters
        ----------
        *others : pyvista.PolyData or manifold3d.Manifold
            Additional meshes to compose with ``self``.

        Returns
        -------
        pyvista.PolyData
            The composed result.

        """
        items = [self.to_manifold()] + [_coerce_manifold(o) for o in others]
        return from_manifold(manifold3d.Manifold.compose(items))


# Type-check compatibility (does not affect runtime)
_: type[pv.DataSetAccessor] = ManifoldAccessor


def _clear_live_accessor_caches() -> None:
    """Release cached Manifold wrappers before interpreter shutdown."""
    for accessor in tuple(_LIVE_ACCESSORS):
        accessor._clear_cached_manifold()


atexit.register(_clear_live_accessor_caches)


__all__ = ['ManifoldAccessor']
