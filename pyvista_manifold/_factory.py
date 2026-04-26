"""Module-level helpers that build :class:`pyvista.PolyData` from non-mesh inputs.

Anything that can start from an existing PolyData lives on the accessor
(``mesh.manifold.<op>``). Functions here cover what PyVista has no direct
equivalent for: iso-surfaces from a scalar field, 2D extrude/revolve, and
convex hulls of a raw point cloud.
"""

from collections.abc import Callable, Sequence

import manifold3d
import numpy as np
import pyvista as pv

from pyvista_manifold._conversion import from_manifold


def level_set(
    f: Callable[[float, float, float], float],
    bounds: Sequence[float],
    edge_length: float,
    *,
    level: float = 0.0,
    tolerance: float = -1.0,
) -> pv.PolyData:
    """Mesh an iso-surface of a scalar field.

    Parameters
    ----------
    f : callable
        Scalar field ``f(x, y, z) -> float``. The interior of the result
        is where ``f(x, y, z) > level``.
    bounds : sequence of float
        ``(xmin, ymin, zmin, xmax, ymax, zmax)`` cubical sample region.
    edge_length : float
        Target edge length for the output mesh.
    level : float, default: 0.0
        Iso-value to extract.
    tolerance : float, default: -1.0
        Geometric tolerance; ``-1`` lets manifold3d choose.

    Returns
    -------
    pyvista.PolyData
        The iso-surface as a triangulated PolyData.

    """
    if len(bounds) != 6:
        msg = (
            f'bounds must have 6 elements (xmin, ymin, zmin, xmax, ymax, zmax), got {len(bounds)}.'
        )
        raise ValueError(msg)
    return from_manifold(
        manifold3d.Manifold.level_set(
            f,
            list(bounds),
            edge_length,
            level=level,
            tolerance=tolerance,
        ),
    )


def extrude(
    polygons: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    height: float,
    *,
    n_divisions: int = 0,
    twist_degrees: float = 0.0,
    scale_top: Sequence[float] = (1.0, 1.0),
    fill_rule: manifold3d.FillRule = manifold3d.FillRule.Positive,
) -> pv.PolyData:
    """Extrude one or more 2D polygons along z.

    Parameters
    ----------
    polygons : array-like
        A single ``(N, 2)`` polygon or a list of such arrays representing
        a polygon-with-holes set.
    height : float
        Extrusion height along z.
    n_divisions : int, default: 0
        Number of intermediate slices. ``0`` produces a single segment.
    twist_degrees : float, default: 0.0
        Rotation about z applied linearly from bottom to top.
    scale_top : sequence of float, default: (1.0, 1.0)
        XY scale at the top relative to the bottom (tapered extrusion).
    fill_rule : manifold3d.FillRule, default: FillRule.Positive
        Polygon fill rule passed to the underlying CrossSection.

    Returns
    -------
    pyvista.PolyData
        The extruded solid as a triangulated PolyData.

    """
    cs = _polygons_to_cross_section(polygons, fill_rule)
    return from_manifold(
        manifold3d.Manifold.extrude(
            cs,
            height,
            n_divisions=n_divisions,
            twist_degrees=twist_degrees,
            scale_top=tuple(scale_top),
        ),
    )


def revolve(
    polygons: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    *,
    segments: int = 0,
    revolve_degrees: float = 360.0,
    fill_rule: manifold3d.FillRule = manifold3d.FillRule.Positive,
) -> pv.PolyData:
    """Revolve a 2D polygon about the y axis.

    Parameters
    ----------
    polygons : array-like
        A single ``(N, 2)`` polygon or a list of such arrays. Coordinates
        are interpreted as ``(x, y)``; the revolution is about the y axis.
    segments : int, default: 0
        Number of circular segments. ``0`` falls back to the
        :func:`manifold3d.set_circular_segments` global.
    revolve_degrees : float, default: 360.0
        Sweep angle in degrees.
    fill_rule : manifold3d.FillRule, default: FillRule.Positive
        Polygon fill rule.

    Returns
    -------
    pyvista.PolyData
        The revolved solid as a triangulated PolyData.

    """
    cs = _polygons_to_cross_section(polygons, fill_rule)
    return from_manifold(
        manifold3d.Manifold.revolve(
            cs,
            circular_segments=segments,
            revolve_degrees=revolve_degrees,
        ),
    )


def hull_points(points: np.ndarray) -> pv.PolyData:
    """Convex hull of a 3D point cloud.

    Parameters
    ----------
    points : array-like
        ``(N, 3)`` point coordinates.

    Returns
    -------
    pyvista.PolyData
        The hull as a triangulated PolyData.

    """
    pts = np.ascontiguousarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        msg = f'points must be (N, 3), got shape {pts.shape}.'
        raise ValueError(msg)
    return from_manifold(manifold3d.Manifold.hull_points(pts))


def _polygons_to_cross_section(
    polygons: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    fill_rule: manifold3d.FillRule,
) -> manifold3d.CrossSection:
    """Coerce assorted 2D-polygon inputs into a CrossSection."""
    arr = np.asarray(polygons, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 2:
        contours = [arr]
    elif arr.ndim == 3 and arr.shape[2] == 2:
        contours = [np.ascontiguousarray(c, dtype=np.float32) for c in arr]
    else:
        contours = [
            np.ascontiguousarray(np.asarray(c, dtype=np.float32))
            for c in polygons  # type: ignore[union-attr]
        ]
        for c in contours:
            if c.ndim != 2 or c.shape[1] != 2:
                msg = f'each polygon must be (N, 2), got shape {c.shape}.'
                raise ValueError(msg)
    return manifold3d.CrossSection(contours, fillrule=fill_rule)
