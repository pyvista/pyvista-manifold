"""PyVista accessor for the Manifold 3D geometry library.

Installing this package registers a ``manifold`` accessor on
:class:`pyvista.PolyData` via PyVista's plugin entry-point system. Any
PolyData then exposes Manifold operations as ``mesh.manifold.<op>``
without an explicit import.

A small set of module-level helpers covers operations that don't start
from an existing mesh: :func:`level_set`, :func:`extrude`,
:func:`revolve`, :func:`hull_points`. For everything else, prefer
PyVista's own primitives (``pv.Cube``, ``pv.Sphere``, ``pv.Cylinder``)
chained through the accessor.
"""

from importlib.metadata import PackageNotFoundError, version as _version

# Re-export Manifold enums for convenience.
from manifold3d import Error, FillRule, JoinType, OpType

# Importing _accessor runs the @register_dataset_accessor decorator.
from pyvista_manifold._accessor import ManifoldAccessor
from pyvista_manifold._conversion import from_manifold, to_manifold
from pyvista_manifold._factory import extrude, hull_points, level_set, revolve

try:
    __version__ = _version('pyvista-manifold')
except PackageNotFoundError:
    try:
        from pyvista_manifold._version import __version__
    except ImportError:
        __version__ = '0.0.0'

__all__ = [
    'Error',
    'FillRule',
    'JoinType',
    'ManifoldAccessor',
    'OpType',
    '__version__',
    'extrude',
    'from_manifold',
    'hull_points',
    'level_set',
    'revolve',
    'to_manifold',
]
