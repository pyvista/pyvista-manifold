# pyvista-manifold

A PyVista accessor for [Manifold](https://github.com/elalish/manifold), a fast and reliable boolean / CSG library for triangle meshes.

<p align="center">
  <img src="https://raw.githubusercontent.com/pyvista/pyvista-manifold/main/assets/gyroid.gif" alt="rotating gold gyroid TPMS sphere" width="520">
</p>

> A gyroid iso-surface intersected with a sphere, rendered in PBR gold. The whole solid was built in three function calls: `level_set` for the gyroid field, `pv.Sphere` for the trim, `mesh.manifold.intersection` to combine them.

![pyvista-manifold examples banner](https://raw.githubusercontent.com/pyvista/pyvista-manifold/main/assets/banner.png)

> From left: a machined aluminum bracket built by chaining `union` and `difference`; a real mesh intersected with a gyroid TPMS lattice; the gold sphere above seen from a fixed angle; a cube fractured by repeated plane cuts.

Once the package is installed, every `pv.PolyData` exposes a `.manifold` accessor. There is nothing to import.

```python
import pyvista as pv

cube = pv.Cube()
sphere = pv.Sphere(radius=0.7, center=(0.4, 0.4, 0.4))
cube.manifold.difference(sphere).plot()
```

## Why

PyVista's built-in boolean filters wrap VTK's `vtkBooleanOperationPolyDataFilter`, which produces non-manifold or self-intersecting output on non-trivial inputs. Manifold solves the same problem with exact arithmetic and topology tracking. This package is the smallest reasonable bridge between the two: a single `.manifold` accessor that converts on demand and always returns a fresh `pv.PolyData`.

## Install

```bash
pip install pyvista-manifold
```

Requires Python 3.10+ and PyVista 0.48+. The accessor registers itself via PyVista's plugin entry-point system; you don't import the package to use it.

## Quick start

```python
import pyvista as pv

# Boolean ops chain through PyVista's filter pipeline
cube = pv.Cube(x_length=2.0, y_length=2.0, z_length=2.0)
sphere = pv.Sphere(radius=0.9)
diff = cube.manifold.difference(sphere)
print(diff.manifold.volume, diff.manifold.is_valid)

# Drill three orthogonal cylinders out of a cube in one call
holes = [
    pv.Cylinder(radius=0.4, height=3, direction=d)
    for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
]
from pyvista_manifold import OpType
drilled = cube.manifold.batch_boolean(holes, op=OpType.Subtract)

# Intersect with an iso-surface from a callable scalar field
import math
from pyvista_manifold import level_set

def gyroid(x, y, z):
    return -(math.sin(2*x)*math.cos(2*y)
             + math.sin(2*y)*math.cos(2*z)
             + math.sin(2*z)*math.cos(2*x))

iso = level_set(gyroid, bounds=(-2, -2, -2, 2, 2, 2), edge_length=0.1)
infilled = pv.Sphere(radius=1.5).manifold.intersection(iso)

# Anything you build chains naturally with PyVista filters
finished = drilled.clean().smooth(n_iter=20).compute_normals()
```

A worked walkthrough lives in [`examples/showcase.ipynb`](examples/showcase.ipynb): mechanical CSG, TPMS infill of a real mesh, topographic slicing, Voronoi-style fracture, Minkowski filleting.

### Gallery

|                                                                                                                               |                                                                                                                                               |
| ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Mechanical CSG.** Stack `union` and `difference` to build a real-looking part.                                              | <img src="https://raw.githubusercontent.com/pyvista/pyvista-manifold/main/assets/bracket.png" alt="machined bracket" width="420">             |
| **TPMS lattice infill.** Intersect a closed mesh with a gyroid field — the standard 3D-printer infill, computed in two lines. | <img src="https://raw.githubusercontent.com/pyvista/pyvista-manifold/main/assets/lattice_cow.png" alt="cow with gyroid infill" width="420">   |
| **Topographic slicing.** `slice_z` at many heights stacks into a contour map of the silhouette.                               | <img src="https://raw.githubusercontent.com/pyvista/pyvista-manifold/main/assets/topographic.png" alt="horse topographic slices" width="420"> |
| **Iso-surface from a callable.** `level_set` extracts a TPMS surface from a Python function, no marching-cubes plumbing.      | <img src="https://raw.githubusercontent.com/pyvista/pyvista-manifold/main/assets/gyroid.png" alt="gyroid TPMS sphere" width="420">            |
| **Voronoi-style fracture.** Repeated `split_by_plane` calls turn a cube into a stack of polyhedral cells.                     | <img src="https://raw.githubusercontent.com/pyvista/pyvista-manifold/main/assets/fracture.png" alt="fractured cube" width="420">              |

## The accessor

`mesh.manifold` is a per-instance accessor that converts the PolyData into a `manifold3d.Manifold` on demand, runs the operation, and converts the result back. The input is left untouched.

```python
mesh.manifold                      # accessor instance, cached on the dataset
mesh.manifold.to_manifold()        # raw manifold3d.Manifold (drop down when needed)
mesh.manifold.<operation>(...)     # any method below; always returns pv.PolyData
```

The conversion runs `pyvista.PolyData.clean()` and triangulates the input by default, so PyVista primitives like `pv.Cube` and `pv.Cylinder` (which ship with seam-duplicated vertices) work directly. Pass `clean=False` to `mesh.manifold.to_manifold()` if you need to preserve every input vertex.

If your mesh isn't a closed manifold solid, the conversion still returns a `Manifold`, but downstream operations may misbehave. Check `mesh.manifold.is_valid` (returns `True` when Manifold's status is `NoError`).

### Boolean operations

| Method                                 | Result                                   |
| -------------------------------------- | ---------------------------------------- |
| `union(other)`                         | self joined with other                   |
| `difference(other)`                    | self with other subtracted               |
| `intersection(other)`                  | overlap of self and other                |
| `batch_boolean(others, op=OpType.Add)` | n-ary union, difference, or intersection |

`other` is either a `pv.PolyData` or a `manifold3d.Manifold`. Mixing is fine.

### Transforms

| Method                 | Notes                                                 |
| ---------------------- | ----------------------------------------------------- |
| `translate(t)`         | 3-vector                                              |
| `rotate(r)`            | XYZ Euler angles in degrees                           |
| `scale(s)`             | scalar or 3-vector                                    |
| `mirror(normal)`       | reflect about a plane through the origin              |
| `transform(matrix)`    | 3x4 column-major affine                               |
| `warp(f, batch=False)` | per-vertex callback (or vectorized with `batch=True`) |

### Hulls

| Method               | Notes                                 |
| -------------------- | ------------------------------------- |
| `hull()`             | convex hull of this mesh's vertices   |
| `hull_with(*others)` | convex hull of this plus other meshes |

### Refinement and smoothing

| Method                                             | Notes                                                           |
| -------------------------------------------------- | --------------------------------------------------------------- |
| `refine(n)`                                        | subdivide every edge into `n` segments                          |
| `refine_to_length(length)`                         | adaptive subdivision until every edge is shorter than `length`  |
| `refine_to_tolerance(tol)`                         | refine until geometric error is below `tol`                     |
| `smooth_out(min_sharp_angle=60, min_smoothness=0)` | smooth without explicit normals                                 |
| `smooth_by_normals(normal_idx)`                    | smooth using stored vertex normals                              |
| `calculate_normals(normal_idx=0)`                  | compute and store per-vertex normals as `point_data['Normals']` |
| `calculate_curvature(gaussian_idx=0, mean_idx=1)`  | store Gaussian + Mean curvature as point arrays                 |

### Splits and decomposition

| Method                             | Returns                                           |
| ---------------------------------- | ------------------------------------------------- |
| `split(cutter)`                    | `(inside, outside)` PolyData pair                 |
| `split_by_plane(normal, offset=0)` | `(positive, negative)` PolyData pair              |
| `trim_by_plane(normal, offset=0)`  | the half-space on the side `normal` points toward |
| `decompose()`                      | list of disconnected components                   |

### Minkowski

| Method                        | Notes                                          |
| ----------------------------- | ---------------------------------------------- |
| `minkowski_sum(other)`        | self offset outward by `other` (rounded edges) |
| `minkowski_difference(other)` | self eroded inward by `other`                  |

### 3D to 2D

| Method         | Returns                                                      |
| -------------- | ------------------------------------------------------------ |
| `slice_z(z=0)` | closed polylines at height `z` (PolyData with lines)         |
| `project()`    | silhouette projected onto the XY plane (PolyData with lines) |

### Properties and queries

| Property / method                 | Returns                                                        |
| --------------------------------- | -------------------------------------------------------------- |
| `volume`                          | signed volume                                                  |
| `surface_area`                    | total surface area                                             |
| `genus`                           | topological genus (number of handles)                          |
| `bounds`                          | `(xmin, xmax, ymin, ymax, zmin, zmax)`, matching PyVista order |
| `num_vert`, `num_edge`, `num_tri` | geometry counts after Manifold reconstruction                  |
| `is_empty`, `is_valid`, `status`  | empty check, manifold validity, raw `Error` enum               |
| `tolerance`                       | numerical tolerance Manifold is using                          |
| `original_id`                     | Manifold's tracking ID, or `-1`                                |
| `min_gap(other, search_length)`   | closest distance to another solid, capped at `search_length`   |

### Tolerance, simplification, properties

| Method                        | Notes                                                           |
| ----------------------------- | --------------------------------------------------------------- |
| `simplify(tolerance)`         | coarsen while keeping geometry within `tolerance`               |
| `set_tolerance(tol)`          | new mesh with updated tolerance                                 |
| `set_properties(num_prop, f)` | rewrite per-vertex property channels via callback               |
| `as_original()`               | mark the result as a fresh original (assigns a new tracking ID) |
| `compose_with(*others)`       | disjointly combine with other meshes (no boolean)               |

## Module-level helpers

For things that don't start from an existing mesh:

```python
from pyvista_manifold import level_set, extrude, revolve, hull_points

# Iso-surface from a scalar field
iso = level_set(f, bounds=(xmin, ymin, zmin, xmax, ymax, zmax), edge_length=0.1)

# Extrude / revolve a 2D polygon
solid = extrude(polygons, height, n_divisions=0, twist_degrees=0, scale_top=(1, 1))
solid = revolve(polygons, segments=0, revolve_degrees=360.0)

# Convex hull of a raw point cloud
hull = hull_points(points)  # (N, 3) array
```

`polygons` is a single `(N, 2)` array or a list of such arrays representing a polygon-with-holes set.

For everything that has an obvious PyVista equivalent (`pv.Cube`, `pv.Sphere`, `pv.Cylinder`, etc.), use PyVista directly and chain through `.manifold`.

## Conversion utilities

The accessor handles conversion automatically. Reach for these only when the accessor isn't enough:

```python
import pyvista as pv
from pyvista_manifold import to_manifold, from_manifold

m = to_manifold(polydata, point_data_keys=['scalar'])  # PolyData -> Manifold
poly = from_manifold(m, property_names=['scalar'])     # Manifold -> PolyData
```

Per-vertex point arrays can be passed through Manifold as extra property channels via `point_data_keys`. Manifold linearly interpolates them across boolean cuts, and `from_manifold` unpacks them back into `point_data`.

## Caveats

- **Inputs must be manifold solids** (closed, non-self-intersecting). Run `pv.PolyData.clean()` and check `mesh.manifold.is_valid` if you're unsure. PyVista's downloaded example meshes vary: `download_cow`, `download_horse`, `download_armadillo` are manifold; `download_bunny` (the Stanford scan) is not.
- **All faces are triangulated and merged** during conversion. The roundtrip preserves vertex coordinates for triangulated, deduplicated input but does not preserve cell-data arrays.
- **Coordinates are `float32`** inside Manifold. For double precision, call `to_manifold().to_mesh64()` directly.
- **Manifold has no built-in I/O.** Use PyVista's readers and writers on the resulting PolyData.

## Development

```bash
git clone https://github.com/pyvista/pyvista-manifold
cd pyvista-manifold
just sync          # uv sync --extra dev
just test          # pytest with coverage
just lint          # pre-commit run --all-files
just typecheck     # mypy
```

Image-regression tests run via [pytest-pyvista](https://github.com/pyvista/pytest-pyvista). To re-seed the cache after intentional visual changes:

```bash
uv run pytest tests/test_image_regression.py --reset_image_cache
```

The hero images at the top of this README are produced by `assets/render_hero.py`.

## Acknowledgements

- [Manifold](https://github.com/elalish/manifold) by Emmett Lalish and contributors.
- [PyVista](https://github.com/pyvista/pyvista) for the accessor system and the rest of the visualization stack.

## License

MIT.
