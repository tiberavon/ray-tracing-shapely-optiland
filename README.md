# ray-tracing-shapely-optiland

3D verification of [`ray_tracing_shapely`](https://github.com/tiberavon/ray-tracing-shapely)
designs with [Optiland](https://github.com/optiland/optiland)'s non-sequential ray tracer.
A 2D design is handed over as a `profile.json`: its outline, the role and mechanism of
every surface, and the light it is designed for. Here it becomes a 3D scene, is traced,
and comes back.

## What it does

- **Solids and sheets from a profile.**
  - **An extruded profile** becomes a solid with polygon end caps.
  - **A revolved profile** becomes a solid or a mirror of revolution, each surface the one
    exact Optiland geometry it describes: planes, cones, conics (zones and azimuth cuts
    included), and two this package adds, a spline of revolution and an even asphere of
    revolution.
  - **Every stored point and normal is checked** against the surface built from it.
  - **Built solids read back** to `profile.json` (`import_from_optiland`,
    `import_revolved_from_optiland`).
- **Sources.** A source specification from the profile is placed where the 2D scene had
  it. A measured LED's rayfile is traced ray for ray (`RayfileSource`).
- **2D against 3D.**
  - **`compare_2d_3d`** traces the same meridional rays in both engines and compares them
    face by face.
  - **`compare_slice_3d`** compares a rayfile's 2D slice with its own records traced in 3D.
- **A third engine.** A collimator's designed faces in Optiland's sequential mode, for an
  independent check.
- **What the light did.**
  - **Event logs** cut into one record per ray, and persisted to SQLite for queries.
  - **Irradiance maps** on a plane past the optic, plotted with their sections.
  - **Static 3D pictures,** rendered off-screen so that they survive a saved notebook,
    and an inline-Jupyter viewer.
- **Agent tools.** Read-only tools over a traced 3D scene and the 2D-against-3D comparison,
  registered into `ray_tracing_shapely`'s tool lists.

The components' own pages say what each part guarantees:

- `dev_doc/architecture/components/optiland_scene_bridge.md`
- `dev_doc/architecture/components/optiland_nsq_run_registry.md`
- `dev_doc/architecture/components/optiland_nsq_viewer.md`

Distinct from the sibling `optiland-sms` package, which is to ship only the
entry-point-discovered SMS geometry plugin Optiland's own plugin loader finds
automatically. This package is found through `ray_tracing_shapely`'s own docs, not
Optiland's.

## Install

Not yet on PyPI. From a checkout:

```bash
pip install -e ".[dev]"
# or, for the inline-notebook viewer:
pip install -e ".[notebook]"
```
