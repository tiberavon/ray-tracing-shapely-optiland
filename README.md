# ray-tracing-shapely-optiland

3D verification of [`ray_tracing_shapely`](https://github.com/micheleben/ray_tracing_shapely)
nonimaging-optics designs using [Optiland](https://github.com/optiland/optiland)'s
non-sequential ray tracer.

Scene assembly (`scene_bridge.py`: `build_faces_from_profile`, `ExtrudedPolygonSolid`,
reading the `profile.json` handoff), generic 2D/3D rendering (`visualization/renderers.py`,
registered against `ExtrudedPolygonSolid`), inline-Jupyter 3D viewing (`visualization
/jupyter_viewer.py`, via `trame`), and NSQ ray/run persistence into SQLite
(`nsq_db.py`, reusing `ray_tracing_shapely`'s own query layer). All four are implemented and
tested — see:

- `dev_doc/architecture/components/optiland_scene_bridge.md`
- `dev_doc/architecture/components/optiland_nsq_run_registry.md`
- `dev_doc/architecture/components/optiland_nsq_viewer.md`

One piece remains open: an `import_from_optiland` entry point, not yet built.

Distinct from the sibling `optiland-sms` package, which ships only the
entry-point-discovered SMS geometry plugin Optiland's own plugin loader finds
automatically — this package is found through `ray_tracing_shapely`'s own docs, not
Optiland's.

## Install (once there is code to install)

```bash
pip install -e ".[dev]"
# or, for the inline-notebook viewer:
pip install -e ".[notebook]"
```
