"""
Python-specific module: compare_jobs
Copyright 2026 ray-tracing-shapely authors and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

===============================================================================
COMPARE JOBS -- the two halves of a 2D-against-3D comparison, as jobs
===============================================================================
The comparison of `compare.compare_2d_3d`, built from profile files only and
cut into two jobs an executor can run anywhere the packages are installed
(`flows.executors`): plain data in, JSON out, nothing from the caller's
kernel.

    trace_2d_job   each ray's 2D main path through the profiles' sections,
                   and sheets, rebuilt from their files
    slice_2d_job   a rayfile's slice, every ``every``-th kept ray, its 2D main
                   path (`compare.slice_main_paths_2d`)
    slice_3d_job   the same records, unturned, in 3D
                   (`compare.trace_slice_records_3d`)
    trace_3d_job   each ray as a 3D pencil through the profiles' solids
                   (`RevolvedProfileSolid`, `ExtrudedPolygonSolid`) and
                   sheets (`RevolvedProfileSheet`)

Both take ``inputs["profiles"]``: a list of ``{"path", "name", "n"}`` for a
solid (its refractive index: a profile does not carry one) or ``{"path",
"name", "reflectance"}`` for a mirror sheet, with the component name its
faces are logged under. A solid's 2D section is rebuilt by
`BasePrism.from_profile`, a sheet's mirrors by `mirrors_from_sheet_profile`;
in 3D, a sheet is a `RevolvedProfileSheet`. `compare.compare_paths` puts
their results together into the per-ray table.

The 2D job needs no Optiland: a 2D hit is named after the profile's own
spans. A 2D mirror reflects everything: a sheet's reflectance below one
lowers the 3D share on the 2D path, not the path.
===============================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ray_tracing_shapely.core.scene import Scene
from ray_tracing_shapely.optical_elements.mirrors import mirrors_from_sheet_profile
from ray_tracing_shapely.optical_elements.prisms.base_prism import BasePrism

from .compare import MainPath2D, trace_2d_main_path, trace_pencils_3d

#: The symmetry kinds a solid profile is built in 3D by.
SOLID_KINDS = ("revolve", "extrude")


def read_profile(path: Any) -> Dict[str, Any]:
    """A profile file's JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_profile(profile: Mapping[str, Any], name: str, material: Optional[Mapping[str, Any]] = None) -> str:
    """What this comparison builds the profile as: ``"revolve"`` or
    ``"extrude"`` (a solid), or ``"sheet"`` (revolved mirrors); else
    ValueError, saying why. With ``material`` (the profile's entry), also
    that a solid is given ``n``, and ``reflectance`` too if it has mirror-coated
    (``reflect``) spans, and that a sheet is given ``reflectance`` only."""
    body = profile.get("body", "solid")
    kind = (profile.get("symmetry") or {}).get("kind")
    if body == "sheet":
        if kind != "revolve":
            raise ValueError(f"{name}: a sheet is built in 3D revolved, got symmetry {kind!r}")
        built = "sheet"
    elif body == "solid":
        if kind not in SOLID_KINDS:
            raise ValueError(f"{name}: symmetry {kind!r}; a solid is built in 3D from {list(SOLID_KINDS)}")
        built = kind
    else:
        raise ValueError(f"{name}: body {body!r}; a profile is a \"solid\" or a \"sheet\"")
    if material is not None:
        coated = [sp.get("short_label") for sp in profile.get("spans", []) if sp.get("mechanism") == "reflect"]
        if body == "sheet" and ("reflectance" not in material or "n" in material):
            raise ValueError(f"{name} is a sheet: give it \"reflectance\", not \"n\"")
        if body == "solid" and "n" not in material:
            raise ValueError(f"{name} is a solid: give it \"n\", not \"reflectance\" alone")
        if body == "solid" and coated and "reflectance" not in material:
            raise ValueError(f"{name} is a solid with mirror-coated spans {coated}: give it \"reflectance\" "
                             "as well as \"n\"")
        if body == "solid" and not coated and "reflectance" in material:
            raise ValueError(f"{name} is a solid with no mirror-coated span: drop \"reflectance\"")
    return built


def optics_2d(profiles: Sequence[Mapping[str, Any]]) -> Callable[[Scene], Dict[str, Any]]:
    """``compare_2d_3d``'s 2D optics: each profile rebuilt in the scene given (a sheet as its mirrors)."""
    def build(scene: Scene) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for p in profiles:
            if check_profile(p["profile"], p["name"], p) == "sheet":
                objs = mirrors_from_sheet_profile(scene, p["profile"])
            else:
                objs = [BasePrism.from_profile(scene, p["profile"], n=float(p["n"]))]
            for obj in objs:
                scene.add_object(obj)
            out[p["name"]] = objs
        return out
    return build


def components_3d(profiles: Sequence[Mapping[str, Any]]) -> List[Any]:
    """Each profile's 3D component: a solid in a glass of its index, a sheet of its reflectance."""
    from optiland.materials.ideal import IdealMaterial  # noqa: PLC0415
    from optiland.nonsequential import NSQMaterial  # noqa: PLC0415

    from .revolve import RevolvedProfileSheet, RevolvedProfileSolid, scene_placement  # noqa: PLC0415
    from .scene_bridge import ExtrudedPolygonSolid  # noqa: PLC0415

    out = []
    for p in profiles:
        built = check_profile(p["profile"], p["name"], p)
        if built == "sheet":
            out.append(RevolvedProfileSheet(p["name"], p["profile"], float(p["reflectance"]),
                                            cs=scene_placement(p["profile"])))
            continue
        glass = NSQMaterial(IdealMaterial(n=float(p["n"])))
        coated = {sp["short_label"]: float(p["reflectance"]) for sp in p["profile"]["spans"]
                  if sp.get("mechanism") == "reflect" and "reflectance" in p}
        if built == "revolve":
            out.append(RevolvedProfileSolid(p["name"], p["profile"], glass, cs=scene_placement(p["profile"]),
                                            reflectances=coated or None))
        else:
            out.append(ExtrudedPolygonSolid(p["name"], p["profile"], glass))
    return out


def _loaded(inputs: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [{**p, "profile": read_profile(p["path"])} for p in inputs["profiles"]]


def trace_2d_job(inputs: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Each ray's 2D main path: ``{"main": [MainPath2D.to_dict()]}``.

    Inputs: ``profiles``, ``rays`` (as ``compare_2d_3d`` takes them), ``max_rays_2d``.
    """
    profiles = _loaded(inputs)
    build = optics_2d(profiles)
    names = {p["name"]: SimpleNamespace(name=p["name"], profile=p["profile"]) for p in profiles}
    rays = inputs["rays"]
    main: List[MainPath2D] = []
    for k, ray in enumerate(rays):
        ctx.progress(k, len(rays))
        main.append(trace_2d_main_path(build, (ray["position"], ray["direction_deg"]),
                                       max_rays=int(inputs.get("max_rays_2d", 10_000)), components=names))
    ctx.progress(len(rays), len(rays))
    return {"main": [m.to_dict() for m in main]}


def trace_3d_job(inputs: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Each ray as a 3D pencil: ``{"pencils": [[PencilRay3D.to_dict()]]}``.

    Inputs: ``profiles``, ``rays``, ``rays_per_pencil``, ``max_depth``,
    ``wavelength_um``, ``seed``, ``backend`` (Optiland's tracer, ``"numpy"`` or
    ``"torch"``, float64 either way); and ``summary_only``: return, instead of
    every 3D ray, ``{"summary": {"rays", "by_end", "by_path"}}`` (how many
    ended each way, and the ten most common face sequences), with
    ``trace_s``, ``build_s`` (reading and building the components) and
    ``cpus`` (the CPUs the process may use; in a container, its host's) -- for a large trace whose rays are not needed
    back, or a measure of the trace's own rate.
    """
    import os  # noqa: PLC0415
    import time  # noqa: PLC0415

    import optiland.backend as be  # noqa: PLC0415

    backend = str(inputs.get("backend", "numpy"))
    if backend not in ("numpy", "torch"):
        raise ValueError(f"backend is 'numpy' or 'torch', got {backend!r}")
    before = (be.get_backend(), be.get_precision())
    try:
        if backend == "torch":                     # the components are built in the tracer's arrays
            be.set_backend("torch")
            be.set_precision("float64")
        t_build = time.perf_counter()
        components = components_3d(_loaded(inputs))
        build_s = time.perf_counter() - t_build
        rays = [(r["position"], r["direction_deg"]) for r in inputs["rays"]]
        t0 = time.perf_counter()
        pencils = trace_pencils_3d(components, rays, rays_per_pencil=int(inputs.get("rays_per_pencil", 8)),
                                   max_depth=int(inputs.get("max_depth", 30)),
                                   wavelength_um=float(inputs.get("wavelength_um", 0.55)),
                                   seed=int(inputs.get("seed", 1)), progress=ctx.progress, backend=backend)
        trace_s = time.perf_counter() - t0
    finally:
        be.set_backend(before[0])
        be.set_precision("float64" if before[1] == 64 else "float32")
    if not inputs.get("summary_only"):
        return {"pencils": [[r.to_dict() for r in recs] for recs in pencils]}
    by_end: Dict[str, int] = {}
    by_path: Dict[str, int] = {}
    for recs in pencils:
        for r in recs:
            by_end[r.end] = by_end.get(r.end, 0) + 1
            key = " > ".join(r.faces)
            by_path[key] = by_path.get(key, 0) + 1
    top = dict(sorted(by_path.items(), key=lambda kv: (-kv[1], kv[0]))[:10])
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {"summary": {"rays": sum(by_end.values()), "by_end": dict(sorted(by_end.items())), "by_path": top},
            "trace_s": trace_s, "build_s": build_s, "cpus": cpus}


def slice_of(source: Mapping[str, Any]) -> Tuple[Any, Any]:
    """A ``rayfile`` source spec's file (checked against its digest) and its slice, as the spec
    recorded it. The spec's ``rayfile.path`` must be absolute: a job may run anywhere."""
    from ray_tracing_shapely.optical_elements.sources import slice_rayfile  # noqa: PLC0415
    from ray_tracing_shapely.optical_elements.sources.source_spec import spec_rayfile  # noqa: PLC0415

    if "slice" not in source:
        raise ValueError(f"source {source.get('name')!r} has no slice: write it with the slicing flow")
    rayfile = spec_rayfile(source)
    slc = slice_rayfile(rayfile, tuple(source["position"][:2]), float(source["direction_deg"]),
                        max_offset_mm=float(source["slice"]["max_offset_mm"]), roll_deg=float(source["roll_deg"]))
    return rayfile, slc


def slice_2d_job(inputs: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Every ``every``-th kept ray of the source's slice, its 2D main path:
    ``{"pick": [...], "main": [MainPath2D.to_dict()]}``.

    Inputs: ``profiles``, ``source`` (a ``rayfile`` spec, its path absolute), ``every``, ``max_rays_2d``.
    """
    from .compare import slice_main_paths_2d  # noqa: PLC0415

    profiles = _loaded(inputs)
    names = {p["name"]: SimpleNamespace(name=p["name"], profile=p["profile"]) for p in profiles}
    _, slc = slice_of(inputs["source"])
    ctx.progress(0, 1, "tracing the slice in 2D")
    pick, main = slice_main_paths_2d(optics_2d(profiles), names, slc, every=int(inputs.get("every", 1)),
                                     max_rays_2d=int(inputs.get("max_rays_2d", 10_000)))
    ctx.progress(1, 1, "traced")
    return {"pick": [int(k) for k in pick], "main": [m.to_dict() for m in main]}


def slice_3d_job(inputs: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """The same records, unturned, in 3D, ``repeats`` rays each:
    ``{"records_3d": [[PencilRay3D.to_dict()]]}``, one list per picked ray.

    Inputs: ``profiles``, ``source``, ``every``, ``repeats``, ``max_depth``, ``seed``.
    """
    import numpy as np  # noqa: PLC0415

    from .compare import trace_slice_records_3d  # noqa: PLC0415

    components = components_3d(_loaded(inputs))
    source = inputs["source"]
    rayfile, slc = slice_of(source)
    pick = np.arange(0, len(slc.indices), max(1, int(inputs.get("every", 1))))
    ctx.progress(0, 1, "tracing the records in 3D")
    recs = trace_slice_records_3d(components, rayfile, slc.indices[pick], source["position"],
                                  float(source["direction_deg"]), roll_deg=float(source["roll_deg"]),
                                  repeats=int(inputs.get("repeats", 4)), max_depth=int(inputs.get("max_depth", 30)),
                                  seed=int(inputs.get("seed", 1)))
    ctx.progress(1, 1, "traced")
    return {"records_3d": [[r.to_dict() for r in rr] for rr in recs]}


__all__ = ["SOLID_KINDS", "check_profile", "components_3d", "optics_2d", "read_profile", "trace_2d_job",
           "slice_2d_job", "slice_3d_job", "slice_of", "trace_3d_job"]
