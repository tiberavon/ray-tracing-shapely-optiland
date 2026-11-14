"""
Python-specific module: compare
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
COMPARE -- the same meridional rays traced in 2D and in 3D, ray by ray
===============================================================================
A 2D design exported to 3D (``profile.json``, ``revolve``, ``plate``) should
trace the same in both engines for rays that stay in the 2D plane. This
module checks that, headless (no VTK):

    rays      each given in the 2D scene: a start point (a, b) and a
              direction in degrees
    2D        each traced alone, with surface hits recorded; at every split
              (a Fresnel reflection beside a refraction) the brighter child
              is followed: the ray's main path
    3D        each launched as a pencil source (the source spec's ``pencil``)
              at the mapped place, x = a, y = c, z = b; several Monte Carlo
              rays per pencil, split out of the event log (``events``)

Faces are named alike on both sides, as the event log names them:
``<component>.<label>``. A 2D hit is named by the component the caller says
the 2D object became, and by the span of that component's ``profile`` whose
outline lies nearest the hit point. A component without a profile (a
``PlanarPlate``) has one face, and every hit on its 2D object is that face.
The profile is the 3D component's own: when the component is not the 2D
object's export, a 2D hit is named after the nearest face it does have, and
the ray is reported as differing all the same.

Per ray (``Comparison.rays``; 2D values in the 2D scene, 3D values in the
world, means over the 3D rays that followed the 2D main path):
- whether 3D rays followed the 2D main path, and their share of the pencil
  (the rest are Fresnel ghosts, or a different path). The ray agrees when
  that share is above ``min_share``, by default when any did: Monte Carlo
  splits a pencil, so the 2D path need not be the commonest;
- the 3D path most rays took;
- the exit-direction and exit-point differences between the 2D main path
  and the 3D rays that followed it. The exit point is the last event, or,
  given a ``plane``, where the final ray crosses it.

The summary keeps the worst differences and lists only the rays whose
sequences differ: the per-ray table can be long (an agent gets the summary).
===============================================================================
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ray_tracing_shapely.core.scene import Scene
from ray_tracing_shapely.core.scene_objs.light_source.single_ray import SingleRay
from ray_tracing_shapely.core.simulator import Simulator

from .events import ESCAPED, split_rays
from .sources import build_source_3d, scene_to_world

RaySpec = Union[Mapping[str, Any], Tuple[Sequence[float], float]]
Optics2D = Callable[[Scene], Mapping[str, Any]]

#: Where the comparison puts the detector Optiland needs before it traces:
#: far out of the 2D plane, too small to meet.
_PARKED_DETECTOR = 1.0e9


def _ray(spec: RaySpec) -> Tuple[np.ndarray, float]:
    if isinstance(spec, Mapping):
        return np.asarray(spec["position"], dtype=float), float(spec["direction_deg"])
    point, angle = spec
    return np.asarray(point, dtype=float), float(angle)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# --- naming a 2D hit as the event log names a face ----------------------------


@dataclass
class _Namer:
    """2D object uuid -> how to name a hit on it."""

    by_uuid: Dict[str, Tuple[str, Optional[Mapping[str, Any]], Optional[str]]] = field(default_factory=dict)

    def name(self, uuid: Optional[str], point: Mapping[str, float]) -> str:
        if uuid not in self.by_uuid:
            return f"2d:{uuid}"
        component, profile, single = self.by_uuid[uuid]
        if profile is None:
            return single
        return f"{component}.{span_at(profile, (point['x'], point['y']))}"


def _into_profile_half(profile: Mapping[str, Any], pts: np.ndarray, p: np.ndarray) -> np.ndarray:
    """A revolved profile keeps one side of its axis: mirror ``p`` onto that side."""
    sym = profile.get("symmetry") or {}
    if sym.get("kind") != "revolve":
        return p
    o = np.asarray(sym["axis"]["point"], dtype=float)
    u = _unit(sym["axis"]["direction"])
    n = np.array([-u[1], u[0]])
    side = np.sign(np.mean((pts - o) @ n)) or 1.0
    along, across = float((p - o) @ u), float((p - o) @ n)
    return o + along * u + side * abs(across) * n


def _distance_to_polyline(p: np.ndarray, pts: np.ndarray) -> float:
    if len(pts) == 1:
        return float(np.linalg.norm(pts[0] - p))
    a, ab = pts[:-1], np.diff(pts, axis=0)
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300), 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None] * ab - p, axis=1).min())


def span_at(profile: Mapping[str, Any], point: Sequence[float]) -> str:
    """The short label of the profile span whose outline lies nearest ``point``.

    A span owns ``points[index_start:index_end]`` and its outline is the
    polyline through them; the joins between spans belong to neither, and a
    closed body's outline is not closed back along the axis. A revolved
    profile keeps one side of its axis, so a point on the other side is
    mirrored onto it first.
    """
    pts = np.asarray(profile["points"], dtype=float)
    p = _into_profile_half(profile, pts, np.asarray(point, dtype=float))
    best, label = math.inf, None
    for span in profile["spans"]:
        lo, hi = int(span["index_start"]), int(span["index_end"])
        if hi <= lo:
            continue
        d = _distance_to_polyline(p, pts[lo:hi])
        if d < best:
            best, label = d, span["short_label"]
    if label is None:
        raise ValueError("the profile has no span with points")
    return label


def _single_face(component) -> str:
    faces = [getattr(s, "name", None) for s in (getattr(component, "surfaces", None) or [])]
    faces = [f for f in faces if f]
    if len(faces) != 1:
        raise ValueError(f"component {component.name!r} has no profile and {len(faces)} faces: "
                         "a 2D hit on it cannot be named")
    return faces[0]


def _namer(objects: Mapping[str, Any], components: Mapping[str, Any]) -> _Namer:
    namer = _Namer()
    for name, objs in objects.items():
        if name not in components:
            raise ValueError(f"the 2D optics name {name!r}, which is not a 3D component "
                             f"(have: {sorted(components)})")
        comp = components[name]
        profile = getattr(comp, "profile", None)
        single = None if profile is not None else _single_face(comp)
        for obj in (objs if isinstance(objs, (list, tuple)) else [objs]):
            namer.by_uuid[obj._uuid] = (name, profile, single)
    return namer


# --- the 2D side ----------------------------------------------------------------


@dataclass(frozen=True)
class MainPath2D:
    """A 2D ray's main path: the faces it met, its end, and its last segment."""

    faces: Tuple[str, ...]
    end: str
    point: np.ndarray            # start of the last segment, (a, b)
    direction: np.ndarray        # its unit direction, (a, b)


def trace_2d_main_path(optics_2d: Optics2D, ray: RaySpec, *, max_rays: int = 10_000,
                       components: Mapping[str, Any]) -> MainPath2D:
    """Trace one ray alone in a fresh 2D scene and follow its brightest children."""
    scene = Scene()
    namer = _namer(optics_2d(scene), components)
    p, angle = _ray(ray)
    d = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
    scene.add_object(SingleRay(scene, {"p1": {"x": p[0], "y": p[1]},
                                       "p2": {"x": p[0] + d[0], "y": p[1] + d[1]}}))
    segments = Simulator(scene, max_rays=max_rays, record_surface_hits=True).run_with_result(name="compare").segments
    children = defaultdict(list)
    for sg in segments:
        children[sg.parent_uuid].append(sg)
    roots = children[None]
    if len(roots) != 1:
        raise RuntimeError(f"expected one 2D root ray, got {len(roots)}")
    sg, faces = roots[0], []
    while True:
        if sg.hit_surface_uuid is not None:
            faces.append(namer.name(sg.hit_surface_uuid, sg.hit_point))
        nxt = children.get(sg.uuid)
        if not nxt:
            break
        sg = max(nxt, key=lambda c: c.total_brightness)
    end = ESCAPED if sg.hit_surface_uuid is None else f"absorbed at {faces[-1]}"
    if sg.hit_surface_uuid is not None:
        point = np.array([sg.hit_point["x"], sg.hit_point["y"]])
    else:
        point = np.array([sg.p1["x"], sg.p1["y"]])
    direction = _unit([sg.p2["x"] - sg.p1["x"], sg.p2["y"] - sg.p1["y"]])
    return MainPath2D(tuple(faces), end, point, direction)


# --- the comparison -------------------------------------------------------------


def _on_plane(point, direction, plane_point, plane_normal) -> np.ndarray:
    denom = float(np.dot(direction, plane_normal))
    if abs(denom) < 1e-15:
        return np.full_like(np.asarray(point, dtype=float), np.nan)
    return point + float(np.dot(plane_point - point, plane_normal)) / denom * direction


@dataclass
class Comparison:
    """The per-ray table and what it adds up to."""

    rays: List[Dict[str, Any]]
    plane: Optional[Dict[str, Any]]

    def summary(self, max_listed: int = 20) -> Dict[str, Any]:
        """The worst differences, and the rays whose face sequences differ (at most ``max_listed``)."""
        agree = [r for r in self.rays if r["agree"]]
        differ = [r for r in self.rays if not r["agree"]]
        keys = ("index", "position", "direction_deg", "path_2d", "end_2d", "path_3d", "end_3d",
                "share_on_2d_path")
        return {
            "rays": len(self.rays),
            "agree": len(agree),
            "differ": len(differ),
            "worst_direction_difference": max((r["direction_difference"] for r in agree), default=None),
            "worst_point_difference_mm": max((r["point_difference_mm"] for r in agree), default=None),
            "least_share_on_2d_path": min((r["share_on_2d_path"] for r in self.rays), default=None),
            "rays_3d_on_2d_paths": sum(r["rays_3d_on_2d_path"] for r in self.rays),
            "exit_point": "on the plane" if self.plane else "at the last event",
            "plane": self.plane,
            "differing_rays": [{k: r[k] for k in keys} for r in differ[:max_listed]],
            "differing_rays_not_listed": max(0, len(differ) - max_listed),
        }


def compare_2d_3d(
    optics_2d: Optics2D,
    components: Sequence[Any],
    rays: Sequence[RaySpec],
    *,
    plane: Optional[Mapping[str, Any]] = None,
    rays_per_pencil: int = 8,
    max_depth: int = 30,
    wavelength_um: float = 0.55,
    seed: int = 1,
    max_rays_2d: int = 10_000,
    source_medium: Any = None,
    min_share: float = 0.0,
) -> Comparison:
    """Trace the same meridional rays in 2D and in 3D and compare them ray by ray.

    Args:
        optics_2d: Adds the 2D optics to the scene it is given and returns
            them by the name of the 3D component each became:
            ``{name: object}`` or ``{name: [objects]}`` (a revolved sheet made
            of two mirrors). Called once per ray, on a fresh scene.
        components: The 3D components, each with its ``name``; a bridge-built
            one keeps its ``profile``.
        rays: Each ``{"position": [a, b], "direction_deg": theta}`` or
            ``((a, b), theta)``, in the 2D scene.
        plane: ``{"point": [a, b], "normal_deg": phi}``: compare where the
            final rays cross this line of the 2D scene (a plane in 3D, through
            it and the out-of-plane axis). Default: at the last event.
        rays_per_pencil: Monte Carlo rays traced per 3D pencil.
        max_depth: Optiland's limit on events per ray.
        wavelength_um: The pencils' wavelength.
        seed: Optiland's seed.
        max_rays_2d: The 2D simulator's ray limit, per ray.
        source_medium: The Optiland material the 3D pencils start in, when
            the rays start inside a solid (an LED in its dome); default air.
        min_share: A ray agrees when more than this share of its pencil's 3D
            rays followed the 2D main path. Default 0: any. Fresnel ghosts,
            and a pencil starting on a face, take the rest.

    Returns:
        A ``Comparison``: ``.rays`` the per-ray table, ``.summary()`` the
        summary.

    Raises:
        ValueError: If a 2D name is not a component, or a component without a
            profile has more than one face.
        RuntimeError: If a pencil received no 3D ray.
    """
    from optiland.coordinate_system import CoordinateSystem  # noqa: PLC0415
    from optiland.nonsequential import IrradianceDetectorConfig, NSQScene  # noqa: PLC0415
    from optiland.nonsequential.backends.numpy_backend import NumpyBackend  # noqa: PLC0415

    by_name = {c.name: c for c in components}
    specs = [_ray(r) for r in rays]

    if plane is not None:
        phi = math.radians(float(plane["normal_deg"]))
        plane_2d = (np.asarray(plane["point"], dtype=float), np.array([math.cos(phi), math.sin(phi)]))
        plane_3d = (scene_to_world([*plane_2d[0], 0.0]), scene_to_world([*plane_2d[1], 0.0]))
        plane = {"point": [float(x) for x in plane_2d[0]], "normal_deg": float(plane["normal_deg"])}

    main = [trace_2d_main_path(optics_2d, (p, a), max_rays=max_rays_2d, components=by_name) for p, a in specs]

    scene = NSQScene()
    names = [f"__ray_{k}" for k in range(len(specs))]
    for name, (p, a) in zip(names, specs):
        cs, config = build_source_3d({"name": name, "kind": "pencil", "position": [float(p[0]), float(p[1])],
                                      "direction_deg": a, "wavelength_um": wavelength_um})
        if source_medium is not None:
            config = dataclasses.replace(config, medium=source_medium)
        scene.add_source(name, cs, config)
        scene.source_registry.get(name).name = name          # logged under it (KW-OPTILAND-09)
    for c in components:
        scene.component_registry.add(c.name, c)
    scene.add_detector("__parked", CoordinateSystem(y=_PARKED_DETECTOR),
                       IrradianceDetectorConfig(width=1e-6, height=1e-6, num_pixels_x=1, num_pixels_y=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = scene.trace(num_rays=rays_per_pencil * len(specs), seed=seed, max_depth=max_depth,
                             backend=NumpyBackend(seed=seed), record_paths=True)
    by_source = defaultdict(list)
    for rec in split_rays(result.ray_paths["events"]):
        by_source[rec.source].append(rec)

    table = []
    for k, ((p, a), m, name) in enumerate(zip(specs, main, names)):
        recs = by_source.get(name, [])
        if not recs:
            raise RuntimeError(f"ray {k} received no 3D ray: raise rays_per_pencil")
        key_2d = (m.faces, m.end)
        counts = Counter((r.faces, r.end) for r in recs)
        (faces_3d, end_3d), _ = counts.most_common(1)[0]
        on_path = [r for r in recs if (r.faces, r.end) == key_2d]
        d2 = scene_to_world([*m.direction, 0.0])
        p2 = scene_to_world([*m.point, 0.0])
        if plane is not None:
            p2 = _on_plane(p2, d2, *plane_3d)
        dd = dp = d3 = p3 = None
        if on_path:
            dirs = np.array([r.directions[-1] for r in on_path])
            pts = np.array([r.points[-1] if plane is None else _on_plane(r.points[-1], r.directions[-1], *plane_3d)
                            for r in on_path])
            dd = float(np.abs(dirs - d2).max())
            dp = float(np.linalg.norm(pts - p2, axis=1).max())
            d3, p3 = [float(x) for x in dirs.mean(axis=0)], [float(x) for x in pts.mean(axis=0)]
        table.append({
            "index": k, "position": [float(p[0]), float(p[1])], "direction_deg": a,
            "path_2d": list(m.faces), "end_2d": m.end,
            "path_3d": list(faces_3d), "end_3d": end_3d,
            "agree": len(on_path) / len(recs) > min_share,
            "share_on_2d_path": len(on_path) / len(recs), "rays_3d": len(recs), "rays_3d_on_2d_path": len(on_path),
            "direction_2d": [float(x) for x in m.direction], "exit_point_2d": [float(x) for x in p2[[0, 2]]],
            "direction_3d": d3, "exit_point_3d": p3,
            "direction_difference": dd, "point_difference_mm": dp,
        })
    return Comparison(table, plane)


# --- a rayfile slice against the rayfile's own rays ------------------------------


def compare_slice_3d(
    optics_2d: Optics2D,
    components: Sequence[Any],
    rayfile: Any,
    slc: Any,
    position: Sequence[float],
    direction_deg: float,
    *,
    roll_deg: float = 0.0,
    every: int = 1,
    repeats: int = 4,
    max_depth: int = 30,
    seed: int = 1,
    max_rays_2d: int = 10_000,
) -> List[Dict[str, Any]]:
    """Each kept ray of a slice in 2D, against its own record traced unturned in 3D.

    The 2D ray is the record turned about the axis with its offset h dropped
    (``slice_rayfile``); the 3D ray is the record itself, placed by
    ``rayfile_frame``. Their exit angles from the axis are compared: no turn
    about the axis changes that angle, so it differs only through h. A
    meridional record (h = 0) must agree to the tracing precision.

    Args:
        optics_2d, components, max_depth, seed, max_rays_2d: as ``compare_2d_3d``.
        rayfile, slc: The rayfile and its slice, taken with the same
            ``position``, ``direction_deg`` and ``roll_deg``.
        every: Compare every ``every``-th kept ray.
        repeats: 3D Monte Carlo rays per record.

    Returns:
        One dict per compared ray: ``record``, ``offset_mm`` (h), ``path_2d``,
        ``end_2d``, ``angle_2d_deg``, ``angle_3d_deg`` (mean over the 3D rays
        that met the same faces and ended the same way, or None),
        ``share_on_2d_path``.
    """
    from optiland.coordinate_system import CoordinateSystem  # noqa: PLC0415
    from optiland.nonsequential import IrradianceDetectorConfig, NSQScene  # noqa: PLC0415
    from optiland.nonsequential.backends.numpy_backend import NumpyBackend  # noqa: PLC0415

    from .sources import add_rayfile_source_3d  # noqa: PLC0415

    by_name = {c.name: c for c in components}
    pick = np.arange(0, len(slc.indices), max(1, int(every)))
    records = slc.indices[pick]
    t = math.radians(direction_deg)
    axis_2d = np.array([math.cos(t), math.sin(t)])
    axis_3d = scene_to_world([*axis_2d, 0.0])

    def from_axis(d) -> float:
        return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(d, axis_3d) / np.linalg.norm(d))))))

    main = [trace_2d_main_path(optics_2d, (slc.points[k], math.degrees(math.atan2(*slc.directions[k][::-1]))),
                               max_rays=max_rays_2d, components=by_name) for k in pick]

    scene = NSQScene()
    add_rayfile_source_3d(scene, "__rayfile", rayfile.subset(records), position, direction_deg, roll_deg)
    for c in components:
        scene.component_registry.add(c.name, c)
    scene.add_detector("__parked", CoordinateSystem(y=_PARKED_DETECTOR),
                       IrradianceDetectorConfig(width=1e-6, height=1e-6, num_pixels_x=1, num_pixels_y=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = scene.trace(num_rays=repeats * len(records), seed=seed, max_depth=max_depth,
                             backend=NumpyBackend(seed=seed), record_paths=True)
    by_record = defaultdict(list)
    for rec in split_rays(result.ray_paths["events"]):
        by_record[rec.ray_id % len(records)].append(rec)

    out = []
    for j, (k, m) in enumerate(zip(pick, main)):
        recs = by_record.get(j, [])
        on_path = [r for r in recs if (r.faces, r.end) == (m.faces, m.end)]
        out.append({
            "record": int(records[j]), "offset_mm": float(slc.offsets_mm[records[j]]),
            "path_2d": list(m.faces), "end_2d": m.end,
            "angle_2d_deg": from_axis(scene_to_world([*m.direction, 0.0])),
            "angle_3d_deg": (float(np.mean([from_axis(r.directions[-1]) for r in on_path])) if on_path else None),
            "share_on_2d_path": len(on_path) / len(recs) if recs else 0.0,
        })
    return out


__all__ = ["Comparison", "MainPath2D", "compare_2d_3d", "compare_slice_3d", "span_at", "trace_2d_main_path"]
