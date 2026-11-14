"""
Python-specific module: scene bridge
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

Python translation Copyright 2026 ray-tracing-shapely authors and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

===============================================================================
SCENE BRIDGE -- profile.json -> Optiland NSQ components
===============================================================================
Reads a profile.json (dev_doc/architecture/contracts/C-PROFILE-01.md, produced
by ray_tracing_shapely.optical_elements.prisms.profile_export) and builds the
Optiland non-sequential components it describes. No import of
ray_tracing_shapely anywhere in this module: profile.json is JSON, read as
plain dicts, matching the tool-independence C-PROFILE-01 already commits to.

Scope of this first implementation: 'extrude' symmetry only, with
extrude_axis == 'y' (what profile_export.py's own callers use today). A
'revolve' profile, or an 'extrude' along a different world axis, raises
NotImplementedError rather than silently building the wrong solid -- loud,
per dev_doc/architecture/02_conventions.md's failure_mode discipline.

Both directions live here. Out: build_faces_from_profile, face_from_span,
span_outward_ray, build_end_caps, ExtrudedPolygonSolid. Back:
import_from_optiland and interior_index, whose result
BasePrism.from_profile (core) rebuilds as a labelled 2D element
(C-BRIDGE-06). In between, check_end_cap_crossings reads a finished
trace for light that left an uncapped solid through its open ends
(C-BRIDGE-04).

Each face is placed from the profile's stored normal, not by hand
trigonometry on the corners. [KB7: the bridge exists so a 2D design can be
checked in 3D and handed back.]
===============================================================================
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import VACUUM
from optiland.nonsequential._utils import as_float
from optiland.nonsequential.components.absorbing import AbsorbingComponent
from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.components.compound import CompoundComponent
from optiland.nonsequential.components.geometry.analytic.plane import FinitePlaneGeometry
from optiland.nonsequential.components.reflective import ReflectiveComponent
from optiland.nonsequential.components.refractive import RefractiveComponent

from .geometry import PolygonPlaneGeometry


def _require_supported_symmetry(symmetry: Dict[str, Any]) -> None:
    """Raise unless ``symmetry`` is the extrusion the extruded half of the bridge builds.

    A ``revolve`` profile is built by revolve.py (build_faces_from_profile
    and import_from_optiland dispatch to it); the extrude-only helpers here
    refuse it.
    """
    if symmetry.get("kind") != "extrude":
        raise NotImplementedError(
            f"this part of scene_bridge builds symmetry kind='extrude', got "
            f"{symmetry.get('kind')!r}. A 'revolve' profile goes through "
            "build_faces_from_profile or RevolvedProfileSolid (revolve.py); "
            "'free' is not supported (C-BRIDGE-02)."
        )
    if symmetry.get("extrude_axis") != "y":
        raise NotImplementedError(
            f"scene_bridge only supports extrude_axis='y' today, got "
            f"{symmetry.get('extrude_axis')!r}. Extend _point_to_world to "
            "support other plane_axes/extrude_axis combinations if needed."
        )
    if symmetry.get("plane_axes") != ["x", "z"]:
        raise NotImplementedError(
            f"scene_bridge only supports plane_axes=['x', 'z'] today, got "
            f"{symmetry.get('plane_axes')!r}."
        )


def _point_to_world(point2d, symmetry: Dict[str, Any]) -> tuple:
    """Map a profile.json 2D (a, b) point to world (x, y, z).

    Only the plane_axes=['x','z'], extrude_axis='y' case is implemented
    (see _require_supported_symmetry) -- world y is always 0 here; callers
    needing depth placement (e.g. an end cap) add their own y offset.
    """
    a, b = point2d
    return (a, 0.0, b)


def face_from_span(
    span: Dict[str, Any],
    points,
    normals,
    symmetry: Dict[str, Any],
    interior_material,
    material_overrides: Optional[Dict[str, Any]] = None,
    depth: Optional[float] = None,
    reflectances: Optional[Dict[str, Any]] = None,
):
    """Build one Optiland NSQ component from one profile.json span.

    Args:
        span: One entry of profile["spans"] -- must have index_start,
            index_end (exactly 2 apart; a curved multi-point span is not
            yet supported), short_label, role, mechanism.
        points: profile["points"].
        normals: profile["normals"].
        symmetry: profile["symmetry"] -- see _require_supported_symmetry
            for what is actually implemented.
        interior_material: NSQMaterial for the solid's interior (this
            project's "prism glass"), used as material_back for every
            refractive/reflective face and as material_front for
            AbsorbingComponent faces (the outside of an unused edge is
            still the interior glass on its inner side; VACUUM would be
            wrong there).
        material_overrides: Optional {short_label: NSQMaterial} for a
            span whose *outside* material is not VACUUM -- e.g. a
            refractometer's measuring surface, whose outside is whatever
            sample is being read, not air.
        depth: Extrusion length along the world extrude_axis. Defaults to
            symmetry['length'].
        reflectances: {short_label: reflectance} for every span whose
            mechanism is ``reflect`` -- a constant in [0, 1], a callable of
            wavelength, or an unpolarised Optiland coating, as
            ReflectiveComponent takes it. profile.json does not record a
            reflectance, and Optiland requires one rather than assuming a
            perfect mirror, so a reflect span without an entry raises.

    Returns:
        A RefractiveComponent, ReflectiveComponent, or AbsorbingComponent,
        named after the span's short_label.

    Raises:
        NotImplementedError: For an unsupported symmetry (see
            _require_supported_symmetry), or a span spanning more than
            two points (a curved profile -- not yet supported).
        ValueError: For a span whose declared mechanism this bridge does
            not recognise -- loud rather than silently dropping the face.
    """
    _require_supported_symmetry(symmetry)
    if span["index_end"] - span["index_start"] != 2:
        raise NotImplementedError(
            f"Span {span['short_label']!r} has "
            f"{span['index_end'] - span['index_start']} points; only "
            "straight two-point spans are supported today (C-BRIDGE-02)."
        )
    depth = symmetry["length"] if depth is None else depth
    material_overrides = material_overrides or {}

    p1 = points[span["index_start"]]
    p2 = points[span["index_start"] + 1]
    normal = normals[span["index_start"]]  # noqa: F841 -- POSSIBLE DEFECT, left for review: the component
    # orientation below is derived from the two points (phi_edge), not from
    # this normal, and material_front/material_back depend on which side is
    # outside. Either the winding is guaranteed consistent and the normal is
    # redundant, or a span could be built with its faces swapped.

    x1, _, z1 = _point_to_world(p1, symmetry)
    x2, _, z2 = _point_to_world(p2, symmetry)
    mid_x, mid_z = (x1 + x2) / 2.0, (z1 + z2) / 2.0
    edge_len = ((x2 - x1) ** 2 + (z2 - z1) ** 2) ** 0.5

    phi_edge = math.atan2(z2 - z1, x2 - x1)
    cs = CoordinateSystem(x=mid_x, y=0.0, z=mid_z, ry=-phi_edge)
    geom = FinitePlaneGeometry(width=edge_len, height=depth)

    label = span["short_label"]
    outside = material_overrides.get(label, VACUUM)
    return _component_for_mechanism(
        span["mechanism"],
        cs,
        geom,
        outside,
        interior_material,
        _span_name(span),
        reflectance=(reflectances or {}).get(label),
    )


def _span_name(span: Dict[str, Any]) -> str:
    """The component name face_from_span gives a span: its short label, or a
    positional fallback for an unlabelled one."""
    return span["short_label"] or f"span_{span['index_start']}"


def _component_for_mechanism(
    mechanism, cs, geom, outside, interior_material, name: str, reflectance=None,
    interior_on_back: bool = True,
):
    """Build the NSQ component a declared mechanism maps to (C-BRIDGE-01).

    Every NSQ geometry's n_geom points from material_front to material_back.
    ``interior_on_back`` says whether it points into the solid (true for
    every extruded face, whose local +z is built to point inwards; decided
    per face from the stored normal for a revolved one). A ``reflect``
    mechanism needs ``reflectance``; there is no default.
    """
    if mechanism in ("refract", "tir"):
        front, back = (outside, interior_material) if interior_on_back else (interior_material, outside)
        return RefractiveComponent(
            cs, geom, material_front=front, material_back=back, name=name
        )
    if mechanism == "reflect":
        if reflectance is None:
            raise ValueError(
                f"Span {name!r} reflects but no reflectance was given -- pass "
                f"reflectances={{{name!r}: ...}}. Optiland will not assume a "
                "perfect mirror, and neither does this bridge."
            )
        return ReflectiveComponent(
            cs, geom, reflectance=reflectance, material_front=outside, name=name
        )
    if mechanism is None:
        return AbsorbingComponent(cs, geom, material_front=interior_material, name=name)
    raise ValueError(
        f"Span {name!r} has unrecognised mechanism {mechanism!r} -- "
        "expected 'refract', 'tir', 'reflect', or null."
    )


def build_faces_from_profile(
    profile: Dict[str, Any],
    interior_material,
    material_overrides: Optional[Dict[str, Any]] = None,
    reflectances: Optional[Dict[str, Any]] = None,
):
    """Build one Optiland NSQ component per span in a profile.json dict.

    Args:
        profile: A profile.json-shaped dict (see
            ray_tracing_shapely.optical_elements.prisms.profile_export
            .export_profile, or any tool-independent producer of the same
            schema).
        interior_material: See face_from_span.
        material_overrides: See face_from_span.
        reflectances: See face_from_span.

    Returns:
        dict mapping each span's short_label to its built component, in
        profile["spans"] order -- pass ``list(result.values())`` to
        ``scene.add_component`` per component, or iterate the dict
        directly to also keep each name. A ``revolve`` profile is built by
        revolve.build_revolved_faces (one analytic surface of revolution
        per half-section span); prefer RevolvedProfileSolid, which also
        validates the closed solid.
    """
    if profile["symmetry"].get("kind") == "revolve":
        from .revolve import build_revolved_faces

        components, _ = build_revolved_faces(
            profile, interior_material, material_overrides, reflectances
        )
        return components
    return {
        span["short_label"]: face_from_span(
            span,
            profile["points"],
            profile["normals"],
            profile["symmetry"],
            interior_material,
            material_overrides=material_overrides,
            reflectances=reflectances,
        )
        for span in profile["spans"]
    }


def span_outward_ray(profile: Dict[str, Any], short_label: str, distance: float):
    """World (position, coordinate_system) for a source/detector placed
    ``distance`` outside the named span's midpoint, aimed straight through
    the solid along that span's own outward normal.

    Reads the normal straight from profile.json rather than re-deriving it
    by trigonometry (contrast notebooks/optiland_nsq_refractometer_intro
    .ipynb's ``outward_normal_angle`` helper, which had to compute this by
    hand because it had no profile.json to read it from) -- this is the
    concrete piece of by-hand bookkeeping this bridge removes.

    Returns:
        (CoordinateSystem, ry) -- a coordinate system positioned outside
        the span along its outward normal, with ry set so local +z points
        back into the solid (matching CollimatedSourceConfig's "local +z
        is the propagation axis" contract); ry is also returned standalone
        since a detector wants the same rotation but does not need to be
        re-derived.
    """
    _require_supported_symmetry(profile["symmetry"])
    span = next(s for s in profile["spans"] if s["short_label"] == short_label)
    p1 = profile["points"][span["index_start"]]
    p2 = profile["points"][span["index_start"] + 1]
    nx, nz = profile["normals"][span["index_start"]]

    x1, _, z1 = _point_to_world(p1, profile["symmetry"])
    x2, _, z2 = _point_to_world(p2, profile["symmetry"])
    mid_x, mid_z = (x1 + x2) / 2.0, (z1 + z2) / 2.0

    pos_x = mid_x + distance * nx
    pos_z = mid_z + distance * nz
    phi_outward = math.atan2(nz, nx)
    # Edge direction = outward normal rotated +90 degrees (the outward-normal
    # formula in profile_export._edge_outward_normal is the inverse of this:
    # normal = edge direction rotated -90 degrees). A face's own local +z
    # points into the interior when ry = -phi_edge (verified against Optiland
    # directly; end to end, a wrong sign would light the base span, which
    # tests/test_scene_bridge.py::test_only_base_span_is_unreached forbids);
    # a source wants the same "into the interior" aim.
    phi_edge = phi_outward + math.pi / 2
    ry = -phi_edge
    return CoordinateSystem(x=pos_x, y=0.0, z=pos_z, ry=ry), ry


# --- End caps ---------------------------------------------------------------

#: Accepted values of ExtrudedPolygonSolid's ``end_caps``, and the span
#: mechanism each builds as (None builds an absorbing face).
END_CAP_MECHANISMS: Dict[str, Optional[str]] = {
    "refract": "refract",
    "absorb": None,
}

#: Component names of the two caps, at world y = -length/2 and +length/2.
END_CAP_NAMES: Tuple[str, str] = ("end_cap_neg", "end_cap_pos")


def _profile_polygon(profile: Dict[str, Any]) -> List[Tuple[float, float]]:
    """The profile's outline as a vertex list, in profile 2D coordinates.

    Each straight span contributes its start point; consecutive spans share
    their end and start points, so this is the closed polygon. Curved spans
    raise, as everywhere else in this module (C-BRIDGE-02).
    """
    vertices = []
    for span in profile["spans"]:
        if span["index_end"] - span["index_start"] != 2:
            raise NotImplementedError(
                f"Span {span['short_label']!r} is not a straight two-point span; "
                "the outline polygon is only defined for straight spans (C-BRIDGE-02)."
            )
        a, b = profile["points"][span["index_start"]]
        vertices.append((float(a), float(b)))
    return vertices


def build_end_caps(
    profile: Dict[str, Any],
    interior_material,
    mechanism: str,
) -> Dict[str, Any]:
    """Build the two flat caps that close an extruded profile into a solid.

    Each cap is a PolygonPlaneGeometry with exactly the profile's outline, at
    world y = -length/2 and +length/2, oriented so its local +z points into
    the solid like every side face. Built from the same points as the side
    faces, so cap and side edges coincide exactly.

    Args:
        profile: A profile.json dict with a supported extrude symmetry.
        interior_material: NSQMaterial inside the solid.
        mechanism: One of END_CAP_MECHANISMS. ``"refract"`` models a polished
            glass end face (light can leave, or TIR back in); ``"absorb"``
            models a blackened end, and books any light reaching it as
            absorbed at a named component, where a flux budget can see it.
            Outside each cap is VACUUM.

    Returns:
        {name: component} for the two caps, keyed by END_CAP_NAMES.

    Raises:
        ValueError: If ``mechanism`` is not one of END_CAP_MECHANISMS.
        NotImplementedError: For an unsupported symmetry or a curved span.
    """
    if mechanism not in END_CAP_MECHANISMS:
        raise ValueError(
            f"end cap mechanism must be one of {sorted(END_CAP_MECHANISMS)}, got {mechanism!r}."
        )
    symmetry = profile["symmetry"]
    _require_supported_symmetry(symmetry)
    half = float(symmetry["length"]) / 2.0
    polygon_world = [_point_to_world(v, symmetry) for v in _profile_polygon(profile)]

    caps: Dict[str, Any] = {}
    # rx = +pi/2 turns local +z to world -y (inward at the +y cap); rx = -pi/2
    # turns it to world +y (inward at the -y cap).
    for name, y, rx in (
        (END_CAP_NAMES[0], -half, -math.pi / 2),
        (END_CAP_NAMES[1], half, math.pi / 2),
    ):
        cs = CoordinateSystem(x=0.0, y=y, z=0.0, rx=rx)
        translation, rotation = _get_transform(cs)
        local = (np.asarray(polygon_world) - translation) @ rotation
        geom = PolygonPlaneGeometry(local[:, :2])
        caps[name] = _component_for_mechanism(
            END_CAP_MECHANISMS[mechanism], cs, geom, VACUUM, interior_material, name
        )
    return caps


# --- Post-trace containment check --------------------------------------------


def _points_in_polygon(xs: np.ndarray, ys: np.ndarray, polygon) -> np.ndarray:
    """Even-odd containment of (xs, ys) in a polygon, NumPy only."""
    inside = np.zeros(xs.shape, dtype=bool)
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        if y1 == y2:
            continue
        straddles = (ys < y1) != (ys < y2)
        x_cross = x1 + (ys - y1) * ((x2 - x1) / (y2 - y1))
        inside ^= straddles & (xs < x_cross)
    return inside


def check_end_cap_crossings(
    events: np.ndarray,
    profile: Dict[str, Any],
    *,
    warn: bool = True,
    tol: float = 1e-9,
) -> Dict[str, Any]:
    """Find rays that passed through where an extruded solid's end caps are.

    The open-ended solid's failure mode (C-BRIDGE-04) is silent: a ray drifts
    along the extrusion axis, leaves through the missing end, and neither
    flux conservation nor the medium stack notices. This reads the recorded
    event log instead. A ray's path between two consecutive events is a
    straight segment; if that segment crosses the plane y = +length/2 or
    -length/2 at a point inside the profile outline, the ray went through a
    cap position without anything being there to interact with it. A ray
    that escaped (a ``death`` event named ``escaped``) is extended from its
    last position along its last direction, so leaving through the open end
    into empty space is caught too.

    With end caps built, a ray reaching a cap produces a hit event on the
    cap plane itself, so no segment crosses it strictly and this reports
    nothing -- which makes it a check that the caps work, as well as a
    detector for an uncapped solid.

    Limits. Only recorded rays are seen: with ``record_paths`` set to a
    subset size, a crossing among the unrecorded rays is missed. A child ray
    spawned by bounded splitting starts at its first logged event, so the
    segment from its parent's hit point is not examined.

    Args:
        events: ``result.ray_paths["events"]``, Optiland's structured event
            array (fields ray_id, event_type, x, y, z, L, M, N, flux,
            component_name, ...), in the order it was logged.
        profile: The profile.json dict the solid was built from.
        warn: Emit a UserWarning when any crossing is found.
        tol: Distance from a cap plane within which an event counts as on
            it rather than across it, in the profile's length unit.

    Returns:
        ``{"crossings": int, "ray_ids": list[int], "flux": float}`` -- the
        number of cap-plane crossings found, the distinct rays making them,
        and the summed flux those rays carried at the start of each
        crossing segment.

    Raises:
        NotImplementedError: For an unsupported symmetry or a curved span.
    """
    symmetry = profile["symmetry"]
    _require_supported_symmetry(symmetry)
    half = float(symmetry["length"]) / 2.0
    polygon = _profile_polygon(profile)

    ev = np.asarray(events)
    if ev.size == 0:
        return {"crossings": 0, "ray_ids": [], "flux": 0.0}
    ev = ev[np.argsort(ev["ray_id"], kind="stable")]
    xyz = np.stack([ev["x"], ev["y"], ev["z"]], axis=1)

    same_ray = ev["ray_id"][1:] == ev["ray_id"][:-1]
    starts = xyz[:-1][same_ray]
    ends = xyz[1:][same_ray]
    seg_ids = ev["ray_id"][:-1][same_ray]
    seg_flux = ev["flux"][:-1][same_ray]

    escaped = (ev["event_type"] == "death") & (ev["component_name"] == "escaped")
    esc_pos = xyz[escaped]
    esc_dir = np.stack([ev["L"][escaped], ev["M"][escaped], ev["N"][escaped]], axis=1)
    esc_ids = ev["ray_id"][escaped]
    esc_flux = ev["flux"][escaped]

    hit_ids: List[np.ndarray] = []
    hit_flux: List[np.ndarray] = []
    for plane_y in (-half, half):
        d0 = starts[:, 1] - plane_y
        d1 = ends[:, 1] - plane_y
        across = (d0 * d1 < 0.0) & (np.abs(d0) > tol) & (np.abs(d1) > tol)
        s = np.where(across, d0 / np.where(across, d0 - d1, 1.0), 0.0)
        cross = starts + s[:, None] * (ends - starts)
        through = across & _points_in_polygon(cross[:, 0], cross[:, 2], polygon)
        hit_ids.append(seg_ids[through])
        hit_flux.append(seg_flux[through])

        dy = esc_dir[:, 1]
        moving = np.abs(dy) > 1e-15
        t = np.where(moving, (plane_y - esc_pos[:, 1]) / np.where(moving, dy, 1.0), -1.0)
        ahead = moving & (t > tol)
        cross = esc_pos + np.where(ahead, t, 0.0)[:, None] * esc_dir
        through = ahead & _points_in_polygon(cross[:, 0], cross[:, 2], polygon)
        hit_ids.append(esc_ids[through])
        hit_flux.append(esc_flux[through])

    all_ids = np.concatenate(hit_ids)
    report = {
        "crossings": int(all_ids.size),
        "ray_ids": sorted(int(i) for i in np.unique(all_ids)),
        "flux": float(np.concatenate(hit_flux).sum()),
    }
    if warn and report["crossings"]:
        warnings.warn(
            f"{report['crossings']} ray segment(s) from {len(report['ray_ids'])} ray(s) "
            f"passed through an end-cap position of an uncapped extrusion "
            f"(flux {report['flux']:.3g}); the trace's flux conservation does not "
            "see this. Build the solid with end_caps=... or keep sources in the "
            "profile plane (C-BRIDGE-04).",
            UserWarning,
            stacklevel=2,
        )
    return report


# --- Import: back from Optiland to profile.json ------------------------------


def import_from_optiland(
    solid: "ExtrudedPolygonSolid",
    *,
    in_plane_tol: float = 1e-9,
) -> Dict[str, Any]:
    """Read a live ExtrudedPolygonSolid back out as a profile.json dict.

    The reverse direction of this bridge. Points, normals and the extrusion
    length are re-derived from the live faces -- each face's width and
    coordinate system -- so a face moved or resized after export (by hand,
    or by an optimiser) comes back moved. Everything the geometry cannot
    carry is taken from the profile the solid was built from: span labels,
    roles, mechanisms, the symmetry declaration's axes, construction
    parameters. That is the point of going through profile.json rather than
    a bare vertex list: the device comes back labelled (C-LABEL-02), and
    BasePrism.from_profile can rebuild it in 2D under the same names.

    The returned dict is a new object; ``solid.profile`` is not modified. End
    caps are not part of a profile and are ignored.

    Args:
        solid: The solid to read.
        in_plane_tol: How far, in the profile's length unit, a face may sit
            out of the profile plane (world y = 0) or tilt its edge out of it
            before the solid is refused as no longer an extrusion of a 2D
            outline.

    Returns:
        A profile.json-shaped dict (C-PROFILE-01).

    Raises:
        NotImplementedError: For an unsupported symmetry or a curved span.
        ValueError: If a face has left the profile plane, its local y axis is
            no longer the extrusion axis, or the faces disagree on the
            extrusion length.
    """
    from .revolve import _RevolvedProfileBase, import_revolved_from_optiland

    if isinstance(solid, _RevolvedProfileBase):     # a RevolvedProfileSolid or RevolvedProfileSheet
        return import_revolved_from_optiland(solid)
    profile = copy.deepcopy(solid.profile)
    _require_supported_symmetry(profile["symmetry"])

    points: List[List[float]] = []
    normals: List[List[float]] = []
    lengths: List[float] = []
    for span in profile["spans"]:
        if span["index_end"] - span["index_start"] != 2:
            raise NotImplementedError(
                f"Span {span['short_label']!r} is not a straight two-point span (C-BRIDGE-02)."
            )
        name = _span_name(span)
        face = solid.face(name)
        translation, rotation = _get_transform(face.cs)
        edge_dir, extrude_dir, inward = rotation[:, 0], rotation[:, 1], rotation[:, 2]
        if abs(translation[1]) > in_plane_tol or abs(edge_dir[1]) > in_plane_tol:
            raise ValueError(
                f"face {name!r} is no longer in the profile plane (centre y = "
                f"{translation[1]:.3g}, edge direction y = {edge_dir[1]:.3g}); it "
                "cannot be read back as a 2D outline."
            )
        if abs(abs(extrude_dir[1]) - 1.0) > in_plane_tol:
            raise ValueError(
                f"face {name!r} has its local y axis at {extrude_dir.tolist()}, not "
                "along the extrusion axis."
            )
        half_width = as_float(face.geometry.width) / 2.0
        p1 = translation - half_width * edge_dir
        p2 = translation + half_width * edge_dir
        outward = np.array([-inward[0], -inward[2]])
        outward /= np.linalg.norm(outward)

        span["index_start"] = len(points)
        points.append([float(p1[0]), float(p1[2])])
        points.append([float(p2[0]), float(p2[2])])
        normals.append([float(outward[0]), float(outward[1])])
        normals.append([float(outward[0]), float(outward[1])])
        span["index_end"] = len(points)
        lengths.append(as_float(face.geometry.height))

    if max(lengths) - min(lengths) > in_plane_tol:
        raise ValueError(
            f"faces disagree on the extrusion length: {sorted(set(lengths))}."
        )
    profile["points"] = points
    profile["normals"] = normals
    profile["symmetry"]["length"] = lengths[0]
    return profile


def interior_index(solid: "ExtrudedPolygonSolid", wavelength_um: float = 0.55) -> float:
    """The refractive index of the solid's interior material at one wavelength.

    Read from the faces themselves: a refracting face's interior side
    (material_back for an extruded face; for a revolved one, whichever side
    its stored normal put the glass on), an absorbing face's material_front
    (reflecting faces carry no interior material). This is the ``n`` BasePrism.from_profile needs, which
    profile.json does not record.

    Args:
        solid: The solid to read.
        wavelength_um: Wavelength in micrometres -- Optiland's unit.

    Returns:
        The index, as a float.

    Raises:
        ValueError: If no face carries an interior material, or faces
            disagree on its index.
    """
    from .revolve import interior_side_material

    indices = []
    for face in solid.surfaces:
        if isinstance(face, RefractiveComponent):
            material = interior_side_material(solid, face)
        elif isinstance(face, AbsorbingComponent):
            material = face.material_front
        else:
            continue
        indices.append(as_float(material.n(wavelength_um)))
    if not indices:
        raise ValueError(f"no face of {solid.name!r} carries an interior material.")
    if max(indices) - min(indices) > 1e-12:
        raise ValueError(
            f"faces of {solid.name!r} disagree on the interior index at "
            f"{wavelength_um} um: {sorted(set(indices))}."
        )
    return indices[0]


class ExtrudedPolygonSolid(CompoundComponent):
    """A CompoundComponent wrapping build_faces_from_profile's output.

    Without this wrapper,
    the faces build_faces_from_profile returns are only ever added to a
    scene one at a time via scene.add_component() (each auto-wrapped in an
    anonymous single-surface compound Optiland's own add_component()
    creates internally), which is why nothing in the renderer registry
    could ever recognise "this project's device" as a type to register a
    renderer against -- the registry is keyed by compound-component
    *class* (viewer_3d.py's own _register_default_renderers), and an
    anonymous per-face wrapper has no stable class of its own to key on.

    Register an instance directly with the scene's component registry --
    NOT via scene.add_component(), which is for a single raw surface:

        solid = ExtrudedPolygonSolid("prism", profile, glass, {"M": sample})
        scene.component_registry.add(solid.name, solid)

    Does not attempt Volume() watertightness validation, deliberately:
    the pilot found that a mesh-capped boundary (the only way to close an
    arbitrary polygon extrusion today, absent a dedicated polygon-plane
    geometry) can never pass that check --
    MeshGeometry contributes no rim points to Optiland's own
    _rim_points(), structurally, not as a defect in this project's faces.
    Attempting and silently swallowing that failure would be worse than
    not attempting it; this class does neither -- it simply does not
    claim a validated closed solid, only a set of correctly-placed faces.

    **This is not a theoretical gap -- it was reproduced on purpose**
    [KD3: the tracer reports clean conservation because the escaped rays
    are simply gone]. By default there are no end caps (not even the mesh
    caps the pilot tried): the solid is open at both ends of the extrusion. Every ray built via
    span_outward_ray() stays in the world x-z plane for its entire life
    (only `ry` is ever set), so today's results never approach the open
    ends -- but that safety is an invariant of *how the source/detector
    helpers are used*, not something this class checks. A source with any
    world-y component (a tilt, a divergent or point source, a genuinely
    3D device) can produce rays that drift past the nominal cap position
    over several bounces and re-strike E/M/X from a position/direction a
    real closed solid would never permit -- with
    SimulationResult.flux_conservation_error and
    Diagnostics.medium_stack_underflows both reporting a clean trace
    regardless (the reproduction: a 0.15 rad tilt, 37 of 20,000 rays
    escaped, one full event history examined). See C-BRIDGE-04 before
    relying on this class for any source with off-plane motion.

    Two remedies, both opt-in so existing scenes trace exactly as before:

    - ``end_caps="refract" | "absorb"`` closes the solid with two
      PolygonPlaneGeometry caps cut to the profile outline (build_end_caps).
      Rays then meet a real surface at the ends instead of leaving.
      Optiland's own Volume() check still cannot validate the result -- it
      sees the cap as its bounding rectangle (see geometry.py) -- so closure
      is by construction: caps and side faces share the same points.
    - ``check_containment(result)`` reads the trace's event log afterwards
      and reports any ray that passed through a cap position
      (check_end_cap_crossings). Use it on any uncapped solid whose sources
      are not all in the profile plane.

    Attributes:
        profile: The profile.json dict this solid was built from -- kept
            for renderers (ExtrudedPolygonRenderer2D/3D) to read the
            original 2D cross-section from, rather than reconstructing it
            from the built faces' geometry.
    """

    def __init__(
        self,
        name: str,
        profile: Dict[str, Any],
        interior_material,
        material_overrides: Optional[Dict[str, Any]] = None,
        end_caps: Optional[str] = None,
        reflectances: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Build every span's face via build_faces_from_profile.

        Args:
            name: Registry name for this compound.
            profile: See build_faces_from_profile.
            interior_material: See build_faces_from_profile.
            material_overrides: See build_faces_from_profile.
            end_caps: None (the default) leaves both ends open, as before.
                ``"refract"`` or ``"absorb"`` closes them -- see
                build_end_caps for what each models.
            reflectances: See face_from_span.

        Raises:
            ValueError: If ``end_caps`` is not None or one of
                END_CAP_MECHANISMS, or a span's label collides with a cap's
                name.
        """
        _require_supported_symmetry(profile["symmetry"])
        self._name = name
        self.profile = profile
        self._faces = build_faces_from_profile(
            profile,
            interior_material,
            material_overrides=material_overrides,
            reflectances=reflectances,
        )
        self._caps: Dict[str, Any] = {}
        if end_caps is not None:
            clash = set(self._faces) & set(END_CAP_NAMES)
            if clash:
                raise ValueError(f"span label(s) {sorted(clash)} collide with end-cap names.")
            self._caps = build_end_caps(profile, interior_material, end_caps)
        self.end_caps = end_caps
        # Identity: profile.json's points are already in the world frame
        # this solid was placed in (there is no separate "local" frame a
        # straight-edge profile needs -- unlike Lens's front-vertex CS,
        # there is no natural single vertex to anchor on here).
        self._cs = CoordinateSystem()

    @property
    def name(self) -> str:
        return self._name

    @property
    def surfaces(self):
        return list(self._faces.values()) + list(self._caps.values())

    @property
    def coordinate_system(self) -> CoordinateSystem:
        return self._cs

    def face(self, short_label: str):
        """Return one named face (e.g. ``solid.face("M")``), or an end cap by
        its END_CAP_NAMES name. An unlabelled span is found by the name
        face_from_span gave it."""
        if short_label in self._faces:
            return self._faces[short_label]
        if short_label in self._caps:
            return self._caps[short_label]
        for surface in self._faces.values():
            if surface.name == short_label:
                return surface
        raise KeyError(f"{self._name!r} has no face named {short_label!r}.")

    def check_containment(self, result, *, warn: bool = True) -> Dict[str, Any]:
        """Run check_end_cap_crossings on a finished trace of a scene holding this solid.

        Args:
            result: The SimulationResult; it must have been traced with
                ``record_paths`` on.
            warn: See check_end_cap_crossings.

        Returns:
            check_end_cap_crossings's report.

        Raises:
            ValueError: If the result recorded no paths -- an empty report
                would then mean nothing was looked at, not that nothing
                crossed.
        """
        if result.ray_paths is None:
            raise ValueError(
                "result.ray_paths is None -- trace with record_paths=True (or a "
                "subset size) so there are events to check."
            )
        return check_end_cap_crossings(result.ray_paths["events"], self.profile, warn=warn)
