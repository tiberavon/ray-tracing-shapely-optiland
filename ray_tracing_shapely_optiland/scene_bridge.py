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

This is what the pilot notebook (notebooks/
optiland_nsq_refractometer_intro.ipynb) built by hand; the same physics,
without the by-hand trigonometry -- see that notebook's own closing section
for what this replaces. [KB7: the bridge exists so a 2D design can be
checked in 3D and handed back.]
===============================================================================
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import VACUUM
from optiland.nonsequential.components.absorbing import AbsorbingComponent
from optiland.nonsequential.components.compound import CompoundComponent
from optiland.nonsequential.components.geometry.analytic.plane import FinitePlaneGeometry
from optiland.nonsequential.components.reflective import ReflectiveComponent
from optiland.nonsequential.components.refractive import RefractiveComponent


def _require_supported_symmetry(symmetry: Dict[str, Any]) -> None:
    if symmetry.get("kind") != "extrude":
        raise NotImplementedError(
            f"scene_bridge only supports symmetry kind='extrude' today, got "
            f"{symmetry.get('kind')!r}. 'revolve' needs NSQ freeform geometry "
            "first, which is not built (C-BRIDGE-02)."
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

    import math

    phi_edge = math.atan2(z2 - z1, x2 - x1)
    cs = CoordinateSystem(x=mid_x, y=0.0, z=mid_z, ry=-phi_edge)
    geom = FinitePlaneGeometry(width=edge_len, height=depth)

    label = span["short_label"]
    mechanism = span["mechanism"]
    outside = material_overrides.get(label, VACUUM)
    name = label or f"span_{span['index_start']}"

    if mechanism in ("refract", "tir"):
        return RefractiveComponent(
            cs, geom, material_front=outside, material_back=interior_material, name=name
        )
    if mechanism == "reflect":
        return ReflectiveComponent(cs, geom, material_front=outside, name=name)
    if mechanism is None:
        return AbsorbingComponent(cs, geom, material_front=interior_material, name=name)
    raise ValueError(
        f"Span {label!r} has unrecognised mechanism {mechanism!r} -- "
        "expected 'refract', 'tir', 'reflect', or null."
    )


def build_faces_from_profile(
    profile: Dict[str, Any],
    interior_material,
    material_overrides: Optional[Dict[str, Any]] = None,
):
    """Build one Optiland NSQ component per span in a profile.json dict.

    Args:
        profile: A profile.json-shaped dict (see
            ray_tracing_shapely.optical_elements.prisms.profile_export
            .export_profile, or any tool-independent producer of the same
            schema).
        interior_material: See face_from_span.
        material_overrides: See face_from_span.

    Returns:
        dict mapping each span's short_label to its built component, in
        profile["spans"] order -- pass ``list(result.values())`` to
        ``scene.add_component`` per component, or iterate the dict
        directly to also keep each name.
    """
    return {
        span["short_label"]: face_from_span(
            span,
            profile["points"],
            profile["normals"],
            profile["symmetry"],
            interior_material,
            material_overrides=material_overrides,
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
    import math

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
    # directly -- see notebooks/optiland_nsq_refractometer_intro.ipynb
    # section 5); a source wants the same "into the interior" aim.
    phi_edge = phi_outward + math.pi / 2
    ry = -phi_edge
    return CoordinateSystem(x=pos_x, y=0.0, z=pos_z, ry=ry), ry


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
    are simply gone]. There are no end caps at all (not even the mesh caps
    the pilot tried): the solid
    is open at both ends of the extrusion. Every ray built via
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
    ) -> None:
        """Build every span's face via build_faces_from_profile.

        Args:
            name: Registry name for this compound.
            profile: See build_faces_from_profile.
            interior_material: See build_faces_from_profile.
            material_overrides: See build_faces_from_profile.
        """
        self._name = name
        self.profile = profile
        self._faces = build_faces_from_profile(
            profile, interior_material, material_overrides=material_overrides
        )
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
        return list(self._faces.values())

    @property
    def coordinate_system(self) -> CoordinateSystem:
        return self._cs

    def face(self, short_label: str):
        """Return one named face (e.g. ``solid.face("M")``)."""
        return self._faces[short_label]
