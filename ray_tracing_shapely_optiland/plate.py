"""
Python-specific module: plate
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
PLATE -- a flat polygonal mirror or absorber, placed by itself
===============================================================================
A projector headlamp's shade, or the board an LED sits on, is a flat plate
in a plane that contains the optical axis, perpendicular to the plane of
the 2D section. The 2D scene shows it edge on, as a segment; revolving that
segment would make a cylinder, and extruding it would make a plate of
constant width with no outline. Its outline in the third dimension is not
in the 2D scene at all: the caller gives it.

PlanarPlate is one face on PolygonPlaneGeometry (the extruded end caps'
geometry, geometry.py): a ReflectiveComponent or an AbsorbingComponent.

    PlanarPlate(name, polygon, cs=..., reflectance=0.9)    # anywhere
    PlanarPlate.from_segment(name, shade, polygon, reflectance=0.9)

``from_segment`` places it from a 2D segment (a Mirror, Blocker or any
object with p1/p2, or two points) under the scene bridge's plane mapping,
2D (a, b) -> world (a, 0, b) (``scene_placement``): the plate's local x
runs along the segment from p1, its local y is world y (out of the 2D
plane), so ``polygon`` is given as (u along the segment, v out of the
plane). It then checks that the plate's trace in the section plane, the
polygon's chord along v = 0, is the segment itself, [0, length]: the 2D
scene must be the 3D section (C-BRIDGE-13).

Optiland's ReflectiveComponent reflects on both faces and absorbs the
light it does not reflect (1 - reflectance); there is no one-sided mirror.
For a shade whose upper face is the mirror, with the light source above
it, that is the same thing.
===============================================================================
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import VACUUM
from optiland.nonsequential.components.absorbing import AbsorbingComponent
from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.components.compound import CompoundComponent
from optiland.nonsequential.components.reflective import ReflectiveComponent

from .geometry import PolygonPlaneGeometry

XY = Tuple[float, float]

#: How far, in scene units, the plate's section-plane chord may be from the 2D segment.
SEGMENT_TOL = 1e-9


def _endpoints(segment) -> Tuple[XY, XY]:
    """(p1, p2) from a 2D object with p1/p2 dicts, or from a pair of points."""
    if hasattr(segment, "p1") and hasattr(segment, "p2"):
        p1, p2 = segment.p1, segment.p2
        return (float(p1["x"]), float(p1["y"])), (float(p2["x"]), float(p2["y"]))
    p1, p2 = segment
    return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))


def segment_placement(p1: XY, p2: XY) -> CoordinateSystem:
    """The frame of a plate standing on the 2D segment p1 -> p2.

    Origin at world (a1, 0, b1); local x along the segment (in the world xz
    plane), local y world y, local z the segment's in-plane normal. Under
    Optiland's ``ry`` rotation, local x is (cos ry, 0, -sin ry).

    Raises:
        ValueError: For a zero-length segment.
    """
    da, db = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(da, db)
    if length == 0.0:
        raise ValueError("a plate needs a segment of non-zero length.")
    return CoordinateSystem(x=p1[0], y=0.0, z=p1[1], ry=math.atan2(-db / length, da / length))


def section_chord(polygon: Sequence[XY]) -> Optional[Tuple[float, float]]:
    """The polygon's chord along v = 0 as ``(u_min, u_max)``, or None if it has none.

    Raises:
        ValueError: If the chord is in more than one piece (the section
            would show more than one segment).
    """
    poly = Polygon(polygon)
    us = [u for u, _ in polygon]
    span = max(us) - min(us) + 1.0
    chord = poly.intersection(LineString([(min(us) - span, 0.0), (max(us) + span, 0.0)]))
    if chord.is_empty or chord.length == 0.0:
        return None
    if chord.geom_type != "LineString":
        raise ValueError(f"the plate's outline meets its section plane in more than one piece "
                         f"({chord.geom_type}); the 2D scene shows one segment.")
    xs = [c[0] for c in chord.coords]
    return min(xs), max(xs)


def _check_outline(name: str, verts) -> None:
    """Raise for an outline PolygonPlaneGeometry would refuse, or one that crosses itself."""
    if len(verts) >= 3 and np.all(np.isfinite(verts)) and not Polygon(verts).is_valid:
        raise ValueError(f"plate {name!r}: its outline crosses itself.")
    PolygonPlaneGeometry(verts)  # three vertices, finite, non-zero area


class PlanarPlate(CompoundComponent):
    """A flat polygonal plate: one mirror or absorber face, placed by ``cs``.

    Args:
        name: The plate's name; its face is ``"<name>.plate"`` in the event log.
        polygon: The outline, at least three (u, v) vertices in the plate's
            own frame (local x, y), either winding, convex or not.
        cs: Places the plate's frame in the world.
        reflectance: A mirror: the fraction reflected, on both faces; the
            rest is absorbed. Required unless ``absorbing``.
        absorbing: An absorber (a blackened board) instead of a mirror.
        medium: The medium around the plate (default vacuum).

    Raises:
        ValueError: Unless exactly one of ``reflectance`` and ``absorbing``
            is given, or for a polygon PolygonPlaneGeometry refuses, or one
            that is not simple (it crosses itself).
    """

    def __init__(
        self,
        name: str,
        polygon: Sequence[XY],
        *,
        cs: CoordinateSystem,
        reflectance: Any = None,
        absorbing: bool = False,
        medium=None,
    ) -> None:
        if (reflectance is None) == (not absorbing):
            raise ValueError("a plate is a mirror (reflectance=...) or an absorber "
                             "(absorbing=True): give exactly one.")
        verts = [(float(u), float(v)) for u, v in polygon]
        _check_outline(name, verts)
        geometry = PolygonPlaneGeometry(verts)
        medium = VACUUM if medium is None else medium
        self._name = name
        self._cs = cs
        self.polygon = geometry.vertices.copy()
        self.reflectance = reflectance
        self.absorbing = bool(absorbing)
        self.segment: Optional[Tuple[XY, XY]] = None  # set by from_segment
        face_name = f"{name}.plate"
        if absorbing:
            self._face = AbsorbingComponent(cs, geometry, material_front=medium, name=face_name)
        else:
            self._face = ReflectiveComponent(cs, geometry, reflectance=reflectance,
                                             material_front=medium, name=face_name)

    @classmethod
    def from_segment(
        cls,
        name: str,
        segment,
        polygon: Sequence[XY],
        *,
        reflectance: Any = None,
        absorbing: bool = False,
        medium=None,
        tol: float = SEGMENT_TOL,
    ) -> "PlanarPlate":
        """A plate standing on a 2D segment, perpendicular to the 2D plane.

        Args:
            segment: A 2D Mirror, Blocker or other object with ``p1``/``p2``,
                or a pair of points ``((a1, b1), (a2, b2))``.
            polygon: The outline as (u, v): u along the segment from p1 (so
                the segment is u in [0, length] at v = 0), v out of the 2D
                plane (world y).
            tol: How far the outline's chord along v = 0 may be from the
                segment's ends.

        Raises:
            ValueError: If the outline's chord along v = 0 is missing, in
                pieces, or not the segment (C-BRIDGE-13), plus whatever the
                constructor raises.
        """
        p1, p2 = _endpoints(segment)
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        cs = segment_placement(p1, p2)
        verts = [(float(u), float(v)) for u, v in polygon]
        _check_outline(name, verts)
        chord = section_chord(verts)
        if chord is None or abs(chord[0]) > tol or abs(chord[1] - length) > tol:
            got = "nothing" if chord is None else f"u in [{chord[0]:.6g}, {chord[1]:.6g}]"
            raise ValueError(
                f"plate {name!r}: its outline meets the 2D plane in {got}, but the 2D segment "
                f"is u in [0, {length:.6g}]: the 2D scene would not be the 3D section.")
        plate = cls(name, verts, cs=cs, reflectance=reflectance, absorbing=absorbing,
                    medium=medium)
        plate.segment = (p1, p2)
        return plate

    @property
    def name(self) -> str:
        return self._name

    @property
    def surfaces(self):
        return [self._face]

    @property
    def coordinate_system(self) -> CoordinateSystem:
        return self._cs

    @property
    def face(self):
        """The plate's one NSQ component."""
        return self._face

    def outline_world(self) -> np.ndarray:
        """The outline's vertices in the world, shape (N, 3), from the live face."""
        t, rot = _get_transform(self._face.cs)
        local = np.column_stack([self._face.geometry.vertices,
                                 np.zeros(len(self._face.geometry.vertices))])
        return local @ rot.T + t


__all__ = ["PlanarPlate", "segment_placement", "section_chord", "SEGMENT_TOL"]
