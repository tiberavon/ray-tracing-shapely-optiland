"""
Python-specific module: revolve
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
REVOLVE -- a revolve profile.json -> an Optiland NSQ solid of revolution
===============================================================================
The revolve half of the scene bridge. A ``symmetry.kind =
"revolve"`` profile (schema 1.1 and later) is a half-section: an open chain of spans
left of a declared axis, from the axis back to it. Each span becomes ONE
Optiland analytic surface, chosen exactly, never approximated:

    span (in the axis frame)                          NSQ geometry
    ------------------------------------------------  -------------------------------
    circle, centre on the axis, one end on it          ConicGeometry(R, 0, r_rim)
    line perpendicular to the axis, one end on it      FinitePlaneGeometry(aperture_radius=r)
    line perpendicular to the axis, off it             AnnularPlaneGeometry(r_in, r_out)
    any other line (cylinder, cone, frustum)           CylindricalFrustumGeometry(r1, r2, z1, z2)
    circle or conic not reaching the axis (a zone)     CutConicGeometry(R, K, r_out, inner_radius=r_in)
    spline, monotonic in r (a designed face, or a      SplineRevolutionGeometry(points, tangents, domain)
      zone of one)
    even asphere, its vertex on the axis, its axis     AsphereRevolutionGeometry(R, K, coefficients,
      along it (or a zone of one)                        r_out, inner_radius=r_in)
    circle off the axis (a torus), an arc past its     NotImplementedError, naming which
    equator, a spline not monotonic in r, or any
    other curve type

A sheet may also be cut to an azimuth range about its axis, declared in
the profile as ``symmetry.azimuth_deg = [start, end]`` (C-BRIDGE-12):
its curved spans then become CutConicGeometry with that range; a line
span refuses the cut. The range is measured in the axis frame from the
half-section's own side (+x) towards +y, and must keep 0 deg, the
half-section itself.

The frame. In the solid's own axis frame the revolution axis is +z, the axis
point the origin, and the half-section's side of the axis (left of the axis
direction) +x: a profile point ``p`` maps to ``x = (p - point) . left``,
``y = 0``, ``z = (p - point) . direction``. The profile plane is the frame's
xz plane, so a 2D trace compares with the 3D rays that stay in it.

Placement. ``RevolvedProfileSolid(..., cs=...)`` places that axis frame in
the world; every face's coordinate system refers to it (``reference_cs``),
as Optiland's own Lens places its faces. Without ``cs`` the axis frame is
the world frame. ``scene_placement(profile)`` is the ``cs`` that puts the
solid where its 2D scene had it, under the scene bridge's own plane mapping
(2D ``(a, b)`` -> world ``(a, 0, b)``, the extruded prism's
``plane_axes=["x", "z"]``), so the elements of one 2D scene -- two lenses, or
a lens and a prism -- land in one 3D world in the same relative places.

Sidedness is read from the profile, not assumed. Each surface's own n_geom
(Optiland's geometric normal, which points from material_front to
material_back) is probed with a ray at a point of the span and compared with
the span's stored outward normal: if n_geom points out of the solid, the
interior is material_front, otherwise material_back. This is the first place
the bridge reads the stored normal for orientation (C-BRIDGE-07).

Curved spans are validated (C-BRIDGE-08): every stored sample must lie on the
built surface within ``sag_tol`` and every stored normal must agree with the
surface's normal there within ``slope_tol`` (on 1 - |cos|) -- slope is what
a ray feels.

The solid is validated (C-BRIDGE-09): RevolvedProfileSolid runs the two
checks of Optiland's Volume() on its faces, a watertightness and inside/outside parity
check that the extruded prism cannot pass (its polygon caps are seen as
rectangles). A revolved solid built only from these analytic surfaces can.
The parity check is cast from a point the profile proves is inside (its
half-section's pole of inaccessibility), not from Optiland's own estimate,
the mean of the rim points: for a meniscus whose edge sags more than its
centre thickness, that mean lies in the hollow, outside the glass, and
Volume() rejects a valid solid (KW-OPTILAND-04). Both checks are the
bridge's own versions (_validated_volume): every
rim a face has is sampled (a zone's inner rim, a spline surface's rims) and
none at a radius-0 end (a cone's tip, KW-OPTILAND-06), and parity is traced
through the live faces, not plain-conic copies (KW-OPTILAND-07). So a solid
may have zones and cone tips.
===============================================================================
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import VACUUM
from optiland.nonsequential._utils import as_float
from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.components.compound import CompoundComponent
from optiland.nonsequential.components.geometry.analytic.annulus import AnnularPlaneGeometry
from optiland.nonsequential.components.geometry.analytic.conic import ConicGeometry
from optiland.nonsequential.components.geometry.analytic.frustum import (
    CylindricalFrustumGeometry,
)
from optiland.nonsequential.components.geometry.analytic.plane import FinitePlaneGeometry
from .geometry import AsphereRevolutionGeometry, CutConicGeometry, SplineRevolutionGeometry
from optiland.nonsequential.components.volume import (
    WATERTIGHT_TOL,
    NonWatertightVolumeError,
    Volume,
)

XY = Tuple[float, float]

#: How close, in profile units, a point must be to the axis to be on it.
AXIS_TOL = 1e-9
#: Allowed distance of a curved span's samples from the built surface.
SAG_TOL = 1e-9
#: Allowed 1 - |cos| between a stored normal and the built surface's normal.
SLOPE_TOL = 1e-9


# --- the axis frame ---------------------------------------------------------------


def profile_azimuth(profile: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """The profile's ``symmetry.azimuth_deg`` as ``(start, end)``, or None for the full turn.

    Raises:
        ValueError: For a malformed or empty range, one over a full turn, or
            one that removes the half-section itself (0 deg).
    """
    az = profile.get("symmetry", {}).get("azimuth_deg")
    if az is None:
        return None
    try:
        start, end = float(az[0]), float(az[1])
        if len(az) != 2 or not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError
    except (TypeError, IndexError, ValueError):
        raise ValueError(f"symmetry azimuth_deg must be [start, end] in degrees, got {az!r}") from None
    if not 0.0 < end - start <= 360.0:
        raise ValueError(f"symmetry azimuth_deg needs 0 < end - start <= 360, got {az!r}")
    if end - start == 360.0:
        return None
    if (0.0 - start) % 360.0 > end - start:
        raise ValueError(f"symmetry azimuth_deg {az!r} removes the half-section itself (0 deg).")
    return start, end



@dataclass(frozen=True)
class AxisFrame:
    """The map between a revolve profile's 2D coordinates and the world.

    Attributes:
        point: The axis point, in profile coordinates (the world origin).
        direction: Unit axis direction in profile coordinates (world +z).
        left: Unit normal left of ``direction`` (world +x): the side the
            half-section lies on.
    """

    point: XY
    direction: XY
    left: XY

    def to_world(self, p: XY) -> Tuple[float, float, float]:
        """A profile point as world (x, y, z); x is its distance from the axis."""
        dx, dy = p[0] - self.point[0], p[1] - self.point[1]
        return (dx * self.left[0] + dy * self.left[1], 0.0,
                dx * self.direction[0] + dy * self.direction[1])

    def vector_to_world(self, v: XY) -> Tuple[float, float, float]:
        """A profile direction (a normal) as a world vector."""
        return (v[0] * self.left[0] + v[1] * self.left[1], 0.0,
                v[0] * self.direction[0] + v[1] * self.direction[1])

    def to_profile(self, xyz) -> XY:
        """A world point in the profile plane (its y is ignored) back to profile coordinates."""
        x, z = float(xyz[0]), float(xyz[2])
        return (self.point[0] + x * self.left[0] + z * self.direction[0],
                self.point[1] + x * self.left[1] + z * self.direction[1])

    def vector_to_profile(self, v) -> XY:
        x, z = float(v[0]), float(v[2])
        return (x * self.left[0] + z * self.direction[0], x * self.left[1] + z * self.direction[1])

    def source_cs(self, z: float, field_angle_deg: float = 0.0, x: float = 0.0) -> CoordinateSystem:
        """A coordinate system on the axis at world ``z`` whose local +z (the
        propagation axis of Optiland's source configs) is the axis direction,
        tilted by ``field_angle_deg`` within the profile plane, positive
        towards the half-section's side (world +x). ``x`` offsets it in that
        plane (an off-axis point source)."""
        return CoordinateSystem(x=x, y=0.0, z=z, ry=math.radians(field_angle_deg))


def axis_frame(profile: Dict[str, Any]) -> AxisFrame:
    """The AxisFrame of a revolve profile.

    Raises:
        ValueError: If the profile is not a revolve, or its axis is malformed.
    """
    symmetry = profile.get("symmetry") or {}
    if symmetry.get("kind") != "revolve":
        raise ValueError(f"not a revolve profile: symmetry kind is {symmetry.get('kind')!r}")
    axis = symmetry.get("axis")
    try:
        point = (float(axis["point"][0]), float(axis["point"][1]))
        ux, uy = float(axis["direction"][0]), float(axis["direction"][1])
    except (TypeError, KeyError, IndexError, ValueError):
        raise ValueError(
            "symmetry kind='revolve' needs axis = {'point': [a, b], 'direction': [ux, uy]}, "
            f"got {axis!r}"
        ) from None
    length = math.hypot(ux, uy)
    if length == 0.0:
        raise ValueError("revolve axis direction is zero")
    ux, uy = ux / length, uy / length
    return AxisFrame(point, (ux, uy), (-uy, ux))


def scene_placement(profile: Dict[str, Any]) -> CoordinateSystem:
    """The coordinate system that puts a revolve profile's solid where its 2D scene had it.

    The 2D scene plane maps to world xz exactly as the extruded bridge maps
    it (``(a, b) -> (a, 0, b)``): the axis point lands at ``(a, 0, b)``, the
    axis direction along ``(d_a, 0, d_b)``, and the half-section's side
    along ``(left_a, 0, left_b)``. Pass it as ``RevolvedProfileSolid(...,
    cs=scene_placement(profile))``. Two elements exported from one 2D scene
    then keep their relative positions and tilts in 3D.
    """
    frame = axis_frame(profile)
    (pa, pb), (da, db), (la, lb) = frame.point, frame.direction, frame.left
    # Columns: the frame's x (the half-section side), y (= z cross x, which is
    # world -y), z (the axis), in world coordinates. A proper rotation.
    rotation = np.array([[la, 0.0, da],
                         [0.0, -1.0, 0.0],
                         [lb, 0.0, db]])
    return cs_from_axes((pa, 0.0, pb), rotation)


def cs_from_axes(origin, rotation) -> CoordinateSystem:
    """The CoordinateSystem at ``origin`` whose local x, y, z are ``rotation``'s columns.

    Optiland's CoordinateSystem rotates by R = Rz @ Ry @ Rx: scipy's extrinsic
    "xyz". A local z along world x is Euler's gimbal lock (ry = 90 deg); scipy
    warns and still returns a decomposition, so the result is rebuilt and
    compared.

    Raises:
        ValueError: If ``rotation`` is not a proper rotation, or the Euler
            angles found do not rebuild it within 1e-12.
    """
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    rotation = np.asarray(rotation, dtype=float)
    if (rotation.shape != (3, 3) or np.abs(rotation.T @ rotation - np.eye(3)).max() > 1e-12
            or np.linalg.det(rotation) < 0.0):
        raise ValueError(f"not a proper rotation: {rotation.tolist()}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock")
        rx, ry, rz = Rotation.from_matrix(rotation).as_euler("xyz")
    cs = CoordinateSystem(x=float(origin[0]), y=float(origin[1]), z=float(origin[2]), rx=rx, ry=ry, rz=rz)
    if np.abs(_get_transform(cs)[1] - rotation).max() > 1e-12:
        raise ValueError(f"internal: Euler angles ({rx}, {ry}, {rz}) do not rebuild the frame "
                         f"{rotation.tolist()}.")
    return cs


def _relative_transform(cs: CoordinateSystem, reference: CoordinateSystem):
    """``cs``'s translation and rotation expressed in ``reference``'s frame."""
    t, r = _get_transform(cs)
    t0, r0 = _get_transform(reference)
    return r0.T @ (t - t0), r0.T @ r


# --- one span -> one surface ------------------------------------------------------


@dataclass
class RevolvedSurface:
    """What a half-section span becomes: a coordinate system, a geometry, and
    which side of the geometry is the solid's interior."""

    name: str
    cs: CoordinateSystem
    geometry: Any
    interior_on_back: bool
    kind: str  # "conic", "asphere", "spline", "disc", "annulus", "frustum"


def _span_samples(span, profile, frame: AxisFrame):
    k0, k1 = span["index_start"], span["index_end"]
    pts = [frame.to_world(tuple(profile["points"][k])) for k in range(k0, k1)]
    nrm = [frame.vector_to_world(tuple(profile["normals"][k])) for k in range(k0, k1)]
    return np.array(pts, dtype=float), np.array(nrm, dtype=float)


def _probe_n_geom(geometry, cs_z: float, point_world, outward) -> np.ndarray:
    """Optiland's n_geom at ``point_world``, found by shooting a ray at it along -outward."""
    local = np.array(point_world, dtype=float) - np.array([0.0, 0.0, cs_z])
    origin = local + 1e-3 * np.asarray(outward)
    _, _, hit, n_geom = geometry.ray_intersect(origin[None, :], -np.asarray(outward)[None, :])
    if not bool(np.asarray(hit)[0]):
        raise ValueError(
            "internal: a probe ray aimed straight at a span's own point missed the surface "
            f"built for it ({type(geometry).__name__}) -- the surface is misplaced."
        )
    return np.asarray(n_geom, dtype=float)[0]


def surface_from_span(span: Dict[str, Any], profile: Dict[str, Any], *,
                      axis_tol: float = AXIS_TOL, sag_tol: float = SAG_TOL,
                      slope_tol: float = SLOPE_TOL, allow_zone: bool = True,
                      azimuth_deg: Optional[Tuple[float, float]] = None) -> RevolvedSurface:
    """Choose, place and orient the one NSQ geometry a half-section span revolves into.

    Args:
        allow_zone: Accept a curved span that does not reach the axis (a
            zone: a CutConicGeometry or SplineRevolutionGeometry with a
            central hole). False refuses it. Solids may have zones: their
            closure is checked by the bridge, rims included
            (_validated_volume).
        azimuth_deg: Cut a curved span to this ``(start, end)`` range about
            the axis, in degrees from the frame's +x towards +y.

    Raises:
        NotImplementedError: For a span no analytic NSQ geometry describes
            exactly (see the module table).
        ValueError: For a span lying on the axis, or a curved span whose
            samples or normals disagree with the surface (C-BRIDGE-08).
    """
    frame = axis_frame(profile)
    name = span.get("short_label") or f"span_{span['index_start']}"
    curve = span.get("curve") or {"type": "line"}
    pts, nrm = _span_samples(span, profile, frame)
    (r1, _, z1), (r2, _, z2) = pts[0], pts[-1]

    if curve["type"] == "line":
        if azimuth_deg is not None:
            raise NotImplementedError(
                f"line span {name!r}: an azimuth cut is built for curved spans only; a cut "
                "disc, annulus or frustum needs its own geometry (C-BRIDGE-12)."
            )
        if pts.shape[0] != 2:
            raise ValueError(f"line span {name!r} has {pts.shape[0]} points, not 2")
        if r1 <= axis_tol and r2 <= axis_tol:
            raise ValueError(f"span {name!r} lies along the revolve axis; revolving it makes no surface.")
        if abs(z1 - z2) <= axis_tol:
            z = 0.5 * (z1 + z2)
            r_in, r_out = min(r1, r2), max(r1, r2)
            if r_in <= axis_tol:
                geometry, kind = FinitePlaneGeometry(aperture_radius=r_out), "disc"
            else:
                geometry, kind = AnnularPlaneGeometry(r_in, r_out, z_offset=0.0), "annulus"
            cs_z = z
        else:
            (rf, zf), (rb, zb) = sorted(((r1, z1), (r2, z2)), key=lambda rz: rz[1])
            geometry = CylindricalFrustumGeometry(rf, rb, zf, zb)
            kind, cs_z = "frustum", 0.0
        # A stored line normal must be perpendicular to the line (slope check).
        d = pts[1] - pts[0]
        d /= np.linalg.norm(d)
        if abs(float(nrm[0] @ d)) > math.sqrt(2 * slope_tol) + 1e-12:
            raise ValueError(f"span {name!r}: its stored normal is not perpendicular to it.")
        probe_at, outward = 0.5 * (pts[0] + pts[1]), nrm[0]

    elif curve["type"] in ("circle", "conic"):
        if curve["type"] == "circle":
            rc, _, zc = frame.to_world(tuple(curve["center"]))
            radius = float(curve["radius"])
            if abs(rc) > axis_tol * max(1.0, radius):
                raise NotImplementedError(
                    f"span {name!r}: circle centre {abs(rc):.3g} off the axis -- revolving it "
                    "makes a torus, which no NSQ analytic geometry describes."
                )
            conic = 0.0
        else:
            vr, _, vz = frame.to_world(tuple(curve["vertex"]))
            ar, _, az = frame.vector_to_world(tuple(curve["axis"]))
            a_len = math.hypot(ar, az)
            if abs(vr) > axis_tol * max(1.0, abs(float(curve["radius"]))) or abs(ar) > 1e-12 * a_len:
                raise NotImplementedError(
                    f"span {name!r}: a conic whose axis is not the revolve axis revolves into a "
                    "surface that is not a conic of revolution (a CPC, out of scope: C-BRIDGE-02)."
                )
            conic = float(curve["conic"])
            # the profile's R is along the conic's own axis; Optiland's along the frame's z
            r_frame = float(curve["radius"]) * (1.0 if az > 0.0 else -1.0)
        on_axis = [i for i, r in ((0, r1), (-1, r2)) if r <= axis_tol]
        if not on_axis and not allow_zone:
            raise NotImplementedError(
                f"span {name!r}: a {curve['type']} that does not reach the axis revolves into a "
                f"{'spherical' if curve['type'] == 'circle' else 'conic'} zone, a conic with a "
                "hole, and allow_zone is False."
            )
        r_in = 0.0 if on_axis else float(min(r1, r2))
        r_rim = float(max(r1, r2))
        if curve["type"] == "circle":
            # the vertex on the samples' side of the centre; a span on both sides
            # crosses the equator, where z(r) is not single-valued
            side = np.sign(pts[:, 2] - zc)
            if np.any(side == 0.0) or np.any(side != side[0]):
                raise NotImplementedError(
                    f"span {name!r}: the arc reaches past its sphere's equator, so it is not "
                    "single-valued as z(r); ConicGeometry cannot describe it."
                )
            z_v = float(zc + side[0] * radius)
            r_frame = zc - z_v
        else:
            z_v = float(vz)
        if r_in > 0.0 or azimuth_deg is not None:
            geometry = CutConicGeometry(r_frame, conic, r_rim, inner_radius=r_in,
                                        azimuth_deg=azimuth_deg)
        else:
            geometry = ConicGeometry(r_frame, conic, r_rim)
        kind, cs_z = "conic", z_v
        # C-BRIDGE-08: every sample on the surface, every normal along its normal.
        # The distance is measured along the surface normal, to first order
        # |dz| |n_z|: near an ellipsoid's equator the surface turns parallel to
        # the axis, and a sample a rounding error from it is far from it in z.
        r = pts[:, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            sag = np.asarray(geometry._sag(r, np.zeros_like(r)), dtype=float)
            n_surf = np.asarray(geometry._normal_local(r, np.zeros_like(r)), dtype=float)
            n_surf /= np.linalg.norm(n_surf, axis=1, keepdims=True)
        n_surf = np.where(np.isfinite(n_surf), n_surf, np.array([1.0, 0.0, 0.0]))  # at the equator
        off = np.abs(pts[:, 2] - z_v - sag) * np.abs(n_surf[:, 2])
        if not np.all(np.isfinite(off)):
            raise ValueError(
                f"span {name!r}: some samples lie beyond where the conic built for it is "
                "single-valued as z(r)."
            )
        if off.max() > sag_tol:
            k = int(off.argmax())
            raise ValueError(
                f"span {name!r}: sample {k} lies {off[k]:.3g} from the conic built for it "
                f"(sag_tol {sag_tol:g})."
            )
        cos = np.abs(np.sum(n_surf * nrm, axis=1))
        if (1.0 - cos).max() > slope_tol:
            k = int((1.0 - cos).argmax())
            raise ValueError(
                f"span {name!r}: the stored normal at sample {k} is "
                f"{math.degrees(math.acos(min(1.0, cos[k]))):.3g} deg from the conic's "
                f"(slope_tol {slope_tol:g})."
            )
        mid = pts.shape[0] // 2
        probe_at, outward = pts[mid], nrm[mid]
    elif curve["type"] == "spline":
        if azimuth_deg is not None:
            raise NotImplementedError(
                f"spline span {name!r}: an azimuth cut is built for conic spans only.")
        knots = np.array([frame.to_world(tuple(q)) for q in curve["points"]], dtype=float)[:, [0, 2]]
        tangents = np.array([frame.vector_to_world(tuple(v)) for v in curve["tangents"]],
                            dtype=float)[:, [0, 2]]
        try:
            geometry = SplineRevolutionGeometry(knots, tangents, curve.get("domain"))
        except ValueError as exc:
            raise NotImplementedError(f"span {name!r}: {exc}") from None
        if geometry.inner_radius <= axis_tol:
            geometry.inner_radius = 0.0
        elif not allow_zone:
            raise NotImplementedError(f"span {name!r}: a spline that does not reach the axis "
                                      "revolves into a zone, and allow_zone is False.")
        kind, cs_z = "spline", 0.0
        # C-BRIDGE-08, as for a conic: samples on the surface (distance along its
        # normal), normals along its normal.
        z_m, n_m = geometry.meridian_at(pts[:, 0])
        off = np.abs(pts[:, 2] - z_m) * np.abs(n_m[:, 1])
        if off.max() > sag_tol:
            k = int(off.argmax())
            raise ValueError(f"span {name!r}: sample {k} lies {off[k]:.3g} from the spline surface "
                             f"built for it (sag_tol {sag_tol:g}).")
        cos = np.abs(n_m[:, 0] * nrm[:, 0] + n_m[:, 1] * nrm[:, 2])
        if (1.0 - cos).max() > slope_tol:
            k = int((1.0 - cos).argmax())
            raise ValueError(
                f"span {name!r}: the stored normal at sample {k} is "
                f"{math.degrees(math.acos(min(1.0, cos[k]))):.3g} deg from the spline surface's "
                f"(slope_tol {slope_tol:g}).")
        mid = pts.shape[0] // 2
        probe_at, outward = pts[mid], nrm[mid]
    elif curve["type"] == "asphere":
        if azimuth_deg is not None:
            raise NotImplementedError(
                f"asphere span {name!r}: an azimuth cut is built for conic spans only.")
        vr, _, vz = frame.to_world(tuple(curve["vertex"]))
        ar, _, az = frame.vector_to_world(tuple(curve["axis"]))
        a_len = math.hypot(ar, az)
        r_vertex = curve.get("radius")
        if abs(vr) > axis_tol * max(1.0, abs(float(r_vertex or 0.0))) or abs(ar) > 1e-12 * a_len:
            raise NotImplementedError(
                f"span {name!r}: an asphere whose axis is not the revolve axis revolves into a "
                "surface that is not an asphere of revolution.")
        # the profile's sag runs along the asphere's own axis; the geometry's along the frame's z
        sign = 1.0 if az > 0.0 else -1.0
        on_axis = [i for i, r in ((0, r1), (-1, r2)) if r <= axis_tol]
        if not on_axis and not allow_zone:
            raise NotImplementedError(f"span {name!r}: an asphere that does not reach the axis "
                                      "revolves into a zone, and allow_zone is False.")
        r_in = 0.0 if on_axis else float(min(r1, r2))
        try:
            geometry = AsphereRevolutionGeometry(
                None if r_vertex is None else sign * float(r_vertex), float(curve.get("conic", 0.0)),
                [sign * float(a) for a in curve.get("coefficients", [])], float(max(r1, r2)), inner_radius=r_in)
        except ValueError as exc:
            raise NotImplementedError(f"span {name!r}: {exc}") from None
        kind, cs_z = "asphere", float(vz)
        # C-BRIDGE-08, as for a conic: samples on the surface (distance along its
        # normal), normals along its normal.
        z_m, n_m = geometry.meridian_at(pts[:, 0])
        off = np.abs(pts[:, 2] - cs_z - z_m) * np.abs(n_m[:, 1])
        if off.max() > sag_tol:
            k = int(off.argmax())
            raise ValueError(f"span {name!r}: sample {k} lies {off[k]:.3g} from the asphere built for it "
                             f"(sag_tol {sag_tol:g}).")
        cos = np.abs(n_m[:, 0] * nrm[:, 0] + n_m[:, 1] * nrm[:, 2])
        if (1.0 - cos).max() > slope_tol:
            k = int((1.0 - cos).argmax())
            raise ValueError(
                f"span {name!r}: the stored normal at sample {k} is "
                f"{math.degrees(math.acos(min(1.0, cos[k]))):.3g} deg from the asphere's "
                f"(slope_tol {slope_tol:g}).")
        mid = pts.shape[0] // 2
        probe_at, outward = pts[mid], nrm[mid]
    else:
        raise NotImplementedError(
            f"span {name!r} has curve type {curve['type']!r}; the revolve bridge builds "
            "lines, circles, conics, aspheres and splines."
        )

    probe_geometry = geometry
    if isinstance(geometry, CutConicGeometry):  # the cut may remove azimuth 0, where the probe is
        probe_geometry = ConicGeometry(geometry.radius, geometry.conic, geometry.aperture_radius)
    n_geom = _probe_n_geom(probe_geometry, cs_z, probe_at, outward)
    interior_on_back = float(n_geom @ outward) < 0.0  # n_geom points into the solid
    return RevolvedSurface(name, CoordinateSystem(z=cs_z), geometry, interior_on_back, kind)


def build_revolved_faces(
    profile: Dict[str, Any],
    interior_material,
    material_overrides: Optional[Dict[str, Any]] = None,
    reflectances: Optional[Dict[str, Any]] = None,
    *,
    cs: Optional[CoordinateSystem] = None,
    name_prefix: str = "",
    **tolerances,
) -> Tuple[Dict[str, Any], Dict[str, RevolvedSurface]]:
    """One NSQ component per half-section span, keyed by short label.

    Args:
        cs: Places the axis frame in the world; each face's coordinate
            system refers to it. ``None``: the axis frame is the world.
        name_prefix: Prepended to each component's name (the short label),
            e.g. ``"L2."``, so two solids' faces stay apart in the event log.

    Returns:
        ``(components, surfaces)``: the built components, and for each the
        RevolvedSurface it was built from (its kind and interior side).

    Raises:
        ValueError: For a body other than "solid" or "sheet", a sheet given an
            interior material or a span that does not reflect, two spans
            sharing a label, or anything surface_from_span raises.
    """
    from .scene_bridge import _component_for_mechanism

    body = profile.get("body", "solid")
    if body not in ("solid", "sheet"):
        raise ValueError(f"profile body is {body!r}; expected 'solid' or 'sheet'.")
    azimuth_deg = profile_azimuth(profile)
    if azimuth_deg is not None and body != "sheet":
        raise ValueError("only a sheet can be cut to an azimuth range: a cut solid is not closed.")
    if body == "sheet":
        if interior_material is not None:
            raise ValueError("a sheet encloses nothing: build it with interior_material=None.")
        not_mirrors = [s.get("short_label") for s in profile["spans"] if s.get("mechanism") != "reflect"]
        if not_mirrors:
            raise ValueError(f"every span of a sheet must reflect; these do not: {not_mirrors}")
    names = [s.get("short_label") or f"span_{s['index_start']}" for s in profile["spans"]]
    if len(set(names)) != len(names):
        raise ValueError(f"half-section spans must have distinct labels, got {names}")
    material_overrides = material_overrides or {}
    components: Dict[str, Any] = {}
    surfaces: Dict[str, RevolvedSurface] = {}
    for span, name in zip(profile["spans"], names):
        surface = surface_from_span(span, profile, allow_zone=True,
                                    azimuth_deg=azimuth_deg, **tolerances)
        if cs is not None:
            surface.cs = CoordinateSystem(z=surface.cs.z, reference_cs=cs)
        outside = material_overrides.get(name, VACUUM)
        components[name] = _component_for_mechanism(
            span.get("mechanism"), surface.cs, surface.geometry, outside, interior_material,
            name_prefix + name, reflectance=(reflectances or {}).get(name),
            interior_on_back=surface.interior_on_back,
        )
        surfaces[name] = surface
    return components, surfaces


def interior_point(profile: Dict[str, Any]) -> XY:
    """A point inside a revolve profile's half-section, as far from its edges as possible.

    The pole of inaccessibility of the closed half-section polygon (its
    points, closed along the axis), in profile coordinates. Revolved, it is
    inside the solid: the half-section is the solid's meridional section.
    Being as far as possible from every edge, it stays inside whatever the
    chord error of the sampled curves.
    """
    from shapely.geometry import Polygon  # noqa: PLC0415
    from shapely.ops import polylabel  # noqa: PLC0415

    polygon = Polygon(profile["points"])
    if not polygon.is_valid or polygon.area <= 0.0:
        raise ValueError("the profile's points do not close into a simple half-section polygon")
    pole = polylabel(polygon, tolerance=1e-3 * math.sqrt(polygon.area))
    return (float(pole.x), float(pole.y))


#: Points per rim circle in the watertightness check (Optiland samples 64).
RIM_SAMPLES = 64


def _rim_circles(component) -> List[Tuple[float, float]]:
    """A face's edge circles, (r, z) in its own frame, r > 0.

    Unlike Optiland's _rim_points: a zone's inner rim and a spline surface's
    rims are included, and an end of radius 0 is not a rim (a cone's tip, a
    cap reaching the axis: the surface closes there by itself,
    KW-OPTILAND-06).
    """
    g = component.geometry
    if isinstance(g, (SplineRevolutionGeometry, AsphereRevolutionGeometry)):
        return g.rims()
    if isinstance(g, ConicGeometry):
        rim = as_float(g.aperture_radius)
        circles = [(rim, float(np.asarray(g._sag(np.array([rim]), np.array([0.0])))[0]))]
        inner = float(getattr(g, "inner_radius", 0.0))
        if inner > 0.0:
            circles.append((inner, float(np.asarray(g._sag(np.array([inner]), np.array([0.0])))[0])))
        return circles
    if isinstance(g, FinitePlaneGeometry):
        return [(as_float(g.aperture_radius), 0.0)]
    if isinstance(g, AnnularPlaneGeometry):
        z = as_float(g.z_offset)
        return [(r, z) for r in (as_float(g.inner_radius), as_float(g.outer_radius)) if r > 0.0]
    if isinstance(g, CylindricalFrustumGeometry):
        return [(r, z) for r, z in ((as_float(g.r_front), as_float(g.z_front)),
                                    (as_float(g.r_back), as_float(g.z_back))) if r > 0.0]
    raise TypeError(f"no rim sampling for {type(g).__name__}")


def _check_rims_meet(boundary: List[Any], tol: float = WATERTIGHT_TOL) -> None:
    """Every rim point of every face lies within ``tol`` of another face's rim
    (Optiland's _check_watertight, on _rim_circles)."""
    theta = np.linspace(0.0, 2.0 * np.pi, RIM_SAMPLES, endpoint=False)
    rims = []
    for comp in boundary:
        circles = _rim_circles(comp)
        if not circles:
            continue
        local = np.concatenate([np.stack([r * np.cos(theta), r * np.sin(theta), np.full_like(theta, z)],
                                         axis=1) for r, z in circles])
        t, rot = _get_transform(comp.cs)
        rims.append((comp, local @ rot.T + t))
    for i, (comp, pts) in enumerate(rims):
        others = [q for j, (_, q) in enumerate(rims) if j != i]
        if not others:
            raise NonWatertightVolumeError(f"surface '{comp.name}' has a rim no other surface meets.")
        other = np.concatenate(others)
        gap = np.sqrt(((pts[:, None, :] - other[None, :, :]) ** 2).sum(axis=2)).min(axis=1).max()
        if gap > tol:
            raise NonWatertightVolumeError(
                f"Volume boundary is not watertight: surface '{comp.name}' has a rim point "
                f"{gap:.3g} mm from the nearest point on any other boundary surface's rim "
                f"(tolerance {tol:.1e} mm).")


def _crossings(boundary: List[Any], origin: np.ndarray, direction: np.ndarray) -> int:
    """How many times a ray from ``origin`` crosses the boundary before escaping,
    walked hit by hit through each live face's own intersect() (Optiland's
    _count_crossings intersects detached plain copies instead: a cut conic or
    a spline surface would be checked as a conic)."""
    from optiland.nonsequential.components.volume import (  # noqa: PLC0415
        _PARITY_EPSILON,
        _PARITY_MAX_BOUNCES,
    )
    from optiland.nonsequential.ray_bundle import NSQRayBundle  # noqa: PLC0415

    o, d = np.array(origin, dtype=float), np.array(direction, dtype=float)
    count = 0
    for _ in range(_PARITY_MAX_BOUNCES):
        rays = NSQRayBundle(x=o[:1], y=o[1:2], z=o[2:3], L=d[:1], M=d[1:2], N=d[2:3],
                            flux=np.ones(1), wavelength=np.array([0.55]), n_current=np.ones(1),
                            bounce=np.zeros(1, dtype=np.int32), alive=np.ones(1, dtype=bool),
                            ray_id=np.zeros(1, dtype=np.int64))
        t_min = np.inf
        for comp in boundary:
            t, _, hit, _ = comp.intersect(rays)
            if bool(np.asarray(hit)[0]) and float(np.asarray(t)[0]) < t_min:
                t_min = float(np.asarray(t)[0])
        if not np.isfinite(t_min):
            break
        o = o + (t_min + _PARITY_EPSILON) * d
        count += 1
    return count


def _validated_volume(name: str, boundary: List[Any], interior, inside) -> Volume:
    """Optiland's Volume, validated by the bridge's own versions of Optiland's two
    checks (C-BRIDGE-09).

    - **Rims meet** (_check_rims_meet): Optiland's watertightness test, on rims
      that include a zone's inner rim and a spline surface's rims, and that
      leave out ends of radius 0 (KW-OPTILAND-06).
    - **Ray parity**: every ray from ``inside`` (a world point the profile proves
      is inside) crosses the boundary an odd number of times, traced through
      the live faces (_crossings). Optiland casts it from the mean of the rim
      points, which is outside a steep meniscus (KW-OPTILAND-04),
      through plain-conic copies of every conic subclass
      (KW-OPTILAND-07). Same directions, seed and errors as
      Optiland's.
    """
    from optiland.nonsequential.components.volume import (  # noqa: PLC0415
        _PARITY_DIRECTIONS,
        _PARITY_SEED,
    )
    import optiland.backend as be  # noqa: PLC0415

    if not boundary:
        raise NonWatertightVolumeError(f"Volume '{name}' has no boundary surfaces.")
    _check_rims_meet(boundary)
    rng = np.random.default_rng(_PARITY_SEED)
    dirs = rng.normal(size=(_PARITY_DIRECTIONS, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    previous = be.get_backend()
    try:
        be.set_backend("numpy")
        for d in dirs:
            count = _crossings(boundary, np.asarray(inside, dtype=float), d)
            if count % 2 == 0:
                raise NonWatertightVolumeError(
                    f"Volume boundary failed the inside/outside ray-parity check: a ray from "
                    f"{np.asarray(inside).tolist()} in direction {d.tolist()} crossed the boundary "
                    f"{count} times (expected an odd count from a point inside a closed solid): "
                    "a gap in the boundary, or a face the wrong way round.")
    finally:
        be.set_backend(previous)
    return Volume(name=name, boundary=boundary, interior=interior, _skip_validation=True)


class _RevolvedProfileBase(CompoundComponent):
    """What a revolved solid and a revolved sheet share: faces named
    ``"<name>.<short label>"``, looked up by short label, placed by ``cs``."""

    def _build(self, name, profile, interior_material, material_overrides, reflectances,
               cs, tolerances) -> None:
        self._name = name
        self.profile = profile
        self.frame = axis_frame(profile)
        self._cs = cs if cs is not None else CoordinateSystem()
        self._faces, self._surfaces = build_revolved_faces(
            profile, interior_material, material_overrides, reflectances,
            cs=self._cs, name_prefix=f"{name}.", **tolerances)
        self.azimuth_deg = profile_azimuth(profile)
        self._labels = {id(face): label for label, face in self._faces.items()}
        self.surface_kinds = {k: s.kind for k, s in self._surfaces.items()}

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
        """One named face, e.g. ``solid.face("S1")``."""
        try:
            return self._faces[short_label]
        except KeyError:
            raise KeyError(f"{self._name!r} has no face named {short_label!r}.") from None

    def label_of(self, face) -> str:
        """The short label of one of this component's faces."""
        try:
            return self._labels[id(face)]
        except KeyError:
            raise KeyError(f"{getattr(face, 'name', face)!r} is not a face of {self._name!r}.") from None

    def to_world(self, p: XY) -> np.ndarray:
        """A profile point as a world point, through the axis frame and ``cs``."""
        t, r = _get_transform(self._cs)
        return t + r @ np.array(self.frame.to_world(p))

    def source_cs(self, z: float, field_angle_deg: float = 0.0, x: float = 0.0) -> CoordinateSystem:
        """``AxisFrame.source_cs`` for this placed component: a coordinate system
        at ``z`` along its axis (``x`` off it, towards the half-section's
        side), propagating along the axis tilted by ``field_angle_deg`` within
        the profile plane."""
        return CoordinateSystem(x=x, y=0.0, z=z, ry=math.radians(field_angle_deg),
                                reference_cs=self._cs)


class RevolvedProfileSolid(_RevolvedProfileBase):
    """A solid of revolution built from a revolve profile.json, validated by Optiland.

    Construction builds one analytic surface per half-section span
    (build_revolved_faces) and then runs Optiland's own ``Volume`` checks on
    them: a watertightness check of every shared rim and a ray-parity
    inside/outside check (C-BRIDGE-09), the latter cast from
    ``interior_point(profile)`` (see the module docstring, KW-OPTILAND-04). A boundary that fails raises
    ``NonWatertightVolumeError`` here, before any ray is traced.

    Placement: ``cs`` places the profile's axis frame (see ``axis_frame``:
    the axis along local z through the axis point, the profile plane local
    xz) in the world, as a Lens's ``cs`` places its front vertex. Without it
    the axis frame is the world frame. ``cs=scene_placement(profile)``
    reproduces the element's place in its 2D scene. The faces refer to
    ``cs``, which is fixed at construction (Optiland's Volume check and
    traces read it then).

    Each face is named ``"<name>.<short label>"`` (``"L2.S1"``), as Optiland's
    Lens names its faces, so the faces of two solids built from the same
    kind of profile stay apart in the event log; ``face("S1")`` still looks
    a face up by its short label.

    Register it with the scene's registry, as for ExtrudedPolygonSolid:

        solid = RevolvedProfileSolid("lens", profile, glass, cs=scene_placement(profile))
        scene.component_registry.add(solid.name, solid)

    Attributes:
        profile: The profile.json dict this solid was built from.
        frame: Its AxisFrame (profile coordinates <-> the solid's local frame).
        volume: The validated optiland Volume.
        surface_kinds: {short label: "conic" | "asphere" | "spline" | "disc" | "annulus" | "frustum"}.
    """

    def __init__(
        self,
        name: str,
        profile: Dict[str, Any],
        interior_material,
        material_overrides: Optional[Dict[str, Any]] = None,
        reflectances: Optional[Dict[str, Any]] = None,
        *,
        cs: Optional[CoordinateSystem] = None,
        **tolerances,
    ) -> None:
        if profile.get("body", "solid") != "solid":
            raise ValueError(f"profile body is {profile.get('body')!r}: build a sheet with "
                             "RevolvedProfileSheet.")
        self._build(name, profile, interior_material, material_overrides, reflectances, cs,
                    tolerances)
        self.volume = _validated_volume(name, list(self._faces.values()), interior_material,
                                        self.to_world(interior_point(profile)))

    def interior_on_back(self, short_label: str) -> bool:
        """Whether the named face's interior (glass) side is its material_back."""
        return self._surfaces[short_label].interior_on_back


class RevolvedProfileSheet(_RevolvedProfileBase):
    """Revolved mirrors built from a ``body: "sheet"`` profile: no volume, no interior.

    Each span becomes one ``ReflectiveComponent`` on the exact surface of
    revolution the span table gives it (a paraboloid, ellipsoid, hyperboloid
    or sphere from a conic or circle span with its vertex on the axis; a
    disc, annulus or frustum from a line). A sheet encloses nothing, so no
    ``Volume`` is built: there is nothing to check closed (C-BRIDGE-12).

    Args:
        name: Prefix of the face names (``"<name>.<short label>"``).
        profile: An ``export_sheet_profile`` dict.
        reflectances: One reflectance for every span, or ``{short label:
            reflectance}``. Required: Optiland assumes no perfect mirror, and
            neither does the bridge.
        medium: The medium around the mirrors (default vacuum).
        cs: Places the axis frame in the world, as for RevolvedProfileSolid;
            ``scene_placement(profile)`` puts the mirrors where their 2D scene had them.

    A profile with ``symmetry.azimuth_deg = [start, end]`` is cut to that
    range about the axis, in degrees from the half-section's own side (the
    axis frame's +x) towards the frame's +y: ``[-90, 90]`` keeps the half on
    the half-section's side of the plane through the axis normal to the
    profile plane (the upper half of a projector reflector drawn above its
    axis). ``azimuth_deg`` holds it, None for the full turn.

    A conic or circle span that does not reach the axis becomes a zone (a
    conic with a central hole): a reflector that runs past its ellipsoid's
    equator is a cap and a zone of the far vertex's sheet. Solids may have zones
    too.

    Raises:
        ValueError: For a profile that is not a sheet, a span that does not
            reflect, a span with no reflectance, or an empty azimuth range.
        NotImplementedError: For an azimuth cut on a sheet with a line span.
    """

    def __init__(
        self,
        name: str,
        profile: Dict[str, Any],
        reflectances,
        *,
        medium=None,
        cs: Optional[CoordinateSystem] = None,
        **tolerances,
    ) -> None:
        if profile.get("body") != "sheet":
            raise ValueError(f"profile body is {profile.get('body', 'solid')!r}: build a solid "
                             "with RevolvedProfileSolid.")
        labels = [sp.get("short_label") or f"span_{sp['index_start']}" for sp in profile["spans"]]
        if not isinstance(reflectances, dict):
            reflectances = {label: reflectances for label in labels}
        overrides = {label: medium for label in labels} if medium is not None else None
        self._build(name, profile, None, overrides, reflectances, cs, tolerances)


# --- back from Optiland -------------------------------------------------------------


def _sample_arc_world(z_v: float, R: float, r_from: float, r_to: float, count: int):
    """``count`` points on the meridian of a sphere (vertex z_v, signed radius R),
    at equal angle steps from radius r_from to r_to, with radial unit normals."""
    zc = z_v + R
    a0 = math.asin(max(-1.0, min(1.0, r_from / abs(R))))
    a1 = math.asin(max(-1.0, min(1.0, r_to / abs(R))))
    pts, nrm = [], []
    for k in range(count):
        a = a0 + (a1 - a0) * k / (count - 1)
        r = abs(R) * math.sin(a)
        z = zc - R * math.cos(a)
        pts.append((r, 0.0, z))
        nrm.append(((r - 0.0) / abs(R), 0.0, (z - zc) / abs(R)))
    return pts, nrm


def _sample_conic_world(z_v: float, R: float, K: float, r_from: float, r_to: float, count: int):
    """``count`` points on the meridian of a conic of revolution (vertex z_v,
    signed radius R, conic K), at equal steps in r from r_from to r_to, with
    unit normals towards the vertex's centre of curvature (as the sheet export
    samples a ConicMirror)."""
    pts, nrm = [], []
    for k in range(count):
        r = r_from + (r_to - r_from) * k / (count - 1)
        c = 1.0 / R
        z = c * r * r / (1.0 + math.sqrt(max(0.0, 1.0 - (1.0 + K) * c * c * r * r)))
        nr, nz = -r, R - (1.0 + K) * z
        length = math.hypot(nr, nz)
        pts.append((r, 0.0, z_v + z))
        nrm.append((nr / length, 0.0, nz / length))
    return pts, nrm


def import_revolved_from_optiland(solid: "_RevolvedProfileBase") -> Dict[str, Any]:
    """Read a live RevolvedProfileSolid or RevolvedProfileSheet back out as a revolve profile.json dict.

    The revolve counterpart of import_from_optiland. Each face's geometry and
    coordinate system are read back -- a spline surface's position along the
    axis (its meridian is fixed when it is built), a conic's vertex, radius and aperture,
    a frustum's radii and ends, a disc's or annulus's radii and position -- so
    a face moved or reshaped after export (by hand, or by an optimiser) comes
    back moved. Labels, roles, mechanisms, the axis and construction
    parameters come from the profile the solid was built from. Each span
    keeps its point count and direction of travel; outward normals keep the
    side the original ones pointed to.

    Faces are read relative to the solid's own ``cs``: the profile comes back
    in its own coordinates wherever the solid is placed, and a face counts
    as moved only if it moved relative to its solid.

    Raises:
        ValueError: If a face has been rotated off the axis or moved off it.
    """
    profile = copy.deepcopy(solid.profile)
    frame = solid.frame
    points: List[List[float]] = []
    normals: List[List[float]] = []
    for span in profile["spans"]:
        name = span.get("short_label") or f"span_{span['index_start']}"
        face = solid.face(name)
        translation, rotation = _relative_transform(face.cs, solid.coordinate_system)
        if np.abs(translation[:2]).max() > AXIS_TOL or np.abs(rotation - np.eye(3)).max() > 1e-12:
            raise ValueError(f"face {name!r} is no longer on the axis, unrotated; it cannot be "
                             "read back as a surface of revolution.")
        cz = float(translation[2])
        g = face.geometry
        old_pts, old_nrm = _span_samples(span, solid.profile, frame)
        count = old_pts.shape[0]
        starts_inside = old_pts[0][0] <= old_pts[-1][0]
        if isinstance(g, SplineRevolutionGeometry):
            # Its meridian is fixed at construction; a face moved along the axis
            # comes back moved (cz), its curve's points and its samples with it.
            shift = np.array([0.0, 0.0, cz])
            pts = [tuple(q + shift) for q in old_pts]
            nrm = [tuple(v) for v in old_nrm]
            if cz != 0.0:
                curve = dict(span["curve"])
                curve["points"] = [list(frame.to_profile(tuple(np.array(frame.to_world(tuple(q))) + shift)))
                                   for q in curve["points"]]
                span["curve"] = curve
        elif isinstance(g, AsphereRevolutionGeometry):
            # Its surface is fixed at construction, in the face's frame; the
            # face's position along the axis is its vertex.
            r_from, r_to = ((g.inner_radius, g.outer_radius) if starts_inside
                            else (g.outer_radius, g.inner_radius))
            rs = np.linspace(r_from, r_to, count)
            z_m, n_m = g.meridian_at(rs)
            pts = [(float(r), 0.0, cz + float(z)) for r, z in zip(rs, z_m)]
            nrm = [(float(v[0]), 0.0, float(v[1])) for v in n_m]
            span["curve"] = {"type": "asphere", "vertex": list(frame.to_profile((0.0, 0.0, cz))),
                             "axis": list(frame.vector_to_profile((0.0, 0.0, 1.0))),
                             "radius": g.vertex_radius, "conic": g.asphere_conic,
                             "coefficients": list(g.coefficients)}
        elif isinstance(g, ConicGeometry):
            R, rim = as_float(g.radius), as_float(g.aperture_radius)
            K = as_float(g.conic)
            inner = float(getattr(g, "inner_radius", 0.0))
            r_from, r_to = (inner, rim) if starts_inside else (rim, inner)
            if span.get("curve", {}).get("type") == "conic" or K != 0.0:
                pts, nrm = _sample_conic_world(cz, R, K, r_from, r_to, count)
                span["curve"] = {"type": "conic",
                                 "vertex": list(frame.to_profile((0.0, 0.0, cz))),
                                 "axis": list(frame.vector_to_profile((0.0, 0.0, 1.0))),
                                 "radius": R, "conic": K}
            else:
                pts, nrm = _sample_arc_world(cz, R, r_from, r_to, count)
                span["curve"] = {"type": "circle",
                                 "center": list(frame.to_profile((0.0, 0.0, cz + R))),
                                 "radius": abs(R)}
        else:
            if isinstance(g, CylindricalFrustumGeometry):
                ends = [(as_float(g.r_front), 0.0, cz + as_float(g.z_front)),
                        (as_float(g.r_back), 0.0, cz + as_float(g.z_back))]
            elif isinstance(g, AnnularPlaneGeometry):
                z = cz + as_float(g.z_offset)
                ends = [(as_float(g.inner_radius), 0.0, z), (as_float(g.outer_radius), 0.0, z)]
            elif isinstance(g, FinitePlaneGeometry):
                ends = [(0.0, 0.0, cz), (as_float(g.aperture_radius), 0.0, cz)]
            else:
                raise ValueError(f"face {name!r}: unexpected geometry {type(g).__name__}")
            # keep the original direction of travel: pair ends with the old ends
            e0, e1 = np.array(ends[0]), np.array(ends[1])
            if (np.linalg.norm(e0 - old_pts[0]) + np.linalg.norm(e1 - old_pts[-1])
                    > np.linalg.norm(e1 - old_pts[0]) + np.linalg.norm(e0 - old_pts[-1])):
                e0, e1 = e1, e0
            d = e1 - e0
            n = np.array([d[2], 0.0, -d[0]]) / np.linalg.norm(d)
            pts, nrm = [tuple(e0), tuple(e1)], [tuple(n), tuple(n)]
            span["curve"] = {"type": "line"}
        # outward normals keep the side the originals pointed to
        mid = count // 2
        if float(np.dot(nrm[mid], old_nrm[mid])) < 0.0:
            nrm = [tuple(-np.asarray(v)) for v in nrm]
        span["index_start"] = len(points)
        points.extend(list(frame.to_profile(p)) for p in pts)
        normals.extend(list(frame.vector_to_profile(v)) for v in nrm)
        span["index_end"] = len(points)
    profile["points"] = points
    profile["normals"] = normals
    return profile


def interior_side_material(solid, face):
    """The interior material a refracting face of ``solid`` carries."""
    on_back = (solid.interior_on_back(solid.label_of(face))
               if isinstance(solid, RevolvedProfileSolid) else True)
    return face.material_back if on_back else face.material_front


__all__ = [
    "AxisFrame",
    "axis_frame",
    "profile_azimuth",
    "scene_placement",
    "interior_point",
    "RevolvedSurface",
    "surface_from_span",
    "build_revolved_faces",
    "RevolvedProfileSolid",
    "RevolvedProfileSheet",
    "import_revolved_from_optiland",
    "interior_side_material",
]
