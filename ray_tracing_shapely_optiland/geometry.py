"""
Python-specific module: geometry
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
GEOMETRY -- for Optiland NSQ: a flat polygon, a cut conic, a spline and an asphere of revolution
===============================================================================
Four geometries Optiland lacks. Two are subclasses of the nearest Optiland
one that narrow only the aperture test: PolygonPlaneGeometry (below) and
CutConicGeometry (a conic of revolution with a central hole and/or cut to
an azimuth range; see its class docstring). The third,
SplineRevolutionGeometry, is a surface of revolution whose
meridian is a spline: its own intersection and normals throughout, a
ConicGeometry subclass only so that the backends' lowering accepts it (see
its class docstring). The fourth, AsphereRevolutionGeometry, is the even
asphere of revolution, built the same way.

PolygonPlaneGeometry is a flat surface bounded by an arbitrary simple polygon:
the end cap an extruded profile needs, which none of Optiland's analytic
geometries provides (FinitePlaneGeometry is a rectangle or a disc). It is
analytic, so it traces on both NSQ backends and keeps gradients flowing
through the hit point under TorchBackend, unlike a triangulated
MeshGeometry cap, which is forward-only.

Why it subclasses FinitePlaneGeometry. NSQ's backends lower every scene to a
SceneIR before each trace, and that lowering recognises geometry by class: an
unknown class raises TypeError before any ray moves. Intersection itself
never reads the lowered parameters -- both backends call each live
component's own intersect(), which calls this class's ray_intersect(). So
the subclass is lowered as a "plane" whose width and height are the
polygon's bounding rectangle (a superset of the polygon, never a subset),
while every ray is tested against the true polygon. No Optiland source edit
is involved.

What still sees only the rectangle. Anything in Optiland that inspects the
geometry's parameters instead of calling ray_intersect: NSQScene.to_json
serialisation, and Volume's watertightness and ray-parity checks
(_rim_points samples a rectangle perimeter; _detached_geometry clones a
plain FinitePlaneGeometry). Do not serialise a scene containing this
geometry, and do not wrap it in a Volume: both would describe the bounding
rectangle, not the polygon. ExtrudedPolygonSolid does neither.
===============================================================================
"""

from __future__ import annotations

import inspect
import math
from typing import Optional, Sequence, Tuple

import numpy as np

import optiland.backend as be
from optiland.nonsequential.components.geometry.analytic.conic import ConicGeometry
from optiland.nonsequential.components.geometry.analytic.plane import FinitePlaneGeometry

# Whether this Optiland's plane intersection takes the self-intersection
# threshold. Releases up to 0.6.3 do not: the engine calls
# ray_intersect(origins, directions) and applies its own fixed guard after.
# Later ones pass a threshold, which a subclass must hand on.
_PLANE_TAKES_EPS = "eps" in inspect.signature(FinitePlaneGeometry.ray_intersect).parameters


class PolygonPlaneGeometry(FinitePlaneGeometry):
    """Flat surface at local z=0, bounded by a simple polygon in local (x, y).

    The polygon may be convex or not; containment uses the even-odd rule, so
    the vertices may be given in either winding. Only intersection is
    polygon-aware; bounding_box() is the inherited rectangle's, a superset,
    which is what an AABB only needs to be. See the module docstring for
    what else sees the rectangle instead.

    Attributes:
        vertices: The polygon's vertices in the local frame, as an (N, 2)
            float array, N >= 3. The local origin need not be inside it.
    """

    def __init__(self, vertices: Sequence[Tuple[float, float]]) -> None:
        """Initialize from the polygon's local-frame vertices.

        Args:
            vertices: At least three (x, y) pairs in the local frame, not
                repeating the first vertex at the end.

        Raises:
            ValueError: If fewer than three vertices are given, any
                coordinate is not finite, or the polygon has zero area.
        """
        verts = np.asarray(vertices, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
            raise ValueError(
                f"PolygonPlaneGeometry needs at least three (x, y) vertices, got shape {verts.shape}."
            )
        if not np.all(np.isfinite(verts)):
            raise ValueError("PolygonPlaneGeometry vertices must be finite.")
        x, y = verts[:, 0], verts[:, 1]
        area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        if abs(area) <= 1e-15:
            raise ValueError("PolygonPlaneGeometry polygon has zero area.")

        # A bounding rectangle centred on the local origin, so the parent's
        # aperture test is a superset of the polygon whatever the origin is.
        super().__init__(
            width=2.0 * float(np.max(np.abs(x))),
            height=2.0 * float(np.max(np.abs(y))),
        )
        self.vertices = verts

    def contains(self, hx, hy):
        """Even-odd point-in-polygon test, vectorised over backend arrays.

        Args:
            hx: Local x coordinates, shape (N,), a NumPy array or a torch
                tensor according to the active backend.
            hy: Local y coordinates, same shape.

        Returns:
            Boolean array, shape (N,): True where (hx, hy) is inside.
        """
        crossings = be.zeros_like(hx)
        ones = be.ones_like(hx)
        zeros = be.zeros_like(hx)
        count = len(self.vertices)
        for i in range(count):
            x1, y1 = (float(v) for v in self.vertices[i])
            x2, y2 = (float(v) for v in self.vertices[(i + 1) % count])
            if y1 == y2:
                continue  # a horizontal edge never straddles a horizontal ray
            straddles = (hy < y1) != (hy < y2)
            x_cross = x1 + (hy - y1) * ((x2 - x1) / (y2 - y1))
            crossings = crossings + be.where(straddles & (hx < x_cross), ones, zeros)
        return (crossings % 2) == 1

    def ray_intersect(
        self,
        origins,
        directions,
        eps: Optional[float] = None,
    ):
        """Intersect rays with the polygon.

        The plane solve, normals and n_geom (local +z, the material_back side)
        are FinitePlaneGeometry's own; only the aperture test is narrowed from
        the bounding rectangle to the polygon.

        Args:
            origins: Ray origins in the local frame, shape (N, 3).
            directions: Ray directions in the local frame, shape (N, 3).
            eps: See ComponentGeometry.ray_intersect.

        Returns:
            (t, normals, hit_mask, n_geom), as FinitePlaneGeometry returns them.
        """
        if _PLANE_TAKES_EPS:
            t, normals, hit_mask, n_geom = super().ray_intersect(origins, directions, eps)
        else:
            t, normals, hit_mask, n_geom = super().ray_intersect(origins, directions)
        safe_t = be.where(hit_mask, t, be.zeros_like(t))
        hx = origins[:, 0] + safe_t * directions[:, 0]
        hy = origins[:, 1] + safe_t * directions[:, 1]
        hit_mask = hit_mask & self.contains(hx, hy)
        t = be.where(hit_mask, t, be.ones_like(t) * be.inf)
        return t, normals, hit_mask, n_geom


def _snap(v: float) -> float:
    """``v`` with rounding noise around 0 and +-1 removed (|error| < 1e-15)."""
    for exact in (0.0, 1.0, -1.0):
        if abs(v - exact) < 1e-15:
            return exact
    return v


class CutConicGeometry(ConicGeometry):
    """A conic of revolution cut to a zone ``inner_radius <= r <= aperture_radius``
    and, optionally, to an azimuth range about its axis.

    The reflector of a projector headlamp needs both cuts: past its
    ellipsoid's equator it is a ring of the far vertex's sheet (a hole in the
    middle), and it covers only the upper half (an azimuth range). Optiland's
    ConicGeometry has a full disc aperture only.

    Only the per-root aperture test is narrowed (``_root_valid``), so each
    root of the quadric is judged on its own: a ray whose nearer root falls
    in a removed part still meets the kept part at its farther root. The
    quadric solve, the normals and n_geom are ConicGeometry's own.

    NSQ's lowering recognises geometry by class, so this lowers as a plain
    conic with the same outer aperture (a superset); intersection always
    calls this class's ray_intersect (see the module docstring for what
    still sees the uncut conic: serialisation and Volume's rim sampling;
    the bridge uses it only in sheets, which build no Volume).

    Azimuth is measured in the local frame from +x towards +y. The range
    includes its edges, and the axis point itself (where the azimuth is
    undefined) counts as inside.

    Attributes:
        inner_radius: Radius of the central hole (0: none).
        azimuth_deg: ``(start, end)`` in degrees, ``0 < end - start <= 360``,
            or None for the full turn.
    """

    def __init__(
        self,
        radius: float,
        conic: float,
        aperture_radius: float,
        *,
        inner_radius: float = 0.0,
        azimuth_deg: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Initialize the conic and its cuts.

        Args:
            radius: Vertex radius of curvature, as for ConicGeometry.
            conic: Conic constant K.
            aperture_radius: Outer radius of the kept zone.
            inner_radius: Inner radius of the kept zone.
            azimuth_deg: Kept azimuth range ``(start, end)`` in degrees.

        Raises:
            ValueError: If ``0 <= inner_radius < aperture_radius`` fails, or
                the azimuth range is empty or exceeds a full turn.
        """
        super().__init__(radius, conic, aperture_radius)
        inner_radius = float(inner_radius)
        if not (0.0 <= inner_radius < float(aperture_radius)):
            raise ValueError(
                f"CutConicGeometry needs 0 <= inner_radius < aperture_radius, got "
                f"{inner_radius!r} and {aperture_radius!r}."
            )
        self.inner_radius = inner_radius
        self.azimuth_deg: Optional[Tuple[float, float]] = None
        if azimuth_deg is not None:
            start, end = (float(a) for a in azimuth_deg)
            if not (math.isfinite(start) and math.isfinite(end) and 0.0 < end - start <= 360.0):
                raise ValueError(
                    f"azimuth_deg must be (start, end) with 0 < end - start <= 360, got {azimuth_deg!r}."
                )
            if end - start < 360.0:
                self.azimuth_deg = (start, end)
                a0, a1 = math.radians(start), math.radians(end)
                # exact zeros at multiples of 90 deg: cos(90 deg) is 6e-17 in
                # floating point, which would put the y axis outside (-90, 90)
                self._u0 = tuple(_snap(v) for v in (math.cos(a0), math.sin(a0)))
                self._u1 = tuple(_snap(v) for v in (math.cos(a1), math.sin(a1)))
                self._reflex = end - start > 180.0

    def in_azimuth(self, x, y):
        """Whether local points (x, y) lie in the kept azimuth range.

        Uses half-plane tests (cross products), not angles, so it needs no
        arctan2 on either backend and has no branch cut. A range of at most
        180 deg is the intersection of two half-planes; a wider one is the
        complement of the (narrower) removed range.

        Args:
            x: Local x, shape (N,), a NumPy array or torch tensor.
            y: Local y, same shape.

        Returns:
            Boolean array, shape (N,).
        """
        if self.azimuth_deg is None:
            return be.ones_like(x) > 0.0
        (c0, s0), (c1, s1) = self._u0, self._u1
        after_start = c0 * y - s0 * x  # cross(u0, p): > 0 counter-clockwise of start
        before_end = x * s1 - y * c1  # cross(p, u1): > 0 clockwise of end
        if not self._reflex:
            return (after_start >= 0.0) & (before_end >= 0.0)
        return ~((after_start < 0.0) & (before_end < 0.0))

    def _root_valid(self, t, solvable, origins, directions, eps):
        """ConicGeometry's root test, narrowed to the kept zone and azimuth range.

        Args:
            t, solvable, origins, directions, eps: As ConicGeometry._root_valid.

        Returns:
            (valid, px, py), as ConicGeometry._root_valid.
        """
        valid, px, py = super()._root_valid(t, solvable, origins, directions, eps)
        kept = self.in_azimuth(px, py)
        if self.inner_radius > 0.0:
            kept = kept & ((px**2 + py**2) >= self.inner_radius**2)
        return valid & kept, px, py


#: Sub-intervals per spline segment on which a root is bracketed (see
#: SplineRevolutionGeometry.ray_intersect).
SPLINE_BRACKETS = 16
#: Halvings refining a bracketed root (past double precision on [0, 1]).
SPLINE_HALVINGS = 64


class SplineRevolutionGeometry(ConicGeometry):
    """A surface of revolution about the local z axis whose meridian is a spline.

    The meridian is the 2D spline glass edge's cubic Hermite spline
    (core.glass_curves.SplineCurve): through ``points`` (r, z) with unit
    ``tangents``, each segment's tangents scaled by its chord, over the
    parameter range ``domain`` (default: the whole spline). Its radius must
    be monotonic along it, so the surface is single-valued as z(r), and
    non-negative: it covers ``inner_radius <= r <= outer_radius`` (a zone
    when ``inner_radius > 0``).

    Intersection. A ray o + t d meets the surface where its meridional trace
    meets the spline. Per spline segment, with the segment's parameter u:

    - **a ray mostly across the axis** (|d_perp|^2 >= d_z^2): at the radius
      r(u) the ray is at two distances t±(u) (the roots of
      |o_perp + t d_perp|^2 = r(u)^2); a hit is a root of
      f±(u) = o_z + t±(u) d_z - z(u). Where r(u) is less than the ray's
      distance from the axis, t± do not exist; the edge of that range is
      found and used as a bracket end, where f+ = f-.
    - **a ray mostly along the axis**: t(u) = (z(u) - o_z) / d_z, and a hit
      is a root of r(u)^2 - |o_perp + t(u) d_perp|^2.

    Both forms are well conditioned on their side (a ray square to the axis
    is the first form's simplest case, and the hard case of the second).
    Roots are bracketed by sign changes on SPLINE_BRACKETS equal
    sub-intervals of each segment, after a test of the segment's bounding
    box, and refined by SPLINE_HALVINGS bisections. Two roots inside one
    sub-interval (a ray grazing the surface, tangent to it within 1/16 of a
    segment) are not seen: a grazing ray passes. The normal is exact: the
    spline's tangent turned in the meridional plane.

    Why a ConicGeometry subclass. Both NSQ backends lower a scene before each
    trace, by class, and an unknown class raises TypeError; intersection
    never reads the lowered parameters (each component's own intersect()
    calls this class's ray_intersect()). The conic parameters are therefore
    placeholders: a flat disc of the outer radius. Serialising a scene
    (NSQScene.to_json) or Optiland's Volume checks would see that disc; the
    bridge does neither (RevolvedProfileSolid runs its own checks).
    NumPy only, forward only: no gradients flow through the spline.

    Attributes:
        inner_radius, outer_radius: The radii the surface covers.
        z_inner, z_outer: The meridian's z at those radii.
    """

    def __init__(self, points: Sequence[Tuple[float, float]], tangents: Sequence[Tuple[float, float]],
                 domain: Optional[Tuple[float, float]] = None) -> None:
        pts = np.asarray(points, dtype=float)
        tans = np.asarray(tangents, dtype=float)
        n = pts.shape[0]
        if pts.ndim != 2 or pts.shape[1] != 2 or n < 2 or tans.shape != pts.shape:
            raise ValueError("a spline meridian needs at least 2 (r, z) points and one tangent per point.")
        tans = tans / np.linalg.norm(tans, axis=1, keepdims=True)
        lo, hi = (0.0, float(n - 1)) if domain is None else (float(domain[0]), float(domain[1]))
        lo, hi = min(lo, hi), max(lo, hi)
        if not (0.0 <= lo < hi <= n - 1):
            raise ValueError(f"spline domain must lie in [0, {n - 1}] and not be empty, got {domain!r}.")
        self._pts, self._tans = pts, tans
        if self._point(hi)[0] < self._point(lo)[0]:   # run it outwards: r grows with the parameter
            pts, tans = pts[::-1].copy(), -tans[::-1]
            lo, hi = (n - 1) - hi, (n - 1) - lo
            self._pts, self._tans = pts, tans
        self._lo, self._hi = lo, hi
        self._segments = []                           # (k, u0, u1, r-coef, z-coef, hull)
        for k in range(int(math.floor(lo)), int(math.ceil(hi))):
            u0, u1 = max(lo - k, 0.0), min(hi - k, 1.0)
            if u1 <= u0:
                continue
            a, b, c, d = self._coefficients(k)
            bez = np.array([pts[k], pts[k] + b / 3.0, pts[k + 1] - self._m1(k) / 3.0, pts[k + 1]])
            self._segments.append((k, u0, u1, (a, b, c, d), bez))
        for k, u0, u1, (a, b, c, d), _ in self._segments:   # r'(u) >= 0 on [u0, u1]
            us = [u0, u1]
            if d[0] != 0.0 and u0 < -c[0] / (3.0 * d[0]) < u1:
                us.append(-c[0] / (3.0 * d[0]))
            if min(b[0] + 2.0 * c[0] * u + 3.0 * d[0] * u * u for u in us) < -1e-12 * max(1.0, abs(b[0])):
                raise ValueError("the spline meridian is not monotonic in r: its surface of revolution "
                                 "is not single-valued as z(r).")
        (r_in, z_in), (r_out, z_out) = self._point(lo), self._point(hi)
        if r_in < -1e-9 * max(1.0, r_out):
            raise ValueError("the spline meridian crosses the axis (r < 0); cut it there first.")
        self.inner_radius, self.outer_radius = max(r_in, 0.0), r_out
        self.z_inner, self.z_outer = z_in, z_out
        super().__init__(math.inf, 0.0, r_out)

    # --- the meridian -------------------------------------------------------------

    def _m1(self, k: int) -> np.ndarray:
        h = float(np.linalg.norm(self._pts[k + 1] - self._pts[k]))
        return self._tans[k + 1] * h

    def _coefficients(self, k: int):
        p0, p1 = self._pts[k], self._pts[k + 1]
        h = float(np.linalg.norm(p1 - p0))
        if h == 0.0:
            raise ValueError(f"spline points {k} and {k + 1} coincide.")
        m0, m1 = self._tans[k] * h, self._tans[k + 1] * h
        return p0, m0, -3 * p0 - 2 * m0 + 3 * p1 - m1, 2 * p0 + m0 - 2 * p1 + m1

    def _point(self, p: float) -> np.ndarray:
        k = min(max(int(math.floor(p)), 0), self._pts.shape[0] - 2)
        a, b, c, d = self._coefficients(k)
        u = p - k
        return a + u * (b + u * (c + u * d))

    @staticmethod
    def _eval(coef, u):
        """(r, z, dr/du, dz/du) of one segment at parameters ``u`` (any shape)."""
        a, b, c, d = coef
        u = np.asarray(u, dtype=float)
        r = a[0] + u * (b[0] + u * (c[0] + u * d[0]))
        z = a[1] + u * (b[1] + u * (c[1] + u * d[1]))
        dr = b[0] + u * (2 * c[0] + 3 * u * d[0])
        dz = b[1] + u * (2 * c[1] + 3 * u * d[1])
        return r, z, dr, dz

    def meridian_at(self, r):
        """The meridian at radii ``r`` (within the covered range): (z, unit normal
        (n_r, n_z)), the normal oriented as n_geom (towards +z where the surface
        faces the axis direction)."""
        r = np.atleast_1d(np.asarray(r, dtype=float))
        z = np.empty_like(r)
        nrm = np.empty((r.shape[0], 2))
        done = np.zeros(r.shape, dtype=bool)
        for i, (_, u0, u1, coef, _) in enumerate(self._segments):
            r0 = self._eval(coef, u0)[0]
            r1 = self._eval(coef, u1)[0]
            last = i == len(self._segments) - 1
            sel = ~done & (r >= r0 - 1e-12 * max(1.0, abs(r0))) & ((r <= r1) | last)
            if not np.any(sel):
                continue
            lo_u, hi_u = np.full(sel.sum(), u0), np.full(sel.sum(), u1)
            target = r[sel]
            for _ in range(SPLINE_HALVINGS):
                mid = 0.5 * (lo_u + hi_u)
                below = self._eval(coef, mid)[0] < target
                lo_u, hi_u = np.where(below, mid, lo_u), np.where(below, hi_u, mid)
            _, zz, dr, dz = self._eval(coef, 0.5 * (lo_u + hi_u))
            length = np.hypot(dr, dz)
            z[sel], nrm[sel, 0], nrm[sel, 1] = zz, -dz / length, dr / length
            done |= sel
        return z, nrm

    # --- Optiland's geometry interface -----------------------------------------

    def _sag(self, x, y):
        return self.meridian_at(np.hypot(np.asarray(x, dtype=float), np.asarray(y, dtype=float)))[0]

    def _normal_local(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        rho = np.hypot(x, y)
        _, nrm = self.meridian_at(rho)
        safe = np.where(rho > 0.0, rho, 1.0)
        ex, ey = np.where(rho > 0.0, x / safe, 1.0), np.where(rho > 0.0, y / safe, 0.0)
        return np.stack([nrm[:, 0] * ex, nrm[:, 0] * ey, nrm[:, 1]], axis=1)

    def bounding_box(self, transform):
        from optiland.nonsequential.components.geometry.base import AABB  # noqa: PLC0415

        t_vec = np.array(transform[0], dtype=float)
        rot = np.array(transform[1], dtype=float)
        zs = np.concatenate([seg[4][:, 1] for seg in self._segments])
        r = self.outer_radius
        corners = np.array([[sx * r, sy * r, z] for sx in (-1, 1) for sy in (-1, 1)
                            for z in (zs.min(), zs.max())])
        world = corners @ rot.T + t_vec
        return AABB(world.min(axis=0), world.max(axis=0))

    def rims(self):
        """The surface's edge circles, (r, z) with r > 0."""
        out = [(self.outer_radius, self.z_outer)]
        if self.inner_radius > 0.0:
            out.append((self.inner_radius, self.z_inner))
        return out

    def ray_intersect(self, origins, directions, eps=None):
        o = np.asarray(origins, dtype=float)
        d = np.asarray(directions, dtype=float)
        count = o.shape[0]
        eps_arr = np.broadcast_to(np.asarray(1e-9 if eps is None else eps, dtype=float), (count,))
        ox, oy, oz = o[:, 0], o[:, 1], o[:, 2]
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        A = dx * dx + dy * dy
        B = ox * dx + oy * dy
        C = ox * ox + oy * oy
        across = A >= dz * dz
        safe_a = np.where(A > 0.0, A, 1.0)
        rho_min2 = np.where(across, np.maximum(C - B * B / safe_a, 0.0), 0.0)
        best_t = np.full(count, np.inf)
        best_seg = np.full(count, -1)
        best_u = np.zeros(count)
        steps = np.arange(SPLINE_BRACKETS + 1) / SPLINE_BRACKETS

        def keep(rays, seg_i, u, t):
            ok = np.isfinite(t) & (t > eps_arr[rays])
            rays, u, t = rays[ok], u[ok], t[ok]
            order = np.lexsort((t, rays))              # each ray's nearest root first
            rays, u, t = rays[order], u[order], t[order]
            first = np.r_[True, rays[1:] != rays[:-1]] if rays.size else np.zeros(0, dtype=bool)
            rays, u, t = rays[first], u[first], t[first]
            better = t < best_t[rays]
            rays, u, t = rays[better], u[better], t[better]
            best_t[rays], best_seg[rays], best_u[rays] = t, seg_i, u

        for seg_i, (_, u0, u1, coef, bez) in enumerate(self._segments):
            r_lo, r_hi = self._eval(coef, u0)[0], self._eval(coef, u1)[0]
            z_lo, z_hi = bez[:, 1].min(), bez[:, 1].max()
            # The bounding test: some t > eps with z in [z_lo, z_hi] and rho in [r_lo, r_hi].
            with np.errstate(divide="ignore", invalid="ignore"):
                ta, tb = (z_lo - oz) / dz, (z_hi - oz) / dz
            flat = np.abs(dz) < 1e-300
            t_from = np.where(flat, np.where((oz >= z_lo) & (oz <= z_hi), -np.inf, np.inf), np.minimum(ta, tb))
            t_to = np.where(flat, np.where((oz >= z_lo) & (oz <= z_hi), np.inf, -np.inf), np.maximum(ta, tb))
            t_from = np.maximum(t_from, eps_arr)
            vertex = np.clip(-B / safe_a, t_from, t_to)
            q_min = np.where(A > 0.0, A * vertex * vertex + 2 * B * vertex + C, C)
            with np.errstate(invalid="ignore", over="ignore"):
                q_ends = np.maximum(A * t_from * t_from + 2 * B * t_from + C, A * t_to * t_to + 2 * B * t_to + C)
            q_ends = np.where(np.isnan(q_ends), np.inf, q_ends)
            pad = 1e-9 * max(1.0, r_hi * r_hi)
            cand = (t_from <= t_to) & (q_min <= r_hi * r_hi + pad) & (q_ends >= r_lo * r_lo - pad)
            if not np.any(cand):
                continue
            us = u0 + (u1 - u0) * steps
            r_s, z_s, _, _ = self._eval(coef, us)

            # Rays mostly along the axis: g(u) = r(u)^2 - rho(t(u))^2.
            ia = np.nonzero(cand & ~across)[0]
            if ia.size:
                def g(rays, u):
                    r, z, _, _ = self._eval(coef, u)
                    t = (z - oz[rays]) / dz[rays]
                    return r * r - (A[rays] * t * t + 2 * B[rays] * t + C[rays]), t
                vals = r_s[None, :] ** 2 - (A[ia, None] * ((z_s[None, :] - oz[ia, None]) / dz[ia, None]) ** 2
                                            + 2 * B[ia, None] * (z_s[None, :] - oz[ia, None]) / dz[ia, None]
                                            + C[ia, None])
                rays, lo_u, hi_u = _brackets(ia, vals, us)
                if rays.size:
                    u = _bisect(lambda rr, uu: g(rr, uu)[0], rays, lo_u, hi_u)
                    keep(rays, seg_i, u, g(rays, u)[1])

            # Rays mostly across the axis: f±(u) = o_z + t±(u) d_z - z(u).
            it = np.nonzero(cand & across)[0]
            if it.size:
                rho_min = np.sqrt(rho_min2[it])
                # where the ray's distance from the axis equals r(u): the edge of t±'s range
                u_edge = np.full(it.size, np.nan)
                inside = (rho_min > r_lo) & (rho_min < r_hi)
                if np.any(inside):
                    lo_e, hi_e = np.full(inside.sum(), u0), np.full(inside.sum(), u1)
                    target = rho_min[inside]
                    for _ in range(SPLINE_HALVINGS):
                        mid = 0.5 * (lo_e + hi_e)
                        below = self._eval(coef, mid)[0] < target
                        lo_e, hi_e = np.where(below, mid, lo_e), np.where(below, hi_e, mid)
                    u_edge[inside] = 0.5 * (lo_e + hi_e)
                for sign in (1.0, -1.0):
                    def f(rays, u, sign=sign):
                        r, z, _, _ = self._eval(coef, u)
                        disc = np.maximum(B[rays] ** 2 - A[rays] * (C[rays] - r * r), 0.0)
                        t = (-B[rays] + sign * np.sqrt(disc)) / A[rays]
                        return oz[rays] + t * dz[rays] - z, t
                    disc = B[it, None] ** 2 - A[it, None] * (C[it, None] - r_s[None, :] ** 2)
                    valid = disc >= 0.0
                    t_s = (-B[it, None] + sign * np.sqrt(np.maximum(disc, 0.0))) / A[it, None]
                    vals = oz[it, None] + t_s * dz[it, None] - z_s[None, :]
                    vals = np.where(valid, vals, np.nan)
                    # an interval with one end out of range: replace that end by the edge
                    lo_list, hi_list, ray_list = [], [], []
                    part = valid[:, :-1] != valid[:, 1:]
                    rows, cols = np.nonzero(part)
                    if rows.size:
                        ue = u_edge[rows]
                        ok = np.isfinite(ue)
                        rows, cols, ue = rows[ok], cols[ok], ue[ok]
                        right_valid = valid[rows, cols + 1]
                        other = np.where(right_valid, us[cols + 1], us[cols])
                        f_edge = f(it[rows], ue)[0]
                        f_other = np.where(right_valid, vals[rows, cols + 1], vals[rows, cols])
                        change = f_edge * f_other <= 0.0
                        ray_list.append(it[rows[change]])
                        lo_list.append(np.minimum(ue, other)[change])
                        hi_list.append(np.maximum(ue, other)[change])
                    rays, lo_u, hi_u = _brackets(it, vals, us)
                    rays = np.concatenate([rays] + ray_list)
                    lo_u = np.concatenate([lo_u] + lo_list)
                    hi_u = np.concatenate([hi_u] + hi_list)
                    if rays.size:
                        u = _bisect(lambda rr, uu: f(rr, uu)[0], rays, lo_u, hi_u)
                        keep(rays, seg_i, u, f(rays, u)[1])

        hit = np.isfinite(best_t)
        normals_geom = np.zeros((count, 3))
        normals_geom[:, 2] = 1.0
        for seg_i, (_, _, _, coef, _) in enumerate(self._segments):
            sel = best_seg == seg_i
            if not np.any(sel):
                continue
            _, _, dr, dz_u = self._eval(coef, best_u[sel])
            p = o[sel] + best_t[sel, None] * d[sel]
            rho = np.hypot(p[:, 0], p[:, 1])
            safe = np.where(rho > 0.0, rho, 1.0)
            ex, ey = np.where(rho > 0.0, p[:, 0] / safe, 1.0), np.where(rho > 0.0, p[:, 1] / safe, 0.0)
            n = np.stack([-dz_u * ex, -dz_u * ey, dr], axis=1)
            normals_geom[sel] = n / np.linalg.norm(n, axis=1, keepdims=True)
        facing = np.sum(d * normals_geom, axis=1, keepdims=True) > 0.0
        normals = np.where(facing, -normals_geom, normals_geom)
        return np.where(hit, best_t, np.inf), normals, hit, normals_geom


#: Samples along a ray's candidate stretch on which an asphere's roots are bracketed.
ASPHERE_SAMPLES = 32


class AsphereRevolutionGeometry(ConicGeometry):
    """The even asphere of revolution about the local z axis, its vertex at the origin.

    ``z(r) = c r**2 / (1 + sqrt(1 - (1 + K) c**2 r**2)) + A1 r**2 + A2 r**4 + ...``,
    c = 1 / R (0 for ``radius`` None), the coefficients in Zemax's and
    Optiland's ``even_asphere`` order: the 2D edge's surface
    (core.glass_curves.AsphereCurve), revolved. It covers
    ``inner_radius <= r <= outer_radius`` (a zone when ``inner_radius > 0``).

    Intersection. With q(t) = |o_perp + t d_perp|**2, a quadratic in t, the
    sag is a smooth function of q, so a ray o + t d meets the surface at the
    roots of F(t) = o_z + t d_z - Z(q(t)), smooth even where the ray crosses
    the axis. Only the stretch of t where the ray is within the surface's
    radii and sag bounds is searched (most rays miss it, at the cost of a
    comparison); on it F is sampled ASPHERE_SAMPLES times, roots bracketed by
    sign changes, and where F turns between two samples without changing
    sign it is split at the turn, so two crossings close together are found.
    Roots are refined by SPLINE_HALVINGS bisections. A ray that only grazes,
    crossing twice closer than the refinement separates, can pass. A ray
    parallel to the axis is solved directly. The normal is exact, from the
    sag's derivative.

    Why a ConicGeometry subclass: as for SplineRevolutionGeometry, only so
    that the backends' lowering accepts it; the conic parameters are
    placeholders (a flat disc of the outer radius). NumPy only, forward only.

    Attributes:
        vertex_radius, asphere_conic, coefficients: The surface, in this frame.
        inner_radius, outer_radius: The radii the surface covers.
        z_inner, z_outer: The sag at those radii.
    """

    def __init__(self, radius: Optional[float], conic: float, coefficients: Sequence[float],
                 outer_radius: float, inner_radius: float = 0.0) -> None:
        if radius is not None and (radius == 0.0 or not math.isfinite(radius)):
            raise ValueError(f"an asphere's vertex radius must be finite and non-zero, or None; got {radius!r}")
        coefficients = [float(a) for a in coefficients]
        if not all(math.isfinite(a) for a in coefficients):
            raise ValueError(f"asphere coefficients must be finite, got {coefficients}")
        if not 0.0 <= inner_radius < outer_radius:
            raise ValueError(f"need 0 <= inner_radius < outer_radius, got {inner_radius}, {outer_radius}")
        self.vertex_radius = None if radius is None else float(radius)
        self.asphere_conic = float(conic)
        self.coefficients = coefficients
        self._c = 0.0 if radius is None else 1.0 / float(radius)
        if 1.0 - (1.0 + self.asphere_conic) * self._c ** 2 * outer_radius ** 2 <= 0.0:
            raise ValueError(f"the asphere's conic base stops being single-valued before r = {outer_radius}.")
        self.inner_radius, self.outer_radius = float(inner_radius), float(outer_radius)
        self.z_inner, self.z_outer = (float(self.sag_q(np.array([v * v]))[0])
                                      for v in (self.inner_radius, self.outer_radius))
        rs = np.linspace(self.inner_radius, self.outer_radius, 8 * ASPHERE_SAMPLES + 1)
        zs = self.sag_q(rs * rs)
        step = rs[1] - rs[0]
        # z''(r) = G(q) + 2 q G'(q): bounded here by finite differences of z, widened twice
        bend = np.abs(np.diff(zs, 2)).max() / (step * step) if rs.size > 2 else 0.0
        margin = 2.0 * bend * step * step / 8.0 + 1e-12 * max(1.0, self.outer_radius)
        self._z_lo, self._z_hi = float(zs.min() - margin), float(zs.max() + margin)
        super().__init__(math.inf, 0.0, self.outer_radius)

    # --- the sag, as a function of q = r**2 -------------------------------------

    def sag_q(self, q):
        """z at r**2 = ``q``."""
        q = np.asarray(q, dtype=float)
        root = np.sqrt(np.maximum(1.0 - (1.0 + self.asphere_conic) * self._c ** 2 * q, 0.0))
        z = self._c * q / (1.0 + root)
        power = q.copy()
        for a in self.coefficients:
            z = z + a * power
            power = power * q
        return z

    def slope_over_r(self, q):
        """dz/dr divided by r, at r**2 = ``q``: finite on the axis."""
        q = np.asarray(q, dtype=float)
        g = self._c / np.sqrt(np.maximum(1.0 - (1.0 + self.asphere_conic) * self._c ** 2 * q, 1e-300))
        power = np.ones_like(q)
        for i, a in enumerate(self.coefficients, start=1):
            g = g + 2 * i * a * power
            power = power * q
        return g

    def meridian_at(self, r):
        """(z, unit normal (n_r, n_z)) at radii ``r``, the normal towards +z."""
        r = np.atleast_1d(np.asarray(r, dtype=float))
        slope = self.slope_over_r(r * r) * r
        length = np.hypot(slope, 1.0)
        return self.sag_q(r * r), np.stack([-slope / length, 1.0 / length], axis=1)

    # --- Optiland's geometry interface -----------------------------------------

    def _sag(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return self.sag_q(x * x + y * y)

    def _normal_local(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        g = self.slope_over_r(x * x + y * y)
        n = np.stack([-g * x, -g * y, np.ones_like(g)], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def bounding_box(self, transform):
        from optiland.nonsequential.components.geometry.base import AABB  # noqa: PLC0415

        t_vec = np.array(transform[0], dtype=float)
        rot = np.array(transform[1], dtype=float)
        r = self.outer_radius
        corners = np.array([[sx * r, sy * r, z] for sx in (-1, 1) for sy in (-1, 1)
                            for z in (self._z_lo, self._z_hi)])
        world = corners @ rot.T + t_vec
        return AABB(world.min(axis=0), world.max(axis=0))

    def rims(self):
        """The surface's edge circles, (r, z) with r > 0."""
        out = [(self.outer_radius, self.z_outer)]
        if self.inner_radius > 0.0:
            out.append((self.inner_radius, self.z_inner))
        return out

    def ray_intersect(self, origins, directions, eps=None):
        o = np.asarray(origins, dtype=float)
        d = np.asarray(directions, dtype=float)
        count = o.shape[0]
        eps_arr = np.broadcast_to(np.asarray(1e-9 if eps is None else eps, dtype=float), (count,)).copy()
        ox, oy, oz = o[:, 0], o[:, 1], o[:, 2]
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        A = dx * dx + dy * dy
        B = ox * dx + oy * dy
        C = ox * ox + oy * oy
        r_out2, r_in2 = self.outer_radius ** 2, self.inner_radius ** 2
        best_t = np.full(count, np.inf)

        def F(rays, t):
            q = A[rays] * t * t + 2 * B[rays] * t + C[rays]
            return oz[rays] + t * dz[rays] - self.sag_q(q)

        def Fp(rays, t):
            q = A[rays] * t * t + 2 * B[rays] * t + C[rays]
            return dz[rays] - self.slope_over_r(q) * (A[rays] * t + B[rays])

        # Rays parallel to the axis: one root, directly.
        along = A <= 1e-300
        if np.any(along):
            ia = np.nonzero(along & (C <= r_out2) & (C >= r_in2) & (np.abs(dz) > 0.0))[0]
            t = (self.sag_q(C[ia]) - oz[ia]) / dz[ia]
            ok = t > eps_arr[ia]
            best_t[ia[ok]] = t[ok]

        # The stretch of t inside the radius and sag bounds.
        ib = np.nonzero(~along)[0]
        if ib.size:
            a, b, c = A[ib], B[ib], C[ib]
            disc = b * b - a * (c - r_out2)
            has = disc >= 0.0
            sq = np.sqrt(np.maximum(disc, 0.0))
            t_lo, t_hi = (-b - sq) / a, (-b + sq) / a
            with np.errstate(divide="ignore", invalid="ignore"):
                za, zb = (self._z_lo - oz[ib]) / dz[ib], (self._z_hi - oz[ib]) / dz[ib]
            flat = np.abs(dz[ib]) < 1e-300
            inside_z = (oz[ib] >= self._z_lo) & (oz[ib] <= self._z_hi)
            z_from = np.where(flat, np.where(inside_z, -np.inf, np.inf), np.minimum(za, zb))
            z_to = np.where(flat, np.where(inside_z, np.inf, -np.inf), np.maximum(za, zb))
            lo = np.maximum(np.maximum(t_lo, z_from), eps_arr[ib])
            hi = np.minimum(t_hi, z_to)
            keep = has & (lo < hi)
            rays, lo, hi = ib[keep], lo[keep], hi[keep]
            # A zone: cut out the stretch where the ray is inside the hole.
            pieces = [(rays, lo, hi)]
            if r_in2 > 0.0 and rays.size:
                a, b, c = A[rays], B[rays], C[rays]
                disc_in = b * b - a * (c - r_in2)
                cut = disc_in > 0.0
                sq = np.sqrt(np.maximum(disc_in, 0.0))
                h_lo, h_hi = (-b - sq) / a, (-b + sq) / a
                first_hi = np.where(cut, np.minimum(hi, h_lo), hi)
                second_lo = np.where(cut, np.maximum(lo, h_hi), hi)
                pieces = [(rays, lo, first_hi), (rays[cut], second_lo[cut], hi[cut])]
            steps = np.arange(ASPHERE_SAMPLES + 1) / ASPHERE_SAMPLES
            for rays, lo, hi in pieces:
                ok = lo < hi
                rays, lo, hi = rays[ok], lo[ok], hi[ok]
                if not rays.size:
                    continue
                ts = lo[:, None] + (hi - lo)[:, None] * steps[None, :]
                rr = np.repeat(rays[:, None], steps.size, axis=1)
                f = F(rr, ts)
                fp = Fp(rr, ts)
                left, right = f[:, :-1], f[:, 1:]
                row, col = np.nonzero(left * right <= 0.0)
                br_rays, br_lo, br_hi = [rays[row]], [ts[row, col]], [ts[row, col + 1]]
                # F turns between two samples without changing sign: split at the turn.
                turn_r, turn_c = np.nonzero((left * right > 0.0) & (fp[:, :-1] * fp[:, 1:] < 0.0))
                if turn_r.size:
                    tr = rays[turn_r]
                    a_t, b_t = ts[turn_r, turn_c], ts[turn_r, turn_c + 1]
                    m = _bisect(Fp, tr, a_t, b_t)
                    fm = F(tr, m)
                    crosses = fm * f[turn_r, turn_c] <= 0.0
                    br_rays += [tr[crosses], tr[crosses]]
                    br_lo += [a_t[crosses], m[crosses]]
                    br_hi += [m[crosses], b_t[crosses]]
                br_rays, br_lo, br_hi = np.concatenate(br_rays), np.concatenate(br_lo), np.concatenate(br_hi)
                if not br_rays.size:
                    continue
                t = _bisect(F, br_rays, br_lo, br_hi)
                ok = t > eps_arr[br_rays]
                br_rays, t = br_rays[ok], t[ok]
                np.minimum.at(best_t, br_rays, t)

        hit = np.isfinite(best_t)
        normals_geom = np.zeros((count, 3))
        normals_geom[:, 2] = 1.0
        if np.any(hit):
            p = o[hit] + best_t[hit, None] * d[hit]
            normals_geom[hit] = self._normal_local(p[:, 0], p[:, 1])
        facing = np.sum(d * normals_geom, axis=1, keepdims=True) > 0.0
        normals = np.where(facing, -normals_geom, normals_geom)
        return np.where(hit, best_t, np.inf), normals, hit, normals_geom


def _brackets(rays, vals, us):
    """Sign changes of ``vals`` (rays x samples, NaN where undefined) between
    neighbouring samples ``us``: (ray, u_lo, u_hi) per bracket."""
    left, right = vals[:, :-1], vals[:, 1:]
    with np.errstate(invalid="ignore"):
        change = (left * right <= 0.0) & np.isfinite(left) & np.isfinite(right)
    rows, cols = np.nonzero(change)
    return rays[rows], us[cols], us[cols + 1]


def _bisect(fn, rays, lo, hi):
    """A root of ``fn(rays, u)`` in each bracket [lo, hi] (a sign change)."""
    f_lo = fn(rays, lo)
    for _ in range(SPLINE_HALVINGS):
        mid = 0.5 * (lo + hi)
        f_mid = fn(rays, mid)
        same = (f_mid > 0.0) == (f_lo > 0.0)
        lo, f_lo = np.where(same, mid, lo), np.where(same, f_mid, f_lo)
        hi = np.where(same, hi, mid)
    return 0.5 * (lo + hi)


__all__ = ["PolygonPlaneGeometry", "CutConicGeometry", "SplineRevolutionGeometry", "AsphereRevolutionGeometry"]
