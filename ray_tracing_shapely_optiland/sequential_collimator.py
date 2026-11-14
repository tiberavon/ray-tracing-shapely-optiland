"""
Python-specific module: sequential_collimator
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
SEQUENTIAL COLLIMATOR -- a point-source collimator in Optiland's sequential engine
===============================================================================
A third reference for a designed collimator: the channels of an LED collimator
built from its designed faces in Optiland's sequential engine, a tracer
independent of both the 2D engine and the non-sequential bridge.

Everything is in the collimator's canonical frame: the point source F at the
origin, the axis along +z (the 2D section's axis, a), the meridional
coordinate y (the 2D r). Faces come as points with unit normals, (z, y)
pairs, as ``TIRCollimator.section_faces`` gives them.

- **A face of revolution reaching the axis** (a central lens's front face)
  is an even asphere, fitted to the designed points and normals
  (``fit_even_asphere``): a conic plus terms in r^4, r^6, ..., fitted on sag
  and slope together.
- **A ring channel** (cavity wall, TIR face, exit face) is not a face of
  revolution Optiland's sequential engine can describe: a cone wall is not
  a sag z(r). It is built for meridional rays only (``ring_channel_optic``):
  a straight face is a plane tilted to its line, which a meridional ray
  meets exactly as it meets the cone; a curved face (22c) is a polynomial
  in y in a frame tilted to its chord (``fit_meridional_polynomial``), a
  mirror for total internal reflection.

``trace_from_focus`` launches real rays from F at given angles and returns
where and in which direction they leave: directly through the surfaces, with
no pupil aiming.

The sequential engine solves an asphere's or a polynomial's intersection by
iteration; every surface here is built with a tolerance of 1e-14.
===============================================================================
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from optiland.materials.ideal import IdealMaterial
from optiland.optic import Optic
from optiland.rays import RealRays

#: Iteration tolerance for every iteratively intersected surface.
TOL = 1e-14


def _as_arrays(points, normals):
    p = np.asarray(points, dtype=float)
    n = np.asarray(normals, dtype=float)
    return p, n / np.linalg.norm(n, axis=1, keepdims=True)


def fit_even_asphere(points: Sequence[Tuple[float, float]], normals: Sequence[Tuple[float, float]],
                     terms: int = 8, slope_weight: float = 10.0) -> Dict[str, Any]:
    """An even asphere through a designed face of revolution that reaches the axis.

    Args:
        points: (z, y) points of the face, the first on the axis (its vertex).
        normals: Their unit normals (either sign).
        terms: Asphere terms beyond the conic: r^4, r^6, ..., r^(2 terms + 2).
        slope_weight: Weight of the slope residual (radians) against the sag
            residual (length) in the least-squares fit.

    Returns:
        ``vertex_z``, ``radius``, ``conic``, ``coefficients`` (Optiland's even
        asphere list, starting with r^2), ``sag_error`` and
        ``slope_error_deg``, the worst residuals over the points.
    """
    from scipy.optimize import least_squares  # noqa: PLC0415

    p, n = _as_arrays(points, normals)
    vertex_z = float(p[0, 0])
    r, sag = p[:, 1], p[:, 0] - vertex_z
    slope = -n[:, 1] / n[:, 0]                     # dz/dr: the normal is perpendicular to (dz, dr)
    scale = float(r.max())
    u = r / scale

    def model(x):
        c, k, a = x[0], x[1], x[2:]
        root = np.sqrt(np.maximum(1.0 - (1.0 + k) * c * c * r * r, 1e-15))
        z = c * r * r / (1.0 + root) + sum(a[i] * u ** (2 * i + 4) for i in range(terms))
        dz = c * r / root + sum(a[i] * (2 * i + 4) * u ** (2 * i + 3) / scale for i in range(terms))
        return z, dz

    def residual(x):
        z, dz = model(x)
        return np.r_[z - sag, slope_weight * (np.arctan(dz) - np.arctan(slope))]

    near = int(np.argmin(np.where(r > 0.0, r, np.inf)))
    x0 = np.r_[2.0 * sag[near] / r[near] ** 2, -1.0, np.zeros(terms)]
    fit = least_squares(residual, x0, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=50000)
    z, dz = model(fit.x)
    coefficients = [0.0] + [float(fit.x[2 + i]) / scale ** (2 * i + 4) for i in range(terms)]
    return {"vertex_z": vertex_z, "radius": 1.0 / float(fit.x[0]), "conic": float(fit.x[1]),
            "coefficients": coefficients, "sag_error": float(np.abs(z - sag).max()),
            "slope_error_deg": float(np.degrees(np.abs(np.arctan(dz) - np.arctan(slope)).max()))}


def central_channel_optic(flat_z: float, n: float, front: Dict[str, Any],
                          image_gap: float = 50.0) -> Optic:
    """F (at z = 0) -> a flat rear face at ``flat_z`` -> the glass -> ``front`` (an
    even asphere from fit_even_asphere, or any dict with ``vertex_z``, ``radius``,
    ``conic`` and optional ``coefficients``) -> air -> an image plane."""
    optic = Optic()
    optic.surfaces.add(index=0, z=0.0)
    optic.surfaces.add(index=1, z=flat_z, material=IdealMaterial(n=n), is_stop=True)
    optic.surfaces.add(index=2, z=front["vertex_z"], radius=front["radius"], conic=front["conic"],
                       surface_type="even_asphere", coefficients=list(front.get("coefficients", [])),
                       tol=TOL)
    optic.surfaces.add(index=3, z=front["vertex_z"] + image_gap)
    return optic


def fit_meridional_polynomial(points: Sequence[Tuple[float, float]],
                              normals: Sequence[Tuple[float, float]],
                              degree: int = 12) -> Dict[str, Any]:
    """A meridional curve as Optiland's polynomial surface w(v) in a tilted frame.

    The frame's origin is the chord's middle, its local y along the chord
    (tilted by ``rx`` from the global y towards +z), its local z the chord's
    normal; the curve is w = sum c_j v^j there, fitted on sag and slope.

    Returns:
        ``y``, ``z``, ``rx`` (the frame), ``coefficients`` (Optiland's
        polynomial matrix, x^0 row only), ``sag_error``, ``slope_error_deg``.
    """
    p, n = _as_arrays(points, normals)
    zy = p[:, ::-1]                                 # (y, z) for the frame's arithmetic
    origin = 0.5 * (zy[0] + zy[-1])
    chord = zy[-1] - zy[0]
    rx = math.atan2(chord[1], chord[0])
    ey = np.array([math.cos(rx), math.sin(rx)])     # local y, in global (y, z)
    ez = np.array([-math.sin(rx), math.cos(rx)])    # local z
    v = (zy - origin) @ ey
    w = (zy - origin) @ ez
    nyz = n[:, ::-1]
    slope = -(nyz @ ey) / (nyz @ ez)                 # dw/dv
    half = float(np.abs(v).max())
    s = v / half
    rows = [s ** j for j in range(degree + 1)]
    drows = [j * s ** max(j - 1, 0) / half for j in range(degree + 1)]
    weight = half                                    # slope residual in units comparable to sag
    a = np.vstack([np.stack(rows, 1), weight * np.stack(drows, 1)])
    b = np.r_[w, weight * slope]
    c, *_ = np.linalg.lstsq(a, b, rcond=None)
    coefficients = [float(c[j]) / half ** j for j in range(degree + 1)]
    w_fit = sum(cj * v ** j for j, cj in enumerate(coefficients))
    dw_fit = sum(j * cj * v ** max(j - 1, 0) for j, cj in enumerate(coefficients))
    return {"y": float(origin[0]), "z": float(origin[1]), "rx": rx, "coefficients": [coefficients],
            "sag_error": float(np.abs(w_fit - w).max()),
            "slope_error_deg": float(np.degrees(np.abs(np.arctan(dw_fit) - np.arctan(slope)).max()))}


def _plane_through(p: Tuple[float, float], q: Tuple[float, float]) -> Dict[str, float]:
    """A plane containing the meridional line p -> q ((z, y) points), its local y along it."""
    return {"y": float(p[1]), "z": float(p[0]), "rx": math.atan2(q[0] - p[0], q[1] - p[1])}


def ring_channel_optic(wall: Sequence[Tuple[float, float]], mirror: Dict[str, Any],
                       exit_face: Sequence[Tuple[float, float]], n: float,
                       image_z: float) -> Optic:
    """F -> the wall (a plane, refracting into the glass) -> the mirror (a polynomial
    from fit_meridional_polynomial, reflecting) -> the exit face (a plane,
    refracting into air) -> an image plane at ``image_z``. For meridional rays
    (x = 0) only: the planes stand for cones of revolution, which a meridional
    ray meets along the same line, with the same normal."""
    optic = Optic()
    optic.surfaces.add(index=0, z=0.0)
    optic.surfaces.add(index=1, material=IdealMaterial(n=n), is_stop=True, **_plane_through(*wall))
    optic.surfaces.add(index=2, material="mirror", surface_type="polynomial", radius=math.inf,
                       coefficients=mirror["coefficients"], tol=TOL,
                       y=mirror["y"], z=mirror["z"], rx=mirror["rx"])
    optic.surfaces.add(index=3, material=IdealMaterial(n=1.0), **_plane_through(*exit_face))
    optic.surfaces.add(index=4, z=image_z)
    return optic


def trace_from_focus(optic: Optic, angles_deg: Sequence[float]) -> Dict[str, np.ndarray]:
    """Real meridional rays from F (the origin) at ``angles_deg`` from the axis,
    towards +y, traced through every surface.

    Returns:
        ``angle_deg`` (each ray's direction after the last surface, from +z,
        positive towards +y), ``y`` and ``z`` (its points on each surface,
        shape (surfaces, rays)).
    """
    phi = np.radians(np.asarray(angles_deg, dtype=float))
    zero = np.zeros_like(phi)
    rays = RealRays(x=zero, y=zero, z=zero, L=zero, M=np.sin(phi), N=np.cos(phi),
                    intensity=np.ones_like(phi), wavelength=np.full_like(phi, 0.55))
    optic.surfaces.trace(rays)
    s = optic.surfaces
    m, nn = np.asarray(s.M)[-1], np.asarray(s.N)[-1]
    return {"angle_deg": np.degrees(np.arctan2(m, nn)), "y": np.asarray(s.y), "z": np.asarray(s.z)}


__all__ = ["central_channel_optic", "fit_even_asphere", "fit_meridional_polynomial",
           "ring_channel_optic", "trace_from_focus"]
