"""
Python-specific module: test_spline_revolution
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
SplineRevolutionGeometry
===============================================================================
The oracle is Optiland's own ConicGeometry (and the bridge's CutConicGeometry
for a zone): a spline through exact points of a conic, with its exact
tangents, must meet every random ray the conic meets, nowhere else, at the
same distance to the spline's accuracy, and every hit must lie on the
spline's own surface to rounding.
===============================================================================
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.nonsequential.components.geometry.analytic.conic import ConicGeometry

from ray_tracing_shapely_optiland.geometry import CutConicGeometry, SplineRevolutionGeometry

R, K, RIM = 20.0, -0.6, 12.0


def _conic_meridian(r):
    c = 1.0 / R
    root = np.sqrt(1.0 - (1.0 + K) * c * c * r * r)
    z = c * r * r / (1.0 + root)
    t = np.stack([np.ones_like(r), c * r / root], axis=1)
    return np.stack([r, z], axis=1), t / np.linalg.norm(t, axis=1, keepdims=True)


def _rays(count, seed):
    rng = np.random.default_rng(seed)
    o = rng.uniform(-15.0, 15.0, (count, 3))
    o[:, 2] = rng.uniform(-10.0, 15.0, count)
    d = rng.normal(size=(count, 3))
    return o, d / np.linalg.norm(d, axis=1, keepdims=True)


@pytest.mark.parametrize("r_in, knots, t_tol", [(0.0, 65, 1e-7), (4.0, 65, 1e-7), (0.0, 17, 3e-5)],
                         ids=["cap-65", "zone-65", "cap-17"])
def test_a_spline_through_a_conic_meets_every_ray_the_conic_meets(r_in, knots, t_tol):
    pts, tans = _conic_meridian(np.linspace(r_in, RIM, knots))
    spline = SplineRevolutionGeometry(pts[::-1], -tans[::-1])     # given inwards: it runs itself outwards
    oracle = CutConicGeometry(R, K, RIM, inner_radius=r_in) if r_in else ConicGeometry(R, K, RIM)
    o, d = _rays(8000, seed=int(r_in) + knots)
    ts, normals, hit, n_geom = spline.ray_intersect(o, d)
    tc, _, hit_c, n_geom_c = oracle.ray_intersect(o, d)
    assert hit.sum() > 1000 and np.array_equal(hit, hit_c)
    assert np.abs(ts[hit] - tc[hit]).max() < t_tol
    assert np.abs(n_geom[hit] - n_geom_c[hit]).max() < t_tol           # n_geom: Optiland's orientation
    assert np.all(np.sum(normals[hit] * d[hit], axis=1) <= 0.0)       # faces the incoming ray
    p = o[hit] + ts[hit, None] * d[hit]
    z, _ = spline.meridian_at(np.hypot(p[:, 0], p[:, 1]))
    assert np.abs(p[:, 2] - z).max() < 1e-11                         # on its own surface


def test_rays_square_to_the_axis_and_along_it_are_both_found():
    # The two forms of the solve: a ray across the axis at the height of a sample,
    # and rays parallel to the axis.
    pts, tans = _conic_meridian(np.linspace(0.0, RIM, 33))
    spline = SplineRevolutionGeometry(pts, tans)
    oracle = ConicGeometry(R, K, RIM)
    z = np.linspace(0.1, 3.5, 40)
    o = np.stack([np.full_like(z, -30.0), np.full_like(z, 0.3), z], axis=1)
    d = np.tile([1.0, 0.0, 0.0], (z.size, 1))
    for oo, dd in ((o, d), (np.stack([np.linspace(-11, 11, 40), np.full(40, 2.0), np.full(40, -5.0)], 1),
                       np.tile([0.0, 0.0, 1.0], (40, 1)))):
        ts, _, hit, _ = spline.ray_intersect(oo, dd)
        tc, _, hit_c, _ = oracle.ray_intersect(oo, dd)
        assert hit.all() and hit_c.all() and np.abs(ts - tc).max() < 1e-6


def test_a_ray_starting_on_the_surface_finds_its_far_crossing_not_itself():
    pts, tans = _conic_meridian(np.linspace(0.0, RIM, 33))
    spline = SplineRevolutionGeometry(pts, tans)
    o = np.array([[-11.0, 0.0, 2.0]])
    d = np.array([[1.0, 0.0, 0.0]])
    t1, _, _, _ = spline.ray_intersect(o, d)
    start = o + t1[:, None] * d
    t2, _, hit, _ = spline.ray_intersect(start, d)
    assert hit[0] and t2[0] > 1.0                                    # across the bowl, not t = 0
    assert abs((start + t2[:, None] * d)[0, 0] + start[0, 0]) < 1e-9  # its mirror image


def test_the_rims_and_bounding_box():
    pts, tans = _conic_meridian(np.linspace(4.0, RIM, 17))
    spline = SplineRevolutionGeometry(pts, tans)
    assert spline.inner_radius == pytest.approx(4.0) and spline.outer_radius == pytest.approx(RIM)
    assert [r for r, _ in spline.rims()] == pytest.approx([RIM, 4.0])
    box = spline.bounding_box((np.zeros(3), np.eye(3)))
    assert box.max_corner[0] == pytest.approx(RIM) and box.min_corner[2] <= pts[0, 1]
    assert box.max_corner[2] >= pts[-1, 1]


def test_a_spline_not_single_valued_in_r_is_refused():
    pts = np.array([[1.0, 0.0], [3.0, 1.0], [2.0, 2.0]])               # r turns back
    tans = np.array([[1.0, 0.5], [0.0, 1.0], [-1.0, 0.5]])
    with pytest.raises(ValueError, match="monotonic"):
        SplineRevolutionGeometry(pts, tans)
    with pytest.raises(ValueError, match="domain"):
        SplineRevolutionGeometry(pts[:2], tans[:2], (0.5, 1.5))
