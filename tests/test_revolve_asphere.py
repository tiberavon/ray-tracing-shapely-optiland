"""
Python-specific module: test revolve asphere
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
An asphere of revolution, and one lens in three engines (C-BRIDGE-08)
===============================================================================
AsphereRevolutionGeometry with no coefficients must meet rays exactly where
Optiland's ConicGeometry does. A ray crossing it twice between two samples
must be found. The plano-hyperbolic lens written as an asphere focuses the
whole beam in 3D, skew rays included.

The catalogue asphere, Thorlabs ACL25416U (R 8.818197, K -0.99917152, A4 to
A10 from its published surface data; a spherical back, R -69.99948; 14 mm of
B270): its 2D section, the bridge's revolved solid and Optiland's sequential
even_asphere must bend every meridional ray the same way. The index is
1.52138, the one the design's own back focal distance (7.292189 mm at 633 nm)
implies; with it the sequential engine's focal length is the catalogue's
16 mm.
===============================================================================
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.materials.ideal import IdealMaterial
from optiland.nonsequential import (CollimatedSourceConfig, IrradianceDetectorConfig, NSQMaterial,
                                    NSQScene, Spectrum)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.components.geometry.analytic.conic import ConicGeometry
from optiland.optic import Optic
from optiland.rays import RealRays

from ray_tracing_shapely.core.glass_curves import AsphereCurve, ConicCurve
from ray_tracing_shapely.core.scene import Scene
from ray_tracing_shapely.core.scene_objs.light_source.single_ray import SingleRay
from ray_tracing_shapely.core.simulator import Simulator
from ray_tracing_shapely.optical_elements.prisms.base_prism import BasePrism
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile
from ray_tracing_shapely.optical_elements.roles import Mechanism, Role
from ray_tracing_shapely_optiland.compare import compare_2d_3d
from ray_tracing_shapely_optiland.geometry import AsphereRevolutionGeometry
from ray_tracing_shapely_optiland.revolve import (RevolvedProfileSolid, import_revolved_from_optiland,
                                                  scene_placement)

X0, Y0 = 100.0, 30.0              # away from the world origin (KW-OPTILAND-05)
ACL_R1, ACL_K = 8.818197, -0.99917152
ACL_A = [0.0, 8.6821674e-5, 6.3760123e-8, 2.4073084e-9, -1.7189021e-11]   # from r^2, as Zemax and Optiland
ACL_R2, ACL_T, ACL_RIM, ACL_BFL = -69.99948, 14.0, 12.7, 7.292189047056
N_633 = 1.5213801505833693        # implied by the back focal distance at 633 nm (B270)
HEIGHTS = [-11.0, -9.5, -7.0, -4.5, -2.0, -0.5, 0.25, 1.5, 3.0, 5.5, 8.0, 10.0, 11.0]


def _pair(o, d):
    return np.asarray(o, dtype=float), np.asarray(d, dtype=float)


def test_without_coefficients_it_meets_rays_where_optilands_conic_does():
    rng = np.random.default_rng(1)
    n = 20_000
    o, d = _pair(np.c_[rng.uniform(-15, 15, n), rng.uniform(-15, 15, n), rng.uniform(-10, 10, n)],
                 rng.normal(size=(n, 3)))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    for radius, conic in ((ACL_R1, ACL_K), (-20.0, 0.5), (15.0, -2.25)):
        want = [np.asarray(v) for v in ConicGeometry(radius, conic, 12.0).ray_intersect(o, d)]
        t, _, hit, n_geom = AsphereRevolutionGeometry(radius, conic, [], 12.0).ray_intersect(o, d)
        assert np.array_equal(hit, want[2]) and hit.sum() > 3000
        assert np.abs(t[hit] - want[0][hit]).max() < 1e-12
        assert np.abs(n_geom[hit] - want[3][hit]).max() < 1e-12


def test_a_ray_crossing_twice_between_two_samples_is_found():
    g = AsphereRevolutionGeometry(None, 0.0, [0.05], 10.0)              # z = 0.05 r^2
    # a ray along x at height z = 0.05 * 0.2^2 crosses at x = -0.2 and +0.2, both between two
    # of the samples along its 20 mm candidate stretch
    z = 0.05 * 0.2 ** 2
    t, _, hit, _ = g.ray_intersect(np.array([[-10.0, 0.0, z]]), np.array([[1.0, 0.0, 0.0]]))
    assert hit[0] and t[0] == pytest.approx(9.8, abs=1e-12)
    t, _, hit, _ = g.ray_intersect(np.array([[0.0, 0.0, z]]), np.array([[1.0, 0.0, 0.0]]))
    assert hit[0] and t[0] == pytest.approx(0.2, abs=1e-12)              # the far crossing, from between them
    t, _, hit, _ = g.ray_intersect(np.array([[3.0, 4.0, -1.0]]), np.array([[0.0, 0.0, 1.0]]))
    assert hit[0] and t[0] == pytest.approx(1.0 + 0.05 * 25.0)          # along the axis: solved directly


# --- the plano-hyperbolic lens, written as an asphere ------------------------------------------

N, R, T, H = 1.5, 20.0, 12.0, 10.0
FOCUS = X0 + T + R / (N - 1)


class _PlanoAsphere(BasePrism):
    _edge_roles = [("S1", "Flat Face", Role.ENTRANCE), ("RIM", "Edge", Role.BLOCKER),
                   ("S2", "Aspheric Face", Role.EXIT), ("RIM", "Edge", Role.BLOCKER)]
    _edge_mechanisms = {"S1": Mechanism.REFRACT, "RIM": None, "S2": Mechanism.REFRACT}

    def __init__(self, scene, coefficients=()):
        self._coefficients = list(coefficients)
        super().__init__(scene, size=H, n=N)

    def _compute_path(self):
        c = AsphereCurve((X0 + T, Y0), (-1.0, 0.0), R, -N * N, self._coefficients, -H, H)
        lo, hi = c.point(H), c.point(-H)
        return [{"x": X0, "y": Y0 + H, "arc": False}, {"x": X0, "y": Y0 - H, "arc": False},
                {"x": lo[0], "y": lo[1], "arc": False, "curve": c.descriptor()}, {"x": hi[0], "y": hi[1], "arc": False}]


def _revolve(prism, n):
    profile = export_profile(prism, symmetry={"kind": "revolve", "axis": {"point": [X0, Y0], "direction": [1.0, 0.0]}},
                             physical_params={}, numerical_params={})
    return profile, RevolvedProfileSolid("L", profile, NSQMaterial(IdealMaterial(n=n)), cs=scene_placement(profile))


def test_the_asphere_written_hyperbola_focuses_the_whole_beam_in_3d():
    _, lens = _revolve(_PlanoAsphere(Scene()), N)
    assert lens.surface_kinds["S2"] == "asphere"
    scene = NSQScene()
    scene.add_source("B", CoordinateSystem(x=X0 - 20.0, y=0.0, z=Y0, ry=math.pi / 2),
                     CollimatedSourceConfig(Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=0.9 * H))
    scene.component_registry.add(lens.name, lens)
    scene.add_detector("D", CoordinateSystem(x=FOCUS + 100.0, y=0.0, z=Y0, ry=math.pi / 2),
                       IrradianceDetectorConfig(width=2000, height=2000, num_pixels_x=2, num_pixels_y=2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = scene.trace(num_rays=3000, seed=1, max_depth=10, backend=NumpyBackend(seed=1),
                         record_paths=True).ray_paths["events"]
    ev = ev[np.argsort(ev["ray_id"], kind="stable")]
    _, st = np.unique(ev["ray_id"], return_index=True)
    en = np.r_[st[1:], len(ev)]
    focus = np.array([FOCUS, 0.0, Y0])
    worst, count, skew = 0.0, 0, 0.0
    for s, e in zip(st, en):
        if [str(c) for c in ev["component_name"][s + 1:e]] != ["L.S1", "L.S2", ""]:
            continue                                   # a Fresnel reflection took the ray elsewhere
        p = np.array([ev["x"][e - 1], ev["y"][e - 1], ev["z"][e - 1]])
        d = np.array([ev["L"][e - 1], ev["M"][e - 1], ev["N"][e - 1]])
        worst = max(worst, float(np.linalg.norm(np.cross(focus - p, d))))
        skew = max(skew, abs(float(ev["y"][s])))
        count += 1
    assert count > 2500 and skew > 0.8 * H and worst < 1e-9


def test_the_asphere_reads_back_and_what_the_bridge_refuses():
    profile, lens = _revolve(_PlanoAsphere(Scene(), coefficients=[0.0, 2.0e-4, -1.5e-6]), N)
    back = import_revolved_from_optiland(lens)
    s2 = next(sp for sp in back["spans"] if sp["short_label"] == "S2")
    original = next(sp for sp in profile["spans"] if sp["short_label"] == "S2")
    assert np.allclose(s2["curve"]["vertex"], original["curve"]["vertex"], atol=1e-12)
    # read back along the frame's axis, so possibly the other way round (R and every coefficient
    # negated with it): the same surface either way. Its points (at equal steps in r, as a conic's)
    # lie on the original face, and the original's lie on the read-back one.
    def face_of(curve):
        return AsphereCurve(curve["vertex"], curve["axis"], curve["radius"], curve["conic"],
                            curve["coefficients"], -H, H)
    for written, points in ((original, back["points"][s2["index_start"]:s2["index_end"]]),
                            (s2, profile["points"][original["index_start"]:original["index_end"]])):
        assert max(face_of(written["curve"]).nearest(tuple(p))[0] for p in points) < 1e-12
    rebuilt = RevolvedProfileSolid("L2", back, NSQMaterial(IdealMaterial(n=N)), cs=scene_placement(back))
    assert rebuilt.surface_kinds["S2"] == "asphere"
    # a sample off the surface, and an asphere whose axis is not the revolve axis
    k = original["index_start"] + 3
    off = {**profile, "points": [list(p) for p in profile["points"]]}
    off["points"][k][0] += 1e-5
    with pytest.raises(ValueError, match="from the asphere built for it"):
        RevolvedProfileSolid("L3", off, NSQMaterial(IdealMaterial(n=N)), cs=scene_placement(off))
    tilted = {**profile, "spans": [dict(sp) for sp in profile["spans"]]}
    j = next(i for i, sp in enumerate(tilted["spans"]) if sp["short_label"] == "S2")
    tilted["spans"][j]["curve"] = {**tilted["spans"][j]["curve"], "vertex": [X0 + T, Y0 + 0.5]}
    with pytest.raises(NotImplementedError, match="not the revolve axis"):
        RevolvedProfileSolid("L4", tilted, NSQMaterial(IdealMaterial(n=N)), cs=scene_placement(tilted))


# --- the ACL25416U in three engines ----------------------------------------------------------------

class _ACL25416U(BasePrism):
    """S1 the asphere (vertex at x0, light from -x), S2 the sphere, rims at +-12.7 mm."""

    _edge_roles = [("S1", "Aspheric Face", Role.ENTRANCE), ("RIM", "Edge", Role.BLOCKER),
                   ("S2", "Spherical Face", Role.EXIT), ("RIM", "Edge", Role.BLOCKER)]
    _edge_mechanisms = {"S1": Mechanism.REFRACT, "RIM": None, "S2": Mechanism.REFRACT}

    def __init__(self, scene):
        super().__init__(scene, size=ACL_RIM, n=N_633)

    def _compute_path(self):
        s1 = AsphereCurve((X0, Y0), (1.0, 0.0), ACL_R1, ACL_K, ACL_A, ACL_RIM, -ACL_RIM)    # s = y - y0, top down
        s2 = ConicCurve((X0 + ACL_T, Y0), (1.0, 0.0), ACL_R2, 0.0, -ACL_RIM, ACL_RIM)         # bottom up
        return [{"x": s1.start[0], "y": s1.start[1], "arc": False, "curve": s1.descriptor()},
                {"x": s1.end[0], "y": s1.end[1], "arc": False},
                {"x": s2.start[0], "y": s2.start[1], "arc": False, "curve": s2.descriptor()},
                {"x": s2.end[0], "y": s2.end[1], "arc": False}]


def _optics(scene):
    lens = _ACL25416U(scene)
    scene.add_object(lens)
    return {"L": lens}


def _sequential():
    optic = Optic()
    optic.wavelengths.add(value=0.633, is_primary=True)
    optic.set_aperture(aperture_type="EPD", value=22.86)
    optic.surfaces.add(index=0, z=0.0)
    optic.surfaces.add(index=1, z=20.0, radius=ACL_R1, conic=ACL_K, surface_type="even_asphere",
                       coefficients=ACL_A, material=IdealMaterial(n=N_633), is_stop=True, tol=1e-14)
    optic.surfaces.add(index=2, z=20.0 + ACL_T, radius=ACL_R2, material=IdealMaterial(n=1.0))
    optic.surfaces.add(index=3, z=20.0 + ACL_T + ACL_BFL)
    return optic


def _exits_2d():
    """Per height: the 2D ray's direction after S2 (degrees from the axis) and where it crosses x = image."""
    scene = Scene()
    _optics(scene)
    for h in HEIGHTS:
        scene.add_object(SingleRay(scene, {"p1": {"x": X0 - 20.0, "y": Y0 + h}, "p2": {"x": X0 - 19.0, "y": Y0 + h}}))
    segs = Simulator(scene, max_rays=10_000).run_with_result(name="acl").segments
    image_x = X0 + ACL_T + ACL_BFL
    out = []
    for sg in segs:
        dx, dy = sg.p2["x"] - sg.p1["x"], sg.p2["y"] - sg.p1["y"]
        if sg.p1["x"] > X0 + ACL_T - 2.0 and dx > 0 and sg.total_brightness > 0.5 and sg.parent_uuid is not None:
            y_img = sg.p1["y"] + (image_x - sg.p1["x"]) / dx * dy - Y0
            out.append((math.degrees(math.atan2(dy, dx)), y_img))
    return sorted(out, key=lambda a: -a[1])


def test_the_acl25416u_bends_every_ray_the_same_in_three_engines():
    # 2D against the revolved 3D solid, ray by ray, face by face
    profile = export_profile(_ACL25416U(Scene()), symmetry={"kind": "revolve", "axis": {
        "point": [X0, Y0], "direction": [1.0, 0.0]}}, physical_params={}, numerical_params={})
    solid = RevolvedProfileSolid("L", profile, NSQMaterial(IdealMaterial(n=N_633)), cs=scene_placement(profile))
    assert solid.surface_kinds["S1"] == "asphere" and solid.surface_kinds["S2"] == "conic"
    summary = compare_2d_3d(_optics, [solid], [((X0 - 20.0, Y0 + h), 0.0) for h in HEIGHTS]).summary()
    assert summary["agree"] == summary["rays"] == len(HEIGHTS)
    assert summary["worst_direction_difference"] < 1e-9

    # 2D against Optiland's sequential even asphere with the same numbers
    optic = _sequential()
    h = np.array(HEIGHTS)
    zero = np.zeros_like(h)
    rays = RealRays(x=zero, y=h, z=zero, L=zero, M=zero, N=np.ones_like(h), intensity=np.ones_like(h),
                    wavelength=np.full_like(h, 0.633))
    optic.surfaces.trace(rays)
    m, nn = np.asarray(optic.surfaces.M)[-1], np.asarray(optic.surfaces.N)[-1]
    seq = sorted(zip(np.degrees(np.arctan2(m, nn)), np.asarray(optic.surfaces.y)[-1]), key=lambda a: -a[1])
    two_d = _exits_2d()
    assert len(two_d) == len(HEIGHTS)
    angle_diff = max(abs(a[0] - b[0]) for a, b in zip(two_d, seq))
    image_diff = max(abs(a[1] - b[1]) for a, b in zip(two_d, seq))
    assert angle_diff < 1e-9 and image_diff < 1e-9

    # the sequential engine's paraxial focal length is the catalogue's 16 mm
    assert optic.paraxial.f2() == pytest.approx(16.0, abs=0.01)
