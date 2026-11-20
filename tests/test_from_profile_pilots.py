"""
Python-specific module: test from_profile on the pilots' revolve profiles
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
A revolve profile, read back in 2D, traces as the element it was exported from
===============================================================================
A revolve profile holds only the half-section on one side of the axis.
`BasePrism.from_profile` rebuilds the full section by mirroring it. A flow
that compares 2D with 3D from profile files alone depends on that: its 2D
optics are the rebuilt sections, not the designs.

Each solid element of the revolved pilots is exported, sent through JSON, and
rebuilt; a fan of rays is traced through the original and through the
rebuilt section, each alone, following the brightest branch. Both must meet
the same faces, end the same way, and leave at the same point in the same
direction:
- the Shafer triplet's three singlets (two menisci and a biconvex lens);
- the ACL25416U asphere (an even asphere and a sphere);
- the US 7,762,697 TIR collimator (spline faces), from its focal point;
- the US 7,255,467 headlamp's dome (a hemisphere, its axis across the
  scene's) and its plano-convex lens.
The mirror sheets are rebuilt by `mirrors_from_sheet_profile` and checked the
same way:
- the cone reflector pilot's V of two flat walls;
- the US 7,255,467 headlamp's ellipsoidal reflector (a cap that stops short of
  the axis and a zone of the far vertex's sheet), cut to the upper half as
  the pilot has it, and whole;
- a spherical mirror reaching the axis, facing a flat mirror across it.
The SMS elements, their surfaces exact splines through the chain, are rebuilt
by `from_profile` the same way, their mirror-coated surfaces included (a
`reflect` span is the glass's own mirror):
- the RR lens of Example 9.1, at the book's density and ten times it;
- the RX concentrator, its X surface mirror-coated;
- the complete RXI, its bottom and the centre of its top mirror-coated.
===============================================================================
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from ray_tracing_shapely.core.glass_curves import AsphereCurve, ConicCurve
from ray_tracing_shapely.core.scene import Scene
from ray_tracing_shapely.core.scene_objs.mirror.mirror import Mirror
from ray_tracing_shapely.optical_elements.lenses import SingletLens
from ray_tracing_shapely.optical_elements.lenses.tir_collimator import TIRCollimator
from ray_tracing_shapely.optical_elements.mirrors import ArcMirror, ConicMirror, mirrors_from_sheet_profile
from ray_tracing_shapely.optical_elements.prisms.base_prism import BasePrism
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile, export_sheet_profile
from ray_tracing_shapely.optical_elements.roles import Mechanism, Role
from ray_tracing_shapely_optiland.compare import trace_2d_main_path

X0, Y0 = 100.0, 30.0
ACL = dict(radius=8.818197, conic=-0.99917152,
           coefficients=[0.0, 8.6821674e-5, 6.3760123e-8, 2.4073084e-9, -1.7189021e-11])
ACL_R2, ACL_T, ACL_RIM, ACL_N = -69.99948, 14.0, 12.7, 1.5150


class _ACL25416U(BasePrism):
    """The asphere pilot's lens, as its notebook builds it."""

    _edge_roles = [("S1", "Aspheric Face", Role.ENTRANCE), ("RIM", "Edge", Role.BLOCKER),
                   ("S2", "Spherical Face", Role.EXIT), ("RIM", "Edge", Role.BLOCKER)]
    _edge_mechanisms = {"S1": Mechanism.REFRACT, "RIM": None, "S2": Mechanism.REFRACT}

    def __init__(self, scene):
        super().__init__(scene, size=ACL_RIM, n=ACL_N)

    def _compute_path(self):
        s1 = AsphereCurve((X0, Y0), (1.0, 0.0), ACL["radius"], ACL["conic"], ACL["coefficients"], ACL_RIM, -ACL_RIM)
        s2 = ConicCurve((X0 + ACL_T, Y0), (1.0, 0.0), ACL_R2, 0.0, -ACL_RIM, ACL_RIM)
        return [{"x": s1.start[0], "y": s1.start[1], "arc": False, "curve": s1.descriptor()},
                {"x": s1.end[0], "y": s1.end[1], "arc": False},
                {"x": s2.start[0], "y": s2.start[1], "arc": False, "curve": s2.descriptor()},
                {"x": s2.end[0], "y": s2.end[1], "arc": False}]


def _shafer(k):
    lenses = [dict(radius_front=22.64, radius_back=12.64, thickness=1.99, semi_diameter=10.43),
              dict(radius_front=26.11, radius_back=-32.09, thickness=5.01, semi_diameter=10.43),
              dict(radius_front=-10.96, radius_back=-16.57, thickness=2.02, semi_diameter=7.99)]
    return lambda scene: SingletLens(scene, n=1.4918, position=(X0, Y0), **lenses[k])


def _headlamp_lens(scene):
    r, semi, edge, n = 36.3, 35.5, 4.6, 1.65
    t = r - math.sqrt(r * r - semi * semi) + edge
    return SingletLens(scene, radius_front=math.inf, radius_back=-r, thickness=t, semi_diameter=semi, n=n,
                       position=(X0, Y0))


def _dome(scene):
    return SingletLens(scene, radius_front=math.inf, radius_back=-3.0, thickness=3.0,
                       semi_diameter=3.0 * (1 - 1e-5), n=1.50, position=(X0, Y0), rotation=90.0)


def _parallel(x, heights, tilt_deg=0.0):
    return [((x, Y0 + h), tilt_deg) for h in heights]


def _from(point, angles):
    return [(point, a) for a in angles]


HEIGHTS = (-9.0, -6.0, -3.0, -0.5, 0.25, 2.0, 5.0, 8.0)
PILOTS = {
    "shafer_L1": (_shafer(0), _parallel(X0 - 10.0, HEIGHTS) + _parallel(X0 - 10.0, HEIGHTS[1:-1], 8.0)),
    "shafer_L2": (_shafer(1), _parallel(X0 - 10.0, HEIGHTS) + _parallel(X0 - 10.0, HEIGHTS[1:-1], -8.0)),
    "shafer_L3": (_shafer(2), _parallel(X0 - 10.0, (-6.5, -4.0, -1.0, 0.5, 3.0, 6.5))),
    "acl25416u": (_ACL25416U, _parallel(X0 - 20.0, (-11.0, -7.0, -2.0, -0.5, 0.25, 3.0, 8.0, 11.0))),
    "tir_collimator": (lambda scene: TIRCollimator(scene, n=1.49, position=(X0, Y0), points=33),
                       _from((X0, Y0), np.arange(-85.0, 86.0, 10.0))),
    "headlamp_dome": (_dome, _from((X0 + 0.01, Y0 + 0.0), np.arange(-80.0, 81.0, 20.0) + 90.0)),
    "headlamp_lens": (_headlamp_lens, _parallel(X0 - 10.0, (-30.0, -18.0, -6.0, -1.0, 0.5, 12.0, 25.0, 32.0))),
}


@pytest.mark.parametrize("name", list(PILOTS))
def test_the_rebuilt_section_traces_as_the_design(name):
    make, rays = PILOTS[name]
    design = make(Scene())
    profile = json.loads(json.dumps(export_profile(design, symmetry={"kind": "revolve", "axis": design.revolve_axis()}
                                                   if hasattr(design, "revolve_axis") else
                                                   {"kind": "revolve", "axis": {"point": [X0, Y0],
                                                                                "direction": [1.0, 0.0]}},
                                                   physical_params={}, numerical_params={})))
    assert profile["symmetry"]["kind"] == "revolve"
    component = {"X": SimpleNamespace(name="X", profile=profile)}

    def original(scene):
        obj = make(scene)
        scene.add_object(obj)
        return {"X": obj}

    def rebuilt(scene):
        obj = BasePrism.from_profile(scene, profile, n=design.refIndex)
        scene.add_object(obj)
        return {"X": obj}

    met = 0
    for ray in rays:
        a = trace_2d_main_path(original, ray, components=component)
        b = trace_2d_main_path(rebuilt, ray, components=component)
        assert (b.faces, b.end) == (a.faces, a.end), (name, ray)
        assert np.allclose(b.point, a.point, atol=1e-6), (name, ray, a.point, b.point)
        assert np.allclose(b.direction, a.direction, atol=1e-9), (name, ray, a.direction, b.direction)
        met += bool(a.faces)
    assert met >= len(rays) - 2, f"{name}: only {met} of {len(rays)} rays met the element"


# --- the mirror sheets ------------------------------------------------------------

AXIS = {"point": [X0, Y0], "direction": [1.0, 0.0]}
F1F2, A_ELL, FRONT_X, REAR_GAP = 40.0, 40.0, 40.0, 0.5          # the headlamp pilot's reflector, from F1
C_ELL = F1F2 / 2
B_ELL = math.sqrt(A_ELL ** 2 - C_ELL ** 2)
R_ELL, K_ELL = B_ELL ** 2 / A_ELL, -(C_ELL / A_ELL) ** 2
FRONT_Y = B_ELL * math.sqrt(1.0 - ((FRONT_X - C_ELL) / A_ELL) ** 2)


def _mirror(scene, p, q, name):
    m = Mirror(scene, {"p1": {"x": p[0], "y": p[1]}, "p2": {"x": q[0], "y": q[1]}})
    m.name = name
    return m


def _cone(scene):
    return [_mirror(scene, (X0, Y0 + s * 20.0), (X0 + 60.0, Y0 + s * 8.0), name) for s, name in ((1, "upper"),
                                                                                                 (-1, "lower"))]


def _reflector(whole):
    def make(scene):
        cap, zone = [((REAR_GAP, B_ELL), "cap")], [((-B_ELL, -FRONT_Y), "zone")]
        if whole:                                      # the lower halves: the export keeps the half above
            cap, zone = cap + [((-B_ELL, -REAR_GAP), "cap_far")], zone + [((FRONT_Y, B_ELL), "zone_far")]
        return ([ConicMirror(scene, (X0 - (A_ELL - C_ELL), Y0), (1.0, 0.0), R_ELL, K_ELL, extent=e, name=n)
                 for e, n in cap] +
                [ConicMirror(scene, (X0 + C_ELL + A_ELL, Y0), (-1.0, 0.0), R_ELL, K_ELL, extent=e, name=n)
                 for e, n in zone])
    return make


def _sphere_and_flat(scene):
    return [ArcMirror(scene, (X0 + 40.0, Y0), (1.0, 0.0), -40.0, semi_diameter=20.0, name="M1"),
            _mirror(scene, (X0 - 5.0, Y0 - 3.0), (X0 - 5.0, Y0 + 3.0), "M2")]


def _exported(make, cut):
    symmetry = {**{"kind": "revolve", "axis": AXIS}, **({"azimuth_deg": [-90, 90]} if cut else {})}
    return json.loads(json.dumps(export_sheet_profile(make(Scene()), symmetry=symmetry, physical_params={},
                                                      numerical_params={})))


SHEETS = {
    "cone": (_cone, False, [((X0 - 5.0, Y0 + h - 5.0 * math.tan(math.radians(t))), t)
                            for t in (0.0, 8.0, 17.0, 23.0) for h in (-17.0, -9.0, -2.5, 4.0, 11.0)]),
    "headlamp_reflector": (_reflector(False), True, _from((X0 + 0.01, Y0 + 0.02), np.arange(15.0, 166.0, 15.0))),
    "headlamp_reflector_whole": (_reflector(True), False,
                                 _from((X0 + 0.01, Y0 + 0.02), np.r_[np.arange(15.0, 166.0, 15.0),
                                                                     -np.arange(15.0, 166.0, 15.0)])),
    "sphere_and_flat": (_sphere_and_flat, False, _parallel(X0 - 2.0, (-15.0, -8.0, -2.0, -0.3, 0.4, 2.5, 9.0, 14.0))),
}


@pytest.mark.parametrize("name", list(SHEETS))
def test_the_rebuilt_sheet_traces_as_the_design(name):
    make, cut, rays = SHEETS[name]
    profile = _exported(make, cut)
    assert profile["body"] == "sheet"
    component = {"X": SimpleNamespace(name="X", profile=profile)}

    def original(scene):
        objs = make(scene)
        for o in objs:
            scene.add_object(o)
        return {"X": objs}

    def rebuilt(scene):
        objs = mirrors_from_sheet_profile(scene, profile)
        for o in objs:
            scene.add_object(o)
        return {"X": objs}

    met = 0
    for ray in rays:
        a = trace_2d_main_path(original, ray, components=component)
        b = trace_2d_main_path(rebuilt, ray, components=component)
        assert (b.faces, b.end) == (a.faces, a.end), (name, ray)
        assert np.allclose(b.point, a.point, atol=1e-6), (name, ray, a.point, b.point)
        assert np.allclose(b.direction, a.direction, atol=1e-9), (name, ray, a.direction, b.direction)
        met += bool(a.faces)
    assert met >= len(rays) // 2, f"{name}: only {met} of {len(rays)} rays met the mirrors"


def test_the_sheet_rebuild_joins_what_the_axis_cut():
    count = {name: len(mirrors_from_sheet_profile(Scene(), _exported(make, cut)))
             for name, (make, cut, _) in SHEETS.items()}
    assert count == {"cone": 2, "headlamp_reflector": 2, "headlamp_reflector_whole": 4, "sphere_and_flat": 2}
    rebuilt = mirrors_from_sheet_profile(Scene(), _exported(_sphere_and_flat, False))
    assert isinstance(rebuilt[0], ArcMirror) and rebuilt[0].extent == pytest.approx((-20.0, 20.0))
    with pytest.raises(ValueError, match="from_profile"):
        mirrors_from_sheet_profile(Scene(), export_profile(_shafer(0)(Scene()), symmetry={
            "kind": "revolve", "axis": _shafer(0)(Scene()).revolve_axis()}, physical_params={}, numerical_params={}))


# --- the SMS elements --------------------------------------------------------------

from ray_tracing_shapely.optical_elements.prisms.rr_lens import RRLens  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.rx_concentrator import RXConcentrator  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.rxi_direct import RXIOpticDirect  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.xr_concentrator import XRConcentrator  # noqa: E402

SMS_AXIS = {"kind": "revolve", "axis": {"point": [0.0, 0.0], "direction": [0.0, 1.0]}}
SMS = {
    "rr": lambda s: RRLens(s, (-1.0, 2.0), (1.0, 2.0), (-0.5, -2.0), (0.5, -2.0), seed_n_angle_deg=303.3,
                           seed_x_angle_deg=42.7),
    "rr_dense": lambda s: RRLens(s, (-1.0, 2.0), (1.0, 2.0), (-0.5, -2.0), (0.5, -2.0), seed_n_angle_deg=303.3,
                                 seed_x_angle_deg=42.7, n_points_per_section=50),
    "rx": lambda s: RXConcentrator(s, (-0.5, 0.0), (0.5, 0.0), theta_deg=2.0, n=1.5),
    "rxi": lambda s: RXIOpticDirect(s, (-1.0, 0.0), (1.0, 0.0), theta_deg=10.0, n=1.5),
    "xr_dielectric": lambda s: XRConcentrator(s, (-0.5, 0.0), (0.5, 0.0), theta_deg=2.0, p0=(0.0, -4.0),
                                              n0=(0.0, -1.0), n=1.5),
}


def _extent(design):
    """The x of an outline and its top, from its surfaces: a spline path keeps only the corners
    where surfaces meet, so its highest vertex need not be the top."""
    pts = [c for sp in design.spans for c in sp.coords(1e-3)]
    return [c[0] for c in pts], max(c[1] for c in pts)


@pytest.mark.parametrize("name", list(SMS))
def test_the_rebuilt_sms_element_traces_as_the_design(name):
    make = SMS[name]
    design = make(Scene())
    profile = json.loads(json.dumps(export_profile(design, symmetry=SMS_AXIS, physical_params={},
                                                   numerical_params={})))
    component = {"X": SimpleNamespace(name="X", profile=profile)}

    def original(scene):
        obj = make(scene)
        scene.add_object(obj)
        return {"X": obj}

    def rebuilt(scene):
        obj = BasePrism.from_profile(scene, profile, n=design.refIndex)
        scene.add_object(obj)
        return {"X": obj}

    xs, top = _extent(design)
    top += 1.0
    span = max(xs) - min(xs)
    rays = [((x, top), a) for x in np.linspace(-0.45 * span, 0.45 * span, 11) for a in (-90.0, -88.0)]
    met = reflected = 0
    for ray in rays:
        a = trace_2d_main_path(original, ray, components=component)
        b = trace_2d_main_path(rebuilt, ray, components=component)
        assert (b.faces, b.end) == (a.faces, a.end), (name, ray, a.faces, b.faces)
        assert np.allclose(b.point, a.point, atol=1e-6), (name, ray, a.point, b.point)
        assert np.allclose(b.direction, a.direction, atol=1e-9), (name, ray, a.direction, b.direction)
        met += bool(a.faces)
        reflected += any(f.startswith(("X.X", "X.MB", "X.MT")) for f in a.faces) and name in ("rx", "rxi")
    assert met >= len(rays) // 2, f"{name}: only {met} of {len(rays)} rays met the element"
    if name in ("rx", "rxi"):
        assert reflected, f"{name}: no ray met its mirror"


def test_the_xr_primary_is_a_spline_sheet_that_rebuilds_and_revolves():
    """The XR's primary, a spline mirror, exports as a sheet whose one span is a spline starting on
    the axis; rebuilt, it traces as the design; revolved, it is one mirror of revolution."""
    from ray_tracing_shapely.optical_elements.mirrors import SplineMirror
    from ray_tracing_shapely_optiland.revolve import RevolvedProfileSheet, scene_placement

    def make(scene):
        return XRConcentrator(scene, (-0.5, 0.0), (0.5, 0.0), theta_deg=2.0, p0=(0.0, -4.0), n0=(0.0, -1.0),
                              n=1.5).primary

    profile = json.loads(json.dumps(export_sheet_profile(make(Scene()), symmetry=SMS_AXIS, physical_params={},
                                                         numerical_params={})))
    (span,) = profile["spans"]
    assert span["curve"]["type"] == "spline" and span["curve"]["points"][0][0] == 0.0
    component = {"X": SimpleNamespace(name="X", profile=profile)}

    def original(scene):
        objs = make(scene)
        for o in objs:
            scene.add_object(o)
        return {"X": objs}

    def rebuilt(scene):
        objs = mirrors_from_sheet_profile(scene, profile)
        assert len(objs) == 1 and isinstance(objs[0], SplineMirror)
        for o in objs:
            scene.add_object(o)
        return {"X": objs}

    for x in (-20.0, -14.0, -8.0, 9.0, 15.0, 21.0):
        a = trace_2d_main_path(original, ((x, 1.0), -88.0), components=component)
        b = trace_2d_main_path(rebuilt, ((x, 1.0), -88.0), components=component)
        assert a.faces and (b.faces, b.end) == (a.faces, a.end), (x, a.faces, b.faces)
        assert np.allclose(b.point, a.point, atol=1e-6) and np.allclose(b.direction, a.direction, atol=1e-9)
    sheet = RevolvedProfileSheet("X", profile, 1.0, cs=scene_placement(profile))
    assert sheet.surface_kinds == {"X": "spline"}


def test_the_xr_dielectric_revolves_with_its_lens_in_two_pieces():
    """The XR's lens curls back toward the axis past its widest point, so it is cut there: the cap
    through the axis and the zone beyond each revolve single-valued in r; with the flat top, a
    closed solid."""
    from optiland.materials.ideal import IdealMaterial
    from optiland.nonsequential import NSQMaterial

    from ray_tracing_shapely_optiland.revolve import RevolvedProfileSolid, scene_placement

    design = SMS["xr_dielectric"](Scene())
    profile = json.loads(json.dumps(export_profile(design, symmetry=SMS_AXIS, physical_params={},
                                                   numerical_params={})))
    solid = RevolvedProfileSolid("XR", profile, NSQMaterial(IdealMaterial(n=1.5)), cs=scene_placement(profile))
    assert solid.surface_kinds == {"TOP": "disc", "R2": "spline", "R1": "spline"}
