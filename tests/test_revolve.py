"""
Python-specific module: test revolve
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

A revolve profile.json -> an Optiland solid of revolution.

The oracle is Optiland's own Lens, built from a LensConfig with the same
numbers: the bridge's solid must trace identically to it. The second check
is the 2D trace of the same SingletLens: for an on-axis beam every ray is
meridional, so 2D and 3D must agree on where each ray crosses the axis.
"""

import copy
import math
import warnings

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.materials.ideal import IdealMaterial
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    NSQMaterial,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.components.geometry.analytic.frustum import CylindricalFrustumGeometry
from optiland.nonsequential.components.lens import Lens
from optiland.nonsequential.components.volume import NonWatertightVolumeError

from ray_tracing_shapely.core.scene import Scene
from ray_tracing_shapely.core.scene_objs.light_source.single_ray import SingleRay
from ray_tracing_shapely.core.simulator import Simulator
from ray_tracing_shapely.optical_elements.lenses import SingletLens
from ray_tracing_shapely.optical_elements.prisms.base_prism import BasePrism
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile
from ray_tracing_shapely_optiland.revolve import (
    RevolvedProfileSolid,
    axis_frame,
    surface_from_span,
)
from ray_tracing_shapely_optiland.scene_bridge import (
    ExtrudedPolygonSolid,
    import_from_optiland,
    interior_index,
)

N = 1.5
GLASS = NSQMaterial(IdealMaterial(n=N))
LENSES = {
    "biconvex": dict(r1=60.0, r2=-20.0, t=5.0, h=10.0),
    "meniscus": dict(r1=20.0, r2=40.0, t=5.0, h=8.0),
    "biconcave": dict(r1=-30.0, r2=30.0, t=2.0, h=8.0),
}


def _singlet(r1, r2, t, h, **kw):
    return SingletLens(Scene(), radius_front=r1, radius_back=r2, thickness=t, semi_diameter=h, n=N, **kw)


def _revolve_profile(prism, axis):
    return export_profile(prism, symmetry={"kind": "revolve", "axis": axis},
                          physical_params={}, numerical_params={})


def _lens_profile(**kw):
    lens = _singlet(**kw)
    return lens, _revolve_profile(lens, lens.revolve_axis())


def _trace(component, num_rays=3000, aperture=None):
    scene = NSQScene()
    scene.add_source("S", CoordinateSystem(z=-10.0), CollimatedSourceConfig(
        Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=aperture))
    scene.component_registry.add(component.name, component)
    scene.add_detector("D", CoordinateSystem(z=120.0), IrradianceDetectorConfig(
        width=200, height=200, num_pixels_x=4, num_pixels_y=4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return scene.trace(num_rays=num_rays, seed=3, max_depth=10, backend=NumpyBackend(seed=3),
                           record_paths=True)


# --- the built surfaces ----------------------------------------------------------------

def test_a_lens_becomes_two_conics_and_a_cylinder_that_pass_volume():
    _, profile = _lens_profile(**LENSES["biconvex"])
    solid = RevolvedProfileSolid("L", profile, GLASS)
    assert solid.surface_kinds == {"S2": "conic", "RIM": "frustum", "S1": "conic"}
    assert solid.volume is not None          # Optiland's Volume() ran and passed (C-BRIDGE-09)
    s1, s2 = solid.face("S1").geometry, solid.face("S2").geometry
    assert (float(s1.radius), float(s2.radius)) == pytest.approx((60.0, -20.0))
    assert solid.face("S1").cs.z == pytest.approx(0.0) and solid.face("S2").cs.z == pytest.approx(5.0)
    assert interior_index(solid) == pytest.approx(N)


def test_sidedness_follows_the_stored_normals():
    """The glass is material_back of S1 (n_geom points downstream, into the
    lens), material_front of S2 and of the rim -- as in Optiland's Lens."""
    _, profile = _lens_profile(**LENSES["biconvex"])
    solid = RevolvedProfileSolid("L", profile, GLASS)
    assert solid.interior_on_back("S1") is True
    assert solid.interior_on_back("S2") is False
    assert solid.interior_on_back("RIM") is False
    assert solid.face("S1").material_back is GLASS and solid.face("S2").material_front is GLASS


def test_the_frustums_n_geom_points_radially_outward():
    """A property of Optiland's frustum; the sidedness rule does not depend on it, but relies on probing."""
    g = CylindricalFrustumGeometry(5.0, 5.0, 0.0, 2.0)
    _, _, hit, n_geom = g.ray_intersect(np.array([[6.0, 0.0, 1.0]]), np.array([[-1.0, 0.0, 0.0]]))
    assert bool(hit[0]) and n_geom[0] == pytest.approx([1.0, 0.0, 0.0])


@pytest.mark.parametrize("shape", sorted(LENSES))
def test_the_solid_traces_identically_to_optilands_lens(shape):
    p = LENSES[shape]
    _, profile = _lens_profile(**p)
    ours = _trace(RevolvedProfileSolid("ours", profile, GLASS), aperture=p["h"] * 0.95)
    oracle = _trace(Lens("L", CoordinateSystem(), LensConfig(
        r1=p["r1"], r2=p["r2"], thickness=p["t"], material=GLASS, front_aperture_radius=p["h"])),
        aperture=p["h"] * 0.95)
    a, b = ours.ray_paths["events"], oracle.ray_paths["events"]
    # Within one bounce the log is grouped by component, and the two list
    # their components in a different order: compare ray by ray.
    a = a[np.argsort(a["ray_id"], kind="stable")]
    b = b[np.argsort(b["ray_id"], kind="stable")]
    assert len(a) == len(b) > 3000
    assert np.array_equal(a["event_type"], b["event_type"])
    for k in ("x", "y", "z", "L", "M", "N"):
        assert np.allclose(a[k], b[k], atol=1e-9), k


def test_2d_and_3d_agree_on_the_axis_crossing_of_meridional_rays():
    lens, profile = _lens_profile(**LENSES["biconvex"])
    result = _trace(RevolvedProfileSolid("L", profile, GLASS), num_rays=400, aperture=8.0)
    ev = result.ray_paths["events"]
    checked = 0
    for rid in np.unique(ev["ray_id"])[:60]:
        e = ev[ev["ray_id"] == rid]
        k = np.flatnonzero((e["event_type"] == "hit") & (e["component_name"] == "L.S2"))
        if not len(k) or k[0] + 1 >= len(e):
            continue
        h = math.hypot(e["x"][0], e["y"][0])
        if not 1.0 < h < 7.5:
            continue
        p = np.array([e["x"][k[0]], e["y"][k[0]], e["z"][k[0]]])
        d = np.array([e["L"][k[0] + 1], e["M"][k[0] + 1], e["N"][k[0] + 1]])
        t = -(p[:2] @ d[:2]) / (d[:2] @ d[:2])
        z3d = p[2] + t * d[2]
        assert z3d == pytest.approx(_axis_crossing_2d(lens, h), abs=1e-6)
        checked += 1
        if checked == 5:
            break
    assert checked == 5


def _axis_crossing_2d(lens, h):
    """Where a ray along +x at height h crosses y = 0 after the 2D SingletLens."""
    scene = Scene()
    lens2 = _singlet(**LENSES["biconvex"])
    lens2.scene = scene
    scene.add_object(lens2)
    scene.add_object(SingleRay(scene, {"p1": {"x": -40, "y": h}, "p2": {"x": -39, "y": h}}))
    result = Simulator(scene, max_rays=50).run_with_result(name="2d")
    for seg in result.segments:
        (x1, y1), (x2, y2) = (seg.p1["x"], seg.p1["y"]), (seg.p2["x"], seg.p2["y"])
        if seg.interaction_type == "refract" and x1 > 1.0 and y1 * y2 < 0:
            return x1 - y1 * (x2 - x1) / (y2 - y1)
    raise AssertionError("no 2D axis crossing")


def test_flat_faces_and_steps_become_discs_and_annuli():
    plano = _singlet(r1=40.0, r2=math.inf, t=4.0, h=8.0)
    solid = RevolvedProfileSolid("P", _revolve_profile(plano, plano.revolve_axis()), GLASS)
    assert solid.surface_kinds == {"S2": "disc", "RIM": "frustum", "S1": "conic"}
    # A stepped cylinder: radius 6 for 0 < z < 4, radius 3 for 4 < z < 9.
    outline = [(0, -6), (4, -6), (4, -3), (9, -3), (9, 3), (4, 3), (4, 6), (0, 6)]
    stepped = BasePrism.from_vertices(Scene(), outline, n=N)
    profile = _revolve_profile(stepped, {"point": [0, 0], "direction": [1, 0]})
    solid = RevolvedProfileSolid("step", profile, GLASS)
    assert sorted(solid.surface_kinds.values()) == ["annulus", "disc", "disc", "frustum", "frustum"]


def test_a_boundary_with_a_gap_fails_volume():
    _, profile = _lens_profile(**LENSES["biconvex"])
    broken = copy.deepcopy(profile)
    rim = next(s for s in broken["spans"] if s["short_label"] == "RIM")
    for k in (rim["index_start"], rim["index_start"] + 1):   # shrink the cylinder: rims no longer meet
        x, y = broken["points"][k]
        broken["points"][k] = [x, y - 0.5]
    with pytest.raises(NonWatertightVolumeError):
        RevolvedProfileSolid("broken", broken, GLASS)


#: Shafer's first element: its edge sags 5.5 mm, its centre is 1.99 thick.
STEEP_MENISCUS = dict(r1=22.64, r2=12.64, t=1.99, h=10.43)


def test_a_steep_meniscus_is_validated_from_a_point_inside_it():
    """KW-OPTILAND-04. Volume() casts its parity rays from the mean of the
    rim points, which for this meniscus is on the axis at z = 5.0, in the
    hollow (the glass on the axis spans 0..1.99). The bridge casts the same
    check from the half-section's pole of inaccessibility instead."""
    from optiland.nonsequential import LensConfig
    from shapely.geometry import Point, Polygon

    from ray_tracing_shapely_optiland.revolve import interior_point

    lens, profile = _lens_profile(**STEEP_MENISCUS)
    pole = interior_point(profile)
    assert Polygon(profile["points"]).contains(Point(pole))
    solid = RevolvedProfileSolid("L", profile, GLASS)          # builds, and is validated
    assert solid.volume.boundary == solid.surfaces
    # Optiland's own estimate is outside the glass, so its Lens refuses the same lens.
    # When this stops raising, upstream has fixed it: close KW-OPTILAND-04.
    with pytest.raises(NonWatertightVolumeError, match="estimated interior point"):
        Lens("L", CoordinateSystem(), LensConfig(r1=22.64, r2=12.64, thickness=1.99,
                                                 material=GLASS, front_aperture_radius=10.43))


def test_the_parity_check_really_runs_from_the_point_given():
    """The bridge's check is not a skipped check: from a point outside the
    solid (the rim mean, here), the same faces fail."""
    from ray_tracing_shapely_optiland.revolve import _validated_volume

    _, profile = _lens_profile(**STEEP_MENISCUS)
    solid = RevolvedProfileSolid("L", profile, GLASS)
    hollow = solid.to_world((5.0, 0.0))          # on the axis, beyond the back vertex
    with pytest.raises(NonWatertightVolumeError):
        _validated_volume("L", solid.surfaces, GLASS, hollow)


# --- what is refused ---------------------------------------------------------------------

def _single_arc_profile(center, radius, angles_deg):
    pts = [[center[0] + radius * math.cos(math.radians(a)), center[1] + radius * math.sin(math.radians(a))]
           for a in angles_deg]
    nrm = [[(x - center[0]) / radius, (y - center[1]) / radius] for x, y in pts]
    return {
        "points": pts, "normals": nrm,
        "spans": [{"index_start": 0, "index_end": len(pts), "short_label": "A", "mechanism": "refract",
                   "curve": {"type": "circle", "center": list(center), "radius": radius}}],
        "symmetry": {"kind": "revolve", "axis": {"point": [0, 0], "direction": [1, 0]}},
        "body": "solid",
    }


@pytest.mark.parametrize("center, angles, match", [
    ((0.0, 5.0), (180, 225, 270), "torus"),
    ((0.0, 0.0), (120, 135, 150), "spherical zone"),
    ((0.0, 0.0), (180, 120, 60), "equator"),
])
def test_spans_no_analytic_geometry_describes_raise(center, angles, match):
    # A zone is built by default; allow_zone=False still refuses it.
    profile = _single_arc_profile(center, 10.0, angles)
    with pytest.raises(NotImplementedError, match=match):
        surface_from_span(profile["spans"][0], profile, allow_zone=False)


def test_a_sample_off_the_conic_raises():
    _, profile = _lens_profile(**LENSES["biconvex"])
    s1 = next(s for s in profile["spans"] if s["short_label"] == "S1")
    k = s1["index_start"] + 9
    profile["points"][k][0] += 1e-6
    with pytest.raises(ValueError, match="from the conic"):
        RevolvedProfileSolid("L", profile, GLASS)


def test_a_normal_off_the_conics_raises():
    _, profile = _lens_profile(**LENSES["biconvex"])
    s2 = next(s for s in profile["spans"] if s["short_label"] == "S2")
    k = s2["index_start"] + 4
    nx, ny = profile["normals"][k]
    c, s = math.cos(1e-3), math.sin(1e-3)
    profile["normals"][k] = [c * nx - s * ny, s * nx + c * ny]
    with pytest.raises(ValueError, match="stored normal"):
        RevolvedProfileSolid("L", profile, GLASS)


def test_the_extruded_solid_refuses_a_revolve_profile():
    _, profile = _lens_profile(**LENSES["biconvex"])
    with pytest.raises(NotImplementedError, match="revolve"):
        ExtrudedPolygonSolid("L", profile, GLASS)


# --- the frame and the way back ---------------------------------------------------------

def test_axis_frame_maps_the_profile_plane_to_world_xz():
    lens, profile = _lens_profile(**LENSES["biconvex"], rotation=30.0, position=(2.0, 1.0))
    frame = axis_frame(profile)
    assert frame.to_world(lens.front_vertex) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert frame.to_world(lens.back_vertex) == pytest.approx((0.0, 0.0, 5.0), abs=1e-12)
    p = (3.3, -1.7)
    assert frame.to_profile(frame.to_world(p)) == pytest.approx(p)
    from optiland.nonsequential.components.base import _get_transform
    _, rot = _get_transform(frame.source_cs(-10.0, field_angle_deg=5.0))
    assert rot[:, 2] == pytest.approx([math.sin(math.radians(5)), 0.0, math.cos(math.radians(5))])


def test_import_round_trip_and_back_to_2d():
    lens, profile = _lens_profile(**LENSES["meniscus"], rotation=-20.0)
    solid = RevolvedProfileSolid("L", profile, GLASS)
    back = import_from_optiland(solid)
    assert [s["short_label"] for s in back["spans"]] == [s["short_label"] for s in profile["spans"]]
    assert np.allclose(back["points"], profile["points"], atol=1e-12)
    assert np.allclose(back["normals"], profile["normals"], atol=1e-12)
    rebuilt = BasePrism.from_profile(Scene(), back, n=interior_index(solid))
    assert len(rebuilt.spans) == 4
    assert rebuilt.signed_area() == pytest.approx(lens.signed_area(), rel=1e-9)


def test_import_reads_a_reshaped_face():
    _, profile = _lens_profile(**LENSES["biconvex"])
    solid = RevolvedProfileSolid("L", profile, GLASS)
    solid.face("S1").geometry.radius = 80.0     # what an optimiser would do
    back = import_from_optiland(solid)
    s1 = next(s for s in back["spans"] if s["short_label"] == "S1")
    assert s1["curve"]["radius"] == pytest.approx(80.0)
    assert s1["curve"]["center"] == pytest.approx([80.0, 0.0])
    # The reshaped face no longer meets the rim; the 2D rebuild says so.
    with pytest.raises(ValueError, match="does not close"):
        BasePrism.from_profile(Scene(), back, n=N)
