"""
Python-specific module: test_revolve_cut_sheet
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
CUT SHEETS -- a conic zone and an azimuth cut (C-BRIDGE-12)
===============================================================================
The reflector of the patent US 7,255,467 B2 FIG. 2 projector unit, in 3D:
one ellipse (F1 to F2) drawn above its axis only, as two ConicMirror pieces
(a cap from the rear vertex to the equator, a zone of the front vertex's
sheet beyond it), revolved over the upper half (azimuth -90..90 deg).

- CutConicGeometry against an oracle built from Optiland's own uncut
  ConicGeometry: the first root that lies in the kept part.
- The 2D scene is the 3D section, side for side: rays drawn below the axis
  pass through in both.
- Every ray from F1, skew rays included, reflects through F2.
===============================================================================
"""

from __future__ import annotations

import copy
import math
import warnings

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.materials.ideal import IdealMaterial
from optiland.nonsequential import (CollimatedSourceConfig, IrradianceDetectorConfig, NSQMaterial,
                                    NSQScene, PointSourceConfig, Spectrum)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.components.geometry.analytic.conic import ConicGeometry

from ray_tracing_shapely_optiland.geometry import CutConicGeometry
from ray_tracing_shapely_optiland.revolve import (RevolvedProfileSheet, build_revolved_faces,
                                                   profile_azimuth, scene_placement,
                                                   surface_from_span)
from ray_tracing_shapely_optiland.scene_bridge import import_from_optiland

ray_tracing_shapely = pytest.importorskip("ray_tracing_shapely")
from ray_tracing_shapely.core.scene import Scene  # noqa: E402
from ray_tracing_shapely.core.scene_objs.light_source.single_ray import SingleRay  # noqa: E402
from ray_tracing_shapely.core.scene_objs.mirror.mirror import Mirror  # noqa: E402
from ray_tracing_shapely.core.simulator import Simulator  # noqa: E402
from ray_tracing_shapely.optical_elements.mirrors import ConicMirror  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.profile_export import (  # noqa: E402
    export_profile, export_sheet_profile)


# --- CutConicGeometry against Optiland's own conic ----------------------------------------

def _oracle(radius, conic, rim, inner, kept_azimuth, origins, directions):
    """The first intersection with Optiland's uncut ConicGeometry that lies in the
    kept zone: relaunch past every hit that does not. Returns (hit, t)."""
    plain = ConicGeometry(radius, conic, rim)
    hit = np.zeros(len(origins), bool)
    t_out = np.full(len(origins), np.inf)
    for i, (o, d) in enumerate(zip(origins, directions)):
        travelled, o = 0.0, o.copy()
        for _ in range(3):
            t, _, h, _ = plain.ray_intersect(o[None, :], d[None, :])
            if not bool(h[0]):
                break
            p = o + float(t[0]) * d
            r = math.hypot(p[0], p[1])
            phi = math.degrees(math.atan2(p[1], p[0]))
            if r >= inner and kept_azimuth(phi):
                hit[i], t_out[i] = True, travelled + float(t[0])
                break
            travelled += float(t[0]) + 1e-6
            o = p + 1e-6 * d
    return hit, t_out


CUTS = {
    "hole": (5.0, None),
    "half": (0.0, (-90.0, 90.0)),
    "quarter": (0.0, (10.0, 100.0)),
    "reflex": (0.0, (-45.0, 225.0)),
    "hole and half": (8.0, (-90.0, 90.0)),
}


@pytest.mark.parametrize("cut", list(CUTS))
@pytest.mark.parametrize("radius, conic", [(-30.0, -0.25), (25.0, 0.0), (-40.0, -3.0)])
def test_a_cut_conic_finds_the_first_root_in_the_kept_part(cut, radius, conic):
    """Random rays from all around, through and past the surface: the hit is
    Optiland's uncut conic's first hit in the kept part, at the same distance
    (to within the 1e-6 relaunch step of the oracle)."""
    inner, azimuth = CUTS[cut]
    rim = 0.9 * abs(radius) / math.sqrt(max(1.0 + conic, 1.0))
    geometry = CutConicGeometry(radius, conic, rim, inner_radius=inner, azimuth_deg=azimuth)
    rng = np.random.default_rng(7)
    n = 400
    origins = rng.uniform(-1.5 * rim, 1.5 * rim, (n, 3))
    targets = rng.uniform(-rim, rim, (n, 3))
    targets[:, 2] = rng.uniform(-0.5 * abs(radius), 0.5 * abs(radius), n)
    directions = targets - origins
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    def kept(phi):
        if azimuth is None:
            return True
        return (phi - azimuth[0]) % 360.0 <= azimuth[1] - azimuth[0]

    t, _, hit, _ = geometry.ray_intersect(origins, directions)
    hit_o, t_o = _oracle(radius, conic, rim, inner, kept, origins, directions)
    assert np.array_equal(np.asarray(hit), hit_o)
    assert np.allclose(np.asarray(t)[hit_o], t_o[hit_o], atol=1e-5)
    assert 30 < hit_o.sum() < n            # the rays do meet it, and do miss it


def test_an_uncut_cut_conic_is_optilands_conic_bit_for_bit():
    rng = np.random.default_rng(3)
    origins = rng.uniform(-40, 40, (500, 3))
    directions = rng.normal(size=(500, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    ours = CutConicGeometry(-30.0, -0.25, 30.0, azimuth_deg=(-180.0, 180.0))
    theirs = ConicGeometry(-30.0, -0.25, 30.0)
    assert ours.azimuth_deg is None and ours.inner_radius == 0.0
    for a, b in zip(ours.ray_intersect(origins, directions), theirs.ray_intersect(origins, directions)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_an_azimuth_range_includes_its_edges_exactly():
    g = CutConicGeometry(-30.0, 0.0, 20.0, azimuth_deg=(-90.0, 90.0))
    x = np.array([0.0, 0.0, 1.0, -1.0, 0.0])
    y = np.array([5.0, -5.0, 0.0, 0.0, 0.0])
    assert list(np.asarray(g.in_azimuth(x, y))) == [True, True, True, False, True]
    reflex = CutConicGeometry(-30.0, 0.0, 20.0, azimuth_deg=(0.0, 270.0))
    x = np.array([1.0, 0.0, -1.0, 0.0, 1.0])
    y = np.array([0.0, 1.0, 0.0, -1.0, -1.0])
    assert list(np.asarray(reflex.in_azimuth(x, y))) == [True, True, True, True, False]


@pytest.mark.parametrize("kwargs, match", [
    ({"inner_radius": 20.0}, "inner_radius"),
    ({"inner_radius": -1.0}, "inner_radius"),
    ({"azimuth_deg": (10.0, 10.0)}, "azimuth_deg"),
    ({"azimuth_deg": (0.0, 400.0)}, "azimuth_deg"),
    ({"azimuth_deg": (0.0, float("nan"))}, "azimuth_deg"),
])
def test_what_a_cut_conic_refuses(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CutConicGeometry(-30.0, 0.0, 20.0, **kwargs)


def test_a_cut_conic_traces_on_the_torch_backend():
    torch = pytest.importorskip("torch")
    import optiland.backend as be

    g = CutConicGeometry(-30.0, -0.25, 34.0, inner_radius=10.0, azimuth_deg=(-90.0, 90.0))
    o = np.array([[-50.0, 0.0, -20.0], [20.0, 0.0, -100.0], [-20.0, 0.0, -100.0]])
    d = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    t_np, _, hit_np, _ = g.ray_intersect(o, d)
    previous = be.get_backend()
    be.set_backend("torch")
    try:
        t_t, _, hit_t, _ = g.ray_intersect(torch.tensor(o), torch.tensor(d))
    finally:
        be.set_backend(previous)
    assert hit_t.tolist() == list(np.asarray(hit_np)) == [True, True, False]
    assert np.allclose(t_t.detach().numpy()[:2], np.asarray(t_np)[:2], atol=1e-12)


# --- the patent's reflector: a cap and a zone of one ellipse, upper half --------------------

#: F1, well away from the world origin (KW-OPTILAND-05); the axis tilted 10 deg in 2D.
F1 = np.array([120.0, 45.0])
U = np.array([math.cos(math.radians(10.0)), math.sin(math.radians(10.0))])
LEFT = np.array([-U[1], U[0]])
A_, C_ = 40.0, 20.0                          # semi-major axis, centre to focus
B_ = math.sqrt(A_ * A_ - C_ * C_)
R_, K_ = B_ * B_ / A_, -(C_ / A_) ** 2       # vertex radius 30, conic -0.25
F2 = F1 + 2.0 * C_ * U
FRONT = 10.0                                 # the zone ends 10 mm past the centre
UPPER = [-90.0, 90.0]


def _reflector(scene):
    rear, front = F1 - (A_ - C_) * U, F1 + (A_ + C_) * U
    y_front = B_ * math.sqrt(1.0 - (FRONT / A_) ** 2)
    return [ConicMirror(scene, tuple(rear), tuple(U), R_, K_, extent=(0.0, B_), name="cap"),
            ConicMirror(scene, tuple(front), tuple(-U), R_, K_, extent=(-B_, -y_front), name="zone")]


def _off_ellipse(r, z):
    """Distance, to first order, of points (r from the axis, z along it from F1)
    from the reflector's ellipse, whose centre is C_ from F1."""
    f = ((z - C_) / A_) ** 2 + (r / B_) ** 2 - 1.0
    grad = np.hypot(2.0 * (z - C_) / A_ ** 2, 2.0 * r / B_ ** 2)
    return np.abs(f) / grad


def _symmetry(azimuth=UPPER):
    sym = {"kind": "revolve", "axis": {"point": list(F1), "direction": list(U)}}
    if azimuth is not None:
        sym["azimuth_deg"] = azimuth
    return sym


def _profile(azimuth=UPPER):
    return export_sheet_profile(_reflector(Scene()), symmetry=_symmetry(azimuth),
                                physical_params={}, numerical_params={})


def _sheet():
    profile = _profile()
    return RevolvedProfileSheet("R", profile, 1.0, cs=scene_placement(profile)), profile


def test_the_reflector_is_a_cut_cap_and_a_cut_zone():
    sheet, profile = _sheet()
    assert profile["symmetry"]["azimuth_deg"] == UPPER
    assert sheet.azimuth_deg == (-90.0, 90.0)
    cap, zone = sheet.face("cap").geometry, sheet.face("zone").geometry
    assert isinstance(cap, CutConicGeometry) and isinstance(zone, CutConicGeometry)
    assert cap.inner_radius == 0.0 and cap.azimuth_deg == zone.azimuth_deg == (-90.0, 90.0)
    assert zone.inner_radius == pytest.approx(B_ * math.sqrt(1.0 - (FRONT / A_) ** 2), abs=1e-12)
    assert float(cap.aperture_radius) == pytest.approx(B_, abs=1e-12)
    assert float(zone.aperture_radius) == pytest.approx(B_, abs=1e-12)


def _trace_2d(theta_deg):
    """The 2D ray from F1 at theta from the axis (positive: the drawn, upper side):
    (its last segment's start, unit direction, number of reflections)."""
    scene = Scene()
    for m in _reflector(scene):
        scene.add_object(m)
    th = math.radians(theta_deg)
    d = math.cos(th) * U + math.sin(th) * LEFT
    scene.add_object(SingleRay(scene, {"p1": {"x": F1[0], "y": F1[1]},
                                       "p2": {"x": F1[0] + d[0], "y": F1[1] + d[1]}}))
    segs = Simulator(scene, max_rays=200).run_with_result(name="2d").segments
    last = segs[-1]
    a = np.array([last.p1["x"], last.p1["y"]])
    e = np.array([last.p2["x"], last.p2["y"]]) - a
    return a, e / np.linalg.norm(e), len(segs) - 1


def _box(scene, half=300.0):
    """Four detector planes around F1, perpendicular to the profile plane: every ray ends on one."""
    for k, (n, along) in enumerate(((U, LEFT), (-U, LEFT), (LEFT, U), (-LEFT, U))):
        c = F1 + half * n
        scene.add_detector(f"D{k}", CoordinateSystem(x=c[0], y=0.0, z=c[1],
                                                     ry=math.atan2(n[0], n[1])),
                           IrradianceDetectorConfig(width=4 * half, height=4 * half,
                                                    num_pixels_x=2, num_pixels_y=2))


THETAS = (15.0, 50.0, 85.0, 120.0, 160.0, 175.0, -20.0, -70.0, -130.0, -175.0)


def test_every_meridional_ray_ends_in_3d_where_it_ends_in_2d_on_both_sides():
    """C-BRIDGE-12 for a cut sheet: rays from F1 in the profile plane, above the axis
    (they reflect, off the cap or the zone, through F2) and below it (the cut removed
    that side: they pass in 3D, as they do in the 2D scene, which has nothing there)."""
    sheet, _ = _sheet()
    scene = NSQScene()
    for k, th in enumerate(THETAS):
        d = math.cos(math.radians(th)) * U + math.sin(math.radians(th)) * LEFT
        scene.add_source(f"P{k}", CoordinateSystem(x=F1[0], y=0.0, z=F1[1],
                                                   ry=math.atan2(d[0], d[1])),
                         CollimatedSourceConfig(Spectrum.monochromatic(0.55), total_flux=1.0,
                                                aperture_radius=1e-9))
    scene.component_registry.add(sheet.name, sheet)
    _box(scene)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = scene.trace(num_rays=5 * len(THETAS), seed=2, max_depth=20,
                         backend=NumpyBackend(seed=2), record_paths=True).ray_paths["events"]
    ev = ev[np.argsort(ev["ray_id"], kind="stable")]
    _, starts = np.unique(ev["ray_id"], return_index=True)
    seen, faces = set(), set()
    for i, st in enumerate(starts):
        r = ev[st: starts[i + 1] if i + 1 < len(starts) else len(ev)]
        k = int(str(r["component_name"][0]).split("_")[1])
        a, e, n_reflect = _trace_2d(THETAS[k])
        assert r["component_name"][-1] == "", f"ray at {THETAS[k]} deg did not reach a detector"
        names = [str(c) for c in r["component_name"] if str(c).startswith("R.")]
        assert len(names) == n_reflect
        faces.update(names)
        end = np.array([r["x"][-1], r["z"][-1]])
        # positions to 1e-7: the pencils are 1e-9 wide, grown over ~300 mm and a reflection
        assert abs(r["y"][-1]) < 1e-7
        off = end - a
        assert abs(off[0] * e[1] - off[1] * e[0]) < 1e-7  # on the 2D ray's last line
        assert (end - a) @ e > 0.0
        assert np.allclose([r["L"][-1], r["N"][-1]], e, atol=1e-9)
        seen.add(k)
    assert seen == set(range(len(THETAS)))
    assert faces == {"R.cap", "R.zone"}


def test_every_ray_from_f1_that_meets_the_cut_reflector_reflects_through_f2():
    """The skew rays too. And none meets the removed half."""
    sheet, _ = _sheet()
    scene = NSQScene()
    scene.add_source("LED", CoordinateSystem(x=F1[0], y=0.0, z=F1[1]), PointSourceConfig(
        Spectrum.monochromatic(0.55), total_flux=1.0, half_angle_deg=180.0))
    scene.component_registry.add(sheet.name, sheet)
    _box(scene)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = scene.trace(num_rays=4000, seed=1, max_depth=10, backend=NumpyBackend(seed=1),
                         record_paths=True).ray_paths["events"]
    ev = ev[np.argsort(ev["ray_id"], kind="stable")]
    idx = np.where(np.char.startswith(ev["component_name"].astype(str), "R."))[0]
    assert set(ev["component_name"][idx]) == {"R.cap", "R.zone"}
    assert len(np.unique(ev["ray_id"][idx])) == len(idx)        # one reflection each
    p = np.c_[ev["x"][idx], ev["y"][idx], ev["z"][idx]]
    out = np.c_[ev["L"][idx + 1], ev["M"][idx + 1], ev["N"][idx + 1]]
    f2 = np.array([F2[0], 0.0, F2[1]])
    assert np.linalg.norm(np.cross(f2 - p, out), axis=1).max() < 1e-9
    t, rot = _get_transform(sheet.coordinate_system)
    local = (p - t) @ rot
    assert local[:, 0].min() >= 0.0                                 # the kept half only
    assert np.abs(local[:, 1]).max() > 0.9 * B_                     # skew rays, far out of plane
    # About half of all rays: the upper half of an ellipsoid that nearly encloses F1.
    assert 0.35 < len(idx) / 4000 < 0.5


def test_a_cut_sheet_reads_back_to_its_profile_and_a_reshaped_zone_reads_reshaped():
    sheet, profile = _sheet()
    back = import_from_optiland(sheet)
    assert back["symmetry"]["azimuth_deg"] == UPPER
    assert np.allclose(back["points"], profile["points"], atol=1e-9)
    assert np.allclose(back["normals"], profile["normals"], atol=1e-6)
    sheet.face("zone").geometry.inner_radius = 30.0
    zone = import_from_optiland(sheet)["spans"][1]
    pts = np.array(import_from_optiland(sheet)["points"][zone["index_start"]:zone["index_end"]])
    assert np.abs((pts - F1) @ LEFT).min() == pytest.approx(30.0, abs=1e-9)


# --- what is refused ------------------------------------------------------------------------

def test_the_export_refuses_a_mirror_on_the_side_the_cut_removes():
    scene = Scene()
    below = Mirror(scene, {"p1": {"x": float((F1 - 5 * LEFT)[0]), "y": float((F1 - 5 * LEFT)[1])},
                           "p2": {"x": float((F1 - 5 * LEFT + 10 * U)[0]),
                                  "y": float((F1 - 5 * LEFT + 10 * U)[1])}})
    below.name = "below"
    with pytest.raises(ValueError, match="side the azimuth cut"):
        export_sheet_profile(_reflector(scene) + [below], symmetry=_symmetry(),
                             physical_params={}, numerical_params={})
    # uncut, the one-sided reflector is not a surface of revolution's section
    with pytest.raises(ValueError, match="mirror-symmetric"):
        _profile(azimuth=None)


@pytest.mark.parametrize("azimuth, match", [
    ([90.0, 270.0], "half-section itself"),
    ([10.0, 10.0], "end - start"),
    ([0.0], "start, end"),
])
def test_a_bad_azimuth_is_refused_by_the_export_and_by_the_bridge(azimuth, match):
    with pytest.raises(ValueError, match=match):
        _profile(azimuth=azimuth)
    hand_edited = _profile()
    hand_edited["symmetry"]["azimuth_deg"] = azimuth
    with pytest.raises(ValueError, match=match):
        profile_azimuth(hand_edited)
    with pytest.raises(ValueError, match=match):
        RevolvedProfileSheet("R", hand_edited, 1.0)


def test_a_full_turn_is_no_cut():
    sym = _symmetry([-180.0, 180.0])
    scene = Scene()
    twin = [ConicMirror(scene, tuple(F1 - (A_ - C_) * U), tuple(U), R_, K_, extent=(-15.0, 15.0),
                        name="cap")]
    full = export_sheet_profile(twin, symmetry=sym, physical_params={}, numerical_params={})
    assert "azimuth_deg" not in full["symmetry"]
    sheet = RevolvedProfileSheet("R", full, 1.0)
    assert sheet.azimuth_deg is None and type(sheet.face("cap").geometry) is ConicGeometry


def test_a_line_span_refuses_an_azimuth_cut():
    scene = Scene()
    a, b = F1 + 2.0 * LEFT, F1 + 2.0 * LEFT + 10.0 * U
    wall = Mirror(scene, {"p1": {"x": float(a[0]), "y": float(a[1])},
                          "p2": {"x": float(b[0]), "y": float(b[1])}})
    wall.name = "wall"
    profile = export_sheet_profile([wall], symmetry=_symmetry(), physical_params={},
                                   numerical_params={})
    with pytest.raises(NotImplementedError, match="curved spans only"):
        RevolvedProfileSheet("W", profile, 1.0)


def test_a_solid_refuses_a_cut():
    from ray_tracing_shapely.optical_elements.lenses import SingletLens

    lens = SingletLens(Scene(), radius_front=50.0, radius_back=-50.0, thickness=8.0,
                       semi_diameter=10.0, n=1.5)
    sym = {"kind": "revolve", "axis": {"point": [0.0, 0.0], "direction": [1.0, 0.0]},
           "azimuth_deg": UPPER}
    with pytest.raises(ValueError, match="sheets only"):
        export_profile(lens, symmetry=sym, physical_params={}, numerical_params={})
    solid_profile = export_profile(lens, symmetry={k: v for k, v in sym.items() if k != "azimuth_deg"},
                                   physical_params={}, numerical_params={})
    cut = copy.deepcopy(solid_profile)
    cut["symmetry"]["azimuth_deg"] = UPPER
    glass = NSQMaterial(IdealMaterial(n=1.5))
    with pytest.raises(ValueError, match="only a sheet"):
        build_revolved_faces(cut, glass)
    # a zone: solids may have one (test_revolve_zones_in_solids);
    # allow_zone=False still refuses it
    profile = _profile()
    with pytest.raises(NotImplementedError, match="hole"):
        surface_from_span(profile["spans"][1], profile, allow_zone=False)


# --- drawing --------------------------------------------------------------------------------

def test_the_2d_renderer_draws_only_the_kept_side():
    from optiland.nonsequential.visualization.viewer_2d import NSQViewer2D

    from ray_tracing_shapely_optiland.visualization.renderers import RevolvedProfileSheetRenderer2D

    sheet, profile = _sheet()
    scene = NSQScene()
    scene.component_registry.add(sheet.name, sheet)
    viewer = NSQViewer2D(scene)
    viewer.register_renderer(RevolvedProfileSheet, RevolvedProfileSheetRenderer2D())
    _, ax = viewer.view(num_rays=0, projection="XZ")
    lines = [np.asarray(line.get_xydata()) for line in ax.lines]
    assert len(lines) == len(profile["spans"])                      # no mirror images
    # every drawn point on the ellipse, above the axis, and each line spans its span
    # (near the equator a point may slide 1e-6 along the steep curve, not off it)
    for line, span in zip(lines, profile["spans"]):
        q = line[:, ::-1] - F1                  # XZ: horizontal world z (the scene's b)
        r, z = q @ LEFT, q @ U
        assert r.min() > -1e-9 and _off_ellipse(r, z).max() < 1e-9
        ends = np.array(profile["points"])[[span["index_start"], span["index_end"] - 1]]
        assert np.linalg.norm(line[[0, -1], ::-1] - ends, axis=1).max() < 1e-6
    # looking down the axis: half circles
    viewer = NSQViewer2D(NSQScene())
    s2 = RevolvedProfileSheet("R", profile, 1.0)                    # unplaced: axis along world z
    viewer.scene.component_registry.add(s2.name, s2)
    viewer.register_renderer(RevolvedProfileSheet, RevolvedProfileSheetRenderer2D())
    _, ax = viewer.view(num_rays=0, projection="XY")
    for line in ax.lines:
        xy = np.asarray(line.get_xydata())
        assert xy[:, 0].min() > -1e-9 and np.hypot(*xy.T) == pytest.approx(B_)


def test_the_3d_renderer_sweeps_only_the_kept_half():
    vtk = pytest.importorskip("vtk")
    from ray_tracing_shapely_optiland.visualization.renderers import RevolvedProfileSheetRenderer3D

    sheet, profile = _sheet()
    renderer = vtk.vtkRenderer()
    RevolvedProfileSheetRenderer3D().render(sheet, renderer)
    actors = [renderer.GetActors().GetItemAsObject(i)
              for i in range(renderer.GetActors().GetNumberOfItems())]
    assert len(actors) == 2
    t, rot = _get_transform(sheet.coordinate_system)
    half = np.array([sheet.frame.to_world(tuple(p)) for p in profile["points"]])[:, [0, 2]]
    for actor in actors:
        actor.GetMapper().Update()
        data = actor.GetMapper().GetInput()
        local = (np.array([data.GetPoint(i) for i in range(data.GetNumberOfPoints())]) - t) @ rot
        assert local[:, 0].min() > -1e-9                            # nothing on the cut side
        r_max = half[:, 0].max()
        assert local[:, 1].min() < -0.99 * r_max and local[:, 1].max() > 0.99 * r_max  # both edges
        assert _off_ellipse(np.hypot(local[:, 0], local[:, 1]), local[:, 2]).max() < 1e-9
