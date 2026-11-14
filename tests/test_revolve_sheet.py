"""
Python-specific module: test revolve sheet
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

Revolved mirrors (body "sheet") through the bridge.

Three oracles, as for the lenses:

- the 2D trace of the same mirrors (ConicMirror, ArcMirror, flat Mirror):
  every meridional ray must end in 3D where it ends in 2D, after the same
  number of reflections, for a paraboloid, an ellipsoid, a hyperboloid, a
  sphere and a cone, on an axis tilted in the 2D scene;
- Optiland's own ReflectiveComponent on a ConicGeometry, built by hand:
  identical events;
- the paraboloid's focus, for the skew rays of a full beam, which 2D does not have.
"""

import copy
import math
import warnings

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (CollimatedSourceConfig, IrradianceDetectorConfig, NSQMaterial,
                                    NSQScene, Spectrum)
from optiland.materials.ideal import IdealMaterial
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.components.geometry.analytic.conic import ConicGeometry
from optiland.nonsequential.components.reflective import ReflectiveComponent

from ray_tracing_shapely_optiland.revolve import (RevolvedProfileSheet, RevolvedProfileSolid,
                                                  build_revolved_faces, scene_placement,
                                                  surface_from_span)
from ray_tracing_shapely_optiland.scene_bridge import import_from_optiland

ray_tracing_shapely = pytest.importorskip("ray_tracing_shapely")
from ray_tracing_shapely.core.scene import Scene  # noqa: E402
from ray_tracing_shapely.core.scene_objs.light_source.single_ray import SingleRay  # noqa: E402
from ray_tracing_shapely.core.scene_objs.mirror.mirror import Mirror  # noqa: E402
from ray_tracing_shapely.core.simulator import Simulator  # noqa: E402
from ray_tracing_shapely.optical_elements.mirrors import ArcMirror, ConicMirror  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_sheet_profile  # noqa: E402

#: The revolve axis in the 2D scene: through VERTEX, along U (tilted 10 deg). Well away
#: from the world origin: near it, Optiland's self-intersection threshold is smaller
#: than the rounding a ray carries there, and a conic mirror re-hits itself at the
#: point it just reflected from (KW-OPTILAND-05, witnessed below).
VERTEX = (120.0, 45.0)
U = (math.cos(math.radians(10.0)), math.sin(math.radians(10.0)))
LEFT = (-U[1], U[0])
BACK = (-U[0], -U[1])          # every conic opens towards the light, which travels along +U


def _at(s, z):
    return (VERTEX[0] + s * LEFT[0] + z * U[0], VERTEX[1] + s * LEFT[1] + z * U[1])


def _flat(scene, p1, p2, name):
    m = Mirror(scene, {"p1": {"x": p1[0], "y": p1[1]}, "p2": {"x": p2[0], "y": p2[1]}})
    m.name = name
    return m


def _mirrors(case, scene):
    if case == "paraboloid":
        return [ConicMirror(scene, VERTEX, BACK, 40.0, -1.0, semi_diameter=15.0, name="M")]
    if case == "ellipsoid":
        return [ConicMirror(scene, VERTEX, BACK, 40.0, -0.5, semi_diameter=15.0, name="M")]
    if case == "hyperboloid":
        return [ConicMirror(scene, VERTEX, BACK, 40.0, -3.0, semi_diameter=15.0, name="M")]
    if case == "sphere":
        return [ArcMirror(scene, VERTEX, BACK, 40.0, semi_diameter=15.0, name="M")]
    if case == "cone":   # narrow end (radius 3) at the vertex, opening to radius 12 towards the light
        return [_flat(scene, _at(3.0, 0.0), _at(12.0, -30.0), "M"),
                _flat(scene, _at(-3.0, 0.0), _at(-12.0, -30.0), "M_twin")]
    raise KeyError(case)


AXIS = {"kind": "revolve", "axis": {"point": list(VERTEX), "direction": list(U)}}


def _profile(case):
    return export_sheet_profile(_mirrors(case, Scene()), symmetry=AXIS, physical_params={},
                                numerical_params={})


def _trace_2d(case, h):
    """(end point on its detector plane, unit direction, number of reflections) of the 2D ray
    entering at height h along +U. The detector planes are z = -100 and z = +100."""
    scene = Scene()
    for m in _mirrors(case, scene):
        scene.add_object(m)
    p = _at(h, -60.0)
    scene.add_object(SingleRay(scene, {"p1": {"x": p[0], "y": p[1]}, "p2": {"x": p[0] + U[0], "y": p[1] + U[1]}}))
    segs = Simulator(scene, max_rays=200).run_with_result(name="2d").segments
    last = segs[-1]
    a = np.array([last.p1["x"], last.p1["y"]])
    d = np.array([last.p2["x"], last.p2["y"]]) - a
    d /= np.linalg.norm(d)
    z0, dz = (a - VERTEX) @ U, d @ U
    plane = 100.0 if dz > 0 else -100.0
    return a + (plane - z0) / dz * d, d, len(segs) - 1


def _world(p2d):
    return np.array([p2d[0], 0.0, p2d[1]])


def _pencil_cs(p2d, d2d):
    """A source at the 2D point, along the 2D direction, in the (a, 0, b) world."""
    return CoordinateSystem(x=p2d[0], y=0.0, z=p2d[1], ry=math.atan2(d2d[0], d2d[1]))


def _detectors(scene, placement_axis_point, along):
    """Two planes perpendicular to the axis, at z = -100 and z = +100 along it (2D scene)."""
    for k, z in enumerate((-100.0, 100.0)):
        c = (VERTEX[0] + z * along[0], VERTEX[1] + z * along[1])
        scene.add_detector(f"D{k}", CoordinateSystem(x=c[0], y=0.0, z=c[1],
                                                     ry=math.atan2(along[0], along[1])),
                           IrradianceDetectorConfig(width=4000, height=4000, num_pixels_x=2, num_pixels_y=2))


HEIGHTS = (-13.0, -7.5, -1.0, 2.5, 9.0, 14.0)
CONE_HEIGHTS = (-11.0, -7.0, -4.0, 5.0, 8.5, 11.5)


@pytest.mark.parametrize("case", ["paraboloid", "ellipsoid", "hyperboloid", "sphere", "cone"])
def test_every_meridional_ray_ends_in_3d_where_it_ends_in_2d(case):
    profile = _profile(case)
    sheet = RevolvedProfileSheet("S", profile, 1.0, cs=scene_placement(profile))
    heights = CONE_HEIGHTS if case == "cone" else HEIGHTS
    scene = NSQScene()
    for k, h in enumerate(heights):
        scene.add_source(f"R{k}", _pencil_cs(_at(h, -60.0), U), CollimatedSourceConfig(
            Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=1e-9))
    scene.component_registry.add(sheet.name, sheet)
    _detectors(scene, VERTEX, U)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = scene.trace(num_rays=10 * len(heights), seed=2, max_depth=40,
                         backend=NumpyBackend(seed=2), record_paths=True).ray_paths["events"]
    ev = ev[np.argsort(ev["ray_id"], kind="stable")]
    _, starts = np.unique(ev["ray_id"], return_index=True)
    seen = set()
    reflections_seen = []
    for i, st in enumerate(starts):
        r = ev[st: starts[i + 1] if i + 1 < len(starts) else len(ev)]
        k = int(str(r["component_name"][0]).split("_")[1])
        end, d, n_reflect = _trace_2d(case, heights[k])
        assert r["component_name"][-1] == "", f"ray from height {heights[k]} did not reach a detector"
        assert sum(str(c).startswith("S.") for c in r["component_name"]) == n_reflect
        # 1e-7 over ~100 mm: the 1e-9 pencil width, grown by a cone's repeated reflections
        assert np.allclose([r["x"][-1], r["y"][-1], r["z"][-1]], _world(end), atol=1e-7)
        assert np.allclose([r["L"][-1], r["M"][-1], r["N"][-1]], _world(d), atol=1e-9)
        seen.add(k)
        reflections_seen.append(n_reflect)
    assert seen == set(range(len(heights)))
    if case == "cone":
        assert max(reflections_seen) >= 2          # the cone's rays bounce off both walls


def test_a_sheet_traces_identically_to_a_hand_built_optiland_mirror():
    """Unplaced (the axis frame is the world: the axis along z through the
    profile's axis point), the ellipsoid is Optiland's ConicGeometry with
    R = -40 (its centre is towards -z, against the light), K = -0.5, at the vertex."""
    scene2d = Scene()
    profile = export_sheet_profile(
        [ConicMirror(scene2d, (205.0, 0.0), (-1.0, 0.0), 40.0, -0.5, semi_diameter=15.0, name="M")],
        symmetry={"kind": "revolve", "axis": {"point": [0.0, 0.0], "direction": [1.0, 0.0]}},
        physical_params={}, numerical_params={})
    ours = RevolvedProfileSheet("S", profile, 0.9)
    theirs = ReflectiveComponent(CoordinateSystem(z=205.0), ConicGeometry(-40.0, -0.5, 15.0),
                                 reflectance=0.9, name="S.M")
    logs = []
    for component in (ours, theirs):
        scene = NSQScene()
        scene.add_source("S", CoordinateSystem(z=150.0), CollimatedSourceConfig(
            Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=14.0))
        if isinstance(component, RevolvedProfileSheet):
            scene.component_registry.add("S", component)
        else:
            scene.add_component("S", component)
        scene.add_detector("D", CoordinateSystem(z=100.0), IrradianceDetectorConfig(
            width=400, height=400, num_pixels_x=2, num_pixels_y=2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = scene.trace(num_rays=2000, seed=4, max_depth=10, backend=NumpyBackend(seed=4),
                             record_paths=True).ray_paths["events"]
        logs.append(ev[np.argsort(ev["ray_id"], kind="stable")])
    a, b = logs
    assert len(a) == len(b) > 4000
    assert np.array_equal(a["event_type"], b["event_type"])
    assert np.array_equal(a["component_name"], b["component_name"])
    for k in ("x", "y", "z", "L", "M", "N", "flux"):
        assert np.allclose(a[k], b[k], atol=1e-12), k


def test_every_ray_of_a_full_beam_meets_the_paraboloids_focus():
    """The skew rays too, which no 2D trace has: each reflected ray passes
    through the focus, R/2 from the vertex along the conic's axis."""
    profile = _profile("paraboloid")
    sheet = RevolvedProfileSheet("S", profile, 1.0, cs=scene_placement(profile))
    focus = sheet.to_world(_at(0.0, -20.0))
    scene = NSQScene()
    p = _at(0.0, -60.0)
    scene.add_source("B", _pencil_cs(p, U), CollimatedSourceConfig(
        Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=14.5))
    scene.component_registry.add(sheet.name, sheet)
    _detectors(scene, VERTEX, U)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = scene.trace(num_rays=3000, seed=6, max_depth=10, backend=NumpyBackend(seed=6),
                         record_paths=True).ray_paths["events"]
    final = ev[(ev["component_name"] == "") & (ev["event_type"] == "hit")]
    assert len(final) > 2500
    pts = np.c_[final["x"], final["y"], final["z"]]
    dirs = np.c_[final["L"], final["M"], final["N"]]
    off = np.linalg.norm(np.cross(focus - pts, dirs), axis=1)
    assert np.abs(final["y"]).max() > 5.0          # far out of the profile plane: skew rays
    assert off.max() < 1e-9


def test_a_sheet_reads_back_to_its_profile_and_a_reshaped_face_reads_reshaped():
    profile = _profile("ellipsoid")
    sheet = RevolvedProfileSheet("S", profile, 1.0, cs=scene_placement(profile))
    back = import_from_optiland(sheet)
    assert back["body"] == "sheet"
    assert back["spans"][0]["curve"]["type"] == "conic"
    assert back["spans"][0]["curve"]["conic"] == pytest.approx(-0.5)
    assert np.allclose(back["points"], profile["points"], atol=1e-12)
    assert np.allclose(back["normals"], profile["normals"], atol=1e-12)
    sheet.face("M").geometry.conic = -1.0               # what an optimiser would do
    assert import_from_optiland(sheet)["spans"][0]["curve"]["conic"] == pytest.approx(-1.0)


def test_what_a_sheet_refuses():
    profile = _profile("paraboloid")
    glass = NSQMaterial(IdealMaterial(n=1.5))
    with pytest.raises(ValueError, match="RevolvedProfileSheet"):
        RevolvedProfileSolid("X", profile, glass)
    with pytest.raises(ValueError, match="encloses nothing"):
        build_revolved_faces(profile, glass, reflectances={"M": 1.0})
    with pytest.raises(ValueError, match="no reflectance"):
        RevolvedProfileSheet("X", profile, {"other": 1.0})
    not_a_mirror = copy.deepcopy(profile)
    not_a_mirror["spans"][0]["mechanism"] = "refract"
    with pytest.raises(ValueError, match="must reflect"):
        RevolvedProfileSheet("X", not_a_mirror, 1.0)
    # A conic piece that does not reach the axis (a paraboloid with a hole) is a zone:
    # a sheet accepts it since step 8a (test_revolve_cut_sheet.py), a solid does not.
    scene = Scene()
    ring = [ConicMirror(scene, VERTEX, BACK, 40.0, -1.0, extent=(3.0, 12.0), name="R"),
            ConicMirror(scene, VERTEX, BACK, 40.0, -1.0, extent=(-12.0, -3.0), name="R_twin")]
    holed = export_sheet_profile(ring, symmetry=AXIS, physical_params={}, numerical_params={})
    zone = RevolvedProfileSheet("X", holed, 1.0).face(holed["spans"][0]["short_label"])
    assert zone.geometry.inner_radius == pytest.approx(3.0)
    with pytest.raises(NotImplementedError, match="hole"):
        surface_from_span(holed["spans"][0], holed, allow_zone=False)
    # A conic whose axis is not the revolve axis (hand-edited: the export refuses it first).
    tilted = copy.deepcopy(profile)
    tilted["spans"][0]["curve"]["axis"] = [math.cos(0.3), math.sin(0.3)]
    with pytest.raises(NotImplementedError, match="not a conic of revolution"):
        RevolvedProfileSheet("X", tilted, 1.0)


def test_witness_optilands_conic_mirror_re_hits_itself_near_the_world_origin():
    """KW-OPTILAND-05, on Optiland alone. A hyperboloid mirror with its vertex at
    the world origin: rays arriving from 50 mm away carry ~1e-14 of rounding,
    but the threshold for a new hit is 25 ulps of the hit point's own
    coordinates, ~5.6e-15 there. Many near-axis rays meet the mirror twice at
    the same point and pass through it. When this stops failing to find any,
    upstream has fixed it: close KW-OPTILAND-05."""
    scene = NSQScene()
    scene.add_source("S", CoordinateSystem(x=0.5, z=-50.0), CollimatedSourceConfig(
        Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=1e-6))
    scene.add_component("M", ReflectiveComponent(CoordinateSystem(), ConicGeometry(-40.0, -3.0, 15.0),
                                                  reflectance=1.0, name="M"))
    for k, z in enumerate((-100.0, 100.0)):
        scene.add_detector(f"D{k}", CoordinateSystem(z=z), IrradianceDetectorConfig(
            width=400, height=400, num_pixels_x=2, num_pixels_y=2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = scene.trace(num_rays=1000, seed=1, max_depth=10, backend=NumpyBackend(seed=1),
                         record_paths=True).ray_paths["events"]
    _, hits = np.unique(ev["ray_id"][ev["component_name"] == "M"], return_counts=True)
    assert (hits > 1).sum() > 100


def test_witness_a_long_path_lets_a_ray_re_hit_optilands_frustum_far_from_the_origin():
    """KW-OPTILAND-05's second form, on Optiland alone: a cone mirror (a
    CylindricalFrustumGeometry) 300 mm from the origin, lit by a beam 88 deg off
    its axis that starts ~570 mm upstream. Each hit point carries rounding from
    that long path, more than 25 ulps of its own coordinates, and many rays meet
    the wall again at the point they just left: their two reflections cancel and
    they pass through. When this stops finding any, upstream has fixed it."""
    from optiland.nonsequential.components.geometry.analytic.frustum import CylindricalFrustumGeometry

    axis = CoordinateSystem(x=300.0, y=0.0, z=50.0, ry=math.pi / 2)       # the cone's axis along world x
    wall = ReflectiveComponent(CoordinateSystem(reference_cs=axis), CylindricalFrustumGeometry(20.0, 8.0, 0.0, 60.0),
                               reflectance=1.0, name="cone")
    th = math.radians(88.0)
    d = np.array([math.cos(th), 0.0, math.sin(th)])
    start = np.array([300.0, 0.0, 50.0]) - (5.0 + 20.0 * math.tan(th)) * d
    scene = NSQScene()
    scene.add_source("B", CoordinateSystem(x=start[0], y=start[1], z=start[2], ry=math.atan2(d[0], d[2])),
                     CollimatedSourceConfig(Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=20.0))
    scene.add_component("cone", wall)
    for k, (n, ry, rx) in enumerate((((0, 0, 1), 0.0, 0.0), ((0, 0, -1), math.pi, 0.0),
                                     ((1, 0, 0), math.pi / 2, 0.0), ((-1, 0, 0), -math.pi / 2, 0.0),
                                     ((0, 1, 0), 0.0, -math.pi / 2), ((0, -1, 0), 0.0, math.pi / 2))):
        p = np.array([300.0, 0.0, 50.0]) + 3000.0 * np.array(n, float)
        scene.add_detector(f"D{k}", CoordinateSystem(x=p[0], y=p[1], z=p[2], rx=rx, ry=ry),
                           IrradianceDetectorConfig(width=12000, height=12000, num_pixels_x=2, num_pixels_y=2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = scene.trace(num_rays=3000, seed=1, max_depth=50, backend=NumpyBackend(seed=1),
                         record_paths=True).ray_paths["events"]
    ev = ev[np.argsort(ev["ray_id"], kind="stable")]
    same = (ev["component_name"][1:] == "cone") & (ev["component_name"][:-1] == "cone") \
        & (ev["ray_id"][1:] == ev["ray_id"][:-1])
    gap = np.sqrt(np.diff(ev["x"]) ** 2 + np.diff(ev["y"]) ** 2 + np.diff(ev["z"]) ** 2)
    assert (same & (gap < 1e-6)).sum() > 100


# --- drawing --------------------------------------------------------------------------------

def _drawn_2d(sheet, projection):
    from optiland.nonsequential.visualization.viewer_2d import NSQViewer2D

    from ray_tracing_shapely_optiland.visualization.renderers import RevolvedProfileSheetRenderer2D

    scene = NSQScene()
    scene.component_registry.add(sheet.name, sheet)
    viewer = NSQViewer2D(scene)
    viewer.register_renderer(RevolvedProfileSheet, RevolvedProfileSheetRenderer2D())
    _, ax = viewer.view(num_rays=0, projection=projection)
    return [np.asarray(line.get_xydata()) for line in ax.lines], ax


@pytest.mark.parametrize("case", ["paraboloid", "cone"])
def test_the_2d_sheet_renderer_draws_the_2d_mirrors_unfilled(case):
    """In XZ under scene_placement: every drawn point is on the 2D mirrors
    (the exported spans and their mirror images), every such point is
    drawn, and nothing is filled."""
    profile = _profile(case)
    sheet = RevolvedProfileSheet("S", profile, 1.0, cs=scene_placement(profile))
    lines, ax = _drawn_2d(sheet, "XZ")
    assert len(lines) == 2 * len(profile["spans"]) and not ax.patches
    a = np.array(VERTEX)
    expected = []
    for p in profile["points"]:
        p = np.array(p)
        along = (p - a) @ np.array(U)
        expected += [p, a + 2 * along * np.array(U) - (p - a)]       # and its mirror image
    expected = np.array(expected)
    drawn = np.vstack(lines)[:, ::-1]   # XZ: horizontal world z (the scene's b), vertical world x (a)
    gap = np.linalg.norm(drawn[:, None, :] - expected[None, :, :], axis=2)
    assert gap.min(axis=1).max() < 1e-9 and gap.min(axis=0).max() < 1e-9
    assert sorted(t.get_text() for t in ax.texts) == ["M"]


def test_the_2d_sheet_renderer_looking_down_the_axis_draws_each_rim():
    profile = _profile("paraboloid")
    sheet = RevolvedProfileSheet("S", profile, 1.0, cs=CoordinateSystem(x=1.0, y=2.0))
    (line,), _ = _drawn_2d(sheet, "XY")
    assert np.hypot(line[:, 0] - 1.0, line[:, 1] - 2.0) == pytest.approx(15.0)


@pytest.mark.parametrize("case", ["ellipsoid", "cone"])
def test_the_3d_sheet_renderer_puts_every_point_on_the_placed_mirror(case):
    vtk = pytest.importorskip("vtk")
    from optiland.nonsequential.components.base import _get_transform

    from ray_tracing_shapely_optiland.visualization.renderers import RevolvedProfileSheetRenderer3D

    profile = _profile(case)
    sheet = RevolvedProfileSheet("S", profile, 1.0, cs=scene_placement(profile))
    renderer = vtk.vtkRenderer()
    RevolvedProfileSheetRenderer3D().render(sheet, renderer)
    actors = [renderer.GetActors().GetItemAsObject(i) for i in range(renderer.GetActors().GetNumberOfItems())]
    assert len(actors) == len(profile["spans"])
    t, rot = _get_transform(sheet.coordinate_system)
    half = np.array([sheet.frame.to_world(tuple(p)) for p in profile["points"]])[:, [0, 2]]
    for actor in actors:
        actor.GetMapper().Update()
        data = actor.GetMapper().GetInput()
        world = np.array([data.GetPoint(i) for i in range(data.GetNumberOfPoints())])
        local = (world - t) @ rot
        rz = np.column_stack([np.hypot(local[:, 0], local[:, 1]), local[:, 2]])
        assert np.linalg.norm(rz[:, None, :] - half[None, :, :], axis=2).min(axis=1).max() < 1e-9
