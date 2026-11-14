"""
Python-specific module: test revolve placement
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

Placing RevolvedProfileSolids, and drawing them.

The fixture is a two-lens system -- a biconvex lens, then a meniscus 25
downstream, decentred and tilted -- because one lens on the world axis
cannot tell a placement that works from one that is ignored. Three oracles:

- Optiland's own Lens, placed by the same CoordinateSystem: the pair must
  trace ray by ray identically, with the second lens tilted out of the plane;
- the 2D trace of the same scene: with the second lens decentred and tilted
  in the plane, rays in that plane stay meridional through both lenses, so
  2D and 3D must agree on every exit ray;
- the 2D outline itself: the renderers must draw the solids where the 2D
  scene has them (2D renderer) and where the placement puts them (3D).
"""

import math
import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

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
from optiland.nonsequential.components.lens import Lens
from optiland.nonsequential.visualization.viewer_2d import NSQViewer2D

from ray_tracing_shapely_optiland.revolve import RevolvedProfileSolid, scene_placement
from ray_tracing_shapely_optiland.scene_bridge import import_from_optiland, interior_index
from ray_tracing_shapely_optiland.visualization.renderers import (
    RevolvedProfileRenderer2D,
    RevolvedProfileRenderer3D,
    revolved_half_section,
)

ray_tracing_shapely = pytest.importorskip("ray_tracing_shapely")
from ray_tracing_shapely.core.scene import Scene  # noqa: E402
from ray_tracing_shapely.core.scene_objs.light_source.single_ray import SingleRay  # noqa: E402
from ray_tracing_shapely.core.simulator import Simulator  # noqa: E402
from ray_tracing_shapely.optical_elements.lenses import SingletLens  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile  # noqa: E402

N = 1.5
GLASS = NSQMaterial(IdealMaterial(n=N))
L1 = dict(radius_front=60.0, radius_back=-20.0, thickness=5.0, semi_diameter=10.0)
L2 = dict(radius_front=20.0, radius_back=40.0, thickness=5.0, semi_diameter=8.0)
#: In the 2D scene: L2's front vertex 25 downstream of L1's, 0.5 off its axis, tilted 2 deg.
L2_IN_PLANE = dict(position=(25.0, 0.5), rotation=2.0)
#: In 3D: L2 decentred and tilted out of the profile plane too (radians).
L2_CS = dict(x=0.4, y=-0.3, z=25.0, rx=0.03, ry=-0.02, rz=0.0)


def _profile(lens):
    return export_profile(lens, symmetry={"kind": "revolve", "axis": lens.revolve_axis()},
                          physical_params={}, numerical_params={})


def _two_lenses_2d(scene):
    l1 = SingletLens(scene, n=N, **L1)
    l2 = SingletLens(scene, n=N, **L2, **L2_IN_PLANE)
    return l1, l2


def _two_solids_from_the_2d_scene():
    l1, l2 = _two_lenses_2d(Scene())
    p1, p2 = _profile(l1), _profile(l2)
    return (RevolvedProfileSolid("L1", p1, GLASS, cs=scene_placement(p1)),
            RevolvedProfileSolid("L2", p2, GLASS, cs=scene_placement(p2)))


def _mirror(p, profile):
    """A 2D point mirrored about the profile's revolve axis."""
    a = profile["symmetry"]["axis"]
    P, d = np.array(a["point"], float), np.array(a["direction"], float)
    d /= np.linalg.norm(d)
    v = np.asarray(p, float) - P
    return P + 2.0 * (v @ d) * d - v


# --- placement --------------------------------------------------------------------------

@pytest.mark.parametrize("rotation", [0.0, 2.0, 30.0, 90.0, 180.0, -135.0])
def test_scene_placement_puts_every_point_where_the_2d_scene_has_it(rotation):
    """2D (a, b) -> world (a, 0, b), the extruded bridge's mapping. Rotation 0
    puts the axis along world x, Euler's gimbal lock for Optiland's angles."""
    lens = SingletLens(Scene(), n=N, **L2, position=(3.0, -2.0), rotation=rotation)
    profile = _profile(lens)
    solid = RevolvedProfileSolid("L", profile, GLASS, cs=scene_placement(profile))
    for p in profile["points"]:
        assert solid.to_world(tuple(p)) == pytest.approx([p[0], 0.0, p[1]], abs=1e-12)


def test_faces_are_named_after_their_solid_and_looked_up_by_label():
    s1, s2 = _two_solids_from_the_2d_scene()
    assert sorted(f.name for f in s2.surfaces) == ["L2.RIM", "L2.S1", "L2.S2"]
    assert s2.face("S1").name == "L2.S1" and s2.label_of(s2.face("S1")) == "S1"
    assert interior_index(s2) == pytest.approx(N)


def _trace(components, sources, num_rays):
    scene = NSQScene()
    for name, (cs, radius) in sources.items():
        scene.add_source(name, cs, CollimatedSourceConfig(
            Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=radius))
    for c in components:
        scene.component_registry.add(c.name, c)
    return scene


def test_a_placed_pair_traces_identically_to_two_placed_optiland_lenses():
    l1, l2 = SingletLens(Scene(), n=N, **L1), SingletLens(Scene(), n=N, **L2)
    ours = [RevolvedProfileSolid("L1", _profile(l1), GLASS),
            RevolvedProfileSolid("L2", _profile(l2), GLASS, cs=CoordinateSystem(**L2_CS))]
    oracle = [Lens("L1", CoordinateSystem(), LensConfig(r1=60.0, r2=-20.0, thickness=5.0,
                                                         material=GLASS, front_aperture_radius=10.0)),
              Lens("L2", CoordinateSystem(**L2_CS), LensConfig(r1=20.0, r2=40.0, thickness=5.0,
                                                               material=GLASS, front_aperture_radius=8.0))]
    logs = []
    for components in (ours, oracle):
        scene = _trace(components, {"S": (CoordinateSystem(z=-10.0), 9.5)}, 3000)
        scene.add_detector("D", CoordinateSystem(z=120.0), IrradianceDetectorConfig(
            width=300, height=300, num_pixels_x=4, num_pixels_y=4))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = scene.trace(num_rays=3000, seed=3, max_depth=12, backend=NumpyBackend(seed=3),
                             record_paths=True).ray_paths["events"]
        logs.append(ev[np.argsort(ev["ray_id"], kind="stable")])
    a, b = logs
    assert len(a) == len(b) > 9000
    assert np.array_equal(a["event_type"], b["event_type"])
    for k in ("x", "y", "z", "L", "M", "N"):
        assert np.allclose(a[k], b[k], atol=1e-9), k
    # The second lens was reached, through its tilted faces, by most rays.
    assert np.count_nonzero(a["component_name"] == "L2.S2") > 1500


HEIGHTS = (-6.0, -3.0, 1.5, 4.0, 6.5)
X_CHECK = 60.0  # where the exit rays are compared, in the 2D scene's a


def _exit_height_2d(h):
    """Where the 2D ray entering at height h crosses a = X_CHECK after leaving L2."""
    scene = Scene()
    l1, l2 = _two_lenses_2d(scene)
    for obj in (l1, l2):
        scene.add_object(obj)
    scene.add_object(SingleRay(scene, {"p1": {"x": -15, "y": h}, "p2": {"x": -14, "y": h}}))
    s2 = next(s for s in _profile(l2)["spans"] if s["short_label"] == "S2")["curve"]
    centre, radius = np.array(s2["center"]), s2["radius"]
    best = None
    for seg in Simulator(scene, max_rays=200).run_with_result(name="2d").segments:
        a1 = np.array([seg.p1["x"], seg.p1["y"]])
        a2 = np.array([seg.p2["x"], seg.p2["y"]])
        if abs(np.linalg.norm(a1 - centre) - radius) < 1e-7 and a2[0] > a1[0]:
            if best is None or seg.total_brightness > best[0]:  # the primary, not a ghost
                d = a2 - a1
                best = (seg.total_brightness, a1[1] + (X_CHECK - a1[0]) * d[1] / d[0])
    assert best is not None, f"no 2D ray leaves L2 for h = {h}"
    return best[1]


def test_2d_and_3d_agree_through_a_decentred_tilted_second_lens():
    s1, s2 = _two_solids_from_the_2d_scene()
    # One near-pencil (aperture 1e-9) per height, along the 2D scene's +a: world +x.
    sources = {f"S{k}": (s1.source_cs(-15.0, x=h), 1e-9) for k, h in enumerate(HEIGHTS)}
    scene = _trace([s1, s2], sources, 100)
    scene.add_detector("D", CoordinateSystem(x=100.0, ry=math.pi / 2), IrradianceDetectorConfig(
        width=100, height=100, num_pixels_x=4, num_pixels_y=4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = scene.trace(num_rays=100, seed=3, max_depth=12, backend=NumpyBackend(seed=3),
                         record_paths=True).ray_paths["events"]
    expected = {h: _exit_height_2d(h) for h in HEIGHTS}
    seen = set()
    for rid in np.unique(ev["ray_id"]):
        e = ev[ev["ray_id"] == rid]
        hits = [c for t, c in zip(e["event_type"], e["component_name"]) if t == "hit"]
        if hits != ["L1.S1", "L1.S2", "L2.S1", "L2.S2", ""]:
            continue  # a Fresnel reflection somewhere: not the primary path
        assert np.abs(e["y"]).max() < 1e-8 and np.abs(e["M"]).max() < 1e-8  # stays in the plane
        k = 4  # the L2.S2 hit; the next event carries the exit direction
        p = np.array([e["x"][k], e["z"][k]])
        d = np.array([e["L"][k + 1], e["N"][k + 1]])
        h = min(HEIGHTS, key=lambda hh: abs(hh - e["z"][0]))
        assert p[1] + (X_CHECK - p[0]) * d[1] / d[0] == pytest.approx(expected[h], abs=1e-7)
        seen.add(h)
    assert seen == set(HEIGHTS)


def test_import_reads_a_placed_solid_in_its_own_coordinates():
    _, s2 = _two_solids_from_the_2d_scene()
    back = import_from_optiland(s2)
    assert np.allclose(back["points"], s2.profile["points"], atol=1e-12)
    assert np.allclose(back["normals"], s2.profile["normals"], atol=1e-12)
    # A face moved along its own solid's axis comes back moved along that axis.
    s2.face("S2").cs.z = s2.face("S2").cs.z + 0.5
    moved = import_from_optiland(s2)
    span = next(s for s in moved["spans"] if s["short_label"] == "S2")
    centre_before = next(s for s in back["spans"] if s["short_label"] == "S2")["curve"]["center"]
    step = np.subtract(span["curve"]["center"], centre_before)
    axis = s2.profile["symmetry"]["axis"]["direction"]
    assert step == pytest.approx(0.5 * np.array(axis) / np.linalg.norm(axis), abs=1e-12)


# --- the 2D renderer --------------------------------------------------------------------

def _drawn_outlines_2d(solids, projection):
    scene = NSQScene()
    for s in solids:
        scene.component_registry.add(s.name, s)
    viewer = NSQViewer2D(scene)
    viewer.register_renderer(RevolvedProfileSolid, RevolvedProfileRenderer2D())
    _, ax = viewer.view(num_rays=0, projection=projection)
    return [np.asarray(line.get_xydata()) for line in ax.lines], ax


def test_the_2d_renderer_draws_both_lenses_where_the_2d_scene_has_them():
    s1, s2 = _two_solids_from_the_2d_scene()
    lines, ax = _drawn_outlines_2d([s1, s2], "XZ")
    assert len(lines) == 2
    for solid, line in zip((s1, s2), lines):
        profile = solid.profile
        expected = np.array([p for p in profile["points"]]
                            + [_mirror(p, profile) for p in profile["points"]])
        drawn = line[:, ::-1]  # XZ: horizontal world z (the scene's b), vertical world x (a)
        gap = np.linalg.norm(drawn[:, None, :] - expected[None, :, :], axis=2)
        assert gap.min(axis=1).max() < 1e-9   # everything drawn is on the outline...
        assert gap.min(axis=0).max() < 1e-9   # ...and the whole outline is drawn
    labels = sorted(t.get_text() for t in ax.texts)
    assert labels == ["RIM", "RIM", "S1", "S1", "S2", "S2"]


def test_the_2d_renderer_looking_down_the_axis_draws_the_rim_circle():
    l1 = SingletLens(Scene(), n=N, **L1)
    solid = RevolvedProfileSolid("L1", _profile(l1), GLASS, cs=CoordinateSystem(x=1.0, y=2.0))
    (line,), _ = _drawn_outlines_2d([solid], "XY")
    assert np.hypot(line[:, 0] - 1.0, line[:, 1] - 2.0) == pytest.approx(10.0)


def test_the_renderers_draw_a_reshaped_face_as_it_is_traced():
    l1 = SingletLens(Scene(), n=N, **L1)
    solid = RevolvedProfileSolid("L1", _profile(l1), GLASS)
    solid.face("S1").geometry.radius = 80.0
    profile, local = revolved_half_section(solid)
    span = next(s for s in profile["spans"] if s["short_label"] == "S1")
    r, z = local[span["index_start"]:span["index_end"], 0], local[span["index_start"]:span["index_end"], 2]
    assert np.hypot(r, z - 80.0) == pytest.approx(80.0, abs=1e-9)


# --- the 3D renderer --------------------------------------------------------------------

def _actor_points(actor):
    actor.GetMapper().Update()
    data = actor.GetMapper().GetInput()
    return np.array([data.GetPoint(i) for i in range(data.GetNumberOfPoints())])


def test_the_3d_renderer_matches_optilands_lens_renderer_on_axis():
    vtk = pytest.importorskip("vtk")
    from optiland.nonsequential.visualization.renderers.lens import LensRenderer3D

    l1 = SingletLens(Scene(), n=N, **L1)
    ours, theirs = vtk.vtkRenderer(), vtk.vtkRenderer()
    RevolvedProfileRenderer3D().render(RevolvedProfileSolid("L1", _profile(l1), GLASS), ours)
    LensRenderer3D().render(Lens("L1", CoordinateSystem(), LensConfig(
        r1=60.0, r2=-20.0, thickness=5.0, material=GLASS, front_aperture_radius=10.0)), theirs)
    assert ours.ComputeVisiblePropBounds() == pytest.approx(theirs.ComputeVisiblePropBounds(), abs=1e-9)


def test_the_3d_renderer_puts_every_point_on_the_placed_solid():
    """Each drawn point, taken back into L2's frame by a rotation computed
    here (not by the renderer's code), sits on a vertex of the half-section
    revolved about the axis."""
    vtk = pytest.importorskip("vtk")
    l2 = SingletLens(Scene(), n=N, **L2)
    solid = RevolvedProfileSolid("L2", _profile(l2), GLASS, cs=CoordinateSystem(**L2_CS))
    renderer = vtk.vtkRenderer()
    RevolvedProfileRenderer3D().render(solid, renderer)
    (actor,) = [renderer.GetActors().GetItemAsObject(i) for i in range(renderer.GetActors().GetNumberOfItems())]
    world = _actor_points(actor)
    rot = Rotation.from_euler("xyz", [L2_CS["rx"], L2_CS["ry"], L2_CS["rz"]]).as_matrix()
    local = (world - [L2_CS["x"], L2_CS["y"], L2_CS["z"]]) @ rot
    rz = np.column_stack([np.hypot(local[:, 0], local[:, 1]), local[:, 2]])
    half = np.array([solid.frame.to_world(tuple(p)) for p in solid.profile["points"]])[:, [0, 2]]
    gap = np.linalg.norm(rz[:, None, :] - half[None, :, :], axis=2).min(axis=1)
    assert len(world) > 256 * len(half) * 0.9 and gap.max() < 1e-9


def test_the_3d_viewer_registry_draws_one_actor_per_solid():
    vtk = pytest.importorskip("vtk")
    from optiland.nonsequential.visualization.viewer_3d import NSQViewer3D
    from ray_tracing_shapely_optiland.visualization.jupyter_viewer import build_vtk_renderer

    scene = NSQScene()
    solids = _two_solids_from_the_2d_scene()
    for s in solids:
        scene.component_registry.add(s.name, s)
    viewer = NSQViewer3D(scene)
    viewer.register_renderer(RevolvedProfileSolid, RevolvedProfileRenderer3D())
    renderer = build_vtk_renderer(viewer, num_rays=0)
    assert isinstance(renderer, vtk.vtkRenderer)
    assert renderer.GetActors().GetNumberOfItems() == 2
    xmin, xmax, ymin, ymax, zmin, zmax = renderer.ComputeVisiblePropBounds()
    # Along the 2D scene's a (world x) and b (world z), the bounds are the 2D
    # outlines' (the extremes lie in the plane holding both axes); across it
    # (world y), L1's semi-diameter.
    outline = np.array([q for s in solids for p in s.profile["points"]
                        for q in (p, _mirror(p, s.profile))])
    assert (xmin, xmax) == pytest.approx((outline[:, 0].min(), outline[:, 0].max()), abs=1e-9)
    assert (zmin, zmax) == pytest.approx((outline[:, 1].min(), outline[:, 1].max()), abs=1e-9)
    assert (ymin, ymax) == pytest.approx((-10.0, 10.0), abs=1e-9)
