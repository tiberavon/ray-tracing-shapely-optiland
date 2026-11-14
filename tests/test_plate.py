"""
Python-specific module: test_plate
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
PLATES -- a flat polygonal mirror or absorber standing on a 2D segment
(C-BRIDGE-13)
===============================================================================
A shade like a projector headlamp's: a plate perpendicular to the 2D plane,
standing on a 2D mirror segment, with a stepped (concave) front edge.

- Placement: the plate's trace in the 2D plane is the segment.
- 2D against 3D: rays in the 2D plane end where they end in 2D.
- Out of the plane: a ray meets the plate exactly where the outline says.
- Absorbing and partly reflecting plates; refusals; drawing.
===============================================================================
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (CollimatedSourceConfig, IrradianceDetectorConfig, NSQScene,
                                    Spectrum)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.components.absorbing import AbsorbingComponent
from optiland.nonsequential.components.reflective import ReflectiveComponent

from ray_tracing_shapely_optiland.plate import PlanarPlate, section_chord, segment_placement

ray_tracing_shapely = pytest.importorskip("ray_tracing_shapely")
from ray_tracing_shapely.core.scene import Scene  # noqa: E402
from ray_tracing_shapely.core.scene_objs.blocker.blocker import Blocker  # noqa: E402
from ray_tracing_shapely.core.scene_objs.light_source.single_ray import SingleRay  # noqa: E402
from ray_tracing_shapely.core.scene_objs.mirror.mirror import Mirror  # noqa: E402
from ray_tracing_shapely.core.simulator import Simulator  # noqa: E402

#: The segment, tilted 10 deg, well away from the world origin.
P1 = np.array([150.0, 40.0])
U = np.array([math.cos(math.radians(10.0)), math.sin(math.radians(10.0))])
N2 = np.array([-U[1], U[0]])                 # the segment's normal in the 2D plane
L = 21.0
P2 = P1 + L * U
#: A shade outline (u along the segment, v out of the 2D plane): a front edge
#: stepped back 3 mm for v < -2, the step a low-beam cut line's shade has.
OUTLINE = [(0.0, -12.0), (L - 3.0, -12.0), (L - 3.0, -2.0), (L, -2.0), (L, 12.0), (0.0, 12.0)]


def _segment_obj(scene, cls=Mirror):
    return cls(scene, {"p1": {"x": float(P1[0]), "y": float(P1[1])},
                       "p2": {"x": float(P2[0]), "y": float(P2[1])}})


def _plate(**kw):
    kw = kw or {"reflectance": 1.0}
    return PlanarPlate.from_segment("shade", _segment_obj(Scene()), OUTLINE, **kw)


def _world(p2d, y=0.0):
    return np.array([p2d[0], y, p2d[1]])


def _box(scene, centre, half=200.0):
    """Six detector planes around ``centre`` (2D point): every ray ends on one."""
    c3 = _world(centre)
    for k, (n, ry, rx) in enumerate((((0, 0, 1), 0.0, 0.0), ((0, 0, -1), math.pi, 0.0),
                                     ((1, 0, 0), math.pi / 2, 0.0), ((-1, 0, 0), -math.pi / 2, 0.0),
                                     ((0, 1, 0), 0.0, -math.pi / 2), ((0, -1, 0), 0.0, math.pi / 2))):
        p = c3 + half * np.array(n, float)
        scene.add_detector(f"D{k}", CoordinateSystem(x=p[0], y=p[1], z=p[2], rx=rx, ry=ry),
                           IrradianceDetectorConfig(width=4 * half, height=4 * half,
                                                    num_pixels_x=2, num_pixels_y=2))


def _pencil(scene, name, origin3, direction3):
    d = np.asarray(direction3, float) / np.linalg.norm(direction3)
    rx = -math.asin(d[1])
    ry = math.atan2(d[0], d[2])
    scene.add_source(name, CoordinateSystem(x=origin3[0], y=origin3[1], z=origin3[2], rx=rx, ry=ry),
                     CollimatedSourceConfig(Spectrum.monochromatic(0.55), total_flux=1.0,
                                            aperture_radius=1e-9))


def _trace(scene, n_sources, rays_each=4):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = scene.trace(num_rays=rays_each * n_sources, seed=3, max_depth=10,
                         backend=NumpyBackend(seed=3), record_paths=True).ray_paths["events"]
    ev = ev[np.argsort(ev["ray_id"], kind="stable")]
    _, starts = np.unique(ev["ray_id"], return_index=True)
    bounds = list(starts) + [len(ev)]
    return [ev[bounds[i]:bounds[i + 1]] for i in range(len(starts))]


def test_the_plate_stands_on_its_segment():
    plate = _plate()
    cs = plate.coordinate_system
    from optiland.nonsequential.components.base import _get_transform

    t, rot = _get_transform(cs)
    assert np.allclose(t + rot @ [0.0, 0.0, 0.0], _world(P1), atol=1e-12)
    assert np.allclose(t + rot @ [L, 0.0, 0.0], _world(P2), atol=1e-12)
    assert np.allclose(rot @ [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], atol=1e-15)     # v is world y
    assert np.allclose(rot @ [0.0, 0.0, 1.0], [N2[0], 0.0, N2[1]], atol=1e-15)  # normal in plane
    assert plate.segment == (tuple(P1), tuple(P2))
    assert isinstance(plate.face, ReflectiveComponent) and plate.face.name == "shade.plate"
    assert section_chord(OUTLINE) == pytest.approx((0.0, L))
    assert segment_placement(tuple(P1), tuple(P2)).ry == pytest.approx(cs.ry)


def _trace_2d(origin, d):
    scene = Scene()
    scene.add_object(_segment_obj(scene))
    scene.add_object(SingleRay(scene, {"p1": {"x": origin[0], "y": origin[1]},
                                       "p2": {"x": origin[0] + d[0], "y": origin[1] + d[1]}}))
    segs = Simulator(scene, max_rays=50).run_with_result(name="2d").segments
    last = segs[-1]
    a = np.array([last.p1["x"], last.p1["y"]])
    e = np.array([last.p2["x"], last.p2["y"]]) - a
    return a, e / np.linalg.norm(e), len(segs) - 1


#: (start, direction) in 2D: onto the segment from both sides at several angles, and two
#: that pass just beyond its ends.
IN_PLANE = [(P1 + 5 * U + 30 * N2, -N2), (P1 + 2 * U + 20 * N2, -(N2 + 0.4 * U)),
            (P1 + 19 * U + 15 * N2, -(N2 - 0.7 * U)), (P1 + 11 * U - 25 * N2, N2 + 0.2 * U),
            (P1 + (L + 0.5) * U + 10 * N2, -N2), (P1 - 0.5 * U - 10 * N2, N2)]


def test_every_ray_in_the_2d_plane_ends_in_3d_where_it_ends_in_2d():
    plate = _plate()
    scene = NSQScene()
    for k, (o, d) in enumerate(IN_PLANE):
        _pencil(scene, f"P{k}", _world(o), _world(d))
    scene.component_registry.add(plate.name, plate)
    _box(scene, P1 + 0.5 * L * U)
    reflected = 0
    for r in _trace(scene, len(IN_PLANE)):
        k = int(str(r["component_name"][0]).split("_")[1])
        a, e, n_reflect = _trace_2d(*IN_PLANE[k])
        assert r["component_name"][-1] == ""
        assert sum(str(c) == "shade.plate" for c in r["component_name"]) == n_reflect
        reflected += n_reflect
        end = np.array([r["x"][-1], r["z"][-1]])
        off = end - a
        assert abs(off[0] * e[1] - off[1] * e[0]) < 1e-7 and off @ e > 0.0
        assert abs(r["y"][-1]) < 1e-7
        assert np.allclose([r["L"][-1], r["N"][-1]], e, atol=1e-9)
    assert reflected >= 4


def test_out_of_the_plane_a_ray_meets_the_plate_exactly_where_the_outline_is():
    """Rays along the plate's normal, at a grid of (u, v) on and around the outline,
    the step's notch included: reflected back exactly when (u, v) is inside."""
    plate = _plate()
    grid = [(u, v) for u in (-1.0, 1.0, 10.0, L - 4.0, L - 2.0, L - 1.0, L + 1.0)
            for v in (-13.0, -11.0, -5.0, -1.0, 0.5, 6.0, 11.5, 13.0)]
    n3 = _world(N2)
    scene = NSQScene()
    for k, (u, v) in enumerate(grid):
        on_plate = _world(P1 + u * U, v)
        _pencil(scene, f"G{k}", on_plate + 20.0 * n3, -n3)
    scene.component_registry.add(plate.name, plate)
    _box(scene, P1 + 0.5 * L * U)
    outline = Polygon(OUTLINE)
    hits = {}
    for r in _trace(scene, len(grid), rays_each=2):
        k = int(str(r["component_name"][0]).split("_")[1])
        hit = "shade.plate" in [str(c) for c in r["component_name"]]
        hits.setdefault(k, hit)
        assert hits[k] == hit
        if hit:
            assert np.allclose([r["L"][-1], r["M"][-1], r["N"][-1]], n3, atol=1e-12)
    assert len(hits) == len(grid)
    for k, (u, v) in enumerate(grid):
        assert hits[k] == outline.contains(Point(u, v)), (u, v)
    assert any(hits.values()) and not all(hits.values())
    # the notch: in front of the step, below it, the ray passes
    assert not hits[grid.index((L - 2.0, -5.0))] and hits[grid.index((L - 2.0, 6.0))]


def test_an_absorbing_plate_from_a_2d_blocker_stops_light_and_a_mirror_keeps_its_reflectance():
    absorber = PlanarPlate.from_segment("board", _segment_obj(Scene(), Blocker), OUTLINE,
                                        absorbing=True)
    assert isinstance(absorber.face, AbsorbingComponent)
    for plate, expect in ((absorber, 0.0), (_plate(reflectance=0.8), 0.8)):
        scene = NSQScene()
        _pencil(scene, "S", _world(P1 + 5 * U + 30 * N2), _world(-N2))
        scene.component_registry.add(plate.name, plate)
        _box(scene, P1)
        (r, *_) = _trace(scene, 1, rays_each=1)
        assert str(r["component_name"][1]).endswith(".plate")
        detected = r["flux"][-1] if r["component_name"][-1] == "" else 0.0
        assert detected == pytest.approx(expect, abs=1e-12)


@pytest.mark.parametrize("outline, match", [
    ([(0.0, -5.0), (L - 1.0, -5.0), (L - 1.0, 5.0), (0.0, 5.0)], "2D segment"),     # short
    ([(-1.0, -5.0), (L, -5.0), (L, 5.0), (-1.0, 5.0)], "2D segment"),               # long
    ([(0.0, 1.0), (L, 1.0), (L, 5.0), (0.0, 5.0)], "nothing"),                      # off it
    ([(0.0, -5.0), (L, -5.0), (L, 5.0), (12.0, 5.0), (10.0, -1.0), (8.0, 5.0), (0.0, 5.0)],
     None),                                                                          # a V notch
    ([(0.0, -5.0), (L, 5.0), (L, -5.0), (0.0, 5.0)], "crosses itself"),
])
def test_a_plate_whose_section_is_not_its_segment_is_refused(outline, match):
    if match is None:   # a notch reaching below v = 0 cuts the chord in two
        with pytest.raises(ValueError, match="more than one piece"):
            PlanarPlate.from_segment("x", (tuple(P1), tuple(P2)), outline, reflectance=1.0)
        return
    with pytest.raises(ValueError, match=match):
        PlanarPlate.from_segment("x", (tuple(P1), tuple(P2)), outline, reflectance=1.0)


def test_what_a_plate_refuses():
    cs = CoordinateSystem(z=100.0)
    with pytest.raises(ValueError, match="exactly one"):
        PlanarPlate("x", OUTLINE, cs=cs)
    with pytest.raises(ValueError, match="exactly one"):
        PlanarPlate("x", OUTLINE, cs=cs, reflectance=1.0, absorbing=True)
    with pytest.raises(ValueError, match="non-zero length"):
        PlanarPlate.from_segment("x", (tuple(P1), tuple(P1)), OUTLINE, reflectance=1.0)
    with pytest.raises(ValueError, match="three"):
        PlanarPlate("x", [(0.0, 0.0), (1.0, 0.0)], cs=cs, reflectance=1.0)


# --- drawing --------------------------------------------------------------------------------

def test_the_2d_renderer_shows_the_plate_as_its_segment_in_the_2d_plane():
    from optiland.nonsequential.visualization.viewer_2d import NSQViewer2D

    from ray_tracing_shapely_optiland.visualization.renderers import PlanarPlateRenderer2D

    plate = _plate()
    scene = NSQScene()
    scene.component_registry.add(plate.name, plate)
    viewer = NSQViewer2D(scene)
    viewer.register_renderer(PlanarPlate, PlanarPlateRenderer2D())
    _, ax = viewer.view(num_rays=0, projection="XZ")
    (line,) = ax.lines
    drawn = np.asarray(line.get_xydata())[:, ::-1] - P1          # XZ: horizontal world z
    assert np.abs(drawn @ N2).max() < 1e-9                        # on the segment's line
    assert (drawn @ U).min() == pytest.approx(0.0, abs=1e-9)
    assert (drawn @ U).max() == pytest.approx(L, abs=1e-9)


def test_the_3d_renderer_draws_the_outline_at_its_place():
    vtk = pytest.importorskip("vtk")
    from ray_tracing_shapely_optiland.visualization.renderers import PlanarPlateRenderer3D

    plate = _plate()
    renderer = vtk.vtkRenderer()
    PlanarPlateRenderer3D().render(plate, renderer)
    (actor,) = [renderer.GetActors().GetItemAsObject(i)
                for i in range(renderer.GetActors().GetNumberOfItems())]
    actor.GetMapper().Update()
    data = actor.GetMapper().GetInput()
    pts = np.array([data.GetPoint(i) for i in range(data.GetNumberOfPoints())])
    assert np.allclose(pts, plate.outline_world(), atol=1e-12)
    area = 0.0
    for c in range(data.GetNumberOfCells()):
        ids = data.GetCell(c).GetPointIds()
        a, b, cc = (pts[ids.GetId(i)] for i in range(3))
        area += 0.5 * np.linalg.norm(np.cross(b - a, cc - a))
    assert area == pytest.approx(Polygon(OUTLINE).area, rel=1e-12)   # the notch is not filled
