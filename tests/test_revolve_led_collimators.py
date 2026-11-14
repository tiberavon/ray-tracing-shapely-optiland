"""
Python-specific module: test_revolve_led_collimators
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
The LED collimators, revolved
===============================================================================
A straight-facet Fresnel lens with a conic cap revolves into cones, a disc and
a conic reaching the axis (step 2). The TIR collimator's designed faces, 22c
(a zone) and 12a, are splines, which revolve into SplineRevolutionGeometry
since step 3. A point source on the axis sends every ray in a plane through
the axis, so each 3D ray must leave at the angle the 2D trace gives for the
same launch angle.
===============================================================================
"""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.materials.ideal import IdealMaterial
from optiland.nonsequential import (IrradianceDetectorConfig, NSQMaterial, NSQScene, PointSourceConfig,
                                    Spectrum)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend

from ray_tracing_shapely_optiland.revolve import RevolvedProfileSolid, scene_placement

ray_tracing_shapely = pytest.importorskip("ray_tracing_shapely")
from ray_tracing_shapely.core.scene import Scene  # noqa: E402
from ray_tracing_shapely.core.scene_objs.light_source.single_ray import SingleRay  # noqa: E402
from ray_tracing_shapely.core.simulator import Simulator  # noqa: E402
from ray_tracing_shapely.optical_elements.lenses import FresnelLens, TIRCollimator  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile  # noqa: E402

N = 1.49
F = (100.0, 30.0)                 # away from the world origin (KW-OPTILAND-05)
LENS = dict(focal_distance=20.0, semi_diameter=15.0, zones=5, n=N)


def _profile(element):
    return export_profile(element, symmetry={"kind": "revolve", "axis": element.revolve_axis()},
                          physical_params={}, numerical_params={})


def _trace_3d(lens, half_angle_deg):
    """Per ray: (launch angle from the axis, exit angle from the axis, the faces it met), degrees."""
    profile = _profile(lens)
    solid = RevolvedProfileSolid("L", profile, NSQMaterial(IdealMaterial(n=N)), cs=scene_placement(profile))
    scene = NSQScene()
    scene.add_source("S", CoordinateSystem(x=F[0], y=0.0, z=F[1], ry=math.pi / 2),
                     PointSourceConfig(Spectrum.monochromatic(0.55), total_flux=1.0,
                                       half_angle_deg=half_angle_deg))
    scene.component_registry.add(solid.name, solid)
    scene.add_detector("D", CoordinateSystem(x=F[0] + 300.0, y=0.0, z=F[1], ry=math.pi / 2),
                       IrradianceDetectorConfig(width=4000, height=4000, num_pixels_x=2, num_pixels_y=2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = scene.trace(num_rays=2000, seed=3, max_depth=20, backend=NumpyBackend(seed=3),
                         record_paths=True).ray_paths["events"]
    ev = ev[np.argsort(ev["ray_id"], kind="stable")]
    _, st = np.unique(ev["ray_id"], return_index=True)
    en = np.r_[st[1:], len(ev)]
    out = []
    for s, e in zip(st, en):
        names = tuple(str(c) for c in ev["component_name"][s + 1:e])
        launch = math.degrees(math.acos(min(1.0, float(ev["L"][s]))))
        leave = math.degrees(math.acos(min(1.0, float(ev["L"][e - 1]))))
        out.append((launch, leave, names))
    return solid, out


def _exit_2d(make, launch_deg):
    """The 2D exit angle from the axis of the ray launched at ``launch_deg`` above it."""
    scene = Scene()
    lens = make(scene)
    scene.add_object(lens)
    t = math.radians(launch_deg)
    scene.add_object(SingleRay(scene, {"p1": {"x": F[0], "y": F[1]},
                                       "p2": {"x": F[0] + math.cos(t), "y": F[1] + math.sin(t)}}))
    segs = Simulator(scene, max_rays=2000).run_with_result(name="2d").segments
    leaving = [sg for sg in segs if sg.p1 != {"x": F[0], "y": F[1]}
               and lens.check_ray_intersects(SimpleNamespace(p1=sg.p1, p2=sg.p2)) is None]
    sg = max(leaving, key=lambda x: x.total_brightness)
    return abs(math.degrees(math.atan2(sg.p2["y"] - sg.p1["y"], sg.p2["x"] - sg.p1["x"])))


@pytest.mark.parametrize("kw", [dict(grooves="entrance"), dict(grooves="exit", cap="conic", riser_draft_deg=2.0)],
                         ids=["grooves-entrance", "grooves-exit"])
def test_a_revolved_fresnel_lens_bends_every_ray_as_its_2d_section_does(kw):
    def make(scene):
        return FresnelLens(scene, position=F, **LENS, **kw)

    solid, rays = _trace_3d(make(Scene()), 30.0)
    kinds = solid.surface_kinds
    assert kinds["CAP"] == "conic" and kinds["FLAT"] == "disc"
    assert all(kinds[f"{p}{k}"] == "frustum" for k in range(1, 5) for p in "FR")
    clean = [r for r in rays if len(r[2]) == 3 and r[2][-1] == ""
             and {r[2][0], r[2][1]} & {"L.CAP", "L.F1", "L.F2", "L.F3"}]
    assert len(clean) > 1200                    # cap and facets: most of the light (risers take the rest)
    cap = [r for r in clean if "L.CAP" in r[2]]
    assert cap and max(r[1] for r in cap) < (1e-7 if kw["grooves"] == "entrance" else 1e-3)
    by_face = {}
    for r in clean:
        by_face.setdefault(r[2], []).append(r)
    for path, group in by_face.items():         # a few rays of each path, against 2D
        group.sort()
        for launch, leave, _ in group[:: max(1, len(group) // 4)]:
            assert leave == pytest.approx(_exit_2d(make, launch), abs=1e-6), (path, launch)


@pytest.mark.parametrize("central", ["spline", "conic"])
def test_the_revolved_tir_collimator_collimates_both_channels_as_its_2d_section_does(central):
    def make(scene):
        return TIRCollimator(scene, position=F, central=central)

    design = make(Scene()).design
    solid, rays = _trace_3d(make(Scene()), 90.0)
    kinds = solid.surface_kinds
    assert kinds["22c"] == "spline" and kinds["12a"] == ("spline" if central == "spline" else "conic")
    central_rays = [r for r in rays if r[2] == ("L.12b", "L.12a", "")]
    ring_rays = [r for r in rays if r[2] == ("L.22b", "L.22c", "L.22a", "")]
    # every ray from F through a channel, at every azimuth, in the channel's angles
    assert len(central_rays) > 200 and len(ring_rays) > 400
    assert max(r[0] for r in central_rays) <= design["phi_corner"] + 1e-6
    assert min(r[0] for r in ring_rays) >= design["phi_end"] - 1e-6
    assert max(r[0] for r in ring_rays) <= design["phi_start"] + 1e-6
    assert max(r[1] for r in ring_rays) < 1e-4
    assert max(r[1] for r in central_rays) < (1e-4 if central == "spline" else 0.4)
    for group in (central_rays, ring_rays):     # a few rays of each, against 2D
        group.sort()
        for launch, leave, _ in group[:: max(1, len(group) // 5)]:
            assert leave == pytest.approx(_exit_2d(make, launch), abs=1e-6), launch


def test_a_revolved_tir_collimator_reads_back_to_its_profile_and_a_moved_face_moved():
    from optiland.coordinate_system import CoordinateSystem as CS

    from ray_tracing_shapely.optical_elements.prisms.base_prism import BasePrism
    from ray_tracing_shapely_optiland.revolve import import_revolved_from_optiland

    profile = _profile(TIRCollimator(Scene(), position=F))
    solid = RevolvedProfileSolid("T", profile, NSQMaterial(IdealMaterial(n=N)), cs=scene_placement(profile))
    back = import_revolved_from_optiland(solid)
    assert np.abs(np.array(back["points"]) - np.array(profile["points"])).max() < 1e-12
    assert [sp.get("curve") for sp in back["spans"]] == [sp.get("curve") for sp in profile["spans"]]
    assert len(BasePrism.from_profile(Scene(), back, n=N).path) == 14
    face = solid.face("12a")
    face.cs = CS(z=face.cs.z + 0.5, reference_cs=face.cs.reference_cs)
    moved = import_revolved_from_optiland(solid)
    span = next(sp for sp in moved["spans"] if sp["short_label"] == "12a")
    k = span["index_start"]
    assert moved["points"][k][0] == pytest.approx(profile["points"][k][0] + 0.5)
    assert span["curve"]["points"][0][0] == pytest.approx(
        next(sp for sp in profile["spans"] if sp["short_label"] == "12a")["curve"]["points"][0][0] + 0.5)


def test_a_spline_sample_off_its_surface_or_a_tilted_normal_is_refused():
    # C-BRIDGE-08 for spline spans: the profile's samples and normals are checked
    # against the surface built from the descriptor.
    import copy

    profile = _profile(TIRCollimator(Scene(), position=F))
    span = next(sp for sp in profile["spans"] if sp["short_label"] == "22c")
    k = (span["index_start"] + span["index_end"]) // 2
    off = copy.deepcopy(profile)
    off["points"][k][1] += 1e-6
    with pytest.raises(ValueError, match="from the spline surface"):
        RevolvedProfileSolid("T", off, NSQMaterial(IdealMaterial(n=N)), cs=scene_placement(off))
    tilted = copy.deepcopy(profile)
    nx, ny = tilted["normals"][k]
    a = 1e-3
    tilted["normals"][k] = [nx * math.cos(a) - ny * math.sin(a), nx * math.sin(a) + ny * math.cos(a)]
    with pytest.raises(ValueError, match="stored normal"):
        RevolvedProfileSolid("T", tilted, NSQMaterial(IdealMaterial(n=N)), cs=scene_placement(tilted))
