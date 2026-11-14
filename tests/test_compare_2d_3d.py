"""
Python-specific module: test compare 2d 3d
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
The same meridional rays in 2D and in 3D, compared ray by ray, headless
===============================================================================
Three exports, each traced both ways: a revolved singlet (a solid with a
profile), a revolved V of two mirrors (a sheet whose profile holds both
walls), and the same V as two planar plates (components without a profile,
one face each). Each must agree ray by ray. A 3D lens that is not the 2D one
must be reported: by its exit directions when its faces are the same, by
its paths when a ray misses it.
===============================================================================
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from optiland.materials.ideal import IdealMaterial
from optiland.nonsequential import NSQMaterial

from ray_tracing_shapely.analysis.agentic_tools import clear_context
from ray_tracing_shapely.core.scene import Scene
from ray_tracing_shapely.core.scene_objs.mirror.mirror import Mirror
from ray_tracing_shapely.optical_elements.lenses import SingletLens
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile, export_sheet_profile
from ray_tracing_shapely_optiland import agentic
from ray_tracing_shapely_optiland.compare import compare_2d_3d, span_at
from ray_tracing_shapely_optiland.events import split_rays
from ray_tracing_shapely_optiland.plate import PlanarPlate
from ray_tracing_shapely_optiland.revolve import RevolvedProfileSheet, RevolvedProfileSolid, scene_placement

LENS = dict(radius_front=60.0, radius_back=-20.0, thickness=5.0, semi_diameter=10.0, n=1.5)
LENS_RAYS = [((-20.0, h), a) for h in (-8.0, -3.0, 0.5, 3.0, 6.0) for a in (0.0, 5.0, -10.0)]
IMAGE = {"point": [60.0, 0.0], "normal_deg": 0.0}


def _lens_2d(scene, **params):
    lens = SingletLens(scene, **{**LENS, **params})
    scene.add_object(lens)
    return lens


def _solid(name="L1", **params):
    lens = _lens_2d(Scene(), **params)
    profile = export_profile(lens, symmetry={"kind": "revolve", "axis": lens.revolve_axis()},
                             physical_params={}, numerical_params={})
    return RevolvedProfileSolid(name, profile, NSQMaterial(IdealMaterial(n=LENS["n"])), cs=scene_placement(profile))


# The V: wide end at O, 20 mm half-width, 8 mm at 60 mm (the cone pilot's, nearer the origin)
O, A1, A2, LENGTH = np.array([100.0, 0.0]), 20.0, 8.0, 60.0
V_RAYS = [((O[0] - 5.0, s), t) for t in (0.0, 8.0, 17.0, 23.0) for s in (-17.0, -9.0, -2.5, 4.0, 11.0)]


def _walls(scene):
    out = []
    for sign, name in ((1, "upper"), (-1, "lower")):
        m = Mirror(scene, {"p1": {"x": O[0], "y": O[1] + sign * A1}, "p2": {"x": O[0] + LENGTH, "y": O[1] + sign * A2}})
        m.name = name
        scene.add_object(m)
        out.append(m)
    return out


def _launched(rays):
    """The V rays start 5 mm upstream of the entrance, aimed to cross it at s."""
    out = []
    for (a, s), t in rays:
        d = np.array([math.cos(math.radians(t)), math.sin(math.radians(t))])
        out.append(((O[0] - 5.0 * d[0], s - 5.0 * d[1]), t))
    return out


def test_a_revolved_lens_traces_the_same_ray_by_ray():
    cmp = compare_2d_3d(lambda sc: {"L1": _lens_2d(sc)}, [_solid()], LENS_RAYS, plane=IMAGE)
    s = cmp.summary()
    assert (s["rays"], s["agree"], s["differ"]) == (len(LENS_RAYS), len(LENS_RAYS), 0)
    assert s["worst_direction_difference"] < 1e-9 and s["worst_point_difference_mm"] < 1e-7
    through = [r for r in cmp.rays if r["path_2d"]]
    assert through and all(r["path_2d"] == ["L1.S1", "L1.S2"] for r in through)
    assert any(r["position"][1] < 0 for r in through)              # below the axis: named by the mirrored profile
    json.dumps(s)
    ghosts = [r for r in cmp.rays if r["share_on_2d_path"] < 1.0]   # Fresnel reflections took the rest
    strict = compare_2d_3d(lambda sc: {"L1": _lens_2d(sc)}, [_solid()], LENS_RAYS, plane=IMAGE, min_share=0.999)
    assert ghosts and strict.summary()["differ"] == len(ghosts)


def test_a_different_3d_lens_is_reported():
    bent = compare_2d_3d(lambda sc: {"L1": _lens_2d(sc)}, [_solid(radius_back=-15.0)], LENS_RAYS, plane=IMAGE)
    s = bent.summary()
    assert s["differ"] == 0 and s["worst_direction_difference"] > 1e-3      # same faces, other directions

    small = compare_2d_3d(lambda sc: {"L1": _lens_2d(sc)}, [_solid(semi_diameter=5.0)], LENS_RAYS)
    s = small.summary()
    assert s["differ"] > 0
    missed = [r for r in s["differing_rays"] if r["path_3d"] == []]
    assert missed and all(len(r["path_2d"]) == 2 for r in missed)   # 2D met the lens, 3D missed it


def test_a_revolved_mirror_sheet_traces_the_same():
    profile = export_sheet_profile(_walls(Scene()), symmetry={"kind": "revolve",
                                                            "axis": {"point": list(O), "direction": [1.0, 0.0]}},
                                   physical_params={}, numerical_params={})
    cone = RevolvedProfileSheet("cone", profile, 1.0, cs=scene_placement(profile))
    cmp = compare_2d_3d(lambda sc: {"cone": _walls(sc)}, [cone], _launched(V_RAYS), max_depth=100)
    s = cmp.summary()
    assert s["agree"] == len(V_RAYS) and s["worst_direction_difference"] < 1e-9
    assert max(len(r["path_2d"]) for r in cmp.rays) >= 2                       # some rays reflect twice or more
    assert all(f.startswith("cone.") for r in cmp.rays for f in r["path_2d"])


def test_plates_without_a_profile_trace_the_same():
    wall_len = math.hypot(LENGTH, A1 - A2)
    plates = [PlanarPlate.from_segment(f"trough_{m.name}", m,
                                       [(0, -100), (wall_len, -100), (wall_len, 100), (0, 100)], reflectance=1.0)
              for m in _walls(Scene())]

    def optics(scene):
        upper, lower = _walls(scene)
        return {"trough_upper": upper, "trough_lower": lower}

    cmp = compare_2d_3d(optics, plates, _launched(V_RAYS), max_depth=100)
    s = cmp.summary()
    assert s["agree"] == len(V_RAYS) and s["worst_direction_difference"] < 1e-9
    assert {f for r in cmp.rays for f in r["path_2d"]} == {"trough_upper.plate", "trough_lower.plate"}


def test_span_at_and_bad_names():
    profile = _solid().profile
    assert span_at(profile, (0.05, 3.0)) == span_at(profile, (0.05, -3.0)) == "S1"
    assert span_at(profile, (4.9, -2.0)) == "S2"
    with pytest.raises(ValueError, match="not a 3D component"):
        compare_2d_3d(lambda sc: {"L9": _lens_2d(sc)}, [_solid()], LENS_RAYS[:1])


def test_split_rays_reads_ends_and_faces():
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential import IrradianceDetectorConfig, NSQScene
    from optiland.nonsequential.backends.numpy_backend import NumpyBackend
    from ray_tracing_shapely_optiland.sources import add_source_3d

    scene = NSQScene()
    scene.component_registry.add("L1", _solid())
    add_source_3d(scene, {"name": "p", "kind": "pencil", "position": [-20.0, 3.0], "direction_deg": 0.0})
    scene.add_detector("far", CoordinateSystem(x=80.0, ry=math.pi / 2),
                       IrradianceDetectorConfig(width=200.0, height=200.0, num_pixels_x=2, num_pixels_y=2))
    result = scene.trace(num_rays=50, seed=3, max_depth=20, backend=NumpyBackend(seed=3), record_paths=True)
    recs = split_rays(result.ray_paths["events"])
    assert len(recs) == 50 and all(r.source == "p" for r in recs)
    main = [r for r in recs if r.faces == ("L1.S1", "L1.S2", "")]
    assert main and all(r.end == "detector" for r in main)
    assert all(r.points.shape == r.directions.shape == (len(r.path) + 1, 3) for r in recs)


def test_the_summary_tool(monkeypatch):
    clear_context()
    out = agentic.compare_2d_3d_summary()
    assert out["status"] == "error" and "set_comparison_context" in out["message"]
    small = compare_2d_3d(lambda sc: {"L1": _lens_2d(sc)}, [_solid(semi_diameter=5.0)], LENS_RAYS)
    agentic.set_comparison_context(small)
    try:
        data = agentic.compare_2d_3d_summary(max_listed=1)["data"]
        assert len(data["differing_rays"]) == 1 and data["differing_rays_not_listed"] == data["differ"] - 1
        assert "direction_2d" not in data["differing_rays"][0]                  # the summary, not the table
        assert agentic.compare_2d_3d_summary(max_listed=-1)["status"] == "error"
    finally:
        clear_context()
