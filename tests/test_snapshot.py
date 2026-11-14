"""
Python-specific module: test_snapshot
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
Static 3D pictures and rays by path
===============================================================================
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

vtk = pytest.importorskip("vtk")

from optiland.coordinate_system import CoordinateSystem  # noqa: E402
from optiland.materials.ideal import IdealMaterial  # noqa: E402
from optiland.nonsequential import (ExtendedSourceConfig, IrradianceDetectorConfig, NSQMaterial,  # noqa: E402
                                    NSQScene, Spectrum)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend  # noqa: E402
from optiland.nonsequential.visualization.viewer_3d import NSQViewer3D  # noqa: E402

from ray_tracing_shapely_optiland.revolve import RevolvedProfileSolid, scene_placement  # noqa: E402
from ray_tracing_shapely_optiland.visualization.jupyter_viewer import build_vtk_renderer  # noqa: E402
from ray_tracing_shapely_optiland.visualization.renderers import RevolvedProfileRenderer3D  # noqa: E402
from ray_tracing_shapely_optiland.visualization.snapshot import (group_polylines, polyline_actor,  # noqa: E402
                                                                 ray_polylines, snapshot_png)

ray_tracing_shapely = pytest.importorskip("ray_tracing_shapely")
from ray_tracing_shapely.core.scene import Scene  # noqa: E402
from ray_tracing_shapely.optical_elements.lenses import TIRCollimator  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile  # noqa: E402

F = (100.0, 30.0)


@pytest.fixture(scope="module")
def traced():
    c = TIRCollimator(Scene(), position=F)
    p = export_profile(c, symmetry={"kind": "revolve", "axis": c.revolve_axis()}, physical_params={},
                       numerical_params={})
    solid = RevolvedProfileSolid("T", p, NSQMaterial(IdealMaterial(n=1.49)), cs=scene_placement(p))
    scene = NSQScene()
    scene.add_source("S", CoordinateSystem(x=F[0], y=0.0, z=F[1], ry=math.pi / 2),
                     ExtendedSourceConfig(Spectrum.monochromatic(0.55), width=1e-9, height=1e-9))
    scene.component_registry.add(solid.name, solid)
    scene.add_detector("D", CoordinateSystem(x=F[0] + 500.0, y=0.0, z=F[1], ry=math.pi / 2),
                       IrradianceDetectorConfig(width=10, height=10, num_pixels_x=2, num_pixels_y=2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = scene.trace(num_rays=300, seed=1, max_depth=20, backend=NumpyBackend(seed=1),
                         record_paths=True).ray_paths["events"]
    return solid, ev


def test_ray_polylines_follow_each_ray_and_extend_the_escaped_ones(traced):
    _, ev = traced
    lines = ray_polylines(ev, end_length=50.0)
    assert len(lines) == len(np.unique(ev["ray_id"]))
    for path, pts in lines:
        assert pts.shape[1] == 3 and np.allclose(pts[0], [F[0], 0.0, F[1]], atol=1e-6)
        if path and path[-1] == "escaped":
            assert np.linalg.norm(pts[-1] - pts[-2]) == pytest.approx(50.0)
    groups = group_polylines(lines, {"ring": lambda p: p[:3] == ("T.22b", "T.22c", "T.22a"),
                                     "rest": lambda p: True})
    assert len(groups["ring"]) > 30 and len(groups["ring"]) + len(groups["rest"]) == len(lines)


def test_a_snapshot_is_a_png_of_the_solid_and_its_rays(traced):
    solid, ev = traced
    scene = NSQScene()
    scene.component_registry.add(solid.name, solid)
    viewer = NSQViewer3D(scene)
    viewer.register_renderer(RevolvedProfileSolid, RevolvedProfileRenderer3D())
    renderer = build_vtk_renderer(viewer, None, num_rays=0)
    renderer.AddActor(polyline_actor([pts for _, pts in ray_polylines(ev)], (0.1, 0.5, 0.7)))
    png = snapshot_png(renderer, (320, 200), direction=(-0.4, -1.0, 0.4),
                       focal_point=(F[0] + 25.0, 0.0, F[1]), distance=200.0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    reader = vtk.vtkPNGReader()
    reader.SetMemoryBuffer(png)
    reader.SetMemoryBufferLength(len(png))
    reader.Update()
    assert reader.GetOutput().GetDimensions()[:2] == (320, 200)
