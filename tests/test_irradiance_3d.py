"""
Python-specific module: test irradiance 3d
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
A 3D trace's light on a plane, named in the 2D scene's frame
===============================================================================
A pencil placed off the 2D plane must land on the map at its own (b, c) as
(u, v): the map's frame is the scene's, not the world's. A lens lit from a
point on its axis gives a round map, centred. The plot is written as SVG and
PNG, and its descriptor is plain JSON.
===============================================================================
"""

from __future__ import annotations

import json
import math
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from optiland.coordinate_system import CoordinateSystem  # noqa: E402
from optiland.materials.ideal import IdealMaterial  # noqa: E402
from optiland.nonsequential import (ExtendedSourceConfig, IrradianceDetectorConfig, NSQMaterial,  # noqa: E402
                                    NSQScene, Spectrum)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend  # noqa: E402

from ray_tracing_shapely.analysis.irradiance import ExitPlane, exit_rays_2d, line_profile  # noqa: E402
from ray_tracing_shapely.core.scene import Scene  # noqa: E402
from ray_tracing_shapely.core.simulator import Simulator  # noqa: E402
from ray_tracing_shapely.optical_elements.lenses import SingletLens  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile  # noqa: E402
from ray_tracing_shapely.optical_elements.sources import build_source_2d  # noqa: E402
from ray_tracing_shapely_optiland.irradiance import exit_rays_3d, irradiance_map_3d  # noqa: E402
from ray_tracing_shapely_optiland.revolve import RevolvedProfileSolid, scene_placement  # noqa: E402
from ray_tracing_shapely_optiland.sources import add_source_3d  # noqa: E402
from ray_tracing_shapely_optiland.visualization.irradiance_plot import save_irradiance_plot  # noqa: E402

LENS = dict(radius_front=60.0, radius_back=-20.0, thickness=5.0, semi_diameter=10.0, n=1.5)


def _trace(scene, n):
    scene.add_detector("far", CoordinateSystem(x=1e5, ry=math.pi / 2),
                       IrradianceDetectorConfig(width=1.0, height=1.0, num_pixels_x=1, num_pixels_y=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return scene.trace(num_rays=n, seed=1, max_depth=10, backend=NumpyBackend(seed=1),
                           record_paths=True).ray_paths["events"]


def test_a_pencil_off_the_plane_lands_at_its_own_b_and_c():
    scene = NSQScene()
    add_source_3d(scene, {"name": "p", "kind": "pencil", "position": [-20.0, 1.25, -0.75], "direction_deg": 0.0})
    events = _trace(scene, 20)
    points, directions, flux = exit_rays_3d(events)
    assert np.allclose(points[:, 1:], [1.25, -0.75], atol=1e-6) and np.allclose(directions, [1.0, 0.0, 0.0])
    imap = irradiance_map_3d(events, ExitPlane.at((50.0, 0.0), 0.0), half_size_mm=2.0, pixels=8)
    k_v, k_u = np.unravel_index(int(np.argmax(imap.values)), imap.values.shape)
    assert (imap.u_centres_mm[k_u], imap.v_centres_mm[k_v]) == pytest.approx((1.25, -0.75))
    assert imap.flux_on_map == pytest.approx(flux.sum())


def test_a_lens_lit_from_its_axis_gives_a_round_map_and_a_plot(tmp_path):
    lens = SingletLens(Scene(), **LENS)
    profile = export_profile(lens, symmetry={"kind": "revolve", "axis": lens.revolve_axis()},
                             physical_params={}, numerical_params={})
    scene = NSQScene()
    scene.add_source("s", CoordinateSystem(x=-20.0, ry=math.pi / 2), ExtendedSourceConfig(
        Spectrum.monochromatic(0.55), total_flux=1.0, width=1e-9, height=1e-9, half_angle_deg=30.0))
    scene.component_registry.add("L", RevolvedProfileSolid("L", profile, NSQMaterial(IdealMaterial(n=1.5)),
                                                           cs=scene_placement(profile)))
    plane = ExitPlane.at((40.0, 0.0), 0.0)
    imap = irradiance_map_3d(_trace(scene, 20_000), plane, half_size_mm=30.0, pixels=61)
    s = imap.summary()
    assert np.allclose(s["centroid_mm"], 0.0, atol=0.3)
    assert s["rms_width_mm"][0] == pytest.approx(s["rms_width_mm"][1], rel=0.05)

    scene_2d = Scene()
    scene_2d.add_object(SingletLens(scene_2d, **LENS))
    scene_2d.add_object(build_source_2d(scene_2d, {"name": "s", "kind": "point", "position": [-20.0, 0.0],
                                                   "direction_deg": 0.0, "emission": "uniform",
                                                   "half_angle_deg": 30.0}, n_angles=61))
    segments = Simulator(scene_2d, max_rays=20_000, record_surface_hits=True).run_with_result(name="s").segments
    profile_2d = line_profile(*exit_rays_2d(segments), plane, half_size_mm=30.0, bins=61)
    out = save_irradiance_plot(imap, tmp_path / "plots" / "lens", profile_2d=profile_2d, title="singlet")
    assert (tmp_path / "plots" / "lens.svg").is_file() and (tmp_path / "plots" / "lens.png").is_file()
    assert out["png_available"] and out["width"] > 0 and out["summary"] == json.loads(json.dumps(s))
