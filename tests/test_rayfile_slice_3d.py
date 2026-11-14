"""
Python-specific module: test rayfile slice 3d
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
A rayfile slice in 2D against the same records traced unturned in 3D
===============================================================================
Through a revolved singlet, each kept ray of a slice is traced in 2D and its
own record in 3D, and their exit angles from the axis compared. Meridional
records (h = 0) must agree to the tracing precision; for skew records the
difference must grow with h, which is what the skew share measures.
===============================================================================
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.materials.ideal import IdealMaterial
from optiland.nonsequential import NSQMaterial

from ray_tracing_shapely.core.scene import Scene
from ray_tracing_shapely.core.simulator import Simulator
from ray_tracing_shapely.optical_elements.lenses import SingletLens
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile
from ray_tracing_shapely.optical_elements.sources import Rayfile, slice_rayfile
from ray_tracing_shapely_optiland.compare import compare_slice_3d
from ray_tracing_shapely_optiland.revolve import RevolvedProfileSolid, scene_placement

LENS = dict(radius_front=60.0, radius_back=-20.0, thickness=5.0, semi_diameter=10.0, n=1.5)
AT = (-20.0, 0.0)


def _optics(scene):
    lens = SingletLens(scene, **LENS)
    scene.add_object(lens)
    return {"L1": lens}


def _solid():
    lens = SingletLens(Scene(), **LENS)
    profile = export_profile(lens, symmetry={"kind": "revolve", "axis": lens.revolve_axis()},
                             physical_params={}, numerical_params={})
    return RevolvedProfileSolid("L1", profile, NSQMaterial(IdealMaterial(n=1.5)), cs=scene_placement(profile))


def _disc(n, seed, meridional):
    """Rays from a 2 mm disc within 15 deg of the axis; meridional ones lie in a plane through it."""
    rng = np.random.default_rng(seed)
    rho, phi = 2.0 * np.sqrt(rng.random(n)), 2 * np.pi * rng.random(n)
    pos = np.c_[rho * np.cos(phi), rho * np.sin(phi), np.zeros(n)]
    t = np.radians(15.0 * rng.random(n))
    az = np.where(rng.random(n) < 0.5, phi, phi + np.pi) if meridional else 2 * np.pi * rng.random(n)
    dirs = np.c_[np.sin(t) * np.cos(az), np.sin(t) * np.sin(az), np.cos(t)]
    return Rayfile(positions=pos, directions=dirs, flux=np.full(n, 1.0 / n), spectrum_nm=np.array([[550.0, 1.0]]))


def _differences(rows):
    rows = [r for r in rows if r["angle_3d_deg"] is not None and r["path_2d"]]
    return (np.array([abs(r["offset_mm"]) for r in rows]),
            np.array([abs(r["angle_2d_deg"] - r["angle_3d_deg"]) for r in rows]))


def test_meridional_records_trace_the_same_in_2d_and_3d():
    rf = _disc(300, 4, meridional=True)
    slc = slice_rayfile(rf, AT, 0.0, max_offset_mm=0.0)
    assert len(slc.flux) == rf.n_rays and slc.skew_share == 0.0
    h, diff = _differences(compare_slice_3d(_optics, [_solid()], rf, slc, AT, 0.0, every=3))
    assert len(diff) > 50 and diff.max() < 1e-7


def test_the_difference_grows_with_the_skew_offset():
    rf = _disc(6000, 1, meridional=False)
    slc = slice_rayfile(rf, AT, 0.0, max_offset_mm=1.0)
    assert 0.2 < slc.skew_share < 0.6
    h, diff = _differences(compare_slice_3d(_optics, [_solid()], rf, slc, AT, 0.0, every=10))
    near, far = np.median(diff[h < 0.1]), np.median(diff[h > 0.6])
    assert near < 0.01 and far > 0.1 and far > 20 * near


def test_a_slice_lights_a_2d_scene():
    rf = _disc(200, 2, meridional=False)
    slc = slice_rayfile(rf, AT, 0.0, max_offset_mm=0.5)
    scene = Scene()
    _optics(scene)
    scene.add_object(slc.source_2d(scene, name="led"))
    segments = Simulator(scene, max_rays=20_000).run_with_result(name="slice").segments
    roots = [s for s in segments if s.parent_uuid is None]
    assert len(roots) == len(slc.flux)
    flux_per_brightness = slc.source_2d(Scene()).flux_per_brightness
    assert sum(s.total_brightness for s in roots) * flux_per_brightness == pytest.approx(slc.kept_flux)
