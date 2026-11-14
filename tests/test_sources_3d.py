"""
Python-specific module: test sources 3d
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
A source specification built in 3D, against the chip placed by hand
===============================================================================
The LED collimator pilot placed its chip by hand: a 2 x 1 mm Lambertian
patch, its lower edge on a plane tilted 15 deg, one end at the focus F, the
frame fitted to Euler angles in the notebook. Written as a spec, the same chip
must come out in the same place, with the same frame, and its 2D section must
be exactly where the 3D rectangle crosses the scene plane.
===============================================================================
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (CollimatedSourceConfig, ExtendedSourceConfig, IrradianceDetectorConfig,
                                    NSQScene, PointSourceConfig)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.components.base import _get_transform

from ray_tracing_shapely.optical_elements.sources import patch_section, validate_source_spec
from ray_tracing_shapely_optiland.sources import add_source_3d, build_source_3d

F = (3.0, -2.0)
FW = np.array([F[0], 0.0, F[1]])                  # F in the world: (a, b) -> (a, 0, b)
TILT = math.radians(15.0)
E_EDGE = np.array([0.0, math.cos(TILT), math.sin(TILT)])     # the pilot's lower edge, 2 mm
E_UP = np.array([0.0, -math.sin(TILT), math.cos(TILT)])      # the pilot's height, 1 mm
CHIP = {"name": "chip", "kind": "patch", "position": list(F), "direction_deg": 0.0, "emission": "lambertian",
        "size": [1.0, 2.0], "roll_deg": -15.0, "anchor": [0.0, 0.0], "flux": 2.0}


def _births(spec, num_rays=20000):
    """Start points, directions and flux of the rays a spec's 3D source emits."""
    scene = NSQScene()
    add_source_3d(scene, spec)
    scene.add_detector("far", CoordinateSystem(x=1000.0, ry=math.pi / 2),
                       IrradianceDetectorConfig(width=10.0, height=10.0, num_pixels_x=2, num_pixels_y=2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = scene.trace(num_rays=num_rays, seed=3, max_depth=2, backend=NumpyBackend(seed=3),
                             record_paths=True)
    ev = result.ray_paths["events"]
    ev = ev[np.argsort(ev["ray_id"], kind="stable")]
    _, first = np.unique(ev["ray_id"], return_index=True)
    b = ev[first]
    return (np.stack([b["x"], b["y"], b["z"]], axis=1), np.stack([b["L"], b["M"], b["N"]], axis=1),
            np.asarray(b["flux"]))


def test_the_chip_spec_rebuilds_the_hand_placed_chip():
    cs, config = build_source_3d(CHIP)
    assert isinstance(config, ExtendedSourceConfig)
    assert (config.width, config.height, config.half_angle_deg, config.total_flux) == (2.0, 1.0, 90.0, 2.0)
    translation, rotation = _get_transform(cs)
    want = np.column_stack([E_EDGE, E_UP, [1.0, 0.0, 0.0]])     # the notebook's WANT
    assert np.abs(rotation - want).max() < 1e-12
    assert np.abs(translation - (FW + 0.5 * 2.0 * E_EDGE + 0.5 * 1.0 * E_UP)).max() < 1e-12

    starts, dirs, flux = _births(CHIP)
    u, v = (starts - FW) @ E_EDGE, (starts - FW) @ E_UP
    assert 0.0 <= u.min() < 1e-2 and 2.0 - 1e-2 < u.max() <= 2.0
    assert 0.0 <= v.min() < 1e-2 and 1.0 - 1e-2 < v.max() <= 1.0
    assert np.abs((starts - FW)[:, 0]).max() < 1e-12              # all on the chip's plane
    assert flux.sum() == pytest.approx(2.0, rel=1e-12)
    # Lambertian: the share within 30 deg of the axis is sin^2(30 deg) = 1/4.
    within = float(np.mean(dirs[:, 0] > math.cos(math.radians(30.0))))
    assert within == pytest.approx(0.25, abs=4 * math.sqrt(0.25 * 0.75 / len(dirs)))


def test_the_2d_section_is_where_the_3d_rectangle_crosses_the_plane():
    for spec in (CHIP, {**CHIP, "direction_deg": 70.0, "roll_deg": 35.0, "anchor": [0.3, 0.8],
                        "position": [F[0], F[1], -0.4]}):
        cs, config = build_source_3d(spec)
        t, r = _get_transform(cs)
        corners = [t + r @ np.array([sx * config.width / 2, sy * config.height / 2, 0.0])
                   for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        crossings = []                            # the rectangle's edges crossing world y = 0
        for p, q in zip(corners, corners[1:] + corners[:1]):
            if (p[1] < 0.0) != (q[1] < 0.0):
                k = p[1] / (p[1] - q[1])
                c = p + k * (q - p)
                crossings.append((c[0], c[2]))     # world (x, z) -> 2D (a, b)
        section = patch_section(validate_source_spec(spec))
        assert len(crossings) == 2 and section is not None
        assert np.abs(np.array(sorted(crossings)) - np.array(sorted(section))).max() < 1e-12


def test_point_and_pencil_kinds():
    pencil = {"name": "p", "kind": "pencil", "position": [1.0, 2.0], "direction_deg": 30.0}
    cs, config = build_source_3d(pencil)
    assert isinstance(config, CollimatedSourceConfig)
    starts, dirs, _ = _births(pencil, num_rays=200)
    d = np.array([math.cos(math.radians(30.0)), 0.0, math.sin(math.radians(30.0))])
    assert np.abs(starts - [1.0, 0.0, 2.0]).max() < 1e-8 and np.abs(dirs - d).max() < 1e-12

    cone = {"name": "c", "kind": "point", "position": [0.0, 0.0], "direction_deg": 90.0,
            "emission": "uniform", "half_angle_deg": 20.0}
    assert isinstance(build_source_3d(cone)[1], PointSourceConfig)
    _, dirs, _ = _births(cone, num_rays=5000)
    off_axis = np.degrees(np.arccos(np.clip(dirs[:, 2], -1.0, 1.0)))     # the axis is world +z (2D +b)
    assert off_axis.max() <= 20.0 + 1e-9 and off_axis.max() > 19.0

    lambert_point = {**cone, "emission": "lambertian", "half_angle_deg": 90.0}
    _, config = build_source_3d(lambert_point)
    assert isinstance(config, ExtendedSourceConfig) and config.half_angle_deg == 90.0
