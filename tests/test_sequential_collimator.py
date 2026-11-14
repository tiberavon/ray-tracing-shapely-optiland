"""
Python-specific module: test_sequential_collimator
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
The TIR collimator in Optiland's sequential engine
===============================================================================
A third tracer, independent of the 2D engine and of the non-sequential
bridge, built from the designed faces (TIRCollimator.section_faces):

- **the central channel:** 12b a plane, 12a an even asphere fitted to the
  exact design. F is its front focal point, and every ray leaves parallel.
  With the conic 12a both engines have the same surfaces, and every ray
  leaves at the same angle in both.
- **the ring channel, meridional rays:** 22b and 22a planes tilted to their
  lines, 22c a polynomial mirror fitted to the exact design. Every ray
  leaves parallel, and meets the faces where the 2D design says.
===============================================================================
"""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from ray_tracing_shapely_optiland.sequential_collimator import (central_channel_optic, fit_even_asphere,
                                                                fit_meridional_polynomial,
                                                                ring_channel_optic, trace_from_focus)

ray_tracing_shapely = pytest.importorskip("ray_tracing_shapely")
from ray_tracing_shapely.core.scene import Scene  # noqa: E402
from ray_tracing_shapely.core.scene_objs.light_source.single_ray import SingleRay  # noqa: E402
from ray_tracing_shapely.core.simulator import Simulator  # noqa: E402
from ray_tracing_shapely.optical_elements.lenses import TIRCollimator  # noqa: E402

N = 1.49
F = (200.0, 50.0)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _exit_2d(collimator_kw, launch_deg):
    """The 2D exit angle (from the axis, towards +y positive) of a ray from F."""
    scene = Scene()
    element = TIRCollimator(scene, position=F, **collimator_kw)
    scene.add_object(element)
    t = math.radians(launch_deg)
    scene.add_object(SingleRay(scene, {"p1": {"x": F[0], "y": F[1]},
                                       "p2": {"x": F[0] + math.cos(t), "y": F[1] + math.sin(t)}}))
    segs = Simulator(scene, max_rays=2000).run_with_result(name="2d").segments
    leaving = [s for s in segs if s.p1 != {"x": F[0], "y": F[1]}
               and element.check_ray_intersects(SimpleNamespace(p1=s.p1, p2=s.p2)) is None]
    s = max(leaving, key=lambda sg: sg.total_brightness)
    return math.degrees(math.atan2(s.p2["y"] - s.p1["y"], s.p2["x"] - s.p1["x"]))


@pytest.fixture(scope="module")
def tir():
    collimator = TIRCollimator(Scene())
    return collimator, collimator.section_faces(201)


def test_the_central_channel_as_an_even_asphere(tir):
    c, faces = tir
    asphere = fit_even_asphere(*faces["12a"], terms=8)
    assert asphere["sag_error"] < 1e-9 and asphere["slope_error_deg"] < 1e-7
    optic = central_channel_optic(c.lens_distance, N, asphere)
    angles = np.linspace(0.2, c.design["phi_corner"] - 1e-6, 40)
    out = trace_from_focus(optic, angles)
    assert np.abs(out["angle_deg"]).max() < 1e-7
    # paraxially: F is the front focal point, and the focal length is a plano-convex lens's
    optic.set_aperture(aperture_type="objectNA", value=math.sin(math.radians(c.design["phi_corner"])))
    optic.fields.set_type("object_height")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(0.55, is_primary=True)
    assert optic.paraxial.F1() == pytest.approx(-c.lens_distance, abs=1e-6)
    assert optic.paraxial.f2() == pytest.approx(-asphere["radius"] / (N - 1.0), rel=1e-9)


def test_the_conic_12a_bends_every_ray_as_the_2d_engine_does():
    c = TIRCollimator(Scene(), central="conic")
    front = {"vertex_z": c.lens_distance + c.center_thickness, "radius": -c.design["conic_R"],
             "conic": c.design["conic_K"]}
    angles = np.linspace(0.5, c.design["phi_corner"] - 0.1, 8)
    out = trace_from_focus(central_channel_optic(c.lens_distance, N, front), angles)
    for a, seq in zip(angles, out["angle_deg"]):
        assert seq == pytest.approx(_exit_2d({"central": "conic"}, a), abs=1e-9)
    assert np.abs(out["angle_deg"]).max() > 0.1          # the conic is not exact: the agreement is


def test_the_ring_channel_meridionally(tir):
    c, faces = tir
    mirror = fit_meridional_polynomial(*faces["22c"], degree=14)
    assert mirror["sag_error"] < 1e-11 and mirror["slope_error_deg"] < 1e-9
    optic = ring_channel_optic(faces["22b"][0], mirror, faces["22a"][0], N, image_z=60.0)
    d = c.design
    angles = np.linspace(d["phi_end"] + 1e-9, d["phi_start"] - 1e-9, 40)
    out = trace_from_focus(optic, angles)
    assert np.abs(out["angle_deg"]).max() < 1e-8
    # the end rays meet 22c at its two ends, where the design put them
    (z0, y0), (z1, y1) = faces["22c"][0][0], faces["22c"][0][-1]
    assert out["z"][2, -1] == pytest.approx(z0, abs=1e-6) and out["y"][2, -1] == pytest.approx(y0, abs=1e-6)
    assert out["z"][2, 0] == pytest.approx(z1, abs=1e-6) and out["y"][2, 0] == pytest.approx(y1, abs=1e-6)
    # and against the 2D trace through the spline 22c (33 points), inside the channel (its end
    # rays meet the corners 22c makes with 22a and the mount): the spline's accuracy
    for a in angles[4::8]:
        assert _exit_2d({}, a) == pytest.approx(0.0, abs=2e-4)


def test_a_coarse_polynomial_shows_in_the_beam(tir):
    c, faces = tir
    errors = []
    for degree in (6, 10, 14):
        mirror = fit_meridional_polynomial(*faces["22c"], degree=degree)
        optic = ring_channel_optic(faces["22b"][0], mirror, faces["22a"][0], N, image_z=60.0)
        angles = np.linspace(c.design["phi_end"] + 1e-9, c.design["phi_start"] - 1e-9, 30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            errors.append(np.abs(trace_from_focus(optic, angles)["angle_deg"]).max())
    assert errors[0] > 100 * errors[1] > 1e4 * errors[2]
