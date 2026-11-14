"""
Python-specific module: test_revolve_conic_glass
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
A conic glass edge, revolved (C-BRIDGE-08)
===============================================================================
The 2D plano-hyperbolic lens of developer_tests/test_glass_curves.py: its
convex face is an exact conic edge (K = -n^2), so collimated light from inside
the glass meets in one point. Exported as a revolve profile, the face becomes
Optiland's ConicGeometry, and the whole beam, skew rays included, meets at the
same point in 3D. A spline edge revolves too, and focuses as
its 2D spline does.
===============================================================================
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.materials.ideal import IdealMaterial
from optiland.nonsequential import (CollimatedSourceConfig, IrradianceDetectorConfig, NSQMaterial,
                                    NSQScene, Spectrum)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend

from ray_tracing_shapely_optiland.revolve import RevolvedProfileSolid, scene_placement

ray_tracing_shapely = pytest.importorskip("ray_tracing_shapely")
from ray_tracing_shapely.core.glass_curves import ConicCurve  # noqa: E402
from ray_tracing_shapely.core.scene import Scene  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.base_prism import BasePrism  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile  # noqa: E402
from ray_tracing_shapely.optical_elements.roles import Mechanism, Role  # noqa: E402

N = 1.5
R, T, H = 20.0, 12.0, 10.0
X0, Y0 = 100.0, 30.0          # away from the world origin (KW-OPTILAND-05)
FOCUS = X0 + T + R / (N - 1)


class _PlanoAsphere(BasePrism):
    _edge_roles = [("S1", "Flat Face", Role.ENTRANCE), ("RIM", "Edge", Role.BLOCKER),
                   ("S2", "Convex Face", Role.EXIT), ("RIM", "Edge", Role.BLOCKER)]
    _edge_mechanisms = {"S1": Mechanism.REFRACT, "RIM": None, "S2": Mechanism.REFRACT}

    def __init__(self, scene, *, spline=False):
        self._spline = spline
        super().__init__(scene, size=H, n=N)

    def _compute_path(self):
        c = ConicCurve((X0 + T, Y0), (-1.0, 0.0), R, -N * N, 0.0, 1.0)
        lo, hi = c.point(H), c.point(-H)
        desc = c.descriptor()
        if self._spline:
            ss = [H - 2 * H * k / 16 for k in range(17)]
            desc = {"type": "spline", "points": [list(c.point(s)) for s in ss],
                    "tangents": [[-c.deriv(s)[0], -c.deriv(s)[1]] for s in ss]}
        return [{"x": X0, "y": Y0 + H, "arc": False}, {"x": X0, "y": Y0 - H, "arc": False},
                {"x": lo[0], "y": lo[1], "arc": False, "curve": desc}, {"x": hi[0], "y": hi[1], "arc": False}]


def _profile(spline=False):
    return export_profile(_PlanoAsphere(Scene(), spline=spline),
                          symmetry={"kind": "revolve", "axis": {"point": [X0, Y0], "direction": [1.0, 0.0]}},
                          physical_params={}, numerical_params={})


def test_a_revolved_conic_glass_edge_focuses_every_ray_of_the_beam():
    profile = _profile()
    glass = NSQMaterial(IdealMaterial(n=N))
    lens = RevolvedProfileSolid("L", profile, glass, cs=scene_placement(profile))
    assert lens.surface_kinds["S2"] == "conic"
    assert _worst_miss(lens) < 1e-9


def _worst_miss(lens):
    """Trace a collimated beam through ``lens``: the largest distance at which a ray
    that passes straight through (S1, S2) misses the focus."""
    scene = NSQScene()
    scene.add_source("B", CoordinateSystem(x=X0 - 20.0, y=0.0, z=Y0, ry=math.pi / 2),
                     CollimatedSourceConfig(Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=0.9 * H))
    scene.component_registry.add(lens.name, lens)
    scene.add_detector("D", CoordinateSystem(x=FOCUS + 100.0, y=0.0, z=Y0, ry=math.pi / 2),
                       IrradianceDetectorConfig(width=2000, height=2000, num_pixels_x=2, num_pixels_y=2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = scene.trace(num_rays=3000, seed=1, max_depth=10, backend=NumpyBackend(seed=1),
                         record_paths=True).ray_paths["events"]
    ev = ev[np.argsort(ev["ray_id"], kind="stable")]
    _, st = np.unique(ev["ray_id"], return_index=True)
    en = np.r_[st[1:], len(ev)]
    focus = np.array([FOCUS, 0.0, Y0])
    worst, count, skew = 0.0, 0, 0.0
    for s, e in zip(st, en):
        names = [str(c) for c in ev["component_name"][s + 1:e]]
        if names != ["L.S1", "L.S2", ""]:
            continue                       # a Fresnel reflection took the ray elsewhere
        p = np.array([ev["x"][e - 1], ev["y"][e - 1], ev["z"][e - 1]])
        d = np.array([ev["L"][e - 1], ev["M"][e - 1], ev["N"][e - 1]])
        worst = max(worst, float(np.linalg.norm(np.cross(focus - p, d))))
        skew = max(skew, abs(float(ev["y"][s])))
        count += 1
    assert count > 2500 and skew > 0.8 * H          # the whole beam, far out of the 2D plane
    return worst


def test_a_revolved_spline_glass_edge_focuses_as_its_2d_spline_does():
    # The face as a 17-point spline through the hyperbola: in 2D it
    # focuses collimated light to a spread set by the spline's accuracy; the revolved
    # surface does the same in 3D, for the whole beam.
    profile = _profile(spline=True)
    s2 = next(sp for sp in profile["spans"] if sp["short_label"] == "S2")
    assert s2["curve"]["type"] == "spline"
    lens = RevolvedProfileSolid("L", profile, NSQMaterial(IdealMaterial(n=N)), cs=scene_placement(profile))
    assert lens.surface_kinds["S2"] == "spline"
    assert _worst_miss(lens) < 1e-3                 # 3.8e-4 measured
