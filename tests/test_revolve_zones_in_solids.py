"""
Python-specific module: test_revolve_zones_in_solids
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
Zones and cone tips in solids: the bridge's own closure checks (C-BRIDGE-09)
===============================================================================
Two valid solids Optiland's Volume rejects, built and traced by the bridge:

- **a stepped lens:** a conic cap on the axis, a step, and the same conic
  moved back as a zone (a conic with a hole). Optiland samples only a
  conic's outer rim and checks parity on a plain-conic copy of the zone
  (KW-OPTILAND-07). Both faces are K = -n^2 hyperbolas, so collimated light
  through the cap focuses at one point and through the zone at another, the
  step's depth nearer: each exactly.
- **a cone on a disc:** the cone's tip is on the axis, where Optiland looks
  for a rim another face must meet (KW-OPTILAND-06).

And a solid with a gap is still refused.
===============================================================================
"""

from __future__ import annotations

import copy
import math
import warnings

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.materials.ideal import IdealMaterial
from optiland.nonsequential import (CollimatedSourceConfig, IrradianceDetectorConfig, NSQMaterial,
                                    NSQScene, Spectrum)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.components.volume import NonWatertightVolumeError, Volume

from ray_tracing_shapely_optiland.geometry import CutConicGeometry
from ray_tracing_shapely_optiland.revolve import RevolvedProfileSolid, scene_placement

ray_tracing_shapely = pytest.importorskip("ray_tracing_shapely")
from ray_tracing_shapely.core.glass_curves import ConicCurve  # noqa: E402
from ray_tracing_shapely.core.scene import Scene  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.base_prism import BasePrism  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile  # noqa: E402
from ray_tracing_shapely.optical_elements.roles import Mechanism, Role  # noqa: E402

X0, Y0, N = 100.0, 30.0, 1.5            # away from the world origin (KW-OPTILAND-05)
R, H1, H, T, STEP = 20.0, 5.0, 10.0, 10.0, 1.5
FOCUS = X0 + T + R / (N - 1)
GLASS = NSQMaterial(IdealMaterial(n=N))


class _Stepped(BasePrism):
    """Flat back at x0; in front, a conic cap out to h1, a step back, the same conic
    moved back by the step as a zone out to h."""

    _edge_roles = [("BACK", "Back", Role.ENTRANCE), ("RIM", "Rim", Role.BLOCKER),
                   ("ZONE", "Zone", Role.EXIT), ("STEP", "Step", Role.BLOCKER),
                   ("CAP", "Cap", Role.EXIT), ("STEP", "Step", Role.BLOCKER),
                   ("ZONE", "Zone", Role.EXIT), ("RIM", "Rim", Role.BLOCKER)]
    _edge_mechanisms = {"BACK": Mechanism.REFRACT, "RIM": None, "ZONE": Mechanism.REFRACT,
                        "STEP": None, "CAP": Mechanism.REFRACT}

    def __init__(self, scene):
        super().__init__(scene, size=2 * H, n=N)

    def _compute_path(self):
        cap = ConicCurve((X0 + T, Y0), (-1.0, 0.0), R, -N * N, 0.0, 1.0)
        zone = ConicCurve((X0 + T - STEP, Y0), (-1.0, 0.0), R, -N * N, 0.0, 1.0)
        at = lambda c, y: c.point(-y)        # noqa: E731  (s = -y for an axis along -x)
        corners = [((X0, Y0 + H), None), ((X0, Y0 - H), None), (at(zone, -H), zone),
                   (at(zone, -H1), None), (at(cap, -H1), cap), (at(cap, H1), None),
                   (at(zone, H1), zone), (at(zone, H), None)]
        return [dict(x=p[0], y=p[1], arc=False, **({"curve": c.descriptor()} if c else {}))
                for p, c in corners]


class _Cone(BasePrism):
    """A flat base at x0 and a cone whose tip lies on the axis, length l in front."""

    _edge_roles = [("BASE", "Base", Role.ENTRANCE), ("SIDE", "Side", Role.EXIT),
                   ("SIDE", "Side", Role.EXIT)]
    _edge_mechanisms = {"BASE": Mechanism.REFRACT, "SIDE": Mechanism.REFRACT}

    def __init__(self, scene):
        super().__init__(scene, size=2 * H, n=N)

    def _compute_path(self):
        return [dict(x=X0, y=Y0 + H, arc=False), dict(x=X0, y=Y0 - H, arc=False),
                dict(x=X0 + 15.0, y=Y0, arc=False)]


def _profile(element):
    return export_profile(element, symmetry={"kind": "revolve",
                                             "axis": {"point": [X0, Y0], "direction": [1.0, 0.0]}},
                          physical_params={}, numerical_params={})


def test_optilands_volume_rejects_both_solids():
    # The witnesses for KW-OPTILAND-07 and KW-OPTILAND-06: the faces the bridge builds
    # (and accepts, below), handed to Optiland's own Volume.
    # the step's rim misses the zone's (unsampled) inner rim by the step's depth; the
    # cone's tip misses the base's rim by the cone's slant
    for element, gap in ((_Stepped(Scene()), STEP), (_Cone(Scene()), math.hypot(15.0, H))):
        solid = RevolvedProfileSolid("S", _profile(element), GLASS, cs=scene_placement(_profile(element)))
        with pytest.raises(NonWatertightVolumeError, match=f"{gap:.2g} mm"):
            Volume(name="v", boundary=solid.surfaces, interior=GLASS)


def test_a_conic_zone_in_a_solid_focuses_exactly():
    profile = _profile(_Stepped(Scene()))
    lens = RevolvedProfileSolid("L", profile, GLASS, cs=scene_placement(profile))
    assert isinstance(lens.face("ZONE").geometry, CutConicGeometry)
    assert lens.face("ZONE").geometry.inner_radius == pytest.approx(H1)
    assert lens.surface_kinds == {"CAP": "conic", "STEP": "frustum", "ZONE": "conic", "RIM": "frustum",
                                  "BACK": "disc"}
    scene = NSQScene()
    scene.add_source("B", CoordinateSystem(x=X0 - 20.0, y=0.0, z=Y0, ry=math.pi / 2),
                     CollimatedSourceConfig(Spectrum.monochromatic(0.55), total_flux=1.0,
                                            aperture_radius=0.95 * H))
    scene.component_registry.add(lens.name, lens)
    scene.add_detector("D", CoordinateSystem(x=FOCUS + 100.0, y=0.0, z=Y0, ry=math.pi / 2),
                       IrradianceDetectorConfig(width=2000, height=2000, num_pixels_x=2, num_pixels_y=2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = scene.trace(num_rays=3000, seed=2, max_depth=10, backend=NumpyBackend(seed=2),
                         record_paths=True).ray_paths["events"]
    ev = ev[np.argsort(ev["ray_id"], kind="stable")]
    _, st = np.unique(ev["ray_id"], return_index=True)
    en = np.r_[st[1:], len(ev)]
    worst = {"L.CAP": 0.0, "L.ZONE": 0.0}
    seen = {"L.CAP": 0, "L.ZONE": 0}
    for s, e in zip(st, en):
        names = [str(c) for c in ev["component_name"][s + 1:e]]
        if len(names) != 3 or names[0] != "L.BACK" or names[1] not in worst or names[2] != "":
            continue
        focus = np.array([FOCUS - (STEP if names[1] == "L.ZONE" else 0.0), 0.0, Y0])
        p = np.array([ev["x"][e - 1], ev["y"][e - 1], ev["z"][e - 1]])
        d = np.array([ev["L"][e - 1], ev["M"][e - 1], ev["N"][e - 1]])
        worst[names[1]] = max(worst[names[1]], float(np.linalg.norm(np.cross(focus - p, d))))
        seen[names[1]] += 1
    assert seen["L.CAP"] > 500 and seen["L.ZONE"] > 1500
    assert max(worst.values()) < 1e-9


def test_a_cone_with_its_tip_on_the_axis_is_a_solid():
    profile = _profile(_Cone(Scene()))
    cone = RevolvedProfileSolid("C", profile, GLASS, cs=scene_placement(profile))
    assert cone.surface_kinds == {"BASE": "disc", "SIDE": "frustum"}
    assert min(float(cone.face("SIDE").geometry.r_front), float(cone.face("SIDE").geometry.r_back)) == 0.0


def test_a_solid_with_a_gap_is_still_refused():
    profile = _profile(_Stepped(Scene()))
    broken = copy.deepcopy(profile)
    broken["spans"] = [sp for sp in broken["spans"] if sp["short_label"] != "STEP"]
    with pytest.raises(NonWatertightVolumeError, match="not watertight"):
        RevolvedProfileSolid("L", broken, GLASS, cs=scene_placement(broken))
