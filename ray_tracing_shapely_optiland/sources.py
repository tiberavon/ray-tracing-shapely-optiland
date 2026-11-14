"""
Python-specific module: sources
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
SOURCES -- a source specification built as an Optiland non-sequential source
===============================================================================
The 3D half of the source handoff. A source specification
(ray_tracing_shapely.optical_elements.sources.source_spec) is written in the
2D scene's frame (a, b, c); this module places it in the world the scene
bridge uses, x = a, y = c, z = b, so a source lands where the optics
exported from the same 2D scene land (``revolve.scene_placement``).

Each kind and emission law maps to the one Optiland source that has that
law exactly. Optiland's laws, read from its source code:

    PointSource       uniform (isotropic) within its cone
    ExtendedSource    uniform within its cone below 90 deg; lambertian at 90
    CollimatedSource  every ray along the axis, from a disc

    spec                               Optiland
    ---------------------------------  ----------------------------------------
    pencil                             CollimatedSource, aperture radius 1e-9
    point, uniform                     PointSource(half_angle_deg)
    point, lambertian                  ExtendedSource, 1e-9 square, 90 deg
    patch, uniform (below 90 deg)      ExtendedSource(size, half_angle_deg)
    patch, lambertian                  ExtendedSource(size, 90 deg)

The spec's validation already refuses what has no exact counterpart (a
lambertian cone cut short of 90 deg, a uniform patch filling the hemisphere).

The source's coordinate system: local z the emission axis, local x the
patch's second side (``e_out``, ``size[1]``, Optiland's width), local y its
first side (``e_in``, ``size[0]``, Optiland's height), origin the patch's
centre. ``e_out`` x ``e_in`` = n, so it is a proper rotation.

A measured LED's rays (``optical_elements.sources.rayfile``) become a
``RayfileSource``, placed the same way: the file's z along the emission
axis, its x along ``e_out`` (out of the 2D plane at roll 0), its y along
``e_in`` (in the plane at roll 0), its origin at the given point. Optiland
traces only its three own source classes (``KW-OPTILAND-08``), so the
rayfile source is an ``ExtendedSource`` whose ``generate`` returns the
file's rays; the extended source's own parameters are never used to emit.
===============================================================================
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import numpy as np
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (CollimatedSourceConfig, ExtendedSourceConfig, PointSourceConfig, Spectrum)
from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.ray_bundle import NSQRayBundle
from optiland.nonsequential.sources.extended import ExtendedSource

from ray_tracing_shapely.optical_elements.sources.rayfile import Rayfile
from ray_tracing_shapely.optical_elements.sources.source_spec import (patch_centre, source_axes, spec_rayfile,
                                                                      validate_source_spec)

from .revolve import cs_from_axes

#: The size given to a source that is a point or a pencil, in mm: small
#: enough that every ray starts at the point to well within any tolerance.
POINT_SIZE = 1e-9

SourceConfig = Union[PointSourceConfig, ExtendedSourceConfig, CollimatedSourceConfig]


def scene_to_world(v: Sequence[float]) -> np.ndarray:
    """A scene-frame (a, b, c) point or vector in the world: (a, c, b)."""
    return np.array([v[0], v[2], v[1]], dtype=float)


def source_frame(spec: Mapping[str, Any]) -> np.ndarray:
    """The world rotation of a validated spec's source: columns e_out, e_in, n."""
    ax = source_axes(spec)
    return np.column_stack([scene_to_world(ax["e_out"]), scene_to_world(ax["e_in"]), scene_to_world(ax["n"])])


def build_source_3d(spec: Mapping[str, Any]) -> Tuple[CoordinateSystem, SourceConfig]:
    """The Optiland source of a spec: its coordinate system and config.

    Add it with ``scene.add_source(spec["name"], cs, config)``.

    Raises:
        ValueError: If the spec is invalid (see ``validate_source_spec``),
            or a rayfile's, which is not a config: use ``add_source_3d``.
    """
    spec = validate_source_spec(spec)
    if spec["kind"] == "rayfile":
        raise ValueError(f"source {spec['name']!r}: a rayfile is not an Optiland source config; "
                         "add it with add_source_3d")
    origin = patch_centre(spec) if spec["kind"] == "patch" else spec["position"]
    cs = cs_from_axes(scene_to_world(origin), source_frame(spec))

    flux: Dict[str, Any] = ({"total_flux": spec["flux"]} if spec["flux_unit"] == "W"
                            else {"total_flux_lumens": spec["flux"]})
    spectrum = Spectrum.monochromatic(spec["wavelength_um"])
    kind = spec["kind"]
    if kind == "pencil":
        return cs, CollimatedSourceConfig(spectrum, aperture_radius=POINT_SIZE, **flux)
    if kind == "point" and spec["emission"] == "uniform":
        return cs, PointSourceConfig(spectrum, half_angle_deg=spec["half_angle_deg"], **flux)
    if kind == "point":
        return cs, ExtendedSourceConfig(spectrum, width=POINT_SIZE, height=POINT_SIZE, half_angle_deg=90.0, **flux)
    w_in, w_out = spec["size"]
    return cs, ExtendedSourceConfig(spectrum, width=w_out, height=w_in, half_angle_deg=spec["half_angle_deg"],
                                    **flux)


def add_source_3d(scene, spec: Mapping[str, Any], *, base_dir: Any = None) -> bool:
    """Build a spec's source and add it to an NSQScene under the spec's name.

    A ``rayfile`` spec's file is read (``spec_rayfile``, relative paths from
    ``base_dir``, the profile's folder) and every ray of it traced: the
    spec's ``slice`` is the 2D section's, not the 3D source's.

    The source object is also given that name as its ``name`` attribute:
    the event log names a source by that attribute, not by its registry
    key, and without it every source is logged as ``source_<index>``
    (``KW-OPTILAND-09``).

    Returns:
        True.
    """
    spec = validate_source_spec(spec)
    if spec["kind"] == "rayfile":
        add_rayfile_source_3d(scene, spec["name"], spec_rayfile(spec, base_dir), spec["position"],
                              spec["direction_deg"], spec["roll_deg"], wavelength_um=spec["wavelength_um"])
        return True
    cs, config = build_source_3d(spec)
    scene.add_source(spec["name"], cs, config)
    scene.source_registry.get(spec["name"]).name = spec["name"]
    return True


# --- a measured LED: a rayfile ----------------------------------------------------------


def rayfile_frame(position: Sequence[float], direction_deg: float, roll_deg: float = 0.0) -> CoordinateSystem:
    """The world frame of a rayfile placed in the 2D scene.

    Args:
        position: The file's origin, (a, b) or (a, b, c) in the scene frame.
        direction_deg: The file's +z, an angle in the 2D plane from +a.
        roll_deg: A turn about that axis, as a patch's ``roll_deg``.
    """
    ax = source_axes({"direction_deg": float(direction_deg), "roll_deg": float(roll_deg)})
    rot = np.column_stack([scene_to_world(ax["e_out"]), scene_to_world(ax["e_in"]), scene_to_world(ax["n"])])
    pos = list(position) + [0.0] * (3 - len(position))
    return cs_from_axes(scene_to_world(pos), rot)


class RayfileSource(ExtendedSource):
    """An Optiland source that emits a rayfile's rays.

    Ray ``k`` of a run is record ``k mod n`` of the file: the first ``n`` rays
    of a run that has only this source are the file's, in its order (an ams
    OSRAM file's rays are randomly ordered, so any first ``N`` are a fair
    sample). A run asking for more rays than the file holds reuses them.

    Each ray's flux is its share of the file's flux, scaled to the source's
    ``total_flux`` (the file's own by default), so per-ray weights survive.
    Wavelengths are drawn from the file's spectrum, or ``wavelength_um``.
    """

    def __init__(self, cs: CoordinateSystem, rayfile: Rayfile, *, total_flux: float | None = None,
                 wavelength_um: float | None = None, medium=None) -> None:
        if rayfile.n_rays == 0:
            raise ValueError("a rayfile source needs at least one ray")
        if wavelength_um is not None:
            spectrum = Spectrum.monochromatic(float(wavelength_um))
        elif rayfile.spectrum_nm is not None and np.any(rayfile.spectrum_nm[:, 1] > 0):
            spectrum = Spectrum(rayfile.spectrum_nm[:, 0] / 1000.0, np.clip(rayfile.spectrum_nm[:, 1], 0.0, None))
        else:
            raise ValueError("the rayfile has no spectrum: give wavelength_um")
        extent = np.ptp(rayfile.positions[:, :2], axis=0)
        super().__init__(cs, spectrum, total_flux=float(rayfile.total_flux if total_flux is None else total_flux),
                         width=float(max(extent[0], 1e-9)), height=float(max(extent[1], 1e-9)),
                         half_angle_deg=90.0, medium=medium)
        self.rayfile = rayfile
        self._weights = rayfile.flux / rayfile.flux.mean()

    def generate(self, ray_id: np.ndarray, rng) -> NSQRayBundle:
        """The rays ``ray_id mod n`` of the file, in the world."""
        ray_id = np.asarray(ray_id, dtype=np.int64)
        k = ray_id % self.rayfile.n_rays
        bundle = super().generate(ray_id, rng)          # wavelengths, medium, bookkeeping
        translation, rot = _get_transform(self.cs)
        pos = self.rayfile.positions[k] @ rot.T + translation
        dirs = self.rayfile.directions[k] @ rot.T
        bundle.x, bundle.y, bundle.z = (pos[:, i].copy() for i in range(3))
        bundle.L, bundle.M, bundle.N = (dirs[:, i].copy() for i in range(3))
        bundle.flux = float(self.total_flux) / len(ray_id) * self._weights[k]
        return bundle


def add_rayfile_source_3d(scene, name: str, rayfile: Rayfile, position: Sequence[float], direction_deg: float,
                          roll_deg: float = 0.0, **kwargs) -> RayfileSource:
    """Place a rayfile in the 2D scene's frame and add it to an NSQScene under ``name``.

    ``kwargs`` go to ``RayfileSource`` (``total_flux``, ``wavelength_um``,
    ``medium``). The source is logged under ``name`` (``KW-OPTILAND-09``).
    It is added to the registry directly: ``add_source`` takes only the three
    config classes (``KW-OPTILAND-08``).
    """
    source = RayfileSource(rayfile_frame(position, direction_deg, roll_deg), rayfile, **kwargs)
    source.name = name
    scene.source_registry.add(name, source)
    return source

