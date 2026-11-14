"""
Python-specific module: irradiance
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
IRRADIANCE -- a 3D trace's light on a plane past the optic
===============================================================================
The 3D half of ``ray_tracing_shapely.analysis.irradiance``. The rays of an
event log that left the scene (escaped, or reached a detector) are taken
back into the 2D scene's frame (world (x, y, z) -> scene (a, c, b), the
inverse of ``sources.scene_to_world``) and carried on to the plane, so the
plane is named as a 2D user names it, and its u axis is the 2D trace's line.

Each ray's last free segment starts at its second-to-last event and runs
along its last direction (``events.RayRecord``). Its flux is the flux it
carries at its last event.
===============================================================================
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Union

import numpy as np

from ray_tracing_shapely.analysis.irradiance import ExitPlane, IrradianceMap, irradiance_map

from .events import DETECTOR, ESCAPED, RayRecord, split_rays


def _world_to_scene(v: np.ndarray) -> np.ndarray:
    return v[:, [0, 2, 1]]


def exit_rays_3d(rays: Union[Sequence[RayRecord], np.ndarray], *, ends: Iterable[str] = (ESCAPED, DETECTOR)
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The last free segment of every ray that left, in the scene frame (a, b, c).

    Args:
        rays: ``RayRecord``s, or an event log (``result.ray_paths["events"]``).
        ends: The ends that count as leaving: escaped and detector by default.

    Returns:
        (points, directions, flux): (k, 3), (k, 3), (k,).
    """
    records = split_rays(rays) if isinstance(rays, np.ndarray) else list(rays)
    ends = set(ends)
    out = [r for r in records if r.end in ends and len(r.points) >= 2]
    if not out:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    points = _world_to_scene(np.array([r.points[-2] for r in out]))
    directions = _world_to_scene(np.array([r.directions[-1] for r in out]))
    return points, directions, np.array([float(r.flux[-1]) for r in out])


def irradiance_map_3d(rays: Union[Sequence[RayRecord], np.ndarray], plane: ExitPlane, *, half_size_mm: float,
                      pixels: int = 101, flux_unit: str = "W") -> IrradianceMap:
    """An event log's light on a plane: ``exit_rays_3d`` binned by ``irradiance_map``.

    ``flux_total`` is the flux of every ray that left, not of the source:
    what the scene absorbed is not on any plane.
    """
    points, directions, flux = exit_rays_3d(rays)
    return irradiance_map(points, directions, flux, plane, half_size_mm=half_size_mm, pixels=pixels,
                          flux_unit=flux_unit)


__all__ = ["exit_rays_3d", "irradiance_map_3d"]
