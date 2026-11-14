"""
Python-specific module: events
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
EVENTS -- an Optiland event log cut into one record per ray, headless
===============================================================================
A non-sequential trace run with ``record_paths=True`` keeps every ray's
birth, hits and death in one structured array (``result.ray_paths["events"]``),
rays interleaved. ``split_rays`` cuts it into one ``RayRecord`` per ray:
where it started, the faces it met in order, how it ended, and its points
and directions at each event.

How the log names things:

    birth   component_name is the source's ``name`` attribute, or
            ``source_<index>`` when it has none (KW-OPTILAND-09)
    hit     ``<solid>.<label>`` for a face; ``""`` for any detector
    death   ``"escaped"`` when the ray left the scene; otherwise the face it
            died on, or ``""``

numpy only: the picture (``visualization.snapshot``), the agent tools and
the 2D-against-3D comparison all read the log through this module, and none
of them needs VTK for it.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

#: The end of a ray that hit a detector.
DETECTOR = "detector"
#: The end of a ray that left the scene.
ESCAPED = "escaped"


@dataclass(frozen=True)
class RayRecord:
    """One ray of an event log.

    Attributes:
        ray_id: The log's ray id.
        source: The birth event's component name.
        path: Every component name after the birth, the death's included
            (``""`` a detector, ``"escaped"`` an escape).
        faces: The names of the faces hit, in order (``""`` for a detector).
        end: ``"detector"``, ``"escaped"``, or ``"absorbed at <face>"``.
        points: (k, 3) world points, one per event.
        directions: (k, 3) unit directions, one per event: the direction the
            ray arrives at the event with (at the birth, its launch
            direction). So the last free segment of a ray that left starts
            at ``points[-2]`` and runs along ``directions[-1]``; at an
            escape, ``points[-1]`` repeats ``points[-2]``.
        flux: (k,) flux carried at each event.
    """

    ray_id: int
    source: str
    path: Tuple[str, ...]
    faces: Tuple[str, ...]
    end: str
    points: np.ndarray
    directions: np.ndarray
    flux: np.ndarray


def ray_end(event_types, names) -> str:
    """How a ray ended, from its events' types and component names in order."""
    end = ESCAPED
    for event, name in zip(event_types, names):
        if event == "hit":
            end = DETECTOR if name == "" else f"absorbed at {name}"
        elif event == "death":
            if name == ESCAPED:
                end = ESCAPED
            elif name:
                end = f"absorbed at {name}"
    return end


def split_rays(events) -> List[RayRecord]:
    """One ``RayRecord`` per ray of an event log, in ray-id order.

    Args:
        events: The structured event array (``result.ray_paths["events"]``),
            in any order; a ray's own events keep their recorded order.
    """
    ev = events[np.argsort(events["ray_id"], kind="stable")]
    _, starts = np.unique(ev["ray_id"], return_index=True)
    ends = np.r_[starts[1:], len(ev)].astype(int)
    out = []
    for s, e in zip(starts, ends):
        r = ev[s:e]
        names = [str(c) for c in r["component_name"]]
        types = [str(t) for t in r["event_type"]]
        out.append(RayRecord(
            ray_id=int(r["ray_id"][0]),
            source=names[0],
            path=tuple(names[1:]),
            faces=tuple(n for t, n in zip(types, names) if t == "hit"),
            end=ray_end(types, names),
            points=np.c_[r["x"], r["y"], r["z"]].astype(float),
            directions=np.c_[r["L"], r["M"], r["N"]].astype(float),
            flux=np.asarray(r["flux"], dtype=float),
        ))
    return out


__all__ = ["DETECTOR", "ESCAPED", "RayRecord", "ray_end", "split_rays"]
