"""
Python-specific module: snapshot
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
SNAPSHOT -- a static picture of a 3D scene, and rays coloured by path
===============================================================================
The interactive view (jupyter_viewer.view_in_notebook) needs a live kernel:
a saved notebook shows nothing where it was. This module renders the same
vtkRenderer off-screen to PNG bytes, which a notebook keeps in its output
(``IPython.display.Image(data=...)``), at a camera given in world terms.

Optiland's NSQRays3D colours rays by source, bounce or segment. To show a
collimator's channels and its stray light, rays are coloured by the path
they took instead: ``ray_polylines`` makes one polyline per ray, with the
faces it met, from the headless splitter (``events.split_rays``), and
``polyline_actor`` draws a set of them in one colour.

vtk only: no trame, no window, no display (C-BRIDGE-05: the bridge itself
never imports this module).
===============================================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..events import ESCAPED, split_rays


def ray_polylines(events, end_length: float = 0.0) -> List[Tuple[Tuple[str, ...], np.ndarray]]:
    """One polyline per ray of an Optiland event log (``ray_paths["events"]``).

    Args:
        events: The structured event array, any order.
        end_length: A ray that escapes ends at its last event; extend it this
            far along its last direction (0: leave it).

    Returns:
        [(path, points)]: the component names the ray met after its birth
        (``""`` is a detector, ``"escaped"`` an escape), and its (k, 3) world points.
    """
    out = []
    for ray in split_rays(events):
        pts = ray.points
        if end_length > 0.0 and ray.path and ray.path[-1] == ESCAPED:
            pts = np.vstack([pts, pts[-1] + end_length * ray.directions[-1]])
        out.append((ray.path, pts))
    return out


def polyline_actor(paths: Sequence[np.ndarray], color: Tuple[float, float, float],
                   width: float = 1.0, opacity: float = 1.0):
    """A vtkActor drawing every (k, 3) polyline in ``paths`` in one colour."""
    import vtk  # noqa: PLC0415

    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    lines = vtk.vtkCellArray()
    for p in paths:
        ids = []
        for xyz in np.asarray(p, dtype=float):
            ids.append(points.InsertNextPoint(*xyz))
        if len(ids) >= 2:
            lines.InsertNextCell(len(ids), ids)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetLineWidth(width)
    prop.SetOpacity(opacity)
    return actor


def snapshot_png(renderer, size: Tuple[int, int] = (1200, 800), *,
                 focal_point: Optional[Sequence[float]] = None,
                 direction: Optional[Sequence[float]] = None,
                 view_up: Sequence[float] = (0.0, 0.0, 1.0),
                 distance: Optional[float] = None) -> bytes:
    """Render ``renderer`` off-screen and return the picture as PNG bytes.

    Args:
        size: (width, height) in pixels.
        focal_point: The world point the camera looks at (default: the scene's centre).
        direction: The world direction from the focal point to the camera
            (default: the renderer's own camera).
        view_up: The world direction shown upwards.
        distance: Camera to focal point (default: fit the scene).
    """
    import vtk  # noqa: PLC0415

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.AddRenderer(renderer)
    window.SetSize(*size)
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    if focal_point is None:
        focal_point = camera.GetFocalPoint()
    if direction is not None:
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        dist = distance if distance is not None else camera.GetDistance()
        camera.SetFocalPoint(*focal_point)
        camera.SetPosition(*(np.asarray(focal_point, dtype=float) + dist * d))
        camera.SetViewUp(*view_up)
    elif distance is not None:
        camera.SetFocalPoint(*focal_point)
        camera.SetDistance(distance)
    renderer.ResetCameraClippingRange()
    window.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(window)
    grab.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(1)
    writer.SetInputConnection(grab.GetOutputPort())
    writer.Write()
    data = writer.GetResult()
    png = bytes(memoryview(data))
    window.Finalize()
    return png


def group_polylines(polylines, groups: Dict[str, object]) -> Dict[str, List[np.ndarray]]:
    """Sort ``ray_polylines`` output into named groups: ``groups`` maps a name to a
    predicate on a ray's path; a ray goes to the first group that accepts it."""
    out: Dict[str, List[np.ndarray]] = {name: [] for name in groups}
    for path, pts in polylines:
        for name, accepts in groups.items():
            if accepts(path):
                out[name].append(pts)
                break
    return out


__all__ = ["group_polylines", "polyline_actor", "ray_polylines", "snapshot_png"]
