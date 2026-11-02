"""
Python-specific module: renderers
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

Python translation Copyright 2026 ray-tracing-shapely authors and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

===============================================================================
GENERIC RENDERERS FOR ExtrudedPolygonSolid
===============================================================================
optiland.nonsequential.visualization's renderer registry is keyed by
compound-component *class*, and every built-in renderer (LensRenderer2D/3D,
MirrorRenderer2D/3D) is shaped for one specific, fixed surface set (front/
back/edge[/rim], or a single revolved sag profile). ExtrudedPolygonSolid has
neither -- an arbitrary number of flat faces, one per profile.json span --
so these renderers are deliberately generic over face count and geometry
type instead of assuming either.

Register against a viewer exactly as any other custom renderer:

    from ray_tracing_shapely_optiland.scene_bridge import ExtrudedPolygonSolid
    from ray_tracing_shapely_optiland.visualization.renderers import (
        ExtrudedPolygonRenderer2D, ExtrudedPolygonRenderer3D,
    )
    from optiland.nonsequential.visualization.viewer_2d import NSQViewer2D
    from optiland.nonsequential.visualization.viewer_3d import NSQViewer3D

    viewer2d = NSQViewer2D(scene)
    viewer2d.register_renderer(ExtrudedPolygonSolid, ExtrudedPolygonRenderer2D())
    viewer2d.view(projection="XZ")   # NOT the default "YZ" -- see below

    viewer3d = NSQViewer3D(scene)
    viewer3d.register_renderer(ExtrudedPolygonSolid, ExtrudedPolygonRenderer3D())
    viewer3d.view()                  # opens a native VTK window (interactor.Start())

Scoped exactly as far as scene_bridge.py itself is: only the flat
FinitePlaneGeometry faces build_faces_from_profile can produce today are
drawn. A future mesh-capped end or a curved span would need its own
geometry-type branch in the 3D renderer's face
loop -- not attempted here, since neither exists in scene_bridge.py yet.
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential.components.geometry.analytic.plane import FinitePlaneGeometry
from optiland.nonsequential.visualization.renderers.base import (
    ComponentRenderer2D,
    ComponentRenderer3D,
)
from optiland.nonsequential.visualization.renderers.lens import _projection_indices

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ray_tracing_shapely_optiland.scene_bridge import ExtrudedPolygonSolid


def _point_to_world_xyz(point2d, symmetry) -> tuple:
    """Local copy of scene_bridge._point_to_world's plane_axes=['x','z'],
    extrude_axis='y' mapping -- kept local rather than imported so this
    module has no dependency on scene_bridge's own internal (underscore)
    helpers, only on ExtrudedPolygonSolid's public ``profile`` attribute.
    """
    a, b = point2d
    return (a, 0.0, b)


class ExtrudedPolygonRenderer2D(ComponentRenderer2D):
    """Draws an ExtrudedPolygonSolid's 2D cross-section as a filled polygon.

    Reads ``component.profile["points"]`` directly -- this is the straight-
    edge polygon case, so unlike LensRenderer2D's sag-profile reconstruction
    there is no curve to evaluate, just the stored vertices.

    **Only the ``projection="XZ"`` view is meaningful** for a device built
    with plane_axes=['x','z'] (the only case scene_bridge.py supports
    today, per its own _require_supported_symmetry): the default
    ``projection="YZ"`` plots world z against world y, and every point on
    this solid has y == 0 (that is the extrusion axis), which draws a
    degenerate flat line. Pass ``projection="XZ"`` explicitly.
    """

    def render(
        self,
        component: "ExtrudedPolygonSolid",
        ax: "Axes",
        theme=None,
        projection: str = "YZ",
    ) -> None:
        h_idx, v_idx = _projection_indices(projection)
        profile = component.profile
        pts = np.array(
            [_point_to_world_xyz(p, profile["symmetry"]) for p in profile["points"]]
        )
        pts = np.vstack([pts, pts[0]])  # close the polygon

        color = "steelblue" if theme is None else getattr(theme, "component_color", "steelblue")
        ax.plot(pts[:, h_idx], pts[:, v_idx], "-", color=color, linewidth=1.5, zorder=3)
        ax.fill(pts[:-1, h_idx], pts[:-1, v_idx], color=color, alpha=0.08, zorder=2)

        for span in profile["spans"]:
            p1 = _point_to_world_xyz(profile["points"][span["index_start"]], profile["symmetry"])
            p2 = _point_to_world_xyz(profile["points"][span["index_start"] + 1], profile["symmetry"])
            mid = ((p1[h_idx] + p2[h_idx]) / 2.0, (p1[v_idx] + p2[v_idx]) / 2.0)
            ax.annotate(span["short_label"], mid, color="firebrick", fontweight="bold", zorder=4)


class ExtrudedPolygonRenderer3D(ComponentRenderer3D):
    """Adds one flat vtkActor per FinitePlaneGeometry face of an
    ExtrudedPolygonSolid to a vtkRenderer.

    Quad-building follows optiland.nonsequential.visualization.renderers
    .detector.DetectorRenderer3D's own pattern exactly (four corners in the
    face's local frame -> world via _get_transform -> vtkQuad/vtkPolyData/
    vtkPolyDataMapper/vtkActor) -- the same shape Optiland's own detector
    renderer already uses for a flat rectangular surface, since a
    FinitePlaneGeometry face *is* that same shape.
    """

    def render(self, component: "ExtrudedPolygonSolid", renderer, theme=None) -> None:
        try:
            import vtk  # type: ignore[import]  # noqa: PLC0415
        except ImportError:
            return

        from optiland.nonsequential._utils import as_float  # noqa: PLC0415
        from optiland.nonsequential.components.base import _get_transform  # noqa: PLC0415

        for face in component.surfaces:
            geom = face.geometry
            if not isinstance(geom, FinitePlaneGeometry):
                continue  # only the flat-face case is handled -- see module docstring

            w = as_float(geom.width) / 2.0
            h = as_float(geom.height) / 2.0
            translation, rot = _get_transform(face.cs)

            corners_local = np.array(
                [[-w, -h, 0.0], [w, -h, 0.0], [w, h, 0.0], [-w, h, 0.0]]
            )
            corners_global = corners_local @ rot.T + translation

            points = vtk.vtkPoints()
            for pt in corners_global:
                points.InsertNextPoint(*pt)

            quad = vtk.vtkQuad()
            for i in range(4):
                quad.GetPointIds().SetId(i, i)

            cells = vtk.vtkCellArray()
            cells.InsertNextCell(quad)

            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            poly.SetPolys(cells)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.6, 0.75, 0.95)
            actor.GetProperty().SetOpacity(0.55)

            renderer.AddActor(actor)
