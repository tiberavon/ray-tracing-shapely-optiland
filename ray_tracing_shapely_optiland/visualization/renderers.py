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

Scoped exactly as far as scene_bridge.py itself is: the flat
FinitePlaneGeometry side faces build_faces_from_profile produces, and the
PolygonPlaneGeometry end caps build_end_caps adds, are drawn.

RevolvedProfileSolid (revolve.py) has its own pair,
RevolvedProfileRenderer2D/3D, registered the same way:

    viewer2d.register_renderer(RevolvedProfileSolid, RevolvedProfileRenderer2D())
    viewer3d.register_renderer(RevolvedProfileSolid, RevolvedProfileRenderer3D())

A PlanarPlate (plate.py) has PlanarPlateRenderer2D/3D: its
outline, from the live face, at its placement.

Both draw the solid Optiland traces, not the profile it was built from: the
half-section is read back from the live faces (import_revolved_from_optiland),
so a face reshaped after construction is drawn reshaped, and both honour the
solid's placement (its ``cs``) -- including a tilted or decentred one, which
Optiland's own LensRenderer3D does not (it revolves the world-frame contour
about the world z axis). The 3D mesh is swept in numpy, in float64, not by
vtkRotationalExtrusionFilter (float32 output): its points are the traced
solid's to rounding.

RevolvedProfileSheet (revolve.py) has RevolvedProfileSheetRenderer2D/3D:
the same live read-back and placement, but span by span and unfilled, since
a sheet's spans are separate mirrors that enclose nothing.
===============================================================================
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.components.geometry.analytic.plane import FinitePlaneGeometry
from optiland.nonsequential.visualization.renderers.base import (
    ComponentRenderer2D,
    ComponentRenderer3D,
)
from optiland.nonsequential.visualization.renderers.lens import _projection_indices

from ray_tracing_shapely_optiland.geometry import PolygonPlaneGeometry

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ray_tracing_shapely_optiland.revolve import RevolvedProfileSolid
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
    """Adds one flat vtkActor per face of an ExtrudedPolygonSolid to a
    vtkRenderer: a quad per FinitePlaneGeometry side face, and a polygon per
    PolygonPlaneGeometry end cap.

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
            translation, rot = _get_transform(face.cs)
            # PolygonPlaneGeometry first: it subclasses FinitePlaneGeometry,
            # whose width/height are only its bounding rectangle.
            if isinstance(geom, PolygonPlaneGeometry):
                corners_local = np.column_stack(
                    [geom.vertices, np.zeros(len(geom.vertices))]
                )
                cell = vtk.vtkPolygon()
            elif isinstance(geom, FinitePlaneGeometry):
                w = as_float(geom.width) / 2.0
                h = as_float(geom.height) / 2.0
                corners_local = np.array(
                    [[-w, -h, 0.0], [w, -h, 0.0], [w, h, 0.0], [-w, h, 0.0]]
                )
                cell = vtk.vtkQuad()
            else:
                continue  # only flat faces are handled -- see module docstring
            corners_global = corners_local @ rot.T + translation

            points = vtk.vtkPoints()
            for pt in corners_global:
                points.InsertNextPoint(*pt)

            cell.GetPointIds().SetNumberOfIds(len(corners_global))
            for i in range(len(corners_global)):
                cell.GetPointIds().SetId(i, i)

            cells = vtk.vtkCellArray()
            cells.InsertNextCell(cell)

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


# --- RevolvedProfileSolid --------------------------------------------------------------

_LENS_FACE = (0.8, 0.8, 0.8, 0.6)
_LENS_EDGE = (0.5, 0.5, 0.5)


def revolved_half_section(component: "RevolvedProfileSolid"):
    """The live half-section of a RevolvedProfileSolid (or the spans of a
    RevolvedProfileSheet) in its local frame.

    Returns:
        ``(profile, local)``: the profile read back from the faces, and its
        points as an (N, 3) array ``(r, 0, z)`` in the solid's axis frame
        (r from the axis, z along it), one chain from the axis back to it.
        If a face can no longer be read back as a surface of revolution
        (rotated or moved off the axis), the stored profile is used and a
        warning says so: the drawing then shows what was built, not what is
        traced.
    """
    from ray_tracing_shapely_optiland.revolve import import_revolved_from_optiland  # noqa: PLC0415

    try:
        profile = import_revolved_from_optiland(component)
    except ValueError as exc:
        warnings.warn(f"{component.name}: drawing the profile it was built from: {exc}",
                      stacklevel=3)
        profile = component.profile
    frame = component.frame
    local = np.array([frame.to_world(tuple(p)) for p in profile["points"]], dtype=float)
    return profile, local


class RevolvedProfileRenderer2D(ComponentRenderer2D):
    """Draws a RevolvedProfileSolid's meridional section: the half-section
    and its mirror image about the axis, as one filled outline, with each
    span's short label.

    The section drawn is the one whose plane contains the solid's axis and
    lies along the view: the radial direction is the in-view direction
    perpendicular to the axis (any meridian of a solid of revolution is the
    same). For a solid placed with ``scene_placement`` and the ``"XZ"``
    projection that is exactly the 2D scene's outline (transposed: world z,
    the scene's second coordinate, is horizontal). Looking straight down
    the axis, the outer rim is drawn as a circle.
    """

    def render(
        self,
        component: "RevolvedProfileSolid",
        ax: "Axes",
        theme=None,
        projection: str = "YZ",
    ) -> None:
        h_idx, v_idx = _projection_indices(projection)
        profile, local = revolved_half_section(component)
        t, axis, radial = _meridian_basis(component, h_idx, v_idx)

        face_color, edge_color = _LENS_FACE, _LENS_EDGE
        if theme is not None:
            face_color = theme.parameters.get("lens.color", face_color)
            edge_color = theme.parameters.get("axes.edgecolor", edge_color)

        if radial is None:  # looking along the axis
            phi = np.linspace(0.0, 2.0 * np.pi, 129)
            r_max = float(local[:, 0].max())
            hs = t[h_idx] + r_max * np.cos(phi)
            vs = t[v_idx] + r_max * np.sin(phi)
            ax.fill(hs, vs, facecolor=face_color, zorder=2)
            ax.plot(hs, vs, color=edge_color, linewidth=1.0, zorder=3)
            return

        upper = t + np.outer(local[:, 0], radial) + np.outer(local[:, 2], axis)
        lower = t - np.outer(local[:, 0], radial) + np.outer(local[:, 2], axis)
        contour = np.vstack([upper, lower[::-1]])

        ax.fill(contour[:, h_idx], contour[:, v_idx], facecolor=face_color, zorder=2)
        ax.plot(contour[:, h_idx], contour[:, v_idx], color=edge_color, linewidth=1.0, zorder=3)
        for span in profile["spans"]:
            mid = upper[(span["index_start"] + span["index_end"] - 1) // 2]
            ax.annotate(span.get("short_label") or "", (mid[h_idx], mid[v_idx]),
                        color="firebrick", fontweight="bold", zorder=4)


class RevolvedProfileRenderer3D(ComponentRenderer3D):
    """Adds one vtkActor for a RevolvedProfileSolid: its live half-section
    revolved about the solid's own axis and placed by the solid's ``cs``.

    The mesh is built here rather than by ``vtkRotationalExtrusionFilter``
    (Optiland's revolve_contour): that filter writes float32 points whatever
    its input, which rounds a world point by ~1e-7 relative -- invisible, but
    it would make the drawing only approximately the traced solid. Here each
    contour point is swept to ``resolution`` angles about the local z axis
    and carried to the world in float64 (``t + R @ p``, Optiland's own
    transform of the solid's ``cs``); neighbouring meridians are joined by
    quads. The actor's points and bounds are world coordinates.
    """

    #: Steps around the axis; 256, as Optiland's own revolve_contour.
    resolution = 256

    def render(self, component: "RevolvedProfileSolid", renderer, theme=None) -> None:
        try:
            import vtk  # type: ignore[import]  # noqa: PLC0415
        except ImportError:
            return

        _, local = revolved_half_section(component)
        actor = _swept_actor(vtk, local, component.coordinate_system, self.resolution)

        color = (0.9, 0.9, 1.0)
        if theme is not None:
            from matplotlib.colors import to_rgb  # noqa: PLC0415

            color = to_rgb(theme.parameters.get("lens.color", "#E5E5FF"))
        prop = actor.GetProperty()
        prop.SetOpacity(0.5)
        prop.SetColor(color)
        prop.SetSpecular(1.0)
        prop.SetSpecularPower(50.0)
        renderer.AddActor(actor)


def _swept_actor(vtk, local, cs, resolution: int, azimuth_deg=None):
    """A vtkActor: the (r, 0, z) chain ``local`` swept about the local z axis
    at ``resolution`` angles, carried to the world in float64 by ``cs``
    (``t + R @ p``), neighbouring meridians joined by quads. With
    ``azimuth_deg = (start, end)`` only that range is swept, both edges
    included, and the sweep does not close."""
    n = len(local)
    if azimuth_deg is None:
        m, closed = resolution, True
        phi = 2.0 * np.pi * np.arange(m) / m
    else:
        start, end = azimuth_deg
        m = max(2, int(math.ceil(resolution * (end - start) / 360.0)) + 1)
        closed = False
        phi = np.radians(np.linspace(start, end, m))
    r, z = local[:, 0][:, None], local[:, 2][:, None]
    swept = np.stack([r * np.cos(phi), r * np.sin(phi), np.broadcast_to(z, (n, m))], axis=-1)
    t, rot = _get_transform(cs)
    world = swept.reshape(-1, 3) @ rot.T + t  # point (i, k) is row i * m + k

    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    points.SetNumberOfPoints(len(world))
    for idx, xyz in enumerate(world):
        points.SetPoint(idx, *xyz)
    quads = vtk.vtkCellArray()
    for i in range(n - 1):
        for k in range(m if closed else m - 1):
            k1 = (k + 1) % m
            quads.InsertNextCell(4, (i * m + k, i * m + k1, (i + 1) * m + k1, (i + 1) * m + k))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(quads)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


def _meridian_basis(component, h_idx: int, v_idx: int):
    """(t, axis, radial) for drawing a meridional section in a projection;
    radial is None when looking straight down the axis."""
    t, rot = _get_transform(component.coordinate_system)
    axis = rot[:, 2]
    view_normal = np.cross(np.eye(3)[h_idx], np.eye(3)[v_idx])
    radial = np.cross(view_normal, axis)
    if np.linalg.norm(radial) < 1e-9:
        return t, axis, None
    radial /= np.linalg.norm(radial)
    if radial @ rot[:, 0] < 0.0:  # draw the half-section itself on its own side
        radial = -radial
    return t, axis, radial


# --- RevolvedProfileSheet -----------------------------------------------------------------

_MIRROR_EDGE = (0.5, 0.5, 0.5)
_MIRROR_FACE_3D = (0.75, 0.75, 0.75)


def _kept(azimuth_deg, angle_deg: float, tol: float = 1e-9) -> bool:
    """Whether an azimuth range (None: the full turn) keeps ``angle_deg``."""
    if azimuth_deg is None:
        return True
    start, end = azimuth_deg
    return (angle_deg - start + tol) % 360.0 <= end - start + 2.0 * tol


class RevolvedProfileSheetRenderer2D(ComponentRenderer2D):
    """Draws a RevolvedProfileSheet's meridional section: each span, and its
    mirror image about the axis, as an open line (a sheet encloses nothing,
    so nothing is filled), with each span's short label. A sheet cut to an
    azimuth range (``symmetry.azimuth_deg``) shows a side of the section only
    where the cut keeps it, and, looking down the axis, an arc of that range.

    Read from the live faces, as for the solid. Under ``scene_placement``
    and the ``"XZ"`` projection, the lines are exactly the 2D mirrors
    (transposed). Looking down the axis, each span's outer rim is a circle.
    """

    def render(self, component, ax: "Axes", theme=None, projection: str = "YZ") -> None:
        h_idx, v_idx = _projection_indices(projection)
        profile, local = revolved_half_section(component)
        t, axis, radial = _meridian_basis(component, h_idx, v_idx)
        azimuth = getattr(component, "azimuth_deg", None)
        rot = _get_transform(component.coordinate_system)[1]
        color = _MIRROR_EDGE
        if theme is not None:
            color = theme.parameters.get("axes.edgecolor", color)
        sides = (1.0, -1.0)
        if radial is not None:
            rl = rot.T @ radial  # the drawn radial direction in the sheet's own frame
            phi_plus = math.degrees(math.atan2(rl[1], rl[0]))
            sides = tuple(sign for sign, phi in ((1.0, phi_plus), (-1.0, phi_plus + 180.0))
                          if _kept(azimuth, phi))
        for span in profile["spans"]:
            pts = local[span["index_start"]:span["index_end"]]
            if radial is None:  # looking along the axis
                start, end = azimuth if azimuth is not None else (0.0, 360.0)
                ang = np.radians(np.linspace(start, end, 129))
                ring = np.c_[np.cos(ang), np.sin(ang), np.zeros_like(ang)] @ rot.T
                r_max = float(pts[:, 0].max())
                ax.plot(t[h_idx] + r_max * ring[:, h_idx], t[v_idx] + r_max * ring[:, v_idx],
                        color=color, linewidth=2.0, zorder=3)
                continue
            for sign in sides:
                line = t + np.outer(sign * pts[:, 0], radial) + np.outer(pts[:, 2], axis)
                ax.plot(line[:, h_idx], line[:, v_idx], color=color, linewidth=2.0, zorder=3)
            if not sides:  # the cut keeps neither side of this view's section
                continue
            mid = pts[len(pts) // 2]
            label_at = t + sides[0] * mid[0] * radial + mid[2] * axis
            ax.annotate(span.get("short_label") or "", (label_at[h_idx], label_at[v_idx]),
                        color="firebrick", fontweight="bold", zorder=4)


class RevolvedProfileSheetRenderer3D(ComponentRenderer3D):
    """One vtkActor per span of a RevolvedProfileSheet: the span swept about
    the sheet's own axis and placed by its ``cs``, in float64, as for the
    solid. Spans are swept separately: a sheet's spans need not join. A cut
    sheet is swept over its azimuth range only."""

    resolution = 256

    def render(self, component, renderer, theme=None) -> None:
        try:
            import vtk  # type: ignore[import]  # noqa: PLC0415
        except ImportError:
            return

        profile, local = revolved_half_section(component)
        color = _MIRROR_FACE_3D
        if theme is not None:
            from matplotlib.colors import to_rgb  # noqa: PLC0415

            color = to_rgb(theme.parameters.get("axes.edgecolor", "#BFBFBF"))
        for span in profile["spans"]:
            pts = local[span["index_start"]:span["index_end"]]
            actor = _swept_actor(vtk, pts, component.coordinate_system, self.resolution,
                                 getattr(component, "azimuth_deg", None))
            prop = actor.GetProperty()
            prop.SetColor(color)
            prop.SetSpecular(1.0)
            prop.SetSpecularPower(80.0)
            renderer.AddActor(actor)


# --- PlanarPlate ----------------------------------------------------------------------------

_PLATE_MIRROR = (0.75, 0.75, 0.75)
_PLATE_ABSORBER = (0.2, 0.2, 0.2)


class PlanarPlateRenderer2D(ComponentRenderer2D):
    """Draws a PlanarPlate as its outline projected on the view, closed and
    lightly filled, from the live face. A plate standing on a 2D segment,
    seen in the 2D plane's projection ("XZ" under scene_placement), is
    exactly that segment."""

    def render(self, component, ax: "Axes", theme=None, projection: str = "YZ") -> None:
        h_idx, v_idx = _projection_indices(projection)
        world = component.outline_world()
        closed = np.vstack([world, world[:1]])
        color = _PLATE_ABSORBER if component.absorbing else _PLATE_MIRROR
        ax.plot(closed[:, h_idx], closed[:, v_idx], color=color, linewidth=2.0, zorder=3)
        ax.fill(world[:, h_idx], world[:, v_idx], color=color, alpha=0.25, zorder=2)


class PlanarPlateRenderer3D(ComponentRenderer3D):
    """One vtkActor for a PlanarPlate: its outline as a polygon, in the
    world, in float64, from the live face."""

    def render(self, component, renderer, theme=None) -> None:
        try:
            import vtk  # type: ignore[import]  # noqa: PLC0415
        except ImportError:
            return

        world = component.outline_world()
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        for pt in world:
            points.InsertNextPoint(*pt)
        cell = vtk.vtkPolygon()
        cell.GetPointIds().SetNumberOfIds(len(world))
        for i in range(len(world)):
            cell.GetPointIds().SetId(i, i)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(cell)
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(cells)
        triangles = vtk.vtkTriangleFilter()  # a concave outline draws correctly
        triangles.SetInputData(poly)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(triangles.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(_PLATE_ABSORBER if component.absorbing else _PLATE_MIRROR)
        prop.SetSpecular(0.0 if component.absorbing else 1.0)
        renderer.AddActor(actor)

