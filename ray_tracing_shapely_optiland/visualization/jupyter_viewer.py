"""
Python-specific module: jupyter viewer
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
optiland_nsq_viewer -- inline Jupyter 3D interaction
===============================================================================
NSQViewer3D.view() (optiland.nonsequential.visualization.viewer_3d) builds a
vtkRenderer via its renderer registry, then opens a *native* VTK window and
blocks on interactor.Start() -- useful at a desk, useless in a notebook cell.
This module builds the same populated vtkRenderer -- reusing the exact
per-shape rendering calls Optiland's own registry pattern already provides
(LensRenderer3D, our own ExtrudedPolygonRenderer3D, DetectorRenderer3D,
NSQRays3D) -- and hands it to a `trame` view instead, which serves it into
the notebook's own output cell over WebGL: real orbit/pan/zoom/tilt, no
external app.

**The one honest seam, found while building this:** NSQViewer3D.view() has no method that returns a
populated vtkRenderer without also opening a window -- the renderer-registry
loop, detector rendering, and ray overlay are inlined directly in view()'s
own body. `build_vtk_renderer()` below duplicates that ~20-line
orchestration (which compound to look up, which registry to check, when to
overlay rays) -- but never the actual rendering work itself, which stays
fully delegated to whatever ComponentRenderer3D/DetectorRenderer3D/NSQRays3D
instance the caller's own NSQViewer3D already has configured. A one-line
upstream ask (splitting view() into build_renderer() + a thin view()) would
remove even this duplication; not filed yet.

Distribution note: this module is why ray_tracing_shapely_optiland gained a
`notebook` extra (`trame`, `trame-vtk`, `trame-vuetify`) rather than a hard
dependency -- optiland_scene_bridge (scene_bridge.py, visualization/
renderers.py) must never import this module, so headless/batch use of the
bridge never needs a browser-facing stack at all (C-BRIDGE-05) [KC7: the
headless path never acquires a display dependency].
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from optiland.nonsequential.tracer import SimulationResult
    from optiland.nonsequential.visualization.viewer_3d import NSQViewer3D

__all__ = ["build_vtk_renderer", "view_in_notebook"]


def build_vtk_renderer(
    viewer: "NSQViewer3D",
    result: Optional["SimulationResult"] = None,
    *,
    num_rays: int = 100,
    color_by: str = "source",
    dark_mode: bool = False,
):
    """Populate a vtkRenderer from *viewer*'s scene and registry, without
    opening any window -- the piece NSQViewer3D.view() does not expose on
    its own (see module docstring).

    Args:
        viewer: An NSQViewer3D already constructed against the scene to
            render, with any custom renderers already registered via
            ``viewer.register_renderer(...)`` (e.g. ``ExtrudedPolygonSolid``
            -> ``ExtrudedPolygonRenderer3D``). Its own ``_renderer_registry``
            and ``scene`` are read directly -- the same attributes
            NSQViewer3D.view() reads internally.
        result: Optional SimulationResult. If its ``ray_paths`` is
            populated, those exact paths are overlaid (no re-trace); if
            None (or its ``ray_paths`` is None) and ``num_rays > 0``, NSQRays3D
            runs a fresh trace internally -- identical to NSQViewer3D.view()'s
            own contract for this argument.
        num_rays: Ray segments to overlay; 0 skips ray drawing entirely.
        color_by: ``'source'``, ``'bounce'``, or ``'segment'`` -- passed to
            NSQRays3D.plot() unchanged.
        dark_mode: Background gradient choice, passed to
            BaseViewer3D._make_renderer() unchanged.

    Returns:
        A populated ``vtkRenderer`` (no window, no interactor attached).
    """
    from optiland.nonsequential.visualization.renderers.detector import (  # noqa: PLC0415
        DetectorRenderer3D,
    )

    renderer = viewer._make_renderer(dark_mode)

    for compound in viewer.scene.component_registry.compounds:
        component_renderer = viewer._renderer_registry.get(type(compound))
        if component_renderer is not None:
            component_renderer.render(compound, renderer)

    det_renderer = DetectorRenderer3D()
    for det in viewer.scene.detectors:
        det_renderer.render(det, renderer)

    if num_rays > 0:
        from optiland.nonsequential.visualization.rays import NSQRays3D  # noqa: PLC0415

        existing_paths = result.ray_paths if result is not None else None
        rays = NSQRays3D(viewer.scene)
        rays.plot(renderer, num_rays=num_rays, color_by=color_by, ray_paths=existing_paths)

    return renderer


def view_in_notebook(
    viewer: "NSQViewer3D",
    result: Optional["SimulationResult"] = None,
    *,
    num_rays: int = 100,
    color_by: str = "source",
    dark_mode: bool = False,
    figsize: tuple = (900, 600),
):
    """Build a trame layout that serves *viewer*'s scene inline in Jupyter.

    Usage in a notebook cell (trame's own documented Jupyter pattern --
    see ``trame.app.jupyter``'s module docstring):

        layout = view_in_notebook(viewer, result)
        await layout.ready
        layout

    or, equivalently, in one call: ``await layout.display_cell()``.

    Args:
        viewer, result, num_rays, color_by, dark_mode: See
            build_vtk_renderer -- forwarded unchanged.
        figsize: (width, height) in pixels for the view widget.

    Returns:
        A trame ``SinglePageLayout`` instance. Not yet running -- nothing
        is served until the returned layout's ``.ready`` is awaited
        (trame starts its own server as a background task at that point,
        on an OS-assigned port, per ``trame_client``'s own ``ready``
        property); this function never blocks and never opens a native
        window, unlike ``NSQViewer3D.view()``.

    Raises:
        ImportError: If ``trame``/``trame-vtk``/``trame-vuetify`` are not
            installed -- install this package's ``notebook`` extra
            (``pip install "ray-tracing-shapely-optiland[notebook]"``).
    """
    try:
        import vtk  # type: ignore[import]  # noqa: PLC0415
        from trame.app import get_server  # noqa: PLC0415
        from trame.ui.vuetify3 import SinglePageLayout  # noqa: PLC0415
        from trame.widgets import vtk as vtk_widgets  # noqa: PLC0415
        from trame.widgets import vuetify3  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "view_in_notebook needs trame, trame-vtk, and trame-vuetify. "
            'Install with: pip install "ray-tracing-shapely-optiland[notebook]"'
        ) from exc

    renderer = build_vtk_renderer(
        viewer, result, num_rays=num_rays, color_by=color_by, dark_mode=dark_mode
    )

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.OffScreenRenderingOn()  # trame serves the pixels itself

    # A fresh, anonymously-named server per call: two live views in one
    # notebook session must not collide on trame's process-wide server
    # registry (get_server(name) returns the *same* server for a repeated
    # name -- verified against trame_client's own get_server() contract).
    server = get_server(client_type="vue3")

    with SinglePageLayout(server) as layout:
        layout.title.set_text("NSQ Scene -- 3D View")
        with layout.content:
            with vuetify3.VContainer(fluid=True, style="height: 100%; padding: 0;"):
                view = vtk_widgets.VtkRemoteView(render_window)
                view.update()

    layout.iframe_style = (
        f"border: none; width: {figsize[0]}px; height: {figsize[1]}px;"
    )
    return layout
