"""
Python-specific module: test jupyter viewer
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

Tests for ray_tracing_shapely_optiland.visualization.jupyter_viewer
(optiland_nsq_viewer component).

Verifies real behaviour, not just "doesn't crash": build_vtk_renderer's
actor count against a known scene, and view_in_notebook's actual server
lifecycle (a real port bound, layout.server.running flips True, clean
shutdown) -- run via asyncio.run() rather than pytest-asyncio, since this
package does not otherwise need that dependency.
"""

import asyncio

import pytest

from optiland.nonsequential import (
    NSQScene,
    Spectrum,
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
)
from optiland.nonsequential.visualization.viewer_3d import NSQViewer3D

from ray_tracing_shapely_optiland.scene_bridge import ExtrudedPolygonSolid, span_outward_ray
from ray_tracing_shapely_optiland.visualization.renderers import ExtrudedPolygonRenderer3D

pytest.importorskip("trame")
pytest.importorskip("vtk")

from ray_tracing_shapely_optiland.visualization.jupyter_viewer import (  # noqa: E402
    build_vtk_renderer,
    view_in_notebook,
)

ray_tracing_shapely = pytest.importorskip("ray_tracing_shapely")
from optiland.materials.ideal import IdealMaterial  # noqa: E402
from optiland.nonsequential import NSQMaterial, VACUUM  # noqa: E402
from ray_tracing_shapely.core.scene import Scene  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.refractometer import RefractometerPrism  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile  # noqa: E402

N_PRISM = 1.72
N_TARGET = 1.33


def _viewer_and_scene(num_pixels=8):
    scene2d = Scene()
    prism = RefractometerPrism.from_critical_ray_path(
        scene2d, n_prism=N_PRISM, n_target=N_TARGET, ray_path_length=20.0
    )
    profile = export_profile(
        prism,
        symmetry={"kind": "extrude", "plane_axes": ["x", "z"], "extrude_axis": "y", "length": 10.0},
        physical_params={}, numerical_params={},
    )
    glass = NSQMaterial(IdealMaterial(n=N_PRISM))
    sample = NSQMaterial(IdealMaterial(n=1.20))
    solid = ExtrudedPolygonSolid("prism", profile, interior_material=glass, material_overrides={"M": sample})

    scene = NSQScene()
    spec = Spectrum.monochromatic(0.55)
    src_cs, _ = span_outward_ray(profile, "E", distance=5.0)
    scene.add_source("S", src_cs, CollimatedSourceConfig(spec, total_flux=1.0, aperture_radius=1.0))
    scene.component_registry.add(solid.name, solid)
    det_cs, _ = span_outward_ray(profile, "X", distance=5.0)
    scene.add_detector(
        "D", det_cs,
        IrradianceDetectorConfig(width=18.0, height=18.0, num_pixels_x=num_pixels, num_pixels_y=num_pixels),
    )

    viewer = NSQViewer3D(scene)
    viewer.register_renderer(ExtrudedPolygonSolid, ExtrudedPolygonRenderer3D())
    return viewer


def test_build_vtk_renderer_geometry_only_actor_count():
    viewer = _viewer_and_scene()
    renderer = build_vtk_renderer(viewer, num_rays=0)
    actors = renderer.GetActors()
    actors.InitTraversal()
    count = 0
    while actors.GetNextActor() is not None:
        count += 1
    assert count == 5  # 4 faces (B, X, M, E) + 1 detector


def test_build_vtk_renderer_matches_nsqviewer3d_registry():
    """build_vtk_renderer must read the *same* registry NSQViewer3D.view()
    would use, not a fresh/default one."""
    viewer = _viewer_and_scene()
    assert ExtrudedPolygonSolid in viewer._renderer_registry
    renderer = build_vtk_renderer(viewer, num_rays=0)
    # If the custom renderer had not been picked up, only the detector
    # (1 actor) would be present -- the compound would be silently skipped
    # (the original finding about unregistered types).
    actors = renderer.GetActors()
    actors.InitTraversal()
    count = sum(1 for _ in iter(actors.GetNextActor, None))
    assert count == 5


def test_view_in_notebook_real_server_lifecycle():
    viewer = _viewer_and_scene()
    layout = view_in_notebook(viewer, num_rays=0, figsize=(640, 480))
    assert layout.server.running is False  # not started yet -- view_in_notebook never blocks

    async def _run():
        await layout.ready
        assert layout.server.running is True
        assert isinstance(layout.server.port, int) and layout.server.port > 0
        html = layout._jupyter_content()
        assert "iframe" in html
        assert "width: 640px; height: 480px" in html
        await layout.server.stop()
        assert layout.server.running is False

    asyncio.run(_run())


def test_view_in_notebook_raises_helpful_error_without_trame(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "trame.app":
            raise ImportError("simulated missing trame")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    viewer = _viewer_and_scene()
    with pytest.raises(ImportError, match="notebook"):
        view_in_notebook(viewer, num_rays=0)
