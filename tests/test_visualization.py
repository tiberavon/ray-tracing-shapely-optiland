"""
Python-specific module: test visualization
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

Tests for ExtrudedPolygonSolid + its generic 2D/3D renderers."""

import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.materials.ideal import IdealMaterial
from optiland.nonsequential import (
    NSQScene,
    Spectrum,
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQMaterial,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.visualization.viewer_2d import NSQViewer2D

from ray_tracing_shapely_optiland.scene_bridge import ExtrudedPolygonSolid, span_outward_ray
from ray_tracing_shapely_optiland.visualization.renderers import (
    ExtrudedPolygonRenderer2D,
    ExtrudedPolygonRenderer3D,
)

ray_tracing_shapely = pytest.importorskip("ray_tracing_shapely")
from ray_tracing_shapely.core.scene import Scene  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.refractometer import RefractometerPrism  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile  # noqa: E402

N_PRISM = 1.72
N_TARGET = 1.33


def _profile():
    scene = Scene()
    prism = RefractometerPrism.from_critical_ray_path(
        scene, n_prism=N_PRISM, n_target=N_TARGET, ray_path_length=20.0
    )
    return export_profile(
        prism,
        symmetry={"kind": "extrude", "plane_axes": ["x", "z"], "extrude_axis": "y", "length": 10.0},
        physical_params={"n_prism": N_PRISM, "n_target": N_TARGET},
        numerical_params={},
    )


def _solid(n_sample=1.20):
    profile = _profile()
    glass = NSQMaterial(IdealMaterial(n=N_PRISM))
    sample = NSQMaterial(IdealMaterial(n=n_sample))
    return profile, ExtrudedPolygonSolid(
        "prism", profile, interior_material=glass, material_overrides={"M": sample}
    )


def test_extruded_polygon_solid_satisfies_compound_component_interface():
    profile, solid = _solid()
    assert solid.name == "prism"
    assert {f.name for f in solid.surfaces} == {"B", "X", "M", "E"}
    assert solid.coordinate_system is not None
    assert type(solid.face("X")).__name__ == "RefractiveComponent"


def test_registering_via_component_registry_traces_correctly():
    profile, solid = _solid(n_sample=1.20)
    scene = NSQScene()
    spec = Spectrum.monochromatic(0.55)
    src_cs, _ = span_outward_ray(profile, "E", distance=5.0)
    scene.add_source("S", src_cs, CollimatedSourceConfig(spec, total_flux=1.0, aperture_radius=1.0))

    scene.component_registry.add(solid.name, solid)

    det_cs, _ = span_outward_ray(profile, "X", distance=5.0)
    scene.add_detector(
        "D", det_cs, IrradianceDetectorConfig(width=18.0, height=18.0, num_pixels_x=32, num_pixels_y=32)
    )
    result = scene.trace(num_rays=30_000, seed=5, max_depth=8, backend=NumpyBackend(seed=5))
    assert result.flux_conservation_error < 1e-6
    frac = result.total_flux_detected / result.total_flux_in
    expected = (1 - ((N_PRISM - 1) / (N_PRISM + 1)) ** 2) ** 2
    assert frac == pytest.approx(expected, abs=0.03)


def test_renderer_registry_recognises_extruded_polygon_solid():
    """This is the actual point of the CompoundComponent wrapper: a raw
    per-face scene.add_component() loop is invisible to the renderer
    registry; registering the compound directly
    is not."""
    profile, solid = _solid()
    scene = NSQScene()
    scene.component_registry.add(solid.name, solid)

    viewer = NSQViewer2D(scene)
    viewer.register_renderer(ExtrudedPolygonSolid, ExtrudedPolygonRenderer2D())

    assert type(solid) in viewer._renderer_registry
    fig, ax = viewer.view(num_rays=0, projection="XZ")
    # A real polygon patch was drawn (fill + outline), not just an empty axes.
    assert len(ax.patches) >= 1 or len(ax.lines) >= 1


def test_3d_renderer_adds_one_actor_per_face_offscreen():
    vtk = pytest.importorskip("vtk")
    profile, solid = _solid()

    renderer = vtk.vtkRenderer()
    r3d = ExtrudedPolygonRenderer3D()
    r3d.render(solid, renderer)

    actors = renderer.GetActors()
    actors.InitTraversal()
    count = 0
    while actors.GetNextActor() is not None:
        count += 1
    assert count == 4  # B, X, M, E -- one flat quad each
