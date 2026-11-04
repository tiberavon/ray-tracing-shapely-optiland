"""
Python-specific module: test scene bridge
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

Integration test: profile.json -> Optiland NSQ scene, reproducing the
by-hand pilot notebook's physics through ray_tracing_shapely_optiland.scene_bridge instead.

Uses ray_tracing_shapely (core) only to *produce* the profile.json fixture --
scene_bridge.py itself never imports it, matching C-PROFILE-01's
tool-independence.
"""


import pytest

from optiland.materials.ideal import IdealMaterial
from optiland.nonsequential import (
    NSQScene,
    Spectrum,
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQMaterial,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend

from ray_tracing_shapely_optiland.scene_bridge import (
    build_faces_from_profile,
    span_outward_ray,
)

ray_tracing_shapely = pytest.importorskip(
    "ray_tracing_shapely", reason="core package needed to generate the profile.json fixture"
)
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


def _trace_fraction(n_sample: float, num_rays: int = 30_000, seed: int = 3) -> float:
    profile = _profile()
    glass = NSQMaterial(IdealMaterial(n=N_PRISM))
    sample = NSQMaterial(IdealMaterial(n=n_sample))

    faces = build_faces_from_profile(
        profile, interior_material=glass, material_overrides={"M": sample}
    )

    scene = NSQScene()
    spec = Spectrum.monochromatic(0.55)

    src_cs, _ = span_outward_ray(profile, "E", distance=5.0)
    scene.add_source(
        "S", src_cs, CollimatedSourceConfig(spec, total_flux=1.0, aperture_radius=1.0)
    )
    for label, face in faces.items():
        scene.add_component(label, face)

    det_cs, _ = span_outward_ray(profile, "X", distance=5.0)
    scene.add_detector(
        "D", det_cs, IrradianceDetectorConfig(width=18.0, height=18.0, num_pixels_x=32, num_pixels_y=32)
    )

    result = scene.trace(num_rays=num_rays, seed=seed, max_depth=8, backend=NumpyBackend(seed=seed))
    assert result.flux_conservation_error < 1e-6
    return result.total_flux_detected / result.total_flux_in


def test_below_target_gives_high_transmission_matching_fresnel_estimate():
    frac = _trace_fraction(1.20)
    # Two Fresnel losses at normal incidence, ideal TIR at M -- same estimate
    # the pilot notebook validated its by-hand build against.
    expected = (1 - ((N_PRISM - 1) / (N_PRISM + 1)) ** 2) ** 2
    assert frac == pytest.approx(expected, abs=0.03)


def test_above_target_leaks_through_measuring_surface():
    frac_below = _trace_fraction(1.20)
    frac_above = _trace_fraction(1.50)
    assert frac_above < 0.1 * frac_below


def test_only_base_span_is_unreached():
    profile = _profile()
    glass = NSQMaterial(IdealMaterial(n=N_PRISM))
    sample = NSQMaterial(IdealMaterial(n=1.20))
    faces = build_faces_from_profile(profile, interior_material=glass, material_overrides={"M": sample})
    assert set(faces.keys()) == {"B", "X", "M", "E"}
    assert type(faces["B"]).__name__ == "AbsorbingComponent"
    assert type(faces["X"]).__name__ == "RefractiveComponent"
    assert type(faces["M"]).__name__ == "RefractiveComponent"
    assert type(faces["E"]).__name__ == "RefractiveComponent"


# --- C-BRIDGE-01/02: the raise paths, previously undemonstrated -----------

def test_unsupported_symmetry_kind_raises_not_implemented():
    profile = _profile()
    profile["symmetry"] = {"kind": "revolve", "axis": [0, 1, 0]}
    glass = NSQMaterial(IdealMaterial(n=N_PRISM))
    with pytest.raises(NotImplementedError, match="extrude"):
        build_faces_from_profile(profile, interior_material=glass)


def test_unsupported_extrude_axis_raises_not_implemented():
    profile = _profile()
    profile["symmetry"]["extrude_axis"] = "z"
    glass = NSQMaterial(IdealMaterial(n=N_PRISM))
    with pytest.raises(NotImplementedError, match="extrude_axis"):
        build_faces_from_profile(profile, interior_material=glass)


def test_unrecognised_mechanism_raises_value_error():
    profile = _profile()
    for span in profile["spans"]:
        if span["short_label"] == "X":
            span["mechanism"] = "diffract"  # not in {refract, tir, reflect, null}
    glass = NSQMaterial(IdealMaterial(n=N_PRISM))
    with pytest.raises(ValueError, match="unrecognised mechanism"):
        build_faces_from_profile(profile, interior_material=glass)


def test_curved_span_raises_not_implemented():
    profile = _profile()
    profile["points"].append([0.0, 0.0])
    profile["normals"].append([0.0, 1.0])
    profile["spans"][0]["index_end"] = 3  # now spans 3 points, not 2
    glass = NSQMaterial(IdealMaterial(n=N_PRISM))
    with pytest.raises(NotImplementedError, match="two-point"):
        build_faces_from_profile(profile, interior_material=glass)


def test_scene_bridge_does_not_import_jupyter_viewer():
    """C-BRIDGE-05: the directional-dependency invariant -- headless/batch
    use of the bridge must never pull in the notebook/trame stack."""
    import ast
    from pathlib import Path

    for module in ("scene_bridge.py", "visualization/renderers.py"):
        path = Path(__file__).parent.parent / "ray_tracing_shapely_optiland" / module
        tree = ast.parse(path.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not any("jupyter_viewer" in name or "trame" in name for name in imported), (
            f"{module} imports something jupyter_viewer/trame-related: {imported}"
        )
