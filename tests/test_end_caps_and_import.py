"""
Python-specific module: test end caps and import
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

Closing an extruded solid, detecting when an open one leaks, and reading a
solid back out into a labelled profile.json:

- PolygonPlaneGeometry, the cap geometry.
- check_end_cap_crossings / ExtrudedPolygonSolid.check_containment, the
  post-trace detector for C-BRIDGE-04's silent leak.
- end_caps=..., which closes the leak; the two are checked against each other.
- the reflect mechanism, which needs a reflectance.
- import_from_optiland + interior_index + BasePrism.from_profile, the round
  trip 2D -> 3D -> 2D with labels intact.
"""

import copy
import math
import warnings

import numpy as np
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

from ray_tracing_shapely_optiland.geometry import PolygonPlaneGeometry
from ray_tracing_shapely_optiland.scene_bridge import (
    END_CAP_NAMES,
    ExtrudedPolygonSolid,
    build_faces_from_profile,
    check_end_cap_crossings,
    import_from_optiland,
    interior_index,
    span_outward_ray,
)

ray_tracing_shapely = pytest.importorskip("ray_tracing_shapely")
from ray_tracing_shapely.core.scene import Scene  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.base_prism import BasePrism  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.refractometer import RefractometerPrism  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.right_angle import RightAnglePrism  # noqa: E402

N_PRISM = 1.72
SYMMETRY = {"kind": "extrude", "plane_axes": ["x", "z"], "extrude_axis": "y", "length": 10.0}


def _refractometer_profile():
    prism = RefractometerPrism.from_critical_ray_path(
        Scene(), n_prism=N_PRISM, n_target=1.33, ray_path_length=20.0
    )
    return export_profile(
        prism,
        symmetry=copy.deepcopy(SYMMETRY),
        physical_params={"n_prism": N_PRISM},
        numerical_params={},
    )


def _refractometer_solid(profile=None, **kwargs):
    profile = profile or _refractometer_profile()
    glass = NSQMaterial(IdealMaterial(n=N_PRISM))
    sample = NSQMaterial(IdealMaterial(n=1.20))
    return profile, ExtrudedPolygonSolid(
        "prism", profile, interior_material=glass, material_overrides={"M": sample}, **kwargs
    )


def _trace(solid, profile, *, rx=0.0, num_rays=20_000, seed=5, record_paths=True):
    """The C-BRIDGE-04 reproduction: collimated source aimed in through E,
    optionally tilted by rx toward the extrusion axis."""
    scene = NSQScene()
    src_cs, ry = span_outward_ray(profile, "E", distance=5.0)
    src_cs = CoordinateSystem(x=float(src_cs.x), y=0.0, z=float(src_cs.z), rx=rx, ry=ry)
    scene.add_source(
        "S",
        src_cs,
        CollimatedSourceConfig(Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=1.0),
    )
    scene.component_registry.add(solid.name, solid)
    det_cs, _ = span_outward_ray(profile, "X", distance=5.0)
    scene.add_detector(
        "D", det_cs, IrradianceDetectorConfig(width=18.0, height=18.0, num_pixels_x=8, num_pixels_y=8)
    )
    return scene.trace(
        num_rays=num_rays,
        seed=seed,
        max_depth=12,
        backend=NumpyBackend(seed=seed),
        record_paths=record_paths,
    )


def _cap_hit_ray_ids(result):
    ev = result.ray_paths["events"]
    on_cap = (ev["event_type"] == "hit") & np.isin(ev["component_name"], list(END_CAP_NAMES))
    return sorted(int(i) for i in np.unique(ev["ray_id"][on_cap]))


# --- PolygonPlaneGeometry ------------------------------------------------------

L_SHAPE = [(0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (1.0, 1.0), (1.0, 4.0), (0.0, 4.0)]


def test_polygon_contains_is_exact_for_a_non_convex_outline():
    geom = PolygonPlaneGeometry(L_SHAPE)
    xs = np.array([0.5, 3.5, 0.5, 3.0, 2.0, -0.1])
    ys = np.array([0.5, 0.5, 3.5, 3.0, 2.0, 0.5])
    assert geom.contains(xs, ys).tolist() == [True, True, True, False, False, False]


def test_polygon_ray_intersect_misses_inside_the_bounding_rectangle():
    """(3, 3) is inside the inherited rectangle, outside the L: a miss."""
    geom = PolygonPlaneGeometry(L_SHAPE)
    origins = np.array([[0.5, 0.5, -1.0], [3.0, 3.0, -1.0]])
    directions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    t, _normals, hit, _n_geom = geom.ray_intersect(origins, directions)
    assert hit.tolist() == [True, False]
    assert t[0] == pytest.approx(1.0)
    assert np.isinf(t[1])


@pytest.mark.parametrize(
    "vertices", [[(0, 0), (1, 0)], [(0, 0), (1, 1), (2, 2)], [(0, 0), (1, 0), (0, float("nan"))]]
)
def test_polygon_rejects_degenerate_input(vertices):
    with pytest.raises(ValueError):
        PolygonPlaneGeometry(vertices)


# --- the leak, its detector, and the caps ---------------------------------------


def test_containment_check_reports_nothing_for_in_plane_sources():
    """Every committed result uses in-plane sources; the check must not cry wolf."""
    profile, solid = _refractometer_solid()
    result = _trace(solid, profile, rx=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = solid.check_containment(result)
    assert report == {"crossings": 0, "ray_ids": [], "flux": 0.0}


def test_containment_check_catches_the_tilted_source_leak():
    """C-BRIDGE-04's reproduction. Flux conservation stays clean; the check does not."""
    profile, solid = _refractometer_solid()
    result = _trace(solid, profile, rx=0.15)
    assert result.flux_conservation_error < 1e-9
    with pytest.warns(UserWarning, match="end-cap position"):
        report = solid.check_containment(result)
    assert report["crossings"] > 0
    assert report["flux"] > 0.0


@pytest.mark.parametrize("end_caps", ["refract", "absorb"])
def test_end_caps_close_the_leak(end_caps):
    profile, solid = _refractometer_solid(end_caps=end_caps)
    assert {f.name for f in solid.surfaces} == {"B", "X", "M", "E", *END_CAP_NAMES}
    result = _trace(solid, profile, rx=0.15)
    assert result.flux_conservation_error < 1e-9
    assert solid.check_containment(result)["crossings"] == 0
    assert np.abs(result.ray_paths["events"]["y"]).max() <= SYMMETRY["length"] / 2 + 1e-9
    assert _cap_hit_ray_ids(result), "the tilted beam should reach a cap"


def test_caps_face_into_the_solid():
    """Local +z (n_geom, the material_back side) must point inward, as on
    every side face; a flipped cap swaps which medium a ray is leaving."""
    from optiland.nonsequential.components.base import _get_transform

    _, solid = _refractometer_solid(end_caps="refract")
    for name, sign in zip(END_CAP_NAMES, (-1.0, 1.0)):
        translation, rotation = _get_transform(solid.face(name).cs)
        assert translation[1] == pytest.approx(sign * SYMMETRY["length"] / 2)
        assert rotation[1, 2] == pytest.approx(-sign)  # local +z along world -sign*y


def test_grazing_light_tirs_off_a_refracting_cap():
    """Inside the glass the tilted beam meets a cap about 85 degrees off its
    normal, far past the 35.5-degree critical angle for n=1.72: every cap hit
    must be followed by another hit, never by an escape. A cap facing the
    wrong way treats the ray as entering glass from vacuum and lets it out."""
    profile, solid = _refractometer_solid(end_caps="refract")
    ev = _trace(solid, profile, rx=0.15).ray_paths["events"]
    ev = ev[np.argsort(ev["ray_id"], kind="stable")]
    on_cap = np.isin(ev["component_name"][:-1], list(END_CAP_NAMES)) & (
        ev["ray_id"][1:] == ev["ray_id"][:-1]
    )
    assert on_cap.any()
    following = ev[1:][on_cap]
    assert not np.any(following["component_name"] == "escaped")


def test_detector_and_caps_agree_ray_for_ray():
    """The same rays the check flags on the open solid are the rays an
    absorbing cap stops on the closed one: same seed, same paths up to the
    first cap crossing, so each remedy validates the other."""
    profile, open_solid = _refractometer_solid()
    leaked = open_solid.check_containment(_trace(open_solid, profile, rx=0.15), warn=False)
    _, capped = _refractometer_solid(profile, end_caps="absorb")
    stopped = _cap_hit_ray_ids(_trace(capped, profile, rx=0.15))
    assert leaked["ray_ids"] == stopped


def test_check_containment_needs_recorded_paths():
    profile, solid = _refractometer_solid()
    result = _trace(solid, profile, num_rays=200, record_paths=False)
    with pytest.raises(ValueError, match="record_paths"):
        solid.check_containment(result)


def test_check_end_cap_crossings_on_an_empty_log():
    report = check_end_cap_crossings(np.array([]), _refractometer_profile())
    assert report["crossings"] == 0


def test_unknown_end_cap_mechanism_raises():
    with pytest.raises(ValueError, match="end cap mechanism"):
        _refractometer_solid(end_caps="reflect")


def test_end_caps_trace_and_differentiate_under_torch():
    """A polygon cap is analytic: the torch backend traces it and the
    gradient through the glass index survives, where a mesh cap would not."""
    torch = pytest.importorskip("torch")
    import optiland.backend as be
    from optiland.nonsequential.backends.torch_backend import TorchBackend

    profile = _refractometer_profile()
    be.set_backend("torch")
    try:
        n = torch.tensor(N_PRISM, requires_grad=True)
        solid = ExtrudedPolygonSolid(
            "prism",
            profile,
            interior_material=NSQMaterial(IdealMaterial(n=n)),
            material_overrides={"M": NSQMaterial(IdealMaterial(n=1.20))},
            end_caps="refract",
        )
        scene = NSQScene()
        src_cs, ry = span_outward_ray(profile, "E", distance=5.0)
        scene.add_source(
            "S",
            CoordinateSystem(x=float(src_cs.x), z=float(src_cs.z), rx=0.15, ry=ry),
            CollimatedSourceConfig(Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=1.0),
        )
        scene.component_registry.add(solid.name, solid)
        det_cs, _ = span_outward_ray(profile, "X", distance=5.0)
        scene.add_detector(
            "D", det_cs, IrradianceDetectorConfig(width=18.0, height=18.0, num_pixels_x=4, num_pixels_y=4)
        )
        result = scene.trace(
            num_rays=2000, seed=1, max_depth=10, backend=TorchBackend(seed=1), record_paths=True
        )
        result.detectors["D"].data.sum().backward()
        assert n.grad is not None and math.isfinite(float(n.grad)) and float(n.grad) != 0.0
        assert _cap_hit_ray_ids(result)
        assert solid.check_containment(result)["crossings"] == 0
    finally:
        be.set_backend("numpy")


def test_3d_renderer_draws_caps_as_polygons():
    vtk = pytest.importorskip("vtk")
    from ray_tracing_shapely_optiland.visualization.renderers import ExtrudedPolygonRenderer3D

    _, solid = _refractometer_solid(end_caps="absorb")
    renderer = vtk.vtkRenderer()
    ExtrudedPolygonRenderer3D().render(solid, renderer)
    cell_sizes = []
    actors = renderer.GetActors()
    actors.InitTraversal()
    while (actor := actors.GetNextActor()) is not None:
        cell_sizes.append(actor.GetMapper().GetInput().GetCell(0).GetNumberOfPoints())
    # Four quad side faces, two caps with the trapezoid's four vertices.
    assert sorted(cell_sizes) == [4] * 6
    assert len(cell_sizes) == 6


# --- the reflect mechanism ------------------------------------------------------


def _right_angle_profile(mechanism_h="tir"):
    profile = export_profile(
        RightAnglePrism(Scene(), leg_length=30.0, n=1.5),
        symmetry=copy.deepcopy(SYMMETRY),
        physical_params={},
        numerical_params={},
    )
    for span in profile["spans"]:
        if span["short_label"] == "H":
            span["mechanism"] = mechanism_h
    return profile


def _right_angle_transmission(profile, **kwargs):
    solid = ExtrudedPolygonSolid("fold", profile, NSQMaterial(IdealMaterial(n=1.5)), **kwargs)
    scene = NSQScene()
    src_cs, _ = span_outward_ray(profile, "E", distance=5.0)
    scene.add_source(
        "S", src_cs, CollimatedSourceConfig(Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=1.0)
    )
    scene.component_registry.add(solid.name, solid)
    det_cs, _ = span_outward_ray(profile, "X", distance=5.0)
    scene.add_detector(
        "D", det_cs, IrradianceDetectorConfig(width=30.0, height=30.0, num_pixels_x=4, num_pixels_y=4)
    )
    result = scene.trace(num_rays=20_000, seed=1, max_depth=10, backend=NumpyBackend(seed=1))
    return result.total_flux_detected / result.total_flux_in


def test_reflect_span_needs_a_reflectance():
    with pytest.raises(ValueError, match="reflectance"):
        build_faces_from_profile(
            _right_angle_profile("reflect"), interior_material=NSQMaterial(IdealMaterial(n=1.5))
        )


def test_a_perfect_mirror_hypotenuse_folds_like_tir():
    tir = _right_angle_transmission(_right_angle_profile("tir"))
    mirror = _right_angle_transmission(_right_angle_profile("reflect"), reflectances={"H": 1.0})
    half = _right_angle_transmission(_right_angle_profile("reflect"), reflectances={"H": 0.5})
    expected = (1 - ((1.5 - 1) / (1.5 + 1)) ** 2) ** 2  # two normal-incidence Fresnel losses
    assert tir == pytest.approx(expected, abs=0.02)
    assert mirror == pytest.approx(tir, abs=1e-12)
    assert half == pytest.approx(0.5 * tir, abs=0.02)


# --- import: 3D back to a labelled 2D profile -----------------------------------


def test_import_reproduces_the_exported_profile():
    profile, solid = _refractometer_solid(end_caps="refract")
    imported = import_from_optiland(solid)
    assert imported["spans"] == profile["spans"]
    assert imported["symmetry"] == profile["symmetry"]
    assert imported["construction_params"] == profile["construction_params"]
    for a, b in zip(imported["points"], profile["points"]):
        assert a == pytest.approx(b, abs=1e-12)
    for a, b in zip(imported["normals"], profile["normals"]):
        assert a == pytest.approx(b, abs=1e-12)
    assert imported is not solid.profile


def test_interior_index_recovers_the_glass():
    _, solid = _refractometer_solid()
    assert interior_index(solid) == pytest.approx(N_PRISM)


def test_round_trip_reads_live_geometry_and_keeps_labels():
    """Stand-in for a 3D polish: the solid's faces are replaced by those of a
    10% larger outline. The import must return the new geometry under the
    original labels, and the 2D rebuild must stay addressable by them."""
    profile, solid = _refractometer_solid()
    grown = copy.deepcopy(profile)
    grown["points"] = [[1.1 * a, 1.1 * b] for a, b in profile["points"]]
    solid._faces = build_faces_from_profile(
        grown, NSQMaterial(IdealMaterial(n=N_PRISM)), material_overrides={"M": NSQMaterial(IdealMaterial(n=1.2))}
    )

    imported = import_from_optiland(solid)
    for a, b in zip(imported["points"], grown["points"]):
        assert a == pytest.approx(b, abs=1e-12)

    prism = BasePrism.from_profile(Scene(), imported, n=interior_index(solid))
    m = prism.find_edge_by_functional_label("M")
    assert prism.get_functional_label(m) == ("M", "Measuring Surface")
    assert prism.get_edge_role(m).value == "sample_medium"
    assert prism._edge_mechanisms["M"].value == "tir"
    assert prism.refIndex == pytest.approx(N_PRISM)
    m_span = next(s for s in profile["spans"] if s["short_label"] == "M")
    original_length = math.dist(
        profile["points"][m_span["index_start"]], profile["points"][m_span["index_start"] + 1]
    )
    assert prism.get_edge_length(m) == pytest.approx(1.1 * original_length)


def test_import_refuses_a_face_moved_out_of_plane():
    _, solid = _refractometer_solid()
    face = solid.face("X")
    face.cs = CoordinateSystem(x=float(face.cs.x), y=0.5, z=float(face.cs.z), ry=float(face.cs.ry))
    with pytest.raises(ValueError, match="profile plane"):
        import_from_optiland(solid)


def test_a_single_moved_face_cannot_be_rebuilt_silently():
    """Moving one face alone opens the outline: the import reads it faithfully,
    and the 2D rebuild refuses rather than inventing a closing edge."""
    _, solid = _refractometer_solid()
    face = solid.face("X")
    face.cs = CoordinateSystem(x=float(face.cs.x) + 0.3, y=0.0, z=float(face.cs.z), ry=float(face.cs.ry))
    imported = import_from_optiland(solid)
    with pytest.raises(ValueError, match="does not close"):
        BasePrism.from_profile(Scene(), imported, n=N_PRISM)
