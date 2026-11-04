"""
Python-specific module: test nsq db
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

Tests for ray_tracing_shapely_optiland.nsq_db (P-NSQ-RUN-BUNDLE)."""

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

from ray_tracing_shapely_optiland.scene_bridge import build_faces_from_profile, span_outward_ray
from ray_tracing_shapely_optiland.nsq_db import (
    create_nsq_database,
    persist_nsq_run,
    execute_query,
    get_nsq_schema_description,
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


def _trace(n_sample, record_paths, num_rays=5000, seed=4):
    profile = _profile()
    glass = NSQMaterial(IdealMaterial(n=N_PRISM))
    sample = NSQMaterial(IdealMaterial(n=n_sample))
    faces = build_faces_from_profile(profile, interior_material=glass, material_overrides={"M": sample})

    scene = NSQScene()
    spec = Spectrum.monochromatic(0.55)
    src_cs, _ = span_outward_ray(profile, "E", distance=5.0)
    scene.add_source("S", src_cs, CollimatedSourceConfig(spec, total_flux=1.0, aperture_radius=1.0))
    for label, face in faces.items():
        scene.add_component(label, face)
    det_cs, _ = span_outward_ray(profile, "X", distance=5.0)
    scene.add_detector(
        "D", det_cs, IrradianceDetectorConfig(width=18.0, height=18.0, num_pixels_x=16, num_pixels_y=16)
    )
    result = scene.trace(
        num_rays=num_rays, seed=seed, max_depth=8, backend=NumpyBackend(seed=seed),
        record_paths=record_paths,
    )
    return scene, result


def test_create_nsq_database_raises_without_recorded_paths():
    scene, result = _trace(1.20, record_paths=False)
    with pytest.raises(ValueError, match="record_paths"):
        create_nsq_database(scene, result)


def test_create_nsq_database_row_shape():
    scene, result = _trace(1.20, record_paths=True)
    conn = create_nsq_database(scene, result)
    out = execute_query(conn, "SELECT COUNT(*) AS n FROM nsq_events")
    assert out["rows"][0][0] > 0

    out = execute_query(
        conn, "SELECT DISTINCT event_type FROM nsq_events ORDER BY event_type"
    )
    event_types = {row[0] for row in out["rows"]}
    assert event_types <= {"birth", "hit", "death"}


def test_every_ray_has_exactly_one_birth():
    scene, result = _trace(1.20, record_paths=True, num_rays=2000)
    conn = create_nsq_database(scene, result)
    # HAVING references COUNT(*) directly rather than the "AS n" alias: this
    # sqlite3 version (3.37.2) does not reliably resolve a SELECT alias
    # inside HAVING for this query shape -- confirmed by hand, not assumed --
    # and silently returns every group instead of raising, which is exactly
    # the kind of silent-wrong-result this project's own tests should not
    # reproduce by accident.
    out = execute_query(
        conn,
        "SELECT ray_id, COUNT(*) AS n FROM nsq_events "
        "WHERE event_type = 'birth' GROUP BY ray_id HAVING COUNT(*) != 1",
    )
    assert out["rows"] == []


def test_tir_vs_transmit_path_signature_via_self_join():
    """The lineage's actual value: reconstruct which rays TIR'd at M (they
    go on to hit X) versus transmitted (no further hit) from the flat
    event table -- no explicit mechanism field needed (an open gap upstream
    in Optiland's event log), just ray_id + bounce ordering."""
    scene, result = _trace(1.20, record_paths=True, num_rays=5000)  # below n_target: TIR
    conn = create_nsq_database(scene, result)
    out = execute_query(
        conn,
        """
        SELECT COUNT(DISTINCT m.ray_id) AS tir_count
        FROM nsq_events m
        JOIN nsq_events x
          ON x.ray_id = m.ray_id AND x.bounce > m.bounce AND x.component_name = 'X'
        WHERE m.component_name = 'M' AND m.event_type = 'hit'
        """,
    )
    tir_count = out["rows"][0][0]

    out = execute_query(
        conn,
        "SELECT COUNT(DISTINCT ray_id) FROM nsq_events "
        "WHERE component_name = 'M' AND event_type = 'hit'",
    )
    m_hit_count = out["rows"][0][0]

    assert m_hit_count > 0
    # Below n_target, TIR is the design-intended outcome for (nearly) every ray hitting M.
    assert tir_count / m_hit_count > 0.95


def test_get_nsq_schema_description():
    scene, result = _trace(1.20, record_paths=True, num_rays=500)
    conn = create_nsq_database(scene, result)
    schema = get_nsq_schema_description(conn)
    assert set(schema.keys()) == {"nsq_events"}
    assert schema["nsq_events"]["row_count"] > 0
    col_names = {c["name"] for c in schema["nsq_events"]["columns"]}
    assert col_names == {
        "row_id", "ray_id", "event_type", "x", "y", "z", "L", "M", "N",
        "flux", "wavelength", "bounce", "component_name",
    }


def test_detector_hits_carry_empty_component_name():
    """Documents the real Optiland API gap _DETECTOR_HIT_LABEL explains:
    a detector's registry name never reaches its own hit events."""
    scene, result = _trace(1.20, record_paths=True, num_rays=2000)
    conn = create_nsq_database(scene, result)
    out = execute_query(
        conn, "SELECT COUNT(*) FROM nsq_events WHERE component_name = '' AND event_type = 'hit'"
    )
    assert out["rows"][0][0] > 0


def test_persist_nsq_run_writes_db_and_params(tmp_path):
    scene, result = _trace(1.20, record_paths=True, num_rays=2000)
    run_dir = persist_nsq_run(
        scene, result,
        out_dir=tmp_path, design_id="refractometer_critical_path", run_id="n_sample_1.20",
        record_paths=True,
        physical_params={"n_prism": N_PRISM, "n_sample": 1.20},
        numerical_params={"num_rays": 2000},
    )
    assert (run_dir / "nsq_events.sqlite").exists()
    params = (run_dir / "params.json").read_text()
    assert '"record_paths": true' in params
