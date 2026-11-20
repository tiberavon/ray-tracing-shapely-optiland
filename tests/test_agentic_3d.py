"""
Python-specific module: test agentic 3d
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
Read-only agentic tools over a traced 3D scene, registered from outside the core
===============================================================================
A revolved singlet lit by a 35 deg cone from 20 mm: most light goes through
both faces to the detector, some misses the lens, some meets its rim. The
tools must name the faces as the event log does, describe the log, query it
read-only, and group the rays by path with numbers that match a count made
directly from the trace's own event array.
===============================================================================
"""

from __future__ import annotations

import math
import sqlite3
import sys
import warnings
from collections import Counter

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.materials.ideal import IdealMaterial
from optiland.nonsequential import IrradianceDetectorConfig, NSQMaterial, NSQScene
from optiland.nonsequential.backends.numpy_backend import NumpyBackend

from ray_tracing_shapely.analysis import tool_registry
from ray_tracing_shapely.analysis import agentic_tools
from ray_tracing_shapely.analysis.agentic_tools import (clear_context, list_contexts, load_run_context, query_rays,
                                                        save_context_tool, set_active_context,
                                                        set_context_from_result)
from ray_tracing_shapely.analysis.tool_registry import check_tool_lists, get_agentic_tools, get_exposed_tools
from ray_tracing_shapely.core.scene import Scene
from ray_tracing_shapely.core.simulator import Simulator
from ray_tracing_shapely.optical_elements.lenses import SingletLens
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile
from ray_tracing_shapely_optiland import agentic
from ray_tracing_shapely_optiland.nsq_db import create_nsq_database
from ray_tracing_shapely_optiland.revolve import RevolvedProfileSolid, scene_placement
from ray_tracing_shapely_optiland.sources import add_source_3d

NUM_RAYS = 2000


@pytest.fixture(scope="module")
def traced():
    lens = SingletLens(Scene(), radius_front=60.0, radius_back=-20.0, thickness=5.0, semi_diameter=10.0, n=1.5)
    profile = export_profile(lens, symmetry={"kind": "revolve", "axis": lens.revolve_axis()},
                             physical_params={}, numerical_params={})
    scene = NSQScene()
    scene.component_registry.add("L1", RevolvedProfileSolid("L1", profile, NSQMaterial(IdealMaterial(n=1.5)),
                                                            cs=scene_placement(profile)))
    add_source_3d(scene, {"name": "led", "kind": "point", "position": [-20.0, 0.0], "direction_deg": 0.0,
                          "emission": "uniform", "half_angle_deg": 35.0})
    scene.add_detector("far", CoordinateSystem(x=200.0, ry=math.pi / 2),
                       IrradianceDetectorConfig(width=80.0, height=80.0, num_pixels_x=4, num_pixels_y=4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = scene.trace(num_rays=NUM_RAYS, seed=5, max_depth=30, backend=NumpyBackend(seed=5),
                             record_paths=True)
    return scene, result


@pytest.fixture
def context(traced):
    scene, result = traced
    agentic.set_nsq_context(scene, result)
    yield scene, result
    clear_context()


def test_without_a_context_every_tool_says_what_to_call():
    clear_context()
    for tool in (agentic.describe_nsq_scene, agentic.describe_nsq_schema, agentic.nsq_path_spectrum):
        out = tool()
        assert out["status"] == "error" and "set_nsq_context" in out["message"]
    assert agentic.query_nsq("SELECT 1")["status"] == "error"


def test_describe_scene_names_faces_as_the_log_does(context):
    data = agentic.describe_nsq_scene()["data"]
    (lens,) = data["components"]
    assert (lens["name"], lens["symmetry"], lens["body"]) == ("L1", "revolve", "solid")
    faces = {f["label"]: f for f in lens["faces"]}
    assert set(faces) == {"S1", "S2", "RIM"}
    assert (faces["S1"]["name"], faces["S1"]["role"], faces["S1"]["mechanism"], faces["S1"]["surface_kind"]) == \
        ("L1.S1", "entrance", "refract", "conic")
    assert faces["RIM"]["surface_kind"] == "frustum" and faces["RIM"]["role"] == "blocker"
    assert data["sources"] == [{"name": "led", "type": "PointSource", "logged_as": "led"}]
    assert [d["name"] for d in data["detectors"]] == ["far"]
    assert data["event_log"]["rays"] == NUM_RAYS
    logged = {r[0] for r in agentic.query_nsq("SELECT DISTINCT component_name FROM nsq_events "
                                              "WHERE event_type = 'hit'")["data"]["rows"]}
    assert logged - {""} <= {f["name"] for f in lens["faces"]}


def test_schema_describes_every_column(context):
    cols = agentic.describe_nsq_schema()["data"]["nsq_events"]["columns"]
    assert {c["name"] for c in cols} >= {"ray_id", "event_type", "component_name", "flux"}
    assert all(c["meaning"] for c in cols)


def test_query_is_read_only_and_capped(context):
    out = agentic.query_nsq("SELECT COUNT(*) AS n FROM nsq_events")
    assert out["status"] == "ok" and out["data"]["rows"][0][0] > NUM_RAYS
    assert agentic.query_nsq("DELETE FROM nsq_events")["status"] == "error"
    assert agentic.query_nsq("SELECT nope FROM nsq_events")["message"].startswith("SQL error")
    assert agentic.query_nsq("SELECT * FROM nsq_events")["data"]["row_count"] <= 200


def test_path_spectrum_matches_a_direct_count(context):
    scene, result = context
    data = agentic.nsq_path_spectrum(top_n=50)["data"]
    assert data["rays"] == NUM_RAYS and data["others"]["rays"] == 0
    assert sum(p["share"] for p in data["paths"]) == pytest.approx(1.0)
    main = data["paths"][0]
    assert main["path"] == ["led", "L1.S1", "L1.S2", "detector far"] and main["end"] == "detector far"

    ev = result.ray_paths["events"]
    ev = ev[np.argsort(ev["ray_id"], kind="stable")]
    direct = Counter()
    for rid in np.unique(ev["ray_id"]):
        e = ev[ev["ray_id"] == rid]
        direct[tuple(str(c) for c, t in zip(e["component_name"], e["event_type"]) if t != "death")] += 1
    assert main["rays"] == direct[("led", "L1.S1", "L1.S2", "")]
    assert len(data["paths"]) >= 3                       # through, missed, rim: the cone overfills the lens
    ends = Counter(p["end"] for p in data["paths"] for _ in range(p["rays"]))
    assert ends["escaped"] > 0 and sum(ends.values()) == NUM_RAYS


def test_a_persisted_log_is_read_only(traced, tmp_path):
    scene, result = traced
    path = tmp_path / "nsq_events.sqlite"
    create_nsq_database(scene, result, db_path=path).close()
    agentic.set_nsq_context(scene, db=path)
    try:
        assert agentic.nsq_path_spectrum()["data"]["rays"] == NUM_RAYS
        conn = agentic._context()["db"]
        with pytest.raises(Exception, match="readonly"):
            conn.execute("DELETE FROM nsq_events")
    finally:
        clear_context()


def test_a_run_folder_loads_its_2d_run_and_its_3d_log_into_one_context(traced, tmp_path, monkeypatch):
    monkeypatch.setattr(agentic_tools, "_RUN_FILE_LOADERS", {})
    scene, result = traced
    lens_2d = Scene()
    lens = SingletLens(lens_2d, radius_front=60.0, radius_back=-20.0, thickness=5.0, semi_diameter=10.0, n=1.5)
    lens.name = "L1"
    lens_2d.add_object(lens)
    set_context_from_result(lens_2d, Simulator(lens_2d).run_with_result())
    run_dir = save_context_tool(str(tmp_path), "lens", "r1")["data"]["run_dir"]
    create_nsq_database(scene, result, db_path=f"{run_dir}/nsq_events.sqlite").close()

    assert load_run_context(run_dir, "both")["data"]["loaded"] == ["rays.sqlite"]   # the bridge not registered
    clear_context("both")
    agentic_tools.register_run_file_loader("nsq_events.sqlite", agentic.load_nsq_context)
    out = load_run_context(run_dir, "both")
    assert out["data"]["loaded"] == ["rays.sqlite", "nsq_events.sqlite"]
    (both,) = [c for c in list_contexts()["data"]["contexts"] if c["name"] == "both"]
    assert both["extensions"] == ["optiland.nsq"]

    set_active_context("both")
    assert query_rays("SELECT COUNT(*) FROM rays")["status"] == "ok"
    data = agentic.describe_nsq_scene()["data"]                      # from the log: no live 3D scene
    assert [c["name"] for c in data["components"]] == ["L1"]
    assert {f["name"] for f in data["components"][0]["faces"]} <= {"L1.S1", "L1.S2", "L1.RIM"}
    assert data["sources"][0]["logged_as"] == "led" and data["event_log"]["rays"] == NUM_RAYS
    assert "without its scene" in data["note"]
    spectrum = agentic.nsq_path_spectrum()["data"]
    assert spectrum["rays"] == NUM_RAYS and spectrum["paths"][0]["path"] == ["led", "L1.S1", "L1.S2", "detector"]
    conn = agentic._context()["db"]
    clear_context("both")
    with pytest.raises(sqlite3.ProgrammingError):                    # closed with its context
        conn.execute("SELECT 1")


def test_register_adds_the_five_tools_once(monkeypatch):
    monkeypatch.setattr(agentic_tools, "_RUN_FILE_LOADERS", {})
    monkeypatch.setattr(tool_registry, "_REGISTRY", list(tool_registry._REGISTRY))
    monkeypatch.setattr(tool_registry, "_EXTERNAL_TOOLS", [])
    monkeypatch.setattr(tool_registry, "_WITHDRAWN_TOOLS", dict(tool_registry._WITHDRAWN_TOOLS))
    monkeypatch.delitem(sys.modules, "ray_tracing_shapely.analysis.mcp_server", raising=False)
    before = len(get_exposed_tools())
    assert agentic.register() is True
    assert agentic.register() is False
    names = [t["name"] for t in get_agentic_tools()][-5:]
    assert names == ["describe_nsq_scene", "describe_nsq_schema", "query_nsq", "nsq_path_spectrum",
                     "compare_2d_3d_summary"]
    assert len(get_exposed_tools()) == before + 5 and check_tool_lists() == []
    assert agentic_tools._RUN_FILE_LOADERS == {"nsq_events.sqlite": agentic.load_nsq_context}
