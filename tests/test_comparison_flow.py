"""
Python-specific module: test the 2D-against-3D comparison flow
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
The comparison flow: profile files in, the same table as compare_2d_3d out
===============================================================================
A revolved singlet's profile is written to a file and compared through the
flow, its traces run in process and in worker processes. The table equals
`compare_2d_3d`'s with the section and the solid built from the same file;
so does a revolved V of two mirrors, a sheet, given its reflectance.
On ``modal``, against a stand-in for the Modal client, the same table, an
estimate with Modal's start-up and a cost; ``modal`` is refused where the
client is not installed. Answers of the wrong shape are refused in place; a profile given the wrong
material (a sheet an index) sends the flow back. An estimate over the minute is asked;
"no" goes back to the options. Through the generic flow tools, an agent
starts the flow by name and waits on its jobs with flow_wait.
===============================================================================
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("transitions")

from ray_tracing_shapely.core.scene import Scene  # noqa: E402
from ray_tracing_shapely.core.scene_objs.mirror.mirror import Mirror  # noqa: E402
from ray_tracing_shapely.flows import agentic as tools  # noqa: E402
from ray_tracing_shapely.flows import kinds, persistence  # noqa: E402
from ray_tracing_shapely.optical_elements.lenses import SingletLens  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.profile_export import (export_profile,  # noqa: E402
                                                                         export_sheet_profile)
from ray_tracing_shapely_optiland import agentic, comparison_flow  # noqa: E402
from ray_tracing_shapely_optiland.compare import compare_2d_3d  # noqa: E402
from ray_tracing_shapely_optiland.compare_jobs import components_3d, optics_2d, read_profile  # noqa: E402
from ray_tracing_shapely_optiland.comparison_flow import Compare2D3DFlow  # noqa: E402

LENS = dict(radius_front=60.0, radius_back=-20.0, thickness=5.0, semi_diameter=10.0, n=1.5)
RAYS = [{"position": [-20.0, h], "direction_deg": a} for h in (-8.0, -3.0, 0.5, 3.0, 6.0) for a in (0.0, 5.0)]
FAN = {"from": [-20.0, 1.0], "angles_deg": [-10.0, 10.0, 5.0]}
IMAGE = {"point": [60.0, 0.0], "normal_deg": 0.0}


def _lens_profile(folder):
    lens = SingletLens(Scene(), **LENS)
    profile = export_profile(lens, symmetry={"kind": "revolve", "axis": lens.revolve_axis()},
                             physical_params={}, numerical_params={})
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    return str(folder / "profile.json")


def _sheet_profile(path):
    scene = Scene()
    walls = []
    for sign, name in ((1, "upper"), (-1, "lower")):
        m = Mirror(scene, {"p1": {"x": 100.0, "y": sign * 20.0}, "p2": {"x": 160.0, "y": sign * 8.0}})
        m.name = name
        walls.append(m)
    profile = export_sheet_profile(walls, symmetry={"kind": "revolve", "axis": {"point": [100.0, 0.0],
                                                                                "direction": [1.0, 0.0]}},
                                   physical_params={}, numerical_params={})
    path.write_text(json.dumps(profile), encoding="utf-8")
    return str(path)


def _direct(path, rays, material=None, name="lens", **options):
    loaded = [{"path": path, "name": name, **(material or {"n": LENS["n"]}), "profile": read_profile(path)}]
    pairs = [(r["position"], r["direction_deg"]) for r in rays]
    return compare_2d_3d(optics_2d(loaded), components_3d(loaded), pairs, **{"plane": IMAGE, **options})


def test_every_asking_box_has_a_field_and_one_trigger(tmp_path):
    flow = Compare2D3DFlow(tmp_path, render_svg=False)
    states = [s for s in flow.machine.states if s.startswith("awaiting_")]
    assert list(flow.FIELDS) == states
    for state in states:
        assert len([t for t in flow.machine.get_triggers(state) if t.startswith("provide_")]) == 1, state


def test_the_flow_gives_compare_2d_3d_table_in_process(tmp_path):
    path = _lens_profile(tmp_path / "lens")
    flow = Compare2D3DFlow(tmp_path / "runs", render_svg=False)
    q = flow.answer([{"path": path, "n": LENS["n"]}])
    assert q["accepted"] and q["question"]["name"] == "rays"
    (described,) = q["context"]["profiles"]
    assert described["name"] == "lens" and described["symmetry"] == "revolve" and "S1" in described["faces"]
    assert flow.answer(RAYS + [FAN])["question"]["name"] == "options"
    assert flow.answer({"plane": IMAGE})["question"]["name"] == "executor"
    done = flow.answer("in_process")
    assert done["state"] == "done" and done["finished"], done["explanation"]
    rays = RAYS + [{"position": [-20.0, 1.0], "direction_deg": a} for a in (-10.0, -5.0, 0.0, 5.0, 10.0)]
    direct = _direct(path, rays)
    table = json.loads((persistence.run_dir(tmp_path / "runs", flow.run_id) / "rays.json").read_text())
    assert table == json.loads(json.dumps(direct.rays))
    assert flow.result["summary"] == direct.summary() and flow.result["summary"]["agree"] == len(rays)
    states = [e["state"] for e in persistence.read_events(tmp_path / "runs", flow.run_id)]
    assert [s for s in ("loading", "awaiting_go_ahead", "tracing_2d", "tracing_3d", "comparing", "done")
            if s in states] == ["loading", "awaiting_go_ahead", "tracing_2d", "tracing_3d", "comparing", "done"]


def test_a_mirror_sheet_compares_given_its_reflectance(tmp_path):
    path = _sheet_profile(tmp_path / "v.json")
    rays = [{"position": [95.0, h], "direction_deg": a} for a in (0.0, 8.0, 17.0) for h in (-15.0, -6.0, 3.0, 12.0)]
    flow = Compare2D3DFlow(tmp_path / "runs", render_svg=False)
    q = flow.answer([{"path": path, "reflectance": 0.9, "name": "cone"}])
    assert q["question"]["name"] == "rays", q
    (described,) = q["context"]["profiles"]
    assert described["body"] == "sheet" and described["reflectance"] == 0.9
    for v in (rays, {"max_depth": 100}):
        flow.answer(v)
    assert flow.answer("in_process")["state"] == "done"
    direct = _direct(path, rays, {"reflectance": 0.9}, name="cone", plane=None, max_depth=100)
    assert flow.result["summary"] == direct.summary() and direct.summary()["agree"] == len(rays)
    assert flow.result["profiles"] == [{"name": "cone", "path": path, "reflectance": 0.9}]
    assert any(len(r["path_2d"]) >= 2 for r in direct.rays)                    # some reflect twice or more


def test_wrong_answers_are_refused_or_sent_back(tmp_path):
    path = _lens_profile(tmp_path / "lens")
    flow = Compare2D3DFlow(tmp_path / "runs", render_svg=False)
    for bad, reason in (("lens.json", "non-empty list"), ([{"path": path}], "optionally"),
                        ([{"path": str(tmp_path / "none.json"), "n": 1.5}], "no file"),
                        ([{"path": path, "n": 0.5}], "at least 1"),
                        ([{"path": path, "n": 1.5}, {"path": path, "n": 1.5}], "share a name")):
        out = flow.answer(bad)
        assert not out["accepted"] and reason in out["reason"], (bad, out["reason"])
    back = flow.answer([{"path": _sheet_profile(tmp_path / "v.json"), "n": 1.0}])
    assert back["accepted"] and back["sent_back"]["to"] == "profiles"
    assert 'give it "reflectance"' in back["sent_back"]["reason"]
    back = flow.answer([{"path": path, "reflectance": 0.9}])
    assert back["sent_back"]["to"] == "profiles" and 'give it "n"' in back["sent_back"]["reason"]
    back = flow.answer([{"path": path, "n": 1.5, "reflectance": 1.0}])
    assert back["sent_back"]["to"] == "profiles" and "no mirror-coated span" in back["sent_back"]["reason"]
    for bad, reason in (([{"path": path}], "takes \"path\""),
                        ([{"path": path, "reflectance": 1.5}], "(0, 1]")):
        out = flow.answer(bad)
        assert not out["accepted"] and reason in out["reason"], (bad, out["reason"])
    flow.answer([{"path": path, "n": 1.5, "name": "L1"}])
    for bad, reason in (([], "non-empty list"), ([{"position": [0, 0]}], "a ray is"),
                        ([{"from": [0, 0], "angles_deg": [10, -10, 5]}], "first <= last"),
                        ([{"from": [0, 0], "angles_deg": [0, 6000, 1]}], "more than 5000")):
        out = flow.answer(bad)
        assert not out["accepted"] and reason in out["reason"], (bad, out["reason"])
    flow.answer(RAYS[:2])
    for bad, reason in (({"rays": 3}, "unknown"), ({"rays_per_pencil": 0}, "positive whole"),
                        ({"min_share": 1.0}, "[0, 1)"), ({"plane": [0, 0]}, "options.plane")):
        out = flow.answer(bad)
        assert not out["accepted"] and reason in out["reason"], (bad, out["reason"])
    flow.answer_default()
    assert not flow.answer("gpu")["accepted"]


def test_an_estimate_over_the_minute_is_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison_flow, "ASK_OVER_S", 1e-3)
    path = _lens_profile(tmp_path / "lens")
    flow = Compare2D3DFlow(tmp_path / "runs", render_svg=False)
    for v in ([{"path": path, "n": 1.5}], RAYS[:3], {}):
        flow.answer(v)
    asked = flow.answer("in_process")
    assert asked["question"]["name"] == "go_ahead" and asked["context"]["rays_3d"] == 24
    back = flow.answer(False)
    assert back["accepted"] and back["sent_back"]["to"] == "options"
    flow.answer({"rays_per_pencil": 4})
    assert flow.answer("in_process")["context"]["rays_3d"] == 12
    assert flow.answer(True)["state"] == "done"


def test_an_agent_drives_it_by_name_and_waits_on_worker_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(kinds, "_KINDS", dict(kinds._KINDS))
    monkeypatch.setattr(tools, "_RUNS_ROOT", tmp_path / "flow_runs")
    from ray_tracing_shapely.analysis import tool_registry
    monkeypatch.setattr(tool_registry, "_REGISTRY", list(tool_registry._REGISTRY))
    monkeypatch.setattr(tool_registry, "_EXTERNAL_TOOLS", [])
    agentic.register()
    listed = {k["kind"]: k for k in tools.flow_kinds()["data"]["kinds"]}
    assert [q["name"] for q in listed["compare_2d_3d"]["questions"]] == ["profiles", "rays", "options",
                                                                        "executor", "go_ahead"]
    path = _lens_profile(tmp_path / "lens")
    run = tools.flow_start("compare_2d_3d")["data"]["run_id"]
    for value in ([{"path": path, "n": 1.5}], RAYS, {"plane": IMAGE}):
        assert tools.flow_answer(run, value)["data"]["accepted"]
    out = tools.flow_answer(run, use_default=True)["data"]               # local_process
    assert out["state"] == "tracing_2d" and out["job"]["handle"]["executor"] == "local_process"
    for _ in range(4):
        waited = tools.flow_wait(run, 120)["data"]
        if waited["finished"]:
            break
    assert waited["state"] == "done", waited
    result = tools.flow_result(run)["data"]["result"]["result"]
    assert result["summary"] == _direct(path, RAYS).summary()
    listed_run = next(r for r in tools.flow_list()["data"]["runs"] if r["run_id"] == run)
    assert not listed_run.get("flags")


def test_the_traces_on_modal_give_the_same_table(tmp_path, monkeypatch):
    import sys
    from ray_tracing_shapely.developer_tests._fake_modal import FakeModal, reset
    monkeypatch.setitem(sys.modules, "modal", FakeModal)
    reset()
    monkeypatch.setattr(comparison_flow, "MODAL_APP", "run")                # the stand-in runs a call at once
    path = _lens_profile(tmp_path / "lens")
    flow = Compare2D3DFlow(tmp_path / "runs", render_svg=False)
    for v in ([{"path": path, "n": LENS["n"]}], RAYS, {"plane": IMAGE}):
        flow.answer(v)
    out = flow.answer("modal")
    assert out["accepted"] and flow.estimate["executor"] == "modal" and flow.estimate["cost_usd_est"] > 0.0
    assert flow.estimate["tracing_3d_s"] >= comparison_flow.JOB_START_S["modal"]
    assert flow.wait(30.0, poll_s=0.05)["state"] == "done"
    assert flow.result["summary"] == _direct(path, RAYS).summary() and flow.result["executor"] == "modal"
    assert len(FakeModal.calls) == 2                                          # the 2D job and the 3D job

    monkeypatch.setattr(comparison_flow, "_modal_installed", lambda: False)
    other = Compare2D3DFlow(tmp_path / "runs", render_svg=False)
    for v in ([{"path": path, "n": LENS["n"]}], RAYS, {}):
        other.answer(v)
    refused = other.answer("modal")
    assert not refused["accepted"] and "not installed" in refused["reason"]


def test_a_profile_s_pencil_sources_are_rays(tmp_path):
    lens = SingletLens(Scene(), **LENS)
    sources = [{"name": "beam", "kind": "pencil", "position": [-20.0, 2.0], "direction_deg": 3.0},
               {"name": "led", "kind": "point", "position": [-20.0, 0.0], "direction_deg": 0.0,
                "emission": "uniform", "half_angle_deg": 20.0}]
    profile = export_profile(lens, symmetry={"kind": "revolve", "axis": lens.revolve_axis()}, physical_params={},
                             numerical_params={}, sources=sources)
    (tmp_path / "lens").mkdir()
    path = tmp_path / "lens" / "profile.json"
    path.write_text(json.dumps(profile))
    flow = Compare2D3DFlow(tmp_path / "runs", render_svg=False)
    flow.answer([{"path": str(path), "n": LENS["n"]}])
    out = flow.answer([{"source": "led"}])
    assert not out["accepted"] and "a cone of rays" in out["reason"] and '"from": [-20.0, 0.0]' in out["reason"]
    assert not flow.answer([{"source": "nope"}])["accepted"]
    assert flow.answer([{"source": "beam"}, RAYS[0]])["accepted"]
    assert flow.rays == [{"position": [-20.0, 2.0], "direction_deg": 3.0}, RAYS[0]]


def test_a_solid_with_a_mirror_face_compares_given_its_reflectance(tmp_path):
    """The SMS RX: refracting R, mirror-coated X (its glass's own coating). Given n and a reflectance, the
    flow's summary equals compare_2d_3d's, every meridional ray agreeing; without the reflectance it is
    sent back."""
    from ray_tracing_shapely.optical_elements.prisms.rx_concentrator import RXConcentrator
    rx = RXConcentrator(Scene(), (-0.5, 0.0), (0.5, 0.0), theta_deg=2.0, n=1.5)
    profile = export_profile(rx, symmetry={"kind": "revolve", "axis": {"point": [0.0, 0.0], "direction": [0.0, 1.0]}},
                             physical_params={}, numerical_params={})
    (tmp_path / "rx").mkdir()
    path = str(tmp_path / "rx" / "profile.json")
    (tmp_path / "rx" / "profile.json").write_text(json.dumps(profile))
    rays = [{"position": [x, 1.0], "direction_deg": -90.0} for x in (-10.0, -6.0, -2.0, 2.0, 6.0, 10.0)]
    flow = Compare2D3DFlow(tmp_path / "runs", render_svg=False)
    back = flow.answer([{"path": path, "n": 1.5}])
    assert back["sent_back"]["to"] == "profiles" and "mirror-coated spans" in back["sent_back"]["reason"]
    assert flow.answer([{"path": path, "n": 1.5, "reflectance": 1.0}])["question"]["name"] == "rays"
    flow.answer(rays)
    flow.answer({})
    assert flow.answer("in_process")["state"] == "done"
    direct = _direct(path, rays, {"n": 1.5, "reflectance": 1.0}, name="rx", plane=None)
    assert flow.result["summary"] == direct.summary() and direct.summary()["agree"] == len(rays)
