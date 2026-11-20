"""
Python-specific module: test the slice comparison flow
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
The slice comparison flow: a profile's rayfile source, sliced, against 3D
===============================================================================
A revolved singlet's profile carries a sliced rayfile source (a synthetic
LED of skew rays, written beside it). Through the flow, in process, the
per-ray table equals `compare_slice_3d`'s (the angles to 1e-9: the flow's
2D lens is rebuilt from the profile); the summary counts the rays and
bands the exit-angle difference by h. Through the generic flow tools, by
name, its jobs run in worker processes. A source that is not there, or not a
sliced rayfile, and options of the wrong shape are refused in place.
===============================================================================
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("transitions")

from optiland.materials.ideal import IdealMaterial  # noqa: E402
from optiland.nonsequential import NSQMaterial  # noqa: E402

from ray_tracing_shapely.core.scene import Scene  # noqa: E402
from ray_tracing_shapely.flows import agentic as tools  # noqa: E402
from ray_tracing_shapely.flows import kinds, persistence  # noqa: E402
from ray_tracing_shapely.optical_elements.lenses import SingletLens  # noqa: E402
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile  # noqa: E402
from ray_tracing_shapely.optical_elements.sources import slice_rayfile  # noqa: E402
from ray_tracing_shapely.optical_elements.sources.rayfile import read_rayfile, write_tm25  # noqa: E402
from ray_tracing_shapely.optical_elements.sources.source_spec import rayfile_source_spec  # noqa: E402
from ray_tracing_shapely_optiland import agentic  # noqa: E402
from ray_tracing_shapely_optiland.compare import compare_slice_3d  # noqa: E402
from ray_tracing_shapely_optiland.revolve import RevolvedProfileSolid, scene_placement  # noqa: E402
from ray_tracing_shapely_optiland.slice_comparison_flow import SliceCompareFlow  # noqa: E402

LENS = dict(radius_front=60.0, radius_back=-20.0, thickness=5.0, semi_diameter=10.0, n=1.5)
AT = (-20.0, 0.0)


def _profile(folder, n=600, max_offset_mm=0.5):
    """A singlet's profile.json with a sliced rayfile source; the rayfile beside it."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rho, phi = 2.0 * np.sqrt(rng.random(n)), 2 * np.pi * rng.random(n)
    pos = np.c_[rho * np.cos(phi), rho * np.sin(phi), np.zeros(n)]
    t, az = np.radians(15.0 * rng.random(n)), 2 * np.pi * rng.random(n)
    dirs = np.c_[np.sin(t) * np.cos(az), np.sin(t) * np.sin(az), np.cos(t)]
    write_tm25(folder / "led.TM25RAY", pos, dirs, np.full(n, 1.0 / n), spectrum_nm=[(550.0, 1.0)])
    spec = rayfile_source_spec("LED", "led.TM25RAY", AT, 0.0, max_offset_mm=max_offset_mm, base_dir=folder)
    lens = SingletLens(Scene(), **LENS)
    profile = export_profile(lens, symmetry={"kind": "revolve", "axis": lens.revolve_axis()}, physical_params={},
                             numerical_params={}, sources=[spec, {"name": "beam", "kind": "pencil",
                                                                 "position": [-20.0, 1.0], "direction_deg": 0.0}])
    (folder / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    return str(folder / "profile.json")


def _direct(path, every):
    profile = json.loads(open(path).read())
    rf = read_rayfile(str(__import__("pathlib").Path(path).parent / "led.TM25RAY"))
    slc = slice_rayfile(rf, AT, 0.0, max_offset_mm=profile["sources"][0]["slice"]["max_offset_mm"])

    def optics(scene):
        lens = SingletLens(scene, **LENS)
        scene.add_object(lens)
        return {"lens": lens}
    solid = RevolvedProfileSolid("lens", profile, NSQMaterial(IdealMaterial(n=1.5)), cs=scene_placement(profile))
    return compare_slice_3d(optics, [solid], rf, slc, AT, 0.0, every=every)


def test_every_asking_box_has_a_field(tmp_path):
    flow = SliceCompareFlow(tmp_path, render_svg=False)
    states = [s for s in flow.machine.states if s.startswith("awaiting_")]
    assert list(flow.FIELDS) == states
    assert [flow.FIELDS[s].name for s in states] == ["profiles", "source", "options", "executor", "go_ahead"]


def test_the_flow_gives_compare_slice_3d_table(tmp_path):
    path = _profile(tmp_path / "lens")
    flow = SliceCompareFlow(tmp_path / "runs", render_svg=False)
    q = flow.answer([{"path": path, "n": 1.5}])
    assert q["question"]["name"] == "source", q
    (listed,) = q["context"]["sliced_sources"]
    assert listed["source"] == "LED" and listed["rays_kept"] > 0
    assert flow.answer("LED")["question"]["name"] == "options"
    assert flow.answer({"every": 3})["question"]["name"] == "executor"
    done = flow.answer("in_process")
    assert done["state"] == "done", done["explanation"]
    table = json.loads((persistence.run_dir(tmp_path / "runs", flow.run_id) / "slice.json").read_text())
    direct = json.loads(json.dumps(_direct(path, 3)))                  # the 2D lens built, not rebuilt
    assert len(table) == len(direct)
    for got, want in zip(table, direct):
        for key in ("record", "offset_mm", "path_2d", "end_2d", "share_on_2d_path"):
            assert got[key] == want[key], key
        for key in ("angle_2d_deg", "angle_3d_deg"):
            assert (got[key] is None) == (want[key] is None) and (
                got[key] is None or abs(got[key] - want[key]) < 1e-9), key
    s = flow.result["summary"]
    assert s["rays"] == len(table) and 0 < s["met_in_3d"] <= s["rays"] and len(s["by_offset"]) == 4
    assert s["by_offset"][0]["mean_deg"] < s["by_offset"][-1]["mean_deg"]          # larger h, larger difference
    assert flow.result["source"]["name"] == "LED"


def test_wrong_answers_are_refused(tmp_path):
    path = _profile(tmp_path / "lens")
    flow = SliceCompareFlow(tmp_path / "runs", render_svg=False)
    flow.answer([{"path": path, "n": 1.5}])
    for bad, reason in (("nothing", "no source 'nothing'"), ("beam", "a slice comparison needs a sliced rayfile"),
                        (3, "takes a source's name"), ({"profile": "other", "source": "LED"}, "no source")):
        out = flow.answer(bad)
        assert not out["accepted"] and reason in out["reason"], (bad, out["reason"])
    flow.answer({"profile": "lens", "source": "LED"})
    for bad, reason in (({"rays": 3}, "unknown"), ({"every": 0}, "positive whole"), ({"repeats": 1.5}, "positive whole")):
        out = flow.answer(bad)
        assert not out["accepted"] and reason in out["reason"], (bad, out["reason"])
    flow.answer({})
    assert flow.options["every"] == max(1, -(-flow.source["rays_kept"] // 1000))


def test_an_agent_drives_it_by_name_on_worker_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(kinds, "_KINDS", dict(kinds._KINDS))
    monkeypatch.setattr(tools, "_RUNS_ROOT", tmp_path / "flow_runs")
    from ray_tracing_shapely.analysis import tool_registry
    monkeypatch.setattr(tool_registry, "_REGISTRY", list(tool_registry._REGISTRY))
    monkeypatch.setattr(tool_registry, "_EXTERNAL_TOOLS", [])
    agentic.register()
    listed = {k["kind"]: k for k in tools.flow_kinds()["data"]["kinds"]}
    assert [q["name"] for q in listed["compare_slice_3d"]["questions"]] == ["profiles", "source", "options",
                                                                           "executor", "go_ahead"]
    path = _profile(tmp_path / "lens", n=200)
    run = tools.flow_start("compare_slice_3d")["data"]["run_id"]
    for value in ([{"path": path, "n": 1.5}], "LED", {"every": 2}):
        assert tools.flow_answer(run, value)["data"]["accepted"]
    out = tools.flow_answer(run, use_default=True)["data"]                 # local_process
    assert out["state"] == "tracing_2d"
    for _ in range(4):
        waited = tools.flow_wait(run, 120)["data"]
        if waited["finished"]:
            break
    assert waited["state"] == "done", waited
    summary = tools.flow_result(run)["data"]["result"]["result"]["summary"]
    assert summary["rays"] == len(_direct(path, 2))
