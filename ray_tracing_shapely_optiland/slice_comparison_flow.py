"""
Python-specific module: slice_comparison_flow
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
SLICE COMPARISON FLOW -- a rayfile's 2D slice against its own records in 3D
===============================================================================
`compare.compare_slice_3d` as a flow an agent can drive, from profile files:
the comparison flow's boxes, with the rays replaced by a rayfile source the
profiles carry (written by the slicing flow):

    start -> awaiting_profiles -> loading -> awaiting_rays (the source)
          -> awaiting_options -> awaiting_executor -> awaiting_go_ahead
          -> tracing_2d -> tracing_3d -> comparing -> done

- **awaiting_rays** asks for the source: its name, or ``{"profile", "source"}``.
  The question's context lists the profiles' sliced rayfile sources. The
  file is found from the profile's folder and checked against the digest the
  spec recorded.
- **awaiting_options:** ``every`` (compare every ``every``-th kept ray;
  default: as many as keep ``max_compared``), ``max_compared`` (1000),
  ``repeats`` (3D rays per record, 4), ``max_depth``, ``seed``, ``max_rays_2d``.
- **tracing_2d**, **tracing_3d:** `compare_jobs.slice_2d_job` and
  `slice_3d_job`, on any executor; the rayfile goes with them.
- **comparing:** `compare.compare_slice_paths`. ``done`` holds a summary: the
  rays compared, how many met the same faces in 3D, the exit-angle difference
  of the meridional rays (h near 0, which must agree to the tracing
  precision) and its mean and largest by band of h. The table is written
  beside it, as ``slice.json`` in the run's folder.
===============================================================================
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ray_tracing_shapely.flows import persistence
from ray_tracing_shapely.flows.budgets import ASK_OVER_S, estimate_s, format_duration
from ray_tracing_shapely.flows.executors import Job
from ray_tracing_shapely.flows.fields import FlowField, explain_field

from .compare import MainPath2D, PencilRay3D, compare_slice_paths
from .compare_jobs import slice_of
from .comparison_flow import (COMPARISON_FIELDS, COMPARISON_STATES, COMPARISON_TRANSITIONS, JOB_START_S,
                              MODAL_SPEED, Compare2D3DFlow, _material)

#: The options and their defaults. ``every`` None: as many kept rays as keep ``max_compared``.
SLICE_OPTIONS: Dict[str, Any] = {"every": None, "max_compared": 1000, "repeats": 4, "max_depth": 30, "seed": 1,
                                 "max_rays_2d": 10_000}
#: A ray is meridional when its dropped offset h is within this, in mm.
MERIDIONAL_MM = 1e-6
_JOBS = "ray_tracing_shapely_optiland.compare_jobs"

SLICE_FIELDS: Dict[str, FlowField] = {
    **COMPARISON_FIELDS,
    "awaiting_rays": FlowField(
        "source", "json",
        "Which rayfile source? Its name, or {\"profile\": a profile's name, \"source\": its source's name}. The "
        "profiles' sliced rayfile sources are in the question's context.",
        "each kept ray of its slice is traced in 2D, its own record in 3D, and their exit angles compared",
        'e.g. "R608, 50% kept"'),
    "awaiting_options": FlowField(
        "options", "json",
        "Options? {} for the defaults, or any of: every (compare every every-th kept ray; default: enough to "
        "keep max_compared), max_compared (1000), repeats (3D rays per record, 4), max_depth (30), seed (1), "
        "max_rays_2d (10000).",
        "compare_slice_3d's arguments",
        'e.g. {} or {"every": 10, "repeats": 8}', default={}),
}


class SliceCompareFlow(Compare2D3DFlow):
    """A rayfile's 2D slice against its own records in 3D, from profile files: `compare_slice_3d` as a
    flow. Driven as `Compare2D3DFlow` is. At ``done``, ``self.result`` holds the summary and
    ``self.rows`` the per-ray table."""

    FIELDS = SLICE_FIELDS
    PURPOSE = ("Compare a rayfile's 2D slice with the same records traced in 3D, from profile files: each kept "
               "ray is traced in 2D with its offset h dropped, its record unturned in 3D, and their exit angles "
               "from the axis compared, by h.")

    def __init__(self, base_dir: Path, run_id: Optional[str] = None, render_svg: bool = True) -> None:
        self._clear()
        super(Compare2D3DFlow, self).__init__(
            name="compare_slice_3d", states=COMPARISON_STATES, transitions_list=COMPARISON_TRANSITIONS,
            initial="start", base_dir=base_dir, run_id=run_id, render_svg=render_svg)
        self.begin()

    def _clear(self) -> None:
        super()._clear()
        self.options = dict(SLICE_OPTIONS)
        self.source: Optional[Dict[str, Any]] = None
        self.rows: Optional[List[Dict[str, Any]]] = None
        self.picked: Optional[List[int]] = None

    # -- the source ----------------------------------------------------------------

    def _sliced_sources(self) -> List[Dict[str, Any]]:
        out = []
        for p in self.profiles:
            for spec in p["profile"].get("sources") or []:
                if spec.get("kind") == "rayfile" and "slice" in spec:
                    out.append({"profile": p["name"], "source": spec["name"], **spec["slice"],
                                "rayfile": spec["rayfile"]["path"]})
        return out

    def question_context(self) -> Optional[Dict[str, Any]]:
        if self.state == "awaiting_rays" and self.profiles:
            return {"profiles": [self._describe(p) for p in self.profiles], "sliced_sources": self._sliced_sources()}
        if self.state in ("awaiting_options", "awaiting_executor") and self.source:
            return {"source": {k: self.source[k] for k in ("profile", "name", "rays_kept")}, "options": self.options}
        return super().question_context()

    def _provide_rays(self, value: Any) -> None:
        if isinstance(value, str):
            wanted = {"source": value}
        elif isinstance(value, dict) and set(value) <= {"profile", "source"} and "source" in value:
            wanted = dict(value)
        else:
            raise ValueError(f"source takes a source's name, or {{\"profile\", \"source\"}}, got {value!r}")
        found = [(p, spec) for p in self.profiles for spec in p["profile"].get("sources") or []
                 if spec.get("name") == wanted["source"] and wanted.get("profile") in (None, p["name"])]
        if not found:
            names = sorted({s["source"] for s in self._sliced_sources()})
            raise ValueError(f"no source {wanted['source']!r} in the profiles; their sliced rayfile sources: {names}")
        if len(found) > 1:
            raise ValueError(f"several profiles carry a source {wanted['source']!r}: name the profile too")
        (p, spec), = found
        if spec.get("kind") != "rayfile" or "slice" not in spec:
            raise ValueError(f"source {spec['name']!r} is a {spec.get('kind')} source"
                             + (" without a slice" if spec.get("kind") == "rayfile" else "")
                             + ": a slice comparison needs a sliced rayfile (the slicing flow writes one)")
        path = Path(spec["rayfile"]["path"])
        if not path.is_absolute():
            path = Path(p["path"]).resolve().parent / path
        if not path.is_file():
            raise ValueError(f"source {spec['name']!r}: no rayfile at {path}")
        source = json.loads(json.dumps(spec))
        source["rayfile"]["path"] = str(path)
        slice_of(source)                                          # the digest and the slice, checked here
        self.source = {"profile": p["name"], "name": spec["name"], "spec": source,
                       "rays_kept": int(spec["slice"]["rays_kept"]), "rayfile_path": str(path)}

    # -- the options ---------------------------------------------------------------

    def _provide_options(self, value: Any = None) -> None:
        value = {} if value is None else value
        if not isinstance(value, dict):
            raise ValueError(f"options takes an object, got {value!r}")
        unknown = sorted(set(value) - set(SLICE_OPTIONS))
        if unknown:
            raise ValueError(f"options: unknown {unknown}; known {sorted(SLICE_OPTIONS)}")
        out = dict(SLICE_OPTIONS)
        for key in ("every", "max_compared", "repeats", "max_depth", "seed", "max_rays_2d"):
            if key in value and not (key == "every" and value[key] is None):
                v = value[key]
                if isinstance(v, bool) or not isinstance(v, int) or v < (0 if key == "seed" else 1):
                    raise ValueError(f"options.{key} must be a {'whole number' if key == 'seed' else 'positive whole number'}, got {v!r}")
                out[key] = v
        if out["every"] is None:
            out["every"] = max(1, math.ceil(self.source["rays_kept"] / out["max_compared"]))
        self.options = out

    def _compared(self) -> int:
        return math.ceil(self.source["rays_kept"] / self.options["every"])

    # -- the estimate and the jobs ---------------------------------------------------

    def _estimate(self) -> Dict[str, Any]:
        n, per = self._compared(), self.options["repeats"]
        start = JOB_START_S[self.executor_kind]
        speed = MODAL_SPEED if self.executor_kind == "modal" else {}
        s2 = estimate_s("simulate_2d_ray", n) * speed.get("simulate_2d_ray", 1.0) + start
        s3 = estimate_s("trace_3d_ray", n * per) * speed.get("trace_3d_ray", 1.0) + start
        out = {"rays": n, "rays_3d": n * per, "tracing_2d_s": s2, "tracing_3d_s": s3, "total_s": s2 + s3,
               "total": format_duration(s2 + s3), "executor": self.executor_kind, "asks_over_s": ASK_OVER_S}
        if self.executor_kind == "modal":
            from ray_tracing_shapely.flows.modal_executor import CPU_USD_PER_CORE_S, MEMORY_USD_PER_GIB_S  # noqa: PLC0415
            out["cost_usd_est"] = round((s2 + s3) * (CPU_USD_PER_CORE_S + 2.0 * MEMORY_USD_PER_GIB_S), 6)
        return out

    def _job_inputs(self) -> Dict[str, Any]:
        return {"profiles": self._job_profiles(), "source": self.source["spec"], "every": self.options["every"]}

    def _job_files(self) -> Dict[str, str]:
        return {**{p["name"]: p["path"] for p in self._job_profiles()}, "rayfile": self.source["rayfile_path"]}

    def on_enter_tracing_2d(self, *_a: Any, **_k: Any) -> None:
        self.set_explanation(explain_field("tracing_2d", f"Tracing {self._compared()} of the slice's rays in 2D.",
                                           "each alone, its brightest branch followed", ""))
        job = Job(f"{_JOBS}:slice_2d_job", {**self._job_inputs(), "max_rays_2d": self.options["max_rays_2d"]},
                  files=self._job_files(), budget_s=self.estimate["tracing_2d_s"], label="the slice in 2D")
        self.submit_step(self._executor(), job, on_done="traced_2d", ask_over_s=None)

    def _keep_2d(self, result: Dict[str, Any], *_a: Any, **_k: Any) -> None:
        self.picked = result["pick"]
        self.main_2d = [MainPath2D.from_dict(m) for m in result["main"]]

    def on_enter_tracing_3d(self, *_a: Any, **_k: Any) -> None:
        o = self.options
        self.set_explanation(explain_field("tracing_3d", f"Tracing {self._compared()} records, {o['repeats']} "
                                           "rays each, in 3D.", "one Optiland trace", ""))
        job = Job(f"{_JOBS}:slice_3d_job", {**self._job_inputs(), **{k: o[k] for k in ("repeats", "max_depth", "seed")}},
                  files=self._job_files(), budget_s=self.estimate["tracing_3d_s"], label="the records in 3D")
        self.submit_step(self._executor(), job, on_done="traced_3d", ask_over_s=None)

    def _keep_3d(self, result: Dict[str, Any], *_a: Any, **_k: Any) -> None:
        self.pencils = [[PencilRay3D.from_dict(r) for r in rr] for rr in result["records_3d"]]

    def on_enter_comparing(self, *_a: Any, **_k: Any) -> None:
        _, slc = slice_of(self.source["spec"])
        self.rows = compare_slice_paths(slc, self.picked, self.main_2d, self.pencils,
                                        float(self.source["spec"]["direction_deg"]))
        folder = persistence.run_dir(Path(self.base_dir), self.run_id)
        (folder / "slice.json").write_text(json.dumps(self.rows, indent=1), encoding="utf-8")
        summary = slice_summary(self.rows)
        self.result = {"summary": summary, "rays_file": "slice.json",
                       "source": {k: self.source[k] for k in ("profile", "name", "rays_kept")},
                       "profiles": [{"name": p["name"], "path": p["path"], **_material(p)} for p in self.profiles],
                       "options": self.options, "executor": self.executor_kind}
        self.emit_event(message=f"{summary['met_in_3d']} of {summary['rays']} rays met the same faces in 3D; "
                                f"meridional rays differ by at most {summary['meridional']['worst_deg']} deg")
        self._advance()

    def on_enter_done(self, *_a: Any, **_k: Any) -> None:
        self.finish(self.result)
        self.set_explanation(explain_field("done", "The slice comparison is ready.",
                                           "result: the summary; slice.json: the per-ray table", ""))
        self.emit_event(message="done")


def slice_summary(rows: List[Dict[str, Any]], bands: int = 4) -> Dict[str, Any]:
    """The table's summary: rays compared, those whose 3D rays met the same faces, the exit-angle difference
    of the meridional ones, and the difference's mean and largest by band of |h| (quantiles)."""
    met = [r for r in rows if r["angle_3d_deg"] is not None]
    diff = np.array([abs(r["angle_3d_deg"] - r["angle_2d_deg"]) for r in met])
    h = np.array([abs(r["offset_mm"]) for r in met])
    mer = diff[h <= MERIDIONAL_MM] if len(met) else np.array([])
    out: Dict[str, Any] = {
        "rays": len(rows), "met_in_3d": len(met),
        "mean_share_on_2d_path": float(np.mean([r["share_on_2d_path"] for r in rows])) if rows else None,
        "meridional": {"rays": int(len(mer)), "worst_deg": float(mer.max()) if len(mer) else None},
        "by_offset": [],
    }
    if len(met) >= bands:
        edges = np.quantile(h, np.linspace(0.0, 1.0, bands + 1))
        for k in range(bands):
            inside = (h >= edges[k]) & ((h <= edges[k + 1]) if k == bands - 1 else (h < edges[k + 1]))
            if inside.any():
                out["by_offset"].append({"h_mm": [float(edges[k]), float(edges[k + 1])], "rays": int(inside.sum()),
                                         "mean_deg": float(diff[inside].mean()), "worst_deg": float(diff[inside].max())})
    return out


__all__ = ["MERIDIONAL_MM", "SLICE_FIELDS", "SLICE_OPTIONS", "SliceCompareFlow", "slice_summary"]
