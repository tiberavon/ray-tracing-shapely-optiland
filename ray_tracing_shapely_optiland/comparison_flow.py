"""
Python-specific module: comparison_flow
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
COMPARISON FLOW -- 2D against 3D, ray by ray, from profile files
===============================================================================
`compare.compare_2d_3d` as a flow an agent can drive: its inputs are profile
files and rays given as data, never callables, and its two traces are jobs
an executor runs, each in a box of its own:

    start -> awaiting_profiles -> loading -> awaiting_rays -> awaiting_options
          -> awaiting_executor -> awaiting_go_ahead -> tracing_2d -> tracing_3d
          -> comparing -> done

- **awaiting_profiles:** ``[{"path", "n" or "reflectance", "name"?}]``:
  each a solid profile (revolved or extruded) and its refractive index, or
  a mirror sheet (revolved) and its reflectance; a profile carries
  neither. The name its faces are reported under defaults to the
  file's folder for a ``profile.json``, else to the file's stem.
- **loading** reads each file and builds it in both engines: the 2D section
  by `BasePrism.from_profile` (a revolve profile's half-section mirrored)
  or, for a sheet, its mirrors by `mirrors_from_sheet_profile`; the 3D
  component by `RevolvedProfileSolid`, `ExtrudedPolygonSolid` or
  `RevolvedProfileSheet`. A file that cannot be built, or a solid given a
  reflectance (a sheet an index), sends the flow back, with the reason.
- **awaiting_rays:** the rays, in the 2D scene, as ``compare_2d_3d`` takes
  them, or fans: ``{"from": [a, b], "angles_deg": [first, last, step]}``.
  The question's context gives each profile's axis, extent and sources.
- **awaiting_options:** ``{}`` for the defaults, or any of
  rays_per_pencil, min_share, wavelength_um, max_depth, seed, max_rays_2d,
  plane (`compare_2d_3d`'s arguments).
- **awaiting_executor:** where the two traces run: ``local_process`` (a
  worker process each, the default), ``in_process``, or ``modal`` (a
  container each on the deployed Modal app ``MODAL_APP``; refused where the
  Modal client is not installed). With ``modal``, the estimate counts
  Modal's start-up and speed (`MODAL_SPEED`) and carries a cost estimate.
- **awaiting_go_ahead** is always entered. An estimate within the minute a
  box may spend without asking goes straight on; a longer one is asked
  (``true``, or ``false`` to go back to the options).
- **tracing_2d**, **tracing_3d:** the jobs of `compare_jobs`, followed by
  their handles (`flows.executors`): wait with ``flow_wait``.
- **comparing** puts them together (`compare.compare_paths`); ``done``
  holds the summary ``compare_2d_3d_summary`` gives. The per-ray table is
  written beside it, as ``rays.json`` in the run's folder.

Run on the same inputs, the flow's table equals ``compare_2d_3d``'s with
the sections and solids built from the same profiles.
===============================================================================
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ray_tracing_shapely.core.scene import Scene
from ray_tracing_shapely.flows import persistence
from ray_tracing_shapely.flows.budgets import (  # noqa: F401 -- JOB_START_S, MODAL_SPEED: also read from here
    ASK_OVER_S,
    JOB_START_S,
    MODAL_SPEED,
    estimate_s,
    format_duration,
)
from ray_tracing_shapely.flows.executors import InProcessExecutor, Job, LocalProcessExecutor
from ray_tracing_shapely.flows.fields import FlowField, explain_field
from ray_tracing_shapely.flows.flow_base import ObservableFlow

from .compare import MainPath2D, PencilRay3D, compare_paths
from .compare_jobs import check_profile, components_3d, optics_2d, read_profile

#: The most rays one comparison takes.
MAX_RAYS = 5000
#: Where the traces can run.
EXECUTORS = ("local_process", "in_process", "modal")
#: The deployed Modal app the ``modal`` executor sends the traces to.
MODAL_APP = os.environ.get("RTS_MODAL_APP", "rts-jobs")
#: The options and their defaults (`compare_2d_3d`'s).
OPTIONS: Dict[str, Any] = {"rays_per_pencil": 8, "min_share": 0.0, "wavelength_um": 0.55, "max_depth": 30,
                           "seed": 1, "max_rays_2d": 10_000, "plane": None}
_JOBS = "ray_tracing_shapely_optiland.compare_jobs"

COMPARISON_FIELDS: Dict[str, FlowField] = {
    "awaiting_profiles": FlowField(
        "profiles", "json",
        "Which profiles? A list of {\"path\": a profile.json; \"n\": the refractive index of a solid "
        "(revolved or extruded), or \"reflectance\" in (0, 1] of a mirror sheet (revolved); optionally "
        "\"name\": the component name its faces are reported under}.",
        "each is rebuilt in 2D and built in 3D from its file; a profile carries no index or reflectance",
        'e.g. [{"path": "collimator/profile.json", "n": 1.49}, {"path": "reflector/profile.json", '
        '"reflectance": 0.9}]'),
    "awaiting_rays": FlowField(
        "rays", "json",
        "Which rays, in the 2D scene? A list of {\"position\": [a, b], \"direction_deg\": angle from +a}, of "
        "fans {\"from\": [a, b], \"angles_deg\": [first, last, step]}, or of the profiles' own pencil sources "
        "{\"source\": its name}. The profiles' axes, extents and sources are in the question's context.",
        "each ray is traced alone in 2D, and as a pencil of 3D rays; they are compared face by face",
        'e.g. [{"from": [100, 30], "angles_deg": [-80, 80, 10]}]'),
    "awaiting_options": FlowField(
        "options", "json",
        "Options? {} for the defaults, or any of: rays_per_pencil (8), min_share (0), wavelength_um (0.55), "
        "max_depth (30), seed (1), max_rays_2d (10000), plane ({\"point\": [a, b], \"normal_deg\": angle}: "
        "compare where the rays cross it, not at their last event).",
        "compare_2d_3d's arguments",
        'e.g. {} or {"rays_per_pencil": 20, "plane": {"point": [160, 30], "normal_deg": 0}}', default={}),
    "awaiting_executor": FlowField(
        "executor", "choice",
        "Where do the traces run? local_process: a worker process each, waited on with flow_wait; "
        "in_process: here, while the answer is made; modal: a Modal container each (the deployed app "
        "RTS_MODAL_APP, rts-jobs by default), waited on the same way, its estimated cost in the go-ahead's "
        "context.",
        "a long trace runs outside the caller's process and is followed by its record",
        'e.g. "local_process" (the default)', default="local_process", choices=EXECUTORS),
    "awaiting_go_ahead": FlowField(
        "go_ahead", "bool",
        "The traces are estimated over a minute (the estimate is in the question's context). Go ahead? "
        "false goes back to the options.",
        "a box may spend a minute without asking",
        "e.g. true"),
}

COMPARISON_STATES = ["start", "awaiting_profiles", "loading", "awaiting_rays", "awaiting_options",
                     "awaiting_executor", "awaiting_go_ahead", "tracing_2d", "tracing_3d", "comparing", "done"]

COMPARISON_TRANSITIONS = [
    {"trigger": "begin", "source": "start", "dest": "awaiting_profiles"},
    {"trigger": "provide_profiles", "source": "awaiting_profiles", "dest": "loading", "before": "_provide_profiles"},
    {"trigger": "_advance", "source": "loading", "dest": "awaiting_rays", "conditions": "_no_problem"},
    {"trigger": "_advance", "source": "loading", "dest": "awaiting_profiles"},
    {"trigger": "provide_rays", "source": "awaiting_rays", "dest": "awaiting_options", "before": "_provide_rays"},
    {"trigger": "provide_options", "source": "awaiting_options", "dest": "awaiting_executor",
     "before": "_provide_options"},
    {"trigger": "provide_executor", "source": "awaiting_executor", "dest": "awaiting_go_ahead",
     "before": "_provide_executor"},
    {"trigger": "_advance", "source": "awaiting_go_ahead", "dest": "tracing_2d"},
    {"trigger": "provide_go_ahead", "source": "awaiting_go_ahead", "dest": "tracing_2d", "conditions": "_go",
     "prepare": "_provide_go_ahead"},
    {"trigger": "provide_go_ahead", "source": "awaiting_go_ahead", "dest": "awaiting_options"},
    {"trigger": "traced_2d", "source": "tracing_2d", "dest": "tracing_3d", "before": "_keep_2d"},
    {"trigger": "traced_3d", "source": "tracing_3d", "dest": "comparing", "before": "_keep_3d"},
    {"trigger": "_advance", "source": "comparing", "dest": "done"},
    {"trigger": "reset", "source": "*", "dest": "start", "before": "_on_reset"},
]


def _modal_installed() -> bool:
    """Whether the Modal client can be imported here."""
    import sys  # noqa: PLC0415
    if sys.modules.get("modal") is not None:
        return True
    try:
        return importlib.util.find_spec("modal") is not None
    except (ImportError, ValueError):
        return False


def _material(p: Dict[str, Any]) -> Dict[str, float]:
    """A profile entry's ``n`` (a solid) or ``reflectance`` (a sheet)."""
    return {k: p[k] for k in ("n", "reflectance") if k in p}


def _built(p: Dict[str, Any]) -> str:
    prof = p["profile"]
    if "n" not in p:
        return f"{prof.get('body', 'solid')}, reflectance {p['reflectance']:g}"
    coated = f", its mirror faces at reflectance {p['reflectance']:g}" if "reflectance" in p else ""
    return f"{prof['symmetry']['kind']}, n = {p['n']:g}{coated}"


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a number, got {value!r}")
    return float(value)


def _point(value: Any, name: str) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be [a, b], got {value!r}")
    return [_number(v, name) for v in value]


def _rays(value: Any) -> List[Dict[str, Any]]:
    """The rays answered, fans expanded, as ``{"position", "direction_deg"}``."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"rays takes a non-empty list of rays or fans, got {value!r}")
    out: List[Dict[str, Any]] = []
    for k, item in enumerate(value):
        if isinstance(item, dict) and "from" in item:
            unknown = sorted(set(item) - {"from", "angles_deg"})
            first, last, step = (_number(v, f"rays[{k}].angles_deg") for v in item.get("angles_deg") or ())
            if unknown or not step > 0 or last < first:
                raise ValueError(f"rays[{k}]: a fan is {{\"from\": [a, b], \"angles_deg\": [first, last, step]}}, "
                                 f"first <= last, step > 0; got {item!r}")
            start = _point(item["from"], f"rays[{k}].from")
            n = int(math.floor((last - first) / step + 1e-9)) + 1
            out.extend({"position": start, "direction_deg": first + i * step} for i in range(n))
        elif isinstance(item, dict):
            if set(item) != {"position", "direction_deg"}:
                raise ValueError(f"rays[{k}]: a ray is {{\"position\": [a, b], \"direction_deg\": angle}}, got {item!r}")
            out.append({"position": _point(item["position"], f"rays[{k}].position"),
                        "direction_deg": _number(item["direction_deg"], f"rays[{k}].direction_deg")})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            out.append({"position": _point(item[0], f"rays[{k}][0]"),
                        "direction_deg": _number(item[1], f"rays[{k}][1]")})
        else:
            raise ValueError(f"rays[{k}] is not a ray or a fan: {item!r}")
        if len(out) > MAX_RAYS:
            raise ValueError(f"more than {MAX_RAYS} rays: compare fewer, or split them over runs")
    return out


def _default_name(path: Path) -> str:
    return path.parent.name if path.name == "profile.json" and path.parent.name else path.stem


class Compare2D3DFlow(ObservableFlow):
    """2D against 3D, ray by ray, from profile files: `compare_2d_3d` as a flow.

    Construct it (``begin()`` is called), then answer each question
    (``answer(value)``; ``pending_field()`` is the question,
    ``question_state()["context"]`` what it is answered from). The two
    traces are jobs: with ``local_process``, follow them with ``wait`` (or
    the ``flow_wait`` tool). At ``done``, ``self.result`` holds the summary
    and ``self.comparison`` the `compare.Comparison`.

    Args:
        base_dir: Where the run's folder is written; jobs go to ``<base_dir>/jobs``.
        run_id: An explicit run id, or None for a generated one.
        render_svg: Re-render ``state.svg`` on every move (needs Graphviz's ``dot``).
    """

    FIELDS = COMPARISON_FIELDS
    PURPOSE = ("Compare a 2D design with its 3D export ray by ray, from profile files: each ray is traced "
               "alone in 2D and as a pencil of 3D rays, and the faces met, the exit point and the exit "
               "direction are compared.")

    def __init__(self, base_dir: Path, run_id: Optional[str] = None, render_svg: bool = True) -> None:
        self._clear()
        super().__init__(name="compare_2d_3d", states=COMPARISON_STATES, transitions_list=COMPARISON_TRANSITIONS,
                         initial="start", base_dir=base_dir, run_id=run_id, render_svg=render_svg)
        self.begin()

    def _clear(self) -> None:
        self.profiles: List[Dict[str, Any]] = []
        self.rays: List[Dict[str, Any]] = []
        self.options: Dict[str, Any] = dict(OPTIONS)
        self.executor_kind = "local_process"
        self.estimate: Optional[Dict[str, Any]] = None
        self._going = False
        self.main_2d: Optional[List[MainPath2D]] = None
        self.pencils: Optional[List[List[PencilRay3D]]] = None
        self.comparison: Any = None
        self.problem: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None

    def _on_reset(self, *args: Any, **kwargs: Any) -> None:
        super()._on_reset(*args, **kwargs)
        self._clear()

    def send_back_reason(self) -> Optional[str]:
        return self.problem

    def _ask(self) -> None:
        field = self.FIELDS[self.state]
        explanation = field.explanation()
        if self.problem:
            explanation.message = f"{self.problem} -- {explanation.message}"
        self.set_explanation(explanation)
        self.emit_event(field=field.name, message=explanation.message)
        self.update_detail(waiting_on=field.name)

    def _no_problem(self, *_a: Any, **_k: Any) -> bool:
        return self.problem is None

    def question_context(self) -> Optional[Dict[str, Any]]:
        if self.state == "awaiting_go_ahead":
            return self.estimate
        if self.state in ("awaiting_rays", "awaiting_options", "awaiting_executor") and self.profiles:
            out = {"profiles": [self._describe(p) for p in self.profiles], "rays": len(self.rays) or None}
            if self.state != "awaiting_rays":
                out["options"] = self.options
            return out
        return None

    @staticmethod
    def _describe(p: Dict[str, Any]) -> Dict[str, Any]:
        prof = p["profile"]
        xs = [q[0] for q in prof["points"]]
        ys = [q[1] for q in prof["points"]]
        sym = prof.get("symmetry") or {}
        return {"name": p["name"], "path": p["path"], **_material(p), "body": prof.get("body", "solid"),
                "symmetry": sym.get("kind"),
                "axis": sym.get("axis"), "extent": {"a": [min(xs), max(xs)], "b": [min(ys), max(ys)]},
                "faces": [s["short_label"] for s in prof["spans"]],
                "sources": [{k: s.get(k) for k in ("name", "kind", "position", "direction_deg")}
                            for s in prof.get("sources") or []]}

    # -- the profiles -------------------------------------------------------------

    def on_enter_awaiting_profiles(self, *_a: Any, **_k: Any) -> None:
        self._ask()

    def _provide_profiles(self, value: Any) -> None:
        if not isinstance(value, list) or not value:
            raise ValueError(f"profiles takes a non-empty list of {{\"path\", \"n\" or \"reflectance\"}}, "
                             f"got {value!r}")
        out = []
        for k, item in enumerate(value):
            keys = set(item) if isinstance(item, dict) else set()
            if ("path" not in keys or not keys & {"n", "reflectance"}
                    or keys - {"path", "n", "reflectance", "name"}):
                raise ValueError(f"profiles[{k}] takes \"path\", \"n\" (a solid; with \"reflectance\" too if it "
                                 f"has mirror-coated faces) or \"reflectance\" (a sheet), and optionally \"name\", "
                                 f"got {item!r}")
            if not isinstance(item["path"], str):
                raise ValueError(f"profiles[{k}].path must be a string, got {item['path']!r}")
            path = Path(item["path"])
            if not path.is_file():
                raise ValueError(f"profiles[{k}]: no file {path}")
            material = {}
            if "n" in item:
                material["n"] = _number(item["n"], f"profiles[{k}].n")
                if not material["n"] >= 1.0:
                    raise ValueError(f"profiles[{k}].n must be at least 1, got {material['n']}")
            if "reflectance" in item:
                material["reflectance"] = _number(item["reflectance"], f"profiles[{k}].reflectance")
                if not 0.0 < material["reflectance"] <= 1.0:
                    raise ValueError(f"profiles[{k}].reflectance must be in (0, 1], got {material['reflectance']}")
            name = item.get("name", _default_name(path))
            if not isinstance(name, str) or not name:
                raise ValueError(f"profiles[{k}].name must be a non-empty string, got {name!r}")
            out.append({"path": str(path), **material, "name": name})
        names = [p["name"] for p in out]
        if len(set(names)) != len(names):
            raise ValueError(f"two profiles share a name, {names}: give each a \"name\"")
        self.profiles = out

    def on_enter_loading(self, *_a: Any, **_k: Any) -> None:
        self.problem = None
        loaded = []
        try:
            for p in self.profiles:
                prof = read_profile(p["path"])
                check_profile(prof, p["name"], p)
                loaded.append({**p, "profile": prof})
            optics_2d(loaded)(Scene())
            components_3d(loaded)
        except Exception as e:  # noqa: BLE001 -- any reason a file cannot be built is the answer's problem
            self.problem = f"the profiles could not be built: {type(e).__name__}: {e}"
            self.emit_event(message=self.problem)
            self._advance()
            return
        self.profiles = loaded
        self.emit_event(message="built in 2D and 3D: " + ", ".join(
            f"{p['name']} ({_built(p)})" for p in loaded))
        self._advance()

    # -- the rays and the options ------------------------------------------------

    def on_enter_awaiting_rays(self, *_a: Any, **_k: Any) -> None:
        self._ask()

    def _provide_rays(self, value: Any) -> None:
        self.rays = _rays(self._sources_as_rays(value))

    def _sources_as_rays(self, value: Any) -> Any:
        """The answer with each ``{"source": name}`` replaced by that profile source's ray: a pencil is one
        ray; anything else is refused, saying what to give instead."""
        if not isinstance(value, list):
            return value
        specs = {s["name"]: s for p in self.profiles for s in (p["profile"].get("sources") or [])}
        out = []
        for k, item in enumerate(value):
            if not (isinstance(item, dict) and "source" in item):
                out.append(item)
                continue
            if set(item) != {"source"}:
                raise ValueError(f"rays[{k}]: a source is given as {{\"source\": its name}}, got {item!r}")
            spec = specs.get(item["source"])
            if spec is None:
                raise ValueError(f"rays[{k}]: no source {item['source']!r} in the profiles; they carry "
                                 f"{sorted(specs) or 'none'}")
            kind, pos = spec.get("kind"), [float(v) for v in spec["position"][:2]]
            if kind == "pencil":
                out.append({"position": pos, "direction_deg": float(spec["direction_deg"])})
            elif kind == "rayfile":
                raise ValueError(f"rays[{k}]: {spec['name']!r} is a rayfile: compare its slice with the "
                                 "compare_slice_3d flow")
            else:
                half = float(spec.get("half_angle_deg") or 90.0)
                a = float(spec["direction_deg"])
                raise ValueError(f"rays[{k}]: {spec['name']!r} is a {kind} source, a cone of rays, not one ray: "
                                 f"give a fan from it instead, e.g. {{\"from\": {pos}, \"angles_deg\": "
                                 f"[{a - half:g}, {a + half:g}, {max(half / 8.0, 1.0):g}]}}")
        return out

    def on_enter_awaiting_options(self, *_a: Any, **_k: Any) -> None:
        self._ask()

    def _provide_options(self, value: Any = None) -> None:
        value = {} if value is None else value
        if not isinstance(value, dict):
            raise ValueError(f"options takes an object, got {value!r}")
        unknown = sorted(set(value) - set(OPTIONS))
        if unknown:
            raise ValueError(f"options: unknown {unknown}; known {sorted(OPTIONS)}")
        out = dict(OPTIONS)
        for key in ("rays_per_pencil", "max_depth", "seed", "max_rays_2d"):
            if key in value:
                v = value[key]
                if isinstance(v, bool) or not isinstance(v, int) or v < (0 if key == "seed" else 1):
                    raise ValueError(f"options.{key} must be a {'whole number' if key == 'seed' else 'positive whole number'}, got {v!r}")
                out[key] = v
        if "min_share" in value:
            out["min_share"] = _number(value["min_share"], "options.min_share")
            if not 0.0 <= out["min_share"] < 1.0:
                raise ValueError(f"options.min_share must be in [0, 1), got {value['min_share']!r}")
        if "wavelength_um" in value:
            out["wavelength_um"] = _number(value["wavelength_um"], "options.wavelength_um")
            if not out["wavelength_um"] > 0:
                raise ValueError(f"options.wavelength_um must be positive, got {value['wavelength_um']!r}")
        if value.get("plane") is not None:
            plane = value["plane"]
            if not isinstance(plane, dict) or set(plane) != {"point", "normal_deg"}:
                raise ValueError(f"options.plane is {{\"point\": [a, b], \"normal_deg\": angle}}, got {plane!r}")
            out["plane"] = {"point": _point(plane["point"], "options.plane.point"),
                            "normal_deg": _number(plane["normal_deg"], "options.plane.normal_deg")}
        self.options = out

    def on_enter_awaiting_executor(self, *_a: Any, **_k: Any) -> None:
        self._ask()

    def _provide_executor(self, value: Any = "local_process") -> None:
        if value == "modal" and not _modal_installed():
            raise ValueError("modal: the Modal client is not installed here (pip install "
                             "'ray-tracing-shapely[modal]', then modal token new); choose local_process")
        self.executor_kind = value

    # -- the go-ahead --------------------------------------------------------------

    def _estimate(self) -> Dict[str, Any]:
        n, per = len(self.rays), self.options["rays_per_pencil"]
        start = JOB_START_S[self.executor_kind]
        speed = MODAL_SPEED if self.executor_kind == "modal" else {}
        w2 = estimate_s("simulate_2d_ray", n) * speed.get("simulate_2d_ray", 1.0)
        w3 = estimate_s("trace_3d_pencil_ray", n * per) * speed.get("trace_3d_pencil_ray", 1.0)
        s2, s3 = w2 + start, w3 + start
        out = {"rays": n, "rays_3d": n * per, "tracing_2d_s": s2, "tracing_3d_s": s3, "total_s": s2 + s3,
               "total": format_duration(s2 + s3), "executor": self.executor_kind, "asks_over_s": ASK_OVER_S}
        if self.executor_kind == "modal":
            from ray_tracing_shapely.flows.modal_executor import CPU_USD_PER_CORE_S, MEMORY_USD_PER_GIB_S  # noqa: PLC0415
            per_s = CPU_USD_PER_CORE_S + 2.0 * MEMORY_USD_PER_GIB_S          # the jobs' function: 1 core, 2 GiB
            out["cost_usd_est"] = round((s2 + s3) * per_s, 6)
            out["cost_note"] = ("the two jobs' run time at Modal's rates; the bill also counts each container's "
                                "start and idle time (about 2.6 times this so far)")
        return out

    def on_enter_awaiting_go_ahead(self, *_a: Any, **_k: Any) -> None:
        self.estimate = self._estimate()
        if self.estimate["total_s"] <= ASK_OVER_S:
            self.set_explanation(explain_field("go_ahead", f"Estimated at {self.estimate['total']}: no need to "
                                               "ask.", "a box may spend a minute without asking", ""))
            self.emit_event(message=f"estimated at {self.estimate['total']}; going on", estimate=self.estimate)
            self._advance()
            return
        self._ask()

    def _provide_go_ahead(self, value: Any) -> None:
        self._going = bool(value)

    def _go(self, *_a: Any, **_k: Any) -> bool:
        if not self._going:
            self.problem = None
            self.emit_event(message="not confirmed: back to the options")
        return self._going

    # -- the traces ------------------------------------------------------------------

    def _executor(self) -> Any:
        if self.executor_kind == "in_process":
            return InProcessExecutor()
        if self.executor_kind == "modal":
            from ray_tracing_shapely.flows.modal_executor import ModalExecutor  # noqa: PLC0415
            return ModalExecutor(MODAL_APP)
        return LocalProcessExecutor(Path(self.base_dir) / "jobs")

    def _job_profiles(self) -> List[Dict[str, Any]]:
        return [{"path": str(Path(p["path"]).resolve()), "name": p["name"], **_material(p)} for p in self.profiles]

    def on_enter_tracing_2d(self, *_a: Any, **_k: Any) -> None:
        self.set_explanation(explain_field("tracing_2d", f"Tracing {len(self.rays)} rays in 2D.",
                                           "each alone, its brightest branch followed", ""))
        job = Job(f"{_JOBS}:trace_2d_job", {"profiles": self._job_profiles(), "rays": self.rays,
                                            "max_rays_2d": self.options["max_rays_2d"]},
                  files={p["name"]: p["path"] for p in self._job_profiles()},
                  budget_s=self.estimate["tracing_2d_s"], label="2D main paths")
        self.submit_step(self._executor(), job, on_done="traced_2d", ask_over_s=None)

    def _keep_2d(self, result: Dict[str, Any], *_a: Any, **_k: Any) -> None:
        self.main_2d = [MainPath2D.from_dict(m) for m in result["main"]]

    def on_enter_tracing_3d(self, *_a: Any, **_k: Any) -> None:
        o = self.options
        self.set_explanation(explain_field("tracing_3d", f"Tracing {len(self.rays)} pencils of "
                                           f"{o['rays_per_pencil']} rays in 3D.", "one Optiland trace", ""))
        job = Job(f"{_JOBS}:trace_3d_job", {"profiles": self._job_profiles(), "rays": self.rays,
                                            **{k: o[k] for k in ("rays_per_pencil", "max_depth", "wavelength_um",
                                                                 "seed")}},
                  files={p["name"]: p["path"] for p in self._job_profiles()},
                  budget_s=self.estimate["tracing_3d_s"], label="3D pencils")
        self.submit_step(self._executor(), job, on_done="traced_3d", ask_over_s=None)

    def _keep_3d(self, result: Dict[str, Any], *_a: Any, **_k: Any) -> None:
        self.pencils = [[PencilRay3D.from_dict(r) for r in recs] for recs in result["pencils"]]

    def on_enter_comparing(self, *_a: Any, **_k: Any) -> None:
        rays = [(r["position"], r["direction_deg"]) for r in self.rays]
        self.comparison = compare_paths(rays, self.main_2d, self.pencils, plane=self.options["plane"],
                                        min_share=self.options["min_share"])
        folder = persistence.run_dir(Path(self.base_dir), self.run_id)
        (folder / "rays.json").write_text(json.dumps(self.comparison.rays, indent=1), encoding="utf-8")
        summary = self.comparison.summary()
        self.result = {"summary": summary, "rays_file": "rays.json",
                       "profiles": [{"name": p["name"], "path": p["path"], **_material(p)} for p in self.profiles],
                       "options": self.options, "executor": self.executor_kind}
        self.emit_event(message=f"{summary['agree']} of {summary['rays']} rays agree; worst direction "
                                f"difference {summary['worst_direction_difference']}")
        self._advance()

    def on_enter_done(self, *_a: Any, **_k: Any) -> None:
        self.finish(self.result)
        self.set_explanation(explain_field("done", "The comparison is ready.", "result: the summary; rays.json: "
                                           "the per-ray table", ""))
        self.emit_event(message="done")


__all__ = ["COMPARISON_FIELDS", "COMPARISON_STATES", "COMPARISON_TRANSITIONS", "Compare2D3DFlow", "EXECUTORS",
           "MAX_RAYS", "OPTIONS"]
