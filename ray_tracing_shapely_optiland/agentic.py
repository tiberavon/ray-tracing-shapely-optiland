"""
Python-specific module: agentic
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
AGENTIC -- read-only tools over a traced Optiland non-sequential scene
===============================================================================
What an assistant can ask of a 3D trace: what is in the scene, what the
event log holds, a read-only SQL query over it, and where the light went,
grouped by the sequence of faces each ray met.

Two explicit steps, as for the 2D tools:

    from ray_tracing_shapely_optiland import agentic
    agentic.register()                         # once: adds the tools to the core's
                                               # three lists (C-AGENT-02)
    agentic.set_nsq_context(scene, result)     # after each trace
    agentic.set_comparison_context(cmp)        # after compare.compare_2d_3d, if run

``register()`` is never called at import, so importing the bridge in a batch
job does not change what an assistant sees. Register before the MCP server
starts (``mcp_server.start_background``): one server publishes the 2D and
the 3D tools together.

The live objects go into the core's context registry under one namespaced
key, ``optiland.nsq`` (``set_extension_context``), so the core's
``clear_context`` clears them too.

Labels in the event log (``nsq_events.component_name``):
- a face is ``<solid>.<span label>``, as the bridge names it;
- a source's birth carries the source's name, the spec's name when the
  source was added with ``sources.add_source_3d``;
- a detector hit carries ``''``, since Optiland does not forward a
  detector's registry name to its events (``KW-OPTILAND-09``). The path
  tools say ``detector`` for it, or the detector's name when the scene has
  exactly one.
- ``escaped`` is a ray that left the scene.
===============================================================================
"""

from __future__ import annotations

import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union

from ray_tracing_shapely.analysis.agentic_db import execute_query
from ray_tracing_shapely.analysis.agentic_tools import get_extension_context, set_extension_context
from ray_tracing_shapely.analysis.tool_registry import get_agentic_tools, register_tool

from .nsq_db import create_nsq_database, get_nsq_schema_description

if TYPE_CHECKING:
    from optiland.nonsequential.scene import NSQScene

CONTEXT_KEY = "optiland.nsq"
COMPARE_KEY = "optiland.compare"
_MISSING_COMPARISON = ("No 2D-against-3D comparison set. Run ray_tracing_shapely_optiland.compare.compare_2d_3d "
                       "and pass its result to ray_tracing_shapely_optiland.agentic.set_comparison_context.")
_MISSING = ("No 3D context set. Call ray_tracing_shapely_optiland.agentic.set_nsq_context(scene, result) "
            "after tracing with record_paths=True.")

#: What each nsq_events column means, for describe_nsq_schema.
_COLUMN_NOTES = {
    "row_id": "row order; within one ray, events are in the order they happened",
    "ray_id": "the ray; one ray is one Monte Carlo path (it takes one branch at every split)",
    "event_type": "'birth' (leaves its source), 'hit' (meets a face or a detector), 'death' (ends)",
    "x": "position, world mm", "y": "position, world mm", "z": "position, world mm",
    "L": "direction cosine along x after the event", "M": "direction cosine along y", "N": "direction cosine along z",
    "flux": "the ray's flux after the event, W",
    "wavelength": "micrometres",
    "bounce": "number of interactions so far",
    "component_name": ("for a birth, the source's name; for a hit, the face '<solid>.<span label>', or '' for a "
                       "detector; for a death, 'escaped' when the ray left the scene"),
}


def _ok(data: Any) -> Dict[str, Any]:
    return {"status": "ok", "data": data}


def _error(message: str) -> Dict[str, str]:
    return {"status": "error", "message": message}


# --- context --------------------------------------------------------------------


def set_nsq_context(
    scene: Optional["NSQScene"],
    result: Any = None,
    *,
    profiles: Optional[Mapping[str, Mapping[str, Any]]] = None,
    db: Union[sqlite3.Connection, str, Path, None] = None,
    context_name: Optional[str] = None,
) -> bool:
    """
    Register a traced 3D scene for the 3D tools.

    Args:
        scene: The NSQScene that was traced; None for a saved event log
            loaded without its scene (``load_nsq_context``): the tools then
            name what the log names, and roles and surface kinds are unknown.
        result: The trace's SimulationResult, traced with ``record_paths``.
            Needed unless ``db`` is given.
        profiles: The ``profile.json`` dicts the solids were built from,
            keyed by solid name, so each face's role and mechanism can be
            named. By default each solid's own ``profile`` (revolved and
            extruded solids keep theirs).
        db: An ``nsq_events`` database instead of building one from
            ``result``: an open connection, or the path of a persisted
            ``nsq_events.sqlite`` (opened read-only).
        context_name: The core's context to put it in, beside that
            context's 2D run; None (the default): the active one.

    Returns:
        True.

    Raises:
        ValueError: If neither ``result`` nor ``db`` is given, or the result
            has no path log.
    """
    if db is None:
        if result is None:
            raise ValueError("set_nsq_context needs the trace's result (traced with record_paths=True) or a db")
        conn = create_nsq_database(scene, result)
    elif isinstance(db, sqlite3.Connection):
        conn = db
    else:
        conn = sqlite3.connect(f"file:{Path(db)}?mode=ro", uri=True, check_same_thread=False)
    if profiles is None and scene is None:
        profiles = {}
    elif profiles is None:
        profiles = {name: comp.profile for name, comp in _components(scene).items()
                    if isinstance(getattr(comp, "profile", None), Mapping)}
    set_extension_context(CONTEXT_KEY, {"scene": scene, "result": result, "profiles": dict(profiles), "db": conn},
                          **_in(context_name))
    return True


def load_nsq_context(path: Union[str, Path], context_name: str) -> None:
    """Load a saved ``nsq_events.sqlite``, read-only and without its scene, into a named context.

    What the core's ``load_run_context`` calls for a run folder holding one,
    once ``register()`` has run, so a 2D run and its 3D trace saved in one
    folder load into the same context.

    Args:
        path: The ``nsq_events.sqlite`` file.
        context_name: The core's context to put it in.
    """
    set_nsq_context(None, db=path, context_name=context_name)


def set_comparison_context(comparison: Any, *, context_name: Optional[str] = None) -> bool:
    """Register a ``compare.compare_2d_3d`` result for ``compare_2d_3d_summary``.

    Args:
        comparison: The result.
        context_name: As for ``set_nsq_context``.

    Returns:
        True.
    """
    set_extension_context(COMPARE_KEY, {"comparison": comparison}, **_in(context_name))
    return True


def _in(context_name: Optional[str]) -> Dict[str, Any]:
    # the keyword only when given: a core without named contexts is still served
    return {} if context_name is None else {"context_name": context_name}


def _context() -> Dict[str, Any]:
    return get_extension_context(CONTEXT_KEY, _MISSING)


def _named(registry) -> "OrderedDict[str, Any]":
    """A scene registry's entries by name, in the order added.

    Optiland's component, source and detector registries list their objects
    (``compounds()``, ``sources``, ``detectors``) but not the names they were
    added under, which only their dict holds. Read-only use of it.
    """
    return OrderedDict(registry._registry)


def _components(scene) -> "OrderedDict[str, Any]":
    return _named(scene.component_registry)


def _detector_names(scene) -> List[str]:
    return list(_named(scene.detector_registry))


def _detector_label(scene) -> str:
    names = _detector_names(scene)
    return f"detector {names[0]}" if len(names) == 1 else "detector"


# --- tools ----------------------------------------------------------------------


def _faces(name: str, comp: Any, profile: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    spans = {s.get("short_label"): s for s in (profile or {}).get("spans", [])}
    kinds = getattr(comp, "surface_kinds", None) or {}
    faces = []
    for surface in getattr(comp, "surfaces", None) or []:
        face = getattr(surface, "name", None) or ""
        label = face.split(".", 1)[1] if face.startswith(name + ".") else face
        span = spans.get(label, {})
        faces.append({"name": face, "label": label, "type": type(surface).__name__,
                      "surface_kind": kinds.get(label),
                      "role": span.get("role"), "mechanism": span.get("mechanism")})
    return faces


def describe_nsq_scene() -> Dict[str, Any]:
    """
    Describe the traced 3D scene: its solids and their faces, its sources and detectors.

    Each face is named as the event log names it (``<solid>.<span label>``),
    with its role and mechanism from the solid's profile and the kind of
    surface it became (conic, spline, disc, annulus, frustum). Use the face
    names in ``query_nsq`` and to read ``nsq_path_spectrum``.

    Returns:
        ``{"status": "ok", "data": {"components", "sources", "detectors", "event_log"}}``,
        or an error if no 3D context is set.
    """
    try:
        ctx = _context()
    except RuntimeError as e:
        return _error(str(e))
    scene, profiles, db = ctx["scene"], ctx["profiles"], ctx["db"]
    rows, rays = db.execute("SELECT COUNT(*), COUNT(DISTINCT ray_id) FROM nsq_events").fetchone()
    if scene is None:
        return _ok(_describe_from_log(db, rows, rays))
    components = []
    for name, comp in _components(scene).items():
        profile = profiles.get(name)
        components.append({
            "name": name, "type": type(comp).__name__,
            "symmetry": ((profile or {}).get("symmetry") or {}).get("kind"),
            "body": (profile or {}).get("body"),
            "faces": _faces(name, comp, profile),
        })
    sources = [{"name": n, "type": type(src).__name__,
                "logged_as": getattr(src, "name", None) or f"source_{i}"}
               for i, (n, src) in enumerate(_named(scene.source_registry).items())]
    detectors = [{"name": n, "type": type(det).__name__, "logged_as": ""}
                 for n, det in _named(scene.detector_registry).items()]
    return _ok({"components": components, "sources": sources, "detectors": detectors,
                "event_log": {"events": rows, "rays": rays}})


def _describe_from_log(db: sqlite3.Connection, rows: int, rays: int) -> Dict[str, Any]:
    """describe_nsq_scene for a log loaded without its scene: what the log names."""
    hit = [r[0] for r in db.execute("SELECT DISTINCT component_name FROM nsq_events "
                                    "WHERE event_type = 'hit' AND component_name != '' ORDER BY 1")]
    faces: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for face in hit:
        solid, _, label = face.partition(".")
        faces.setdefault(solid, []).append({"name": face, "label": label, "type": None, "surface_kind": None,
                                            "role": None, "mechanism": None})
    births = [r[0] for r in db.execute("SELECT DISTINCT component_name FROM nsq_events "
                                       "WHERE event_type = 'birth' ORDER BY 1")]
    detector_hits = db.execute("SELECT COUNT(*) FROM nsq_events WHERE event_type = 'hit' "
                               "AND component_name = ''").fetchone()[0]
    return {"components": [{"name": n, "type": None, "symmetry": None, "body": None, "faces": f}
                           for n, f in faces.items()],
            "sources": [{"name": b, "type": None, "logged_as": b} for b in births],
            "detectors": [{"name": None, "type": None, "logged_as": ""}] if detector_hits else [],
            "event_log": {"events": rows, "rays": rays},
            "note": ("loaded from a saved event log without its scene: the faces and sources the log "
                     "names; roles, mechanisms, surface kinds and the detector's name are not in it")}


def describe_nsq_schema() -> Dict[str, Any]:
    """
    Describe the 3D event-log database that ``query_nsq`` reads: its one table, nsq_events.

    Returns:
        ``{"status": "ok", "data": {"nsq_events": {"columns": [...], "row_count": N}}}``,
        each column with its type and meaning, or an error if no 3D context is set.
    """
    try:
        db = _context()["db"]
    except RuntimeError as e:
        return _error(str(e))
    schema = get_nsq_schema_description(db)
    for col in schema.get("nsq_events", {}).get("columns", []):
        col["meaning"] = _COLUMN_NOTES.get(col["name"])
    return _ok(schema)


def query_nsq(sql: str) -> Dict[str, Any]:
    """
    Run a read-only SQL SELECT against the 3D event log (table nsq_events).

    One row per event: a ray's birth, each hit, its death. Use
    describe_nsq_schema() for the columns and describe_nsq_scene() for the
    face names. Only SELECT is allowed; results are capped at 200 rows,
    so aggregate or use LIMIT. In HAVING, repeat the aggregate
    (HAVING COUNT(*) > 1) rather than its alias: SQLite silently ignores
    the alias form.

    Args:
        sql: A SQL SELECT query string.

    Returns:
        ``{"status": "ok", "data": {"columns", "rows", "row_count"}}`` or an error.
    """
    try:
        db = _context()["db"]
        return _ok(execute_query(db, sql))
    except ValueError as e:
        return _error(str(e))
    except sqlite3.OperationalError as e:
        return _error(f"SQL error: {e}")
    except RuntimeError as e:
        return _error(str(e))


def path_spectrum(db: sqlite3.Connection, detector_label: str = "detector") -> List[Dict[str, Any]]:
    """
    Rays grouped by their path: source, the faces met in order, and the end.

    The end is the detector (a hit on ``''``), ``escaped``, or ``absorbed``
    at the last face met. Flux is the birth flux (what the source put into
    the path) and the final flux (what the path delivered).

    Returns:
        One dict per path, largest birth flux first: ``path`` (a list of
        names), ``end``, ``rays``, ``flux_emitted``, ``flux_delivered``,
        ``share`` (of all birth flux).
    """
    groups: Dict[tuple, Dict[str, Any]] = {}
    current, seq, first_flux, last_flux, end = None, [], 0.0, 0.0, None

    def close():
        if current is None:
            return
        key = (tuple(seq), end)
        g = groups.setdefault(key, {"path": list(seq), "end": end, "rays": 0,
                                    "flux_emitted": 0.0, "flux_delivered": 0.0})
        g["rays"] += 1
        g["flux_emitted"] += first_flux
        g["flux_delivered"] += last_flux

    for ray_id, event, comp, flux in db.execute(
            "SELECT ray_id, event_type, component_name, flux FROM nsq_events ORDER BY ray_id, row_id"):
        if ray_id != current:
            close()
            current, seq, first_flux, end = ray_id, [], float(flux), None
        last_flux = float(flux)
        if event == "birth":
            seq.append(comp)
        elif event == "hit" and comp == "":
            seq.append(detector_label)
            end = detector_label
        elif event == "hit":
            seq.append(comp)
            end = f"absorbed at {comp}"
        elif event == "death":
            end = "escaped" if comp == "escaped" else (f"absorbed at {comp}" if comp else end)
    close()
    total = sum(g["flux_emitted"] for g in groups.values()) or 1.0
    out = sorted(groups.values(), key=lambda g: -g["flux_emitted"])
    for g in out:
        g["share"] = g["flux_emitted"] / total
    return out


def nsq_path_spectrum(top_n: int = 10) -> Dict[str, Any]:
    """
    Where the light went in the 3D trace: rays grouped by the faces they met, largest flux first.

    Each path is the source, the faces in the order met (as named in
    describe_nsq_scene), and how it ended: at the detector, escaped from the
    scene, or absorbed at its last face. Each ray is one Monte Carlo path.

    Args:
        top_n: How many paths to return; the rest are summed in ``others``.

    Returns:
        ``{"status": "ok", "data": {"paths": [...], "others": {...}, "path_count", "rays", "flux_emitted"}}``,
        each path with ``rays``, ``flux_emitted``, ``flux_delivered`` and
        ``share`` of all emitted flux; or an error.
    """
    try:
        ctx = _context()
    except RuntimeError as e:
        return _error(str(e))
    if not isinstance(top_n, int) or top_n < 1:
        return _error(f"top_n must be a positive integer, got {top_n!r}")
    paths = path_spectrum(ctx["db"], _detector_label(ctx["scene"]) if ctx["scene"] is not None else "detector")
    rest = paths[top_n:]
    others = {"paths": len(rest), "rays": sum(p["rays"] for p in rest),
              "flux_emitted": sum(p["flux_emitted"] for p in rest), "share": sum(p["share"] for p in rest)}
    return _ok({"paths": paths[:top_n], "others": others, "path_count": len(paths),
                "rays": sum(p["rays"] for p in paths), "flux_emitted": sum(p["flux_emitted"] for p in paths)})


def compare_2d_3d_summary(max_listed: int = 20) -> Dict[str, Any]:
    """
    How the same meridional rays traced in 2D and in 3D compare: the summary only.

    Each ray was traced in the 2D scene and, as a pencil at the mapped place,
    in 3D. A ray agrees when its 3D rays followed the 2D ray's main path (the
    faces it met, named ``<component>.<label>`` on both sides, and its end);
    the share that did is given, the rest being Monte Carlo Fresnel ghosts. For the rays that agree, the worst exit-direction and exit-point
    differences; the rays that differ are listed with both paths.

    Args:
        max_listed: At most this many differing rays are listed.

    Returns:
        ``{"status": "ok", "data": {"rays", "agree", "differ",
        "worst_direction_difference", "worst_point_difference_mm",
        "least_share_on_2d_path", "rays_3d_on_2d_paths", "exit_point", "plane", "differing_rays",
        "differing_rays_not_listed"}}``, or an error.
    """
    try:
        ctx = get_extension_context(COMPARE_KEY, _MISSING_COMPARISON)
    except RuntimeError as e:
        return _error(str(e))
    if not isinstance(max_listed, int) or max_listed < 0:
        return _error(f"max_listed must be a non-negative integer, got {max_listed!r}")
    return _ok(ctx["comparison"].summary(max_listed))


# --- registration ---------------------------------------------------------------

_NO_ARGS = {"type": "object", "properties": {}, "required": []}

TOOLS = [
    (describe_nsq_scene, "Describe the traced 3D (Optiland non-sequential) scene: its solids with each face's "
                         "name as the event log uses it, its role, mechanism and surface kind; sources; "
                         "detectors; the size of the event log. Start here for 3D questions.", _NO_ARGS),
    (describe_nsq_schema, "Describe the 3D event-log table nsq_events that query_nsq reads: every column, its "
                          "type and meaning.", _NO_ARGS),
    (query_nsq, "Run a read-only SQL SELECT against the 3D event log nsq_events (one row per ray birth, hit "
                "and death). Results capped at 200 rows: aggregate or use LIMIT.",
     {"type": "object", "properties": {"sql": {"type": "string", "description": "A SQL SELECT query."}},
      "required": ["sql"]}),
    (nsq_path_spectrum, "Where the light went in the 3D trace: rays grouped by the sequence of faces they met "
                        "and how they ended (detector, escaped, absorbed), largest flux first, with each "
                        "path's share of the emitted flux.",
     {"type": "object", "properties": {"top_n": {"type": "integer", "description": "How many paths to list.",
                                                 "default": 10}}, "required": []}),
    (compare_2d_3d_summary, "How the same meridional rays traced in 2D and in 3D compare: how many agree on "
                            "the faces met and the end, the worst exit-direction and exit-point differences, "
                            "and the rays that differ, with both paths. Summary only.",
     {"type": "object", "properties": {"max_listed": {"type": "integer", "description":
                                                      "How many differing rays to list.", "default": 20}},
      "required": []}),
]


def register() -> bool:
    """
    Add the 3D tools to the core's tool lists, once, and the two comparison
    flows to the flows that can be started by name, for the generic flow
    tools: ``compare_2d_3d`` (2D against 3D, ray by ray) and
    ``compare_slice_3d`` (a rayfile's 2D slice against its records in 3D);
    and lets the core's ``load_run_context`` load a run folder's
    ``nsq_events.sqlite`` (``load_nsq_context``). Call before the MCP server
    starts.

    Returns:
        True if the tools were added now, False if they already were.

    Raises:
        RuntimeError: If the MCP server is already running (``register_tool``).
    """
    from ray_tracing_shapely.analysis.agentic_tools import register_run_file_loader  # noqa: PLC0415
    from ray_tracing_shapely.flows.kinds import register_flow_kind  # noqa: PLC0415

    register_run_file_loader("nsq_events.sqlite", load_nsq_context)

    register_flow_kind("compare_2d_3d", "ray_tracing_shapely_optiland.comparison_flow:Compare2D3DFlow")
    register_flow_kind("compare_slice_3d", "ray_tracing_shapely_optiland.slice_comparison_flow:SliceCompareFlow")
    present = {t["name"] for t in get_agentic_tools()}
    if all(fn.__name__ in present for fn, _, _ in TOOLS):
        return False
    for fn, description, schema in TOOLS:
        if fn.__name__ not in present:
            register_tool(fn, description, schema, module="ray_tracing_shapely_optiland.agentic")
    return True
