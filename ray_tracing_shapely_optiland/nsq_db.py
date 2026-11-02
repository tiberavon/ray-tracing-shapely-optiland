"""
Python-specific module: nsq db
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

Python translation Copyright 2026 ray-tracing-shapely authors and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

===============================================================================
NSQ RAY/RUN PERSISTENCE -- P-NSQ-RUN-BUNDLE
===============================================================================
Persists one Optiland NSQ trace's ray-lineage event log into a queryable
SQLite database, and reuses ray_tracing_shapely.analysis.agentic_db's
query/validation layer unchanged against it -- execute_query()/_validate_sql()
never reference the 2D-specific `rays` table by name (verified, C-NSQ-RUN-03),
so they work against this NSQ-specific schema exactly as they do
against the 2D one.

**A correction to the plan, found while building this** [KD2: the plan
said the schema description was engine-agnostic; it was not]:
get_schema_description() is NOT engine-agnostic the way execute_query() is --
it iterates a hardcoded `_COLUMN_DESCRIPTIONS` dict of the 2D engine's own
table names and silently omits anything not in it, `nsq_events` included, so
`agentic_db.get_schema_description(conn)` against an NSQ database returns
`{}` -- not an error, just nothing, for a table that demonstrably exists and
has rows. `get_nsq_schema_description()` below is this module's own small
equivalent, kept separate rather than patched into agentic_db.py's own
2D-specific dict.

`nsq_events` is a SIBLING to `rays.sqlite`'s `rays` table, not a variant of
it: NSQ's ColumnarPathLog is a per-EVENT point-sample log (one row per
birth/hit/death), not the 2D engine's per-SEGMENT record with precomputed
glass-containment/edge-crossing joins -- forcing NSQ's fields into the 2D
schema would either drop real NSQ fields or fabricate 2D-only ones (like
polarization) NSQ's non-sequential engine does not have at all. [KB8: two
sibling layouts, not one generalised one.]

Unlike ray_tracing_shapely_optiland.scene_bridge, this module DOES import
ray_tracing_shapely -- a different, and differently justified, dependency
shape: it reuses agentic_db's generic query/validation code rather than
re-implementing a second SQL layer.
===============================================================================
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from ray_tracing_shapely.analysis.agentic_db import execute_query

if TYPE_CHECKING:
    from optiland.nonsequential.scene import NSQScene
    from optiland.nonsequential.tracer import SimulationResult

# SQL note for anyone (agent included) querying this database: prefer
# repeating the aggregate expression in HAVING (`HAVING COUNT(*) != 1`)
# rather than referencing a SELECT alias (`HAVING n != 1`) -- confirmed by
# hand against sqlite3 3.37.2 that the alias form silently returns every
# group instead of filtering or raising, for a plain
# `SELECT x, COUNT(*) AS n ... GROUP BY x HAVING n != 1` query. A genuine
# SQLite quirk, not specific to this schema, but exactly the kind of
# silently-wrong result that is easy to write by reflex and easy to miss.

# Detector hits carry component_name = "" (empty string), not the
# detector's registry name: NSQScene.add_detector("D", ...) registers "D"
# as the *registry key*, but IrradianceDetectorConfig (and its detector-
# config siblings) has no `name` field to forward into the detector
# object's own `.name` attribute, which is what ArrayBackend/TorchBackend
# actually pass to path_recorder.log_hits() as the event's component_name
# (`getattr(det, "name", f"detector_{di}")` -- confirmed by reading
# array_backend.py). Not a bug in this module; a real Optiland API gap
# between "how a detector is addressed in the scene" and "what name its
# own hit events carry" -- query `component_name = ''` for detector hits.
_DETECTOR_HIT_LABEL = ""

__all__ = [
    "create_nsq_database",
    "persist_nsq_run",
    "execute_query",
    "get_nsq_schema_description",
]


def get_nsq_schema_description(conn: sqlite3.Connection) -> Dict[str, Any]:
    """NSQ-specific equivalent of agentic_db.get_schema_description().

    Returns:
        ``{"nsq_events": {"columns": [...], "row_count": N}}`` if the table
        exists in *conn*, else ``{}``.
    """
    existing = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "nsq_events" not in existing:
        return {}
    columns = [
        {"name": row[1], "type": row[2]}
        for row in conn.execute("PRAGMA table_info(nsq_events)").fetchall()
    ]
    row_count = conn.execute("SELECT COUNT(*) FROM nsq_events").fetchone()[0]
    return {"nsq_events": {"columns": columns, "row_count": row_count}}

_SCHEMA_NSQ_EVENTS = """\
CREATE TABLE nsq_events (
    row_id          INTEGER PRIMARY KEY,
    ray_id          INTEGER NOT NULL,
    event_type      TEXT NOT NULL,
    x               REAL NOT NULL,
    y               REAL NOT NULL,
    z               REAL NOT NULL,
    L               REAL NOT NULL,
    M               REAL NOT NULL,
    N               REAL NOT NULL,
    flux            REAL NOT NULL,
    wavelength      REAL NOT NULL,
    bounce          INTEGER NOT NULL,
    component_name  TEXT NOT NULL
);
"""

_SCHEMA_INDEXES = """\
CREATE INDEX idx_nsq_events_ray_id ON nsq_events(ray_id);
CREATE INDEX idx_nsq_events_component ON nsq_events(component_name);
"""


def create_nsq_database(
    scene: "NSQScene",
    result: "SimulationResult",
    db_path: Optional[Path] = None,
) -> sqlite3.Connection:
    """Persist one NSQ trace's event log into a queryable SQLite database.

    Args:
        scene: The NSQScene the trace ran against (accepted for parity
            with ray_tracing_shapely.analysis.agentic_db.create_database's
            signature and for future use -- e.g. a components/detectors
            table sibling to the 2D schema's glass_objects/detectors --
            not read yet; the event log alone is what is persisted).
        result: The SimulationResult from ``NSQScene.trace(...,
            record_paths=...)``.
        db_path: If given, write to this file path instead of an
            in-memory database. Matches agentic_db.create_database's own
            convention.

    Returns:
        A sqlite3.Connection with ``nsq_events`` populated.

    Raises:
        ValueError: If ``result.ray_paths`` is None -- the trace must have
            been run with ``record_paths`` truthy. This function does not
            silently persist an empty table for a run that recorded
            nothing, since an empty-but-valid database and "nobody asked
            for path recording" would otherwise look identical at query
            time -- exactly the "plausible-looking wrong number" failure
            mode dev_doc/architecture/02_conventions.md flags as this
            project's characteristic defect shape.
    """
    if result.ray_paths is None:
        raise ValueError(
            "result.ray_paths is None -- trace with record_paths=True (or a "
            "positive int for a sampled subset) before persisting. Persisting "
            "an empty table for a run that recorded nothing would be "
            "indistinguishable from a genuinely empty trace at query time."
        )

    target = str(db_path) if db_path is not None else ":memory:"
    conn = sqlite3.connect(target, check_same_thread=False)
    if db_path is None:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")

    conn.executescript(_SCHEMA_NSQ_EVENTS)
    conn.executescript(_SCHEMA_INDEXES)

    events = result.ray_paths["events"]
    rows = [
        (
            int(e["ray_id"]),
            str(e["event_type"]),
            float(e["x"]),
            float(e["y"]),
            float(e["z"]),
            float(e["L"]),
            float(e["M"]),
            float(e["N"]),
            float(e["flux"]),
            float(e["wavelength"]),
            int(e["bounce"]),
            str(e["component_name"]),
        )
        for e in events
    ]
    conn.executemany(
        "INSERT INTO nsq_events "
        "(ray_id, event_type, x, y, z, L, M, N, flux, wavelength, bounce, component_name) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return conn


def persist_nsq_run(
    scene: "NSQScene",
    result: "SimulationResult",
    *,
    out_dir: Path,
    design_id: str,
    run_id: str,
    record_paths: Any,
    physical_params: Optional[Dict[str, Any]] = None,
    numerical_params: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write one NSQ run's nsq_events.sqlite + params.json to
    <out_dir>/<design_id>/<run_id>/, mirroring
    ray_tracing_shapely.analysis.run_registry.persist_run_artifacts's
    directory layout so a 2D run and an NSQ run of "the same" device sit
    in comparable places.

    Args:
        record_paths: The exact ``record_paths`` value passed to
            ``scene.trace(...)`` for this run -- written into
            ``params.json``'s ``numerical_params`` so a later reader can
            tell a full trace from a sampled subset. Required, not
            optional (C-NSQ-RUN-02): silently omitting this is the risk
            the field exists to close.
        physical_params: Merged into ``numerical_params``'s sibling in
            params.json, following the physical/numerical split
            ``run_registry.py`` already established.
        numerical_params: See above; ``record_paths`` is added to a copy
            of this dict, not mutated in place.

    Returns:
        The run directory (``out_dir/design_id/run_id``).
    """
    run_dir = Path(out_dir) / design_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    db_path = run_dir / "nsq_events.sqlite"
    if db_path.exists():
        db_path.unlink()
    create_nsq_database(scene, result, db_path=db_path).close()

    numerical = dict(numerical_params or {})
    numerical["record_paths"] = record_paths
    (run_dir / "params.json").write_text(
        json.dumps(
            {"physical_params": physical_params or {}, "numerical_params": numerical},
            indent=2,
        )
    )
    return run_dir
