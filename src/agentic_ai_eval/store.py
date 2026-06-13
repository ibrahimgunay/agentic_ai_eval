"""Durable, queryable results store — the data layer of a model-eval pipeline.

Eval results are only useful if you can ask questions of them *over time*: is the
candidate model better than last week's? which dimension regressed? how does the
judge agree with humans across releases? This module persists every run into a
normalized **SQLite** database so the answers are a SQL query away — readable by
any BI tool, notebook, or the bundled REST API.

Schema (one row is one fact):

    runs(run_id, spec_name, provider, model, label, overall_score,
         ci_low, ci_high, passed, created_at)
    eval_results(run_id, eval_id, target, dimension, score, ci_low, ci_high,
                 passed, pass_threshold, num_cases, num_passed)
    case_results(run_id, eval_id, case_id, score, passed)
    grader_results(run_id, eval_id, case_id, kind, score, passed, source,
                   uncertainty, pending, rationale)

SQLite is intentional: zero-ops, file-based, and every analyst already has a
client for it. Point a dashboard at the file, or lift the same SQL to Postgres
for a team deployment.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path

from .schema import EvalReport

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    spec_name     TEXT NOT NULL,
    provider      TEXT,
    model         TEXT,
    label         TEXT,
    overall_score REAL,
    ci_low        REAL,
    ci_high       REAL,
    passed        INTEGER,
    created_at    TEXT
);
CREATE TABLE IF NOT EXISTS eval_results (
    run_id         TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    eval_id        TEXT NOT NULL,
    target         TEXT,
    dimension      TEXT,
    score          REAL,
    ci_low         REAL,
    ci_high        REAL,
    passed         INTEGER,
    pass_threshold REAL,
    num_cases      INTEGER,
    num_passed     INTEGER
);
CREATE TABLE IF NOT EXISTS case_results (
    run_id  TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    eval_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    score   REAL,
    passed  INTEGER
);
CREATE TABLE IF NOT EXISTS grader_results (
    run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    eval_id     TEXT NOT NULL,
    case_id     TEXT NOT NULL,
    kind        TEXT,
    score       REAL,
    passed      INTEGER,
    source      TEXT,
    uncertainty REAL,
    pending     INTEGER,
    rationale   TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_spec ON runs(spec_name, created_at);
CREATE INDEX IF NOT EXISTS idx_eval_run ON eval_results(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_dim ON eval_results(dimension);
"""


class EvalStore:
    """A SQLite-backed store for eval runs. Use as a context manager."""

    def __init__(self, path: str | Path = "eval_results.db") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)

    def __enter__(self) -> EvalStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def save_report(self, report: EvalReport, *, label: str | None = None, run_id: str | None = None) -> str:
        """Persist a full report; returns the assigned run_id."""
        run_id = run_id or uuid.uuid4().hex[:12]
        with self.conn:  # transactional
            self.conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id, report.spec_name, report.provider, report.model, label,
                    report.overall_score, report.ci_low, report.ci_high,
                    int(report.passed), report.created_at.isoformat(),
                ),
            )
            for r in report.results:
                self.conn.execute(
                    "INSERT INTO eval_results VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id, r.eval_id, r.target_component, r.dimension.value, r.score,
                        r.ci_low, r.ci_high, int(r.passed), r.pass_threshold,
                        r.num_cases, r.num_passed,
                    ),
                )
                for c in r.case_results:
                    self.conn.execute(
                        "INSERT INTO case_results VALUES (?,?,?,?,?)",
                        (run_id, r.eval_id, c.case_id, c.score, int(c.passed)),
                    )
                    for g in c.grader_results:
                        self.conn.execute(
                            "INSERT INTO grader_results VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (
                                run_id, r.eval_id, c.case_id, g.kind.value, g.score,
                                int(g.passed), g.source, g.uncertainty,
                                int(g.pending), g.rationale,
                            ),
                        )
        return run_id

    # ------------------------------------------------------------------ #
    # Reads — all return plain dicts so the API/CLI can serialize directly.
    # ------------------------------------------------------------------ #

    def query(self, sql: str, params: Iterable = ()) -> list[dict]:
        """Run arbitrary read-only SQL (the escape hatch for analysts)."""
        with closing(self.conn.execute(sql, tuple(params))) as cur:
            return [dict(row) for row in cur.fetchall()]

    def runs(self, spec_name: str | None = None, limit: int = 100) -> list[dict]:
        if spec_name:
            return self.query(
                "SELECT * FROM runs WHERE spec_name = ? ORDER BY created_at DESC LIMIT ?",
                (spec_name, limit),
            )
        return self.query("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))

    def run(self, run_id: str) -> dict | None:
        rows = self.query("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        return rows[0] if rows else None

    def eval_results(self, run_id: str) -> list[dict]:
        return self.query("SELECT * FROM eval_results WHERE run_id = ? ORDER BY score", (run_id,))

    def dimension_scores(self, run_id: str) -> dict[str, float]:
        rows = self.query(
            "SELECT dimension, AVG(score) AS score FROM eval_results "
            "WHERE run_id = ? GROUP BY dimension",
            (run_id,),
        )
        return {r["dimension"]: r["score"] for r in rows}

    def specs(self) -> list[str]:
        return [r["spec_name"] for r in self.query("SELECT DISTINCT spec_name FROM runs ORDER BY spec_name")]
