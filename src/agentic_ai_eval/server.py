"""Optional REST API over the results store (``pip install '.[server]'``).

A thin, read-mostly FastAPI surface so dashboards, notebooks, and CI bots can
pull eval data over HTTP instead of touching the database file directly:

    GET  /health
    GET  /specs
    GET  /runs?spec=<name>
    GET  /runs/{run_id}
    GET  /runs/{run_id}/evals
    GET  /runs/{run_id}/dimensions
    GET  /specs/{spec}/trend
    GET  /compare?baseline=<run>&candidate=<run>   # significance-tested A/B

FastAPI is an optional dependency; importing this module without it raises a
clear, actionable error rather than failing obscurely elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import asdict

from .analytics import compare_runs, score_trend
from .store import EvalStore

try:
    from fastapi import FastAPI, HTTPException
except ImportError:  # pragma: no cover - optional dependency
    FastAPI = None  # type: ignore[assignment]


def create_app(db_path: str | None = None):
    """Build the FastAPI app bound to a results database."""
    if FastAPI is None:  # pragma: no cover
        raise ImportError(
            "The REST API needs FastAPI/uvicorn. Install with: pip install 'agentic-ai-eval[server]'"
        )

    db_path = db_path or os.environ.get("AGENTIC_EVAL_DB", "eval_results.db")
    app = FastAPI(
        title="agentic-ai-eval API",
        version="0.2.0",
        summary="Query agentic-AI evaluation results, trends, and regressions.",
    )

    def store() -> EvalStore:
        return EvalStore(db_path)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "db": db_path}

    @app.get("/specs")
    def specs() -> list[str]:
        with store() as s:
            return s.specs()

    @app.get("/runs")
    def runs(spec: str | None = None, limit: int = 100) -> list[dict]:
        with store() as s:
            return s.runs(spec, limit=limit)

    @app.get("/runs/{run_id}")
    def run(run_id: str) -> dict:
        with store() as s:
            row = s.run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        return row

    @app.get("/runs/{run_id}/evals")
    def run_evals(run_id: str) -> list[dict]:
        with store() as s:
            return s.eval_results(run_id)

    @app.get("/runs/{run_id}/dimensions")
    def run_dimensions(run_id: str) -> dict:
        with store() as s:
            return s.dimension_scores(run_id)

    @app.get("/specs/{spec}/trend")
    def trend(spec: str, limit: int = 50) -> list[dict]:
        with store() as s:
            return [asdict(p) for p in score_trend(s, spec, limit=limit)]

    @app.get("/compare")
    def compare(baseline: str, candidate: str) -> dict:
        with store() as s:
            report = compare_runs(s, baseline, candidate)
        return {
            "spec_name": report.spec_name,
            "baseline_run": report.baseline_run,
            "candidate_run": report.candidate_run,
            "overall_delta": report.overall_delta,
            "overall_p_value": report.overall_test.p_value,
            "overall_significant": report.overall_test.significant,
            "regressed": [d.dimension for d in report.regressed],
            "improved": [d.dimension for d in report.improved],
            "dimensions": [
                {
                    "dimension": d.dimension,
                    "baseline": d.baseline,
                    "candidate": d.candidate,
                    "delta": d.delta,
                    "p_value": d.comparison.p_value,
                    "significant": d.comparison.significant,
                }
                for d in report.dimensions
            ],
        }

    return app


def main() -> None:  # pragma: no cover - entrypoint
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":  # pragma: no cover
    main()
