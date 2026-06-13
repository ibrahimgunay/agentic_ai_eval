"""End-to-end orchestration: ingest -> analyze -> generate evals -> (run) ->
scaffold code -> report.

`Pipeline` holds one LLMClient so online/offline mode is decided once. Each
stage is independently callable, but `run()` wires the common path and persists
artifacts so the whole thing is inspectable on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .analyze import analyze
from .evals import generate_suite, materialize_traces, run_suite
from .evals.runner import TraceProvider
from .ingest import ingest
from .llm import LLMClient
from .report import render_json, render_markdown
from .scaffold import scaffold
from .schema import EvalReport, EvalSuite, SystemAnalysis, SystemSpec, Trace


@dataclass
class PipelineArtifacts:
    spec: SystemSpec
    analysis: SystemAnalysis
    suite: EvalSuite
    report: EvalReport | None = None
    scaffold_files: dict[str, str] = field(default_factory=dict)
    traces: dict[str, Trace] = field(default_factory=dict)


class Pipeline:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    @property
    def online(self) -> bool:
        return self.client.online

    def run(
        self,
        description: str,
        *,
        name: str | None = None,
        traces: TraceProvider | None = None,
        cases_per_eval: int = 5,
        generate_code: bool = True,
    ) -> PipelineArtifacts:
        spec = ingest(description, name=name, client=self.client)
        analysis = analyze(spec, client=self.client)
        suite = generate_suite(spec, analysis, client=self.client, cases_per_eval=cases_per_eval)

        materialized = materialize_traces(suite, traces)
        report = run_suite(suite, materialized, client=self.client, analysis=analysis)

        files = scaffold(spec, client=self.client) if generate_code else {}

        return PipelineArtifacts(
            spec=spec, analysis=analysis, suite=suite, report=report,
            scaffold_files=files, traces=materialized,
        )

    def write_artifacts(self, artifacts: PipelineArtifacts, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        (out / "spec.json").write_text(artifacts.spec.model_dump_json(indent=2))
        (out / "analysis.json").write_text(artifacts.analysis.model_dump_json(indent=2))
        (out / "suite.json").write_text(artifacts.suite.model_dump_json(indent=2))

        if artifacts.report is not None:
            (out / "report.json").write_text(render_json(artifacts.report))
            (out / "report.md").write_text(render_markdown(artifacts.report))

        if artifacts.traces:
            (out / "traces.json").write_text(
                json.dumps({k: v.model_dump() for k, v in artifacts.traces.items()}, indent=2, default=str)
            )

        if artifacts.scaffold_files:
            gen = out / "generated"
            gen.mkdir(exist_ok=True)
            for rel, content in artifacts.scaffold_files.items():
                (gen / rel).write_text(content)

        # A small index for humans.
        (out / "MANIFEST.json").write_text(
            json.dumps(
                {
                    "spec": "spec.json",
                    "analysis": "analysis.json",
                    "suite": "suite.json",
                    "report": "report.md" if artifacts.report else None,
                    "traces": "traces.json" if artifacts.traces else None,
                    "generated": sorted(artifacts.scaffold_files) or None,
                    "provider": self.client.provider_name,
                    "online_mode": self.online,
                },
                indent=2,
            )
        )
        return out
