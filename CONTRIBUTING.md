# Contributing

Thanks for your interest in `agentic-ai-eval`. The bar for this project is
"frontier-lab eval team would recognize their workflow here," so contributions
are held to that standard: typed, tested, and reproducible.

## Development setup

```bash
pip install -e ".[all,dev]"
make check          # ruff + mypy + pytest
```

The whole test suite is **offline and deterministic** — no API keys, no network,
no spend. CI runs it on Python 3.10–3.12. Anything you add must keep that
property: gate any live-provider behaviour behind `client.online` with a
deterministic fallback.

## Guidelines

- **Stay provider-agnostic.** Never import a vendor SDK outside `providers/`.
  Everything else talks to `LLMClient` / `LLMProvider`.
- **Type everything.** `mypy src/agentic_ai_eval` must pass.
- **Test the offline path.** Construct an `LLMClient(api_key="")` and assert the
  deterministic behaviour. Add online behaviour behind the same seam.
- **Keep schemas additive.** Persisted artifacts and the SQL store are a public
  contract; prefer optional fields with defaults over breaking changes.
- **Statistics over vibes.** New aggregate scores should carry a confidence
  interval; new comparisons should carry a significance test.

## Adding a provider

1. Add `providers/<name>_provider.py` implementing `LLMProvider`
   (`parse_json` + `complete_text`), guarding the SDK import so a missing
   dependency degrades to offline rather than crashing.
2. Register it in `providers/__init__.py` (`_REGISTRY`, `DEFAULT_MODELS`,
   `_ENV_KEYS`).
3. Add an optional-dependency extra in `pyproject.toml`.
4. Add a parametrized case to `tests/test_providers.py`.

## Pull requests

Run `make check` before opening a PR and describe what you evaluated. Small,
focused PRs with tests get merged fastest.
