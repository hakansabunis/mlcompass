# Contributing to ml-copilot

Thanks for thinking about contributing! ml-copilot is alpha-stage and
moves quickly, so a short heads-up before you sink time into a change:

- **Open an issue first** for anything larger than a small fix. The
  roadmap in [CHANGELOG.md](CHANGELOG.md) and the design in
  [ARCHITECTURE.md](ARCHITECTURE.md) may already have an opinion on
  what you have in mind, and aligning early saves a rewrite later.
- **Small, focused PRs** are much easier to land than sweeping refactors.

## Development setup

```bash
git clone https://github.com/hakansabunis/ml-copilot
cd ml-copilot

python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -e ".[dev]"
```

That installs ml-copilot in editable mode plus pytest, ruff, mypy,
and tbparse.

## Running the test suite

```bash
python -m pytest tests/             # all
python -m pytest tests/ -v           # verbose
python -m pytest tests/test_cli_advise.py::test_advise_persists_to_project_context
```

The full suite is ~70 tests and runs in under 2 seconds on a laptop.
Tests that use a mock Anthropic client live in `tests/test_advisor.py`
and `tests/test_cli_advise.py` — they don't need an `ANTHROPIC_API_KEY`.

## Style

- `ruff check src/ tests/` for linting
- `ruff format src/ tests/` for formatting
- `mypy src/` for type checking
- All public APIs get docstrings; module-level summaries are required
- Keep diffs surgical: when you change one thing, don't reformat the
  surrounding 200 lines

## Adding a new command

The structure for adding a new command (say `audit`) is:

1. **Tool layer** (`src/ml_copilot/tools/script.py`): pure-Python
   helpers that don't need an LLM (AST parsing, static checks, etc.).
2. **Agent** (`src/ml_copilot/agents/audit.py`): an agentlite Agent
   factory and a `get_<thing>(...)` entry point that takes the tool
   output and returns a parsed recommendation.
3. **UI** (`src/ml_copilot/ui/audit.py`): rich rendering for the
   agent output.
4. **CLI** (`src/ml_copilot/cli.py`): a new `@cli.command()` that
   wires the three layers together.
5. **Tests** (`tests/test_<thing>.py`): unit tests for the tool layer,
   MockClient tests for the agent, CliRunner tests for the command.

See `agents/advise.py`, `tools/dataset.py`, `ui/advise.py`, and the
`advise` command in `cli.py` for the working pattern.

## Tool permissions

If your tool modifies user files or runs user processes, decorate it
with `@tool(requires_confirmation=True)` (from `agentlite`) and let
the permission prompt handle approval. Never auto-run destructive
actions, even with a `--yes` flag, without a clear opt-in.

## Reporting issues

Useful bug reports include:

- ml-copilot version (`ml-copilot --version`)
- Python version (`python --version`)
- OS + terminal
- Minimal reproduction steps
- Full error output (don't paraphrase tracebacks)

For feature requests, please describe the workflow you'd like to support
before proposing an API — there's often a smaller change that gets you
the same result.

## License

By contributing, you agree that your contributions will be licensed
under the project's MIT license.
