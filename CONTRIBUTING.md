# Contributing to mlcompass

Thanks for thinking about contributing! mlcompass is alpha-stage and
moves quickly, so a short heads-up before you sink time into a change:

- **Open an issue first** for anything larger than a small fix. The
  roadmap in [CHANGELOG.md](CHANGELOG.md) and the design in
  [ARCHITECTURE.md](ARCHITECTURE.md) may already have an opinion on
  what you have in mind, and aligning early saves a rewrite later.
- **Small, focused PRs** are much easier to land than sweeping refactors.

## Development setup

Python 3.10 – 3.13. These are the exact commands CI runs, and they are
verified from a cold clone:

```bash
git clone https://github.com/hakansabunis/mlcompass
cd mlcompass

python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

That installs mlcompass in editable mode plus pytest, ruff, mypy,
and tbparse.

Then confirm the install before trusting a test run:

```bash
python -c "import mlcompass; print(mlcompass.__version__, mlcompass.__file__)"
```

This check is not ceremony. `tests/test_ablation_harness.py` puts `src/`
on `sys.path` at module scope, and it collects first alphabetically, so a
full-suite run stays green even when the package is not importable at
all — for example when a `.venv` was built before the checkout moved and
its editable `.pth` still points at the old path. If the line above
raises `ModuleNotFoundError` while `pytest tests/` passes, that is the
situation, and `pip install -e ".[dev]"` again is the fix.

## Running the test suite

```bash
python -m pytest tests/             # all
python -m pytest tests/ -v           # verbose
python -m pytest tests/test_cli_advise.py::test_advise_persists_to_project_context
```

The full suite is 623 tests and runs in about 15 seconds on a laptop.
Two `tests/test_dataset.py` Parquet cases skip unless `pyarrow` is
installed; `pyarrow` is deliberately not in the `dev` extra, so those two
skip in CI as well.

Tests that use a mock Anthropic client live in `tests/test_advisor.py`
and `tests/test_cli_advise.py` — they don't need an `ANTHROPIC_API_KEY`.

## Style

Each of these is a blocking gate in CI, so run them before you push:

- `ruff check src/ tests/` for linting
- `ruff format --check src/ tests/` for formatting (drop `--check` to apply)
- `mypy src/` for type checking
- All public APIs get docstrings; module-level summaries are required
- Keep diffs surgical: when you change one thing, don't reformat the
  surrounding 200 lines

## What CI runs

`.github/workflows/ci.yml`, on every push and pull request to `main`:

- **test** — the suite on ubuntu-latest and windows-latest across Python
  3.10 / 3.11 / 3.12 / 3.13, the eight cells `pyproject.toml` claims to
  support.
- **lint + types** — `ruff check`, `ruff format --check`, `mypy src/`.
- **pandas floor (2.0.0)** — the suite against the oldest pandas
  `pyproject.toml` permits. Development happens on pandas 3.x, so without
  this cell a break at the floor would ship unnoticed. Pinned to Python
  3.11 because pandas 2.0.0 publishes no cp312/cp313 wheel.
- **build + metadata** — builds the sdist and wheel, runs `twine check`,
  and asserts the `slash_commands/*.md` data files are inside the wheel.

## Adding a new command

The structure for adding a new command (say `audit`) is:

1. **Tool layer** (`src/mlcompass/tools/script.py`): pure-Python
   helpers that don't need an LLM (AST parsing, static checks, etc.).
2. **Agent** (`src/mlcompass/agents/audit.py`): an agentlite Agent
   factory and a `get_<thing>(...)` entry point that takes the tool
   output and returns a parsed recommendation.
3. **UI** (`src/mlcompass/ui/audit.py`): rich rendering for the
   agent output.
4. **CLI** (`src/mlcompass/cli.py`): a new `@cli.command()` that
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

- mlcompass version (`mlcompass --version`)
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
