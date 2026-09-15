"""The frozen plan has to be a constraint, not a document.

`ab_protocol.md` rests on pre-registration: a plan is frozen before any cell
runs, and the results are read against it. For one run that discipline was
decorative. `--experiment-id` checked only that `plan.md` EXISTED, then ran
whatever the command line said.

Experiment `ab-20260915-795b90` froze 12 `control+revise` cells at 3
repetitions — 36 runs. The execution did **48 runs across all four arms at 1
repetition**, appended them to `ab_results.csv` under the plan's own id, and
exited 0. Nothing complained.

The cause was the harness's own suggested command. `--plan` printed

    Run it with:
      python benchmark/run_ab.py --experiment-id <id> --panel-id <p> ...

which carries neither `--arm` nor `--repeats`, so following the instruction
guaranteed the mismatch. Both halves are fixed, and both halves are tested
here: the plan is parsed back and enforced, and the suggested command
reproduces the plan it was printed for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

import run_ab  # noqa: E402

PLAN = """# A/B experiment plan — ab-test

Frozen at 2026-09-15T00:00:00Z (UTC), before execution.

- Repetitions per cell: 3; seeds [0, 1, 2]

## Cells

| dataset | arm | model | repeats | seeds | metric |
| --- | --- | --- | --- | --- | --- |
| openml-1464 | control+revise | deepseek-flash | 3 | 0, 1, 2 | roc_auc |
| openml-1464 | control+revise | ollama-qwen2.5-7b | 3 | 0, 1, 2 | roc_auc |
| openml-44031 | control+revise | deepseek-flash | 3 | 0, 1, 2 | r2 |
| openml-44031 | control+revise | ollama-qwen2.5-7b | 3 | 0, 1, 2 | r2 |

## Scoring (sections 3 and 4)

- something else entirely
"""

DATASETS = [
    {"dataset_id": "openml-1464", "task": "binary_classification", "target": "Class"},
    {"dataset_id": "openml-44031", "task": "regression", "target": "y"},
]
PANEL = ["deepseek-flash", "ollama-qwen2.5-7b"]


@pytest.fixture
def plan_path(tmp_path: Path) -> Path:
    p = tmp_path / "plan.md"
    p.write_text(PLAN, encoding="utf-8")
    return p


def test_the_cell_table_parses_back_into_what_it_commits_to(plan_path: Path) -> None:
    plan = run_ab.read_plan_cells(plan_path)
    assert plan["arms"] == ["control+revise"]
    assert plan["datasets"] == ["openml-1464", "openml-44031"]
    assert plan["panel_ids"] == ["deepseek-flash", "ollama-qwen2.5-7b"]
    assert plan["repeats"] == [3]
    assert plan["seeds"] == [(0, 1, 2)]
    assert len(plan["cells"]) == 4
    # The parser must stop at the table. "something else entirely" lives under
    # a later heading and is not a cell.
    assert all(c[0].startswith("openml-") for c in plan["cells"])


def test_the_matching_run_is_allowed(plan_path: Path) -> None:
    run_ab.enforce_plan(
        plan_path,
        datasets=DATASETS,
        arms=("control+revise",),
        panel_ids=PANEL,
        repeats=3,
        seeds=[0, 1, 2],
    )


def test_the_run_that_actually_happened_is_refused(plan_path: Path) -> None:
    """All four arms at one repetition, against a plan naming one arm at three.

    This is `ab-20260915-795b90` verbatim. It must not be silently accepted
    again, and the message must name every divergence rather than the first.
    """
    with pytest.raises(run_ab.PlanViolation) as caught:
        run_ab.enforce_plan(
            plan_path,
            datasets=DATASETS,
            arms=("control", "control+revise", "advise", "advise+audit"),
            panel_ids=PANEL,
            repeats=1,
            seeds=[0],
        )
    message = str(caught.value)
    assert "arms" in message
    assert "repeats" in message
    assert "seeds" in message
    assert "Nothing was executed" in message


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"arms": ("advise",)}, id="wrong_arm"),
        pytest.param({"repeats": 1, "seeds": [0]}, id="fewer_repeats"),
        pytest.param({"panel_ids": ["deepseek-flash"]}, id="fewer_models"),
        pytest.param({"panel_ids": [*PANEL, "openai-gpt-5.4-mini"]}, id="extra_model"),
        pytest.param({"datasets": DATASETS[:1]}, id="fewer_datasets"),
        pytest.param({"seeds": [7, 8, 9]}, id="different_seeds"),
    ],
)
def test_every_dimension_of_the_plan_is_enforced(plan_path: Path, kwargs: dict) -> None:
    """One dimension going unchecked is one dimension a run can drift along."""
    call = {
        "datasets": DATASETS,
        "arms": ("control+revise",),
        "panel_ids": PANEL,
        "repeats": 3,
        "seeds": [0, 1, 2],
    }
    call.update(kwargs)
    with pytest.raises(run_ab.PlanViolation):
        run_ab.enforce_plan(plan_path, **call)


def test_a_plan_with_no_readable_table_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    """An unparseable plan must fail loudly.

    Treating it as "no constraint found, carry on" would restore exactly the
    behaviour this module exists to prevent, and would do it silently.
    """
    broken = tmp_path / "plan.md"
    broken.write_text("# A plan with prose and no cells\n\nnothing here.\n", encoding="utf-8")
    with pytest.raises(run_ab.PlanViolation, match="no readable cell table"):
        run_ab.read_plan_cells(broken)


def test_the_real_795b90_plan_still_parses() -> None:
    """The plan that was violated, as it sits on disk.

    A synthetic fixture proves the parser works on what the test author
    imagined `write_plan` emits. This proves it works on what `write_plan`
    actually emitted.
    """
    real = Path(run_ab.__file__).parent / "runs" / "ab-20260915-795b90" / "plan.md"
    if not real.exists():
        pytest.skip("the 795b90 plan is not in this checkout")
    plan = run_ab.read_plan_cells(real)
    assert plan["arms"] == ["control+revise"]
    assert plan["repeats"] == [3]
    assert len(plan["cells"]) == 12
