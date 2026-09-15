"""The one-difference guarantee of `ab_protocol.md` section 2.

Every arm gets the same dataset, task, model and settings; each differs from
the one above it in exactly one thing. That claim is the whole experiment — if
a stray token diverges, the measured difference is no longer attributable to
the intervention and every row in `ab_results.csv` becomes uninterpretable.

So it is asserted mechanically rather than trusted. Under A/B 1.1 the
assertion has two shapes, because the arms do:

- `control` vs `advise` differ *inside one prompt*, so the control prompt must
  be a **strict prefix** of the advise prompt. When it is not, the failure says
  where the two diverge instead of printing two walls of text and leaving the
  reader to diff them by eye.
- `advise` vs `advise+audit` differ by a **further turn**, not a longer first
  one. The prefix relation is not what is true there and is not asserted;
  what is asserted is that the `advise+audit` arm's first turn is byte-
  identical to the `advise` arm's only turn.

The process-defect checklist gets the same treatment. Section 4 calls it
frozen and deterministic, so it is tested against scripts whose defects are
known by construction. Sections 3 and 5 freeze the split geometry and the
generation temperature, and both are checked against the protocol text rather
than against a number retyped into the harness.

Since A/B 1.2 the split itself is tested rather than only its settings. §9 A4
records the split having been wrong in a way no setting would have revealed —
the holdout shared 38% of its rows with train, so the score rewarded the
defect the checklist penalises — and the tests for it therefore assert
measured properties of the written CSVs, not that the harness calls a
grouping function.

A/B 1.3 adds the execution environment to the prompt (§9 A5), and the same
discipline applies to it: the tests assert that the package list is a function
of what this interpreter can import, never that it contains a particular name.
A list of packages is exactly the kind of thing that gets written down once and
then quietly stops being true.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmark" / "run_ab.py"

_spec = importlib.util.spec_from_file_location("run_ab", SCRIPT)
assert _spec is not None and _spec.loader is not None
run_ab = importlib.util.module_from_spec(_spec)
sys.modules["run_ab"] = run_ab
_spec.loader.exec_module(run_ab)


CELL = {
    "csv_path": r"C:\tmp\mlcab_abc123\train.csv",
    "target": "Class",
    "columns": "  - V1 (int64)\n  - V2 (int64)\n  - Class (int64)",
}

ADVISE = """\
⚠ Warnings
  • 215 exact duplicate row(s) (28.7% of the data). A random split puts copies
of the same row in both train and test.
"""


def _divergence(control: str, treatment: str) -> str:
    """Explain, in one paragraph, where the two prompts stop agreeing."""
    limit = min(len(control), len(treatment))
    index = next((i for i in range(limit) if control[i] != treatment[i]), limit)
    if index == limit and len(treatment) >= len(control):
        return (
            "the prompts agree on every shared character, so the divergence is "
            "not a prefix mismatch. Control is not a substring for another "
            f"reason (control {len(control)} chars, treatment {len(treatment)})."
        )
    line = control.count("\n", 0, index) + 1
    return (
        f"the prompts diverge at character {index} (line {line}). "
        f"control has {control[index : index + 60]!r}; "
        f"treatment has {treatment[index : index + 60]!r}. "
        "ab_protocol.md section 2 allows exactly one difference between the "
        "arms, the appended mlcompass block, so the treatment prompt must "
        "begin with the control prompt verbatim."
    )


@pytest.mark.parametrize(
    ("advise", "label"),
    [
        (ADVISE, "findings present"),
        ("", "no findings — section 2's empty block"),
        ("  • 13 exact duplicate row(s) (2.2% of the data).\n", "small findings"),
    ],
)
def test_control_prompt_is_a_strict_substring_of_treatment(advise: str, label: str) -> None:
    control, treatment = run_ab.build_prompts(
        csv_path=CELL["csv_path"],
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout=advise,
    )
    assert control in treatment, f"[{label}] {_divergence(control, treatment)}"
    assert len(treatment) > len(control), (
        f"[{label}] the treatment prompt is not strictly longer than the control "
        f"prompt ({len(treatment)} vs {len(control)} characters). The arms would "
        "then be identical and the experiment would measure nothing."
    )
    assert treatment.startswith(control), (
        f"[{label}] control is a substring but not a prefix of treatment, which "
        "means the block was interleaved rather than appended: "
        f"{_divergence(control, treatment)}"
    )


def test_the_only_difference_is_the_mlcompass_block() -> None:
    """Removing the block from the advise prompt returns the control prompt."""
    control, treatment = run_ab.build_prompts(
        csv_path=CELL["csv_path"],
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout=ADVISE,
    )
    remainder = treatment[len(control) :]
    assert ADVISE in remainder
    assert treatment.replace(remainder, "") == control
    # The block must not smuggle in advice of its own: everything in it that is
    # not mlcompass's bytes is structural framing.
    framing = remainder.replace(ADVISE, "")
    assert "duplicate" not in framing.lower()
    assert "seed" not in framing.lower()
    assert "leak" not in framing.lower()


def test_an_empty_findings_block_is_still_present() -> None:
    """Section 2: no findings means an empty block, never a hint."""
    control, treatment = run_ab.build_prompts(
        csv_path=CELL["csv_path"],
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout="",
    )
    remainder = treatment[len(control) :]
    assert "mlcompass advise" in remainder
    assert not any(
        word in remainder.lower() for word in ("should", "recommend", "make sure", "remember")
    )


def test_the_advise_prompt_carries_no_audit_block() -> None:
    """A1: `audit` is a separate arm and a separate turn, not a second block here.

    Under A/B 1.0 the treatment prompt carried an `--- mlcompass audit ---`
    section that was always empty, because no script exists when the first
    prompt is built. §9 A1 moved audit to its own arm as a revision round, so
    the first-turn prompt must no longer mention it at all: an empty section
    headed with the command's name tells the model a check ran and found
    nothing, which is exactly the advice §2 forbids the harness to invent.
    """
    _control, advise = run_ab.build_prompts(
        csv_path=CELL["csv_path"],
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout=ADVISE,
    )
    assert "mlcompass audit" not in advise
    assert "audit" not in advise.lower()


def test_a_per_arm_workspace_path_would_break_the_guarantee() -> None:
    """The regression that caught this harness out on its first live run.

    The train.csv path is inside the prompt. When each arm got its own
    `mkdtemp` workspace, the two prompts differed in the directory name as
    well as in the mlcompass block — a second difference, invisible to a test
    that builds both prompts from one path. `run_cell` now takes a workspace
    shared by the arms of a cell and re-asserts the relation at runtime; this
    fixes the shape of the failure it is guarding against.
    """
    control, _ = run_ab.build_prompts(
        csv_path=r"C:\tmp\mlcab_aaaaaaa\train.csv",
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout="",
    )
    _, treatment = run_ab.build_prompts(
        csv_path=r"C:\tmp\mlcab_bbbbbbb\train.csv",
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout=ADVISE,
    )
    assert not treatment.startswith(control)
    assert "diverge at character" in _divergence(control, treatment)


# --------------------------------------------------------------------------- #
# The frozen checklist                                                        #
# --------------------------------------------------------------------------- #

CLEAN = """\
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(0)
df = pd.read_csv("train.csv")
df = df.drop_duplicates()
X = df.drop(columns=["Class"])
y = df["Class"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_tr, y_tr)
print(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
"""

DIRTY = """\
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("train.csv")
scaler = StandardScaler()
Xs = scaler.fit_transform(df)
model = LogisticRegression()
model.fit(Xs, df["Class"])
print(accuracy_score(df["Class"], model.predict(Xs)))
"""


def test_clean_script_scores_zero_defects() -> None:
    result = run_ab.check_defects(CLEAN, target="Class", duplicate_rows=215, minority_fraction=0.24)
    assert result["flags"] == {d: False for d in run_ab.DEFECT_IDS}, result["flags"]
    assert result["defect_count"] == 0


def test_dirty_script_scores_every_defect_its_data_allows() -> None:
    result = run_ab.check_defects(DIRTY, target="Class", duplicate_rows=215, minority_fraction=0.15)
    assert result["flags"] == {
        "no_seed": True,
        "leak_fit_before_split": True,
        "leak_duplicate_rows": True,
        "wrong_metric_for_imbalance": True,
        "no_validation": True,
        "target_in_features": True,
    }, result["flags"]
    assert result["defect_count"] == 6


def test_data_dependent_defects_cannot_fire_without_the_data_condition() -> None:
    """Two of the six are defined against the input, not the source alone."""
    result = run_ab.check_defects(DIRTY, target="Class", duplicate_rows=0, minority_fraction=0.48)
    assert result["flags"]["leak_duplicate_rows"] is False
    assert result["flags"]["wrong_metric_for_imbalance"] is False
    assert result["defect_count"] == 4


# The two scripts below are verbatim from the first live A/B pair on
# openml-1464 with qwen2.5:7b. Each exposed a real bug in the checker, so each
# is pinned here as the regression rather than paraphrased.

LIVE_CONTROL = """\
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")

X = df[['V1', 'V2', 'V3', 'V4']]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

import joblib
joblib.dump(model, 'trained_model.pkl')
"""

LIVE_TREATMENT = """\
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

data = pd.read_csv("train.csv")

X = data.drop(columns=['Class'])
y = data['Class']

# Handle exact duplicate rows (as suggested by the advisor)
data = data.drop_duplicates()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
dump(model, 'trained_model.joblib')
"""


def test_an_explicit_column_list_excludes_the_target() -> None:
    """`X = df[['V1', 'V2', 'V3', 'V4']]` does not leave Class in the features.

    The first version of the checker only recognised `.drop`, and reported a
    defect against a script that had done nothing wrong.
    """
    result = run_ab.check_defects(
        LIVE_CONTROL, target="Class", duplicate_rows=142, minority_fraction=0.239
    )
    assert result["flags"]["target_in_features"] is False
    assert result["flags"]["leak_duplicate_rows"] is True  # it never de-duplicates


def test_a_dedupe_after_the_features_are_taken_does_not_count() -> None:
    """De-duplicating a frame the split no longer reads removes nothing.

    This script calls `drop_duplicates()` before the `train_test_split` line
    but after `X` and `y` were taken from the frame, so nothing that is split
    has had a duplicate removed. Checking line order against the split call
    alone credited it; the deadline is the derivation of the split's inputs.
    """
    result = run_ab.check_defects(
        LIVE_TREATMENT, target="Class", duplicate_rows=142, minority_fraction=0.239
    )
    assert result["flags"]["leak_duplicate_rows"] is True
    assert result["dedupe_deadline"] == 8  # the line assigning X


def test_a_dedupe_before_the_features_are_taken_does_count() -> None:
    late_dedupe = (
        "\n# Handle exact duplicate rows (as suggested by the advisor)"
        "\ndata = data.drop_duplicates()\n"
    )
    fixed = LIVE_TREATMENT.replace(
        "X = data.drop(columns=['Class'])",
        "data = data.drop_duplicates()\nX = data.drop(columns=['Class'])",
    ).replace(late_dedupe, "")
    result = run_ab.check_defects(
        fixed, target="Class", duplicate_rows=142, minority_fraction=0.239
    )
    assert result["flags"]["leak_duplicate_rows"] is False


def test_a_positional_slice_excludes_the_target_only_when_it_is_last() -> None:
    script = CLEAN.replace('X = df.drop(columns=["Class"])', "X = df.iloc[:, :-1]")
    last = run_ab.check_defects(
        script, target="Class", duplicate_rows=0, minority_fraction=0.4, target_is_last_column=True
    )
    not_last = run_ab.check_defects(
        script, target="Class", duplicate_rows=0, minority_fraction=0.4, target_is_last_column=False
    )
    assert last["flags"]["target_in_features"] is False
    assert not_last["flags"]["target_in_features"] is True


def test_a_column_list_containing_the_target_is_not_an_exclusion() -> None:
    script = LIVE_CONTROL.replace("[['V1', 'V2', 'V3', 'V4']]", "[['V1', 'V2', 'Class']]")
    result = run_ab.check_defects(script, target="Class", duplicate_rows=0, minority_fraction=0.4)
    assert result["flags"]["target_in_features"] is True


def test_the_checklist_is_deterministic() -> None:
    first = run_ab.check_defects(DIRTY, target="Class", duplicate_rows=9, minority_fraction=0.1)
    second = run_ab.check_defects(DIRTY, target="Class", duplicate_rows=9, minority_fraction=0.1)
    assert first == second


@pytest.mark.parametrize(
    ("reply", "expected_head"),
    [
        ("Here you go:\n```python\nimport pandas as pd\n```\nEnjoy.", "import pandas"),
        ("```\nimport pandas as pd\n```", "import pandas"),
        ("import pandas as pd\nprint(1)\n", "import pandas"),
    ],
)
def test_extract_python_finds_the_script(reply: str, expected_head: str) -> None:
    source, _how = run_ab.extract_python(reply)
    assert source is not None
    assert source.startswith(expected_head)


def test_extract_python_reports_a_reply_with_no_code() -> None:
    source, how = run_ab.extract_python("I am sorry, I cannot help with that.")
    assert source is None
    assert "no Python" in how


def test_extract_python_keeps_a_truncated_block() -> None:
    """A cut-off reply is kept and allowed to fail at execution, not discarded."""
    source, how = run_ab.extract_python("```python\nimport pandas as pd\ndf = pd.read")
    assert source is not None
    assert "unclosed fence" in how


# --------------------------------------------------------------------------- #
# A/B 1.1 §9 A1 — the third arm                                               #
# --------------------------------------------------------------------------- #

AUDIT = """\
⚠ error    seed              No random seed is set anywhere in the script.
⚠ warning  val_split         No validation split detected in the script.
"""


def test_the_three_arms_are_control_advise_and_advise_plus_audit() -> None:
    """§2 since 1.1 names three arms; `treatment` is a retired 1.0 name.

    The arm list is deliberately not tied to the protocol version here. 1.2
    changed the split and left §2 alone, and a test that re-asserted the
    version on every unrelated amendment would have to be edited each time —
    which is how a test stops being read. The version has its own assertion in
    the A4 section below.
    """
    assert run_ab.ARMS == ("control", "advise", "advise+audit")


def test_the_advise_audit_arms_first_turn_is_byte_identical_to_the_advise_arms_only_turn() -> None:
    """A1's comparability claim, stated as what is actually true of a revision round.

    §2's single-difference rule is a *prefix* relation between `control` and
    `advise`, because those two differ inside one prompt. `advise+audit`
    differs by carrying a further turn rather than a longer first one, so the
    prefix relation is not what holds and is not what is asserted. What is
    asserted — and what makes the two arms comparable — is that the revision
    round begins from the identical first turn, byte for byte. The
    control/advise prefix assertion above is untouched by this.
    """
    control, advise = run_ab.build_prompts(
        csv_path=CELL["csv_path"],
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout=ADVISE,
    )
    messages = run_ab.build_revision_messages(
        advise_prompt=advise,
        first_reply="```python\nimport pandas as pd\n```",
        audit_stdout=AUDIT,
    )
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == advise, _divergence(advise, messages[0]["content"])
    # And it is the *advise* prompt, not the control prompt: the revision round
    # is a third arm downstream of the second, not downstream of the first.
    assert messages[0]["content"] != control
    assert messages[0]["content"].startswith(control)


def test_an_empty_audit_result_still_produces_a_revision_turn() -> None:
    """§2, applied to the new round: an arm never receives advice the tool did not produce.

    `mlcompass audit` finding nothing is a result, not a reason to skip the
    round. Skipping it would silently collapse `advise+audit` into `advise`
    for those cells, and the aggregate would then be a mixture of two arms
    reported as one.
    """
    messages = run_ab.build_revision_messages(
        advise_prompt="PROMPT", first_reply="REPLY", audit_stdout=""
    )
    assert len(messages) == 3
    body = messages[2]["content"]
    assert "mlcompass audit" in body
    # The revision turn is a frame around audit's bytes and carries no advice
    # of its own — not even a hint about what a revision might address.
    assert not any(
        word in body.lower()
        for word in ("should", "recommend", "make sure", "remember", "seed", "leak", "duplicate")
    )


DATASET = {
    "dataset_id": "unit-1",
    "name": "unit fixture",
    "openml_id": 9999,
    "target": "Class",
    "task": "binary_classification",
}
MEMBER = {
    "provider": "openai",
    "base_url": "http://localhost:11434/v1",
    "model": "unit-model",
    "key_name": None,
}

FIRST_TURN_SCRIPT = "# FIRST-TURN-SENTINEL\nimport pandas as pd\nprint('first')\n"
REVISED_SCRIPT = "# REVISED-SENTINEL\nimport pandas as pd\nprint('revised')\n"
CONTROL_ARM_SCRIPT = "# CONTROL-ARM-SENTINEL\nimport pandas as pd\nprint('control')\n"


def _split_fixture(tmp_path):
    train = tmp_path / "train.csv"
    train.write_text("V1,Class\n1,0\n2,1\n", encoding="utf-8")
    holdout = tmp_path / "holdout.csv"
    holdout.write_text("V1,Class\n3,0\n", encoding="utf-8")
    return {
        "train_csv": train,
        "holdout_csv": holdout,
        "columns": "  - V1 (int64)\n  - Class (int64)",
        "seed": 7,
        "holdout_fraction": run_ab.HOLDOUT_FRACTION,
        "stratified": True,
        "train_rows": 2,
        "holdout_rows": 1,
        # §9 A4's pair, carried here too: `scoring.md` prints both on every run
        # and a fixture missing them would let the writer drift unnoticed.
        "grouped_on_duplicates": True,
        "holdout_rows_present_in_train": 0,
        "duplicate_rows": 0,
        "minority_fraction": 0.5,
        "target_is_last_column": True,
    }


def test_the_advise_audit_arm_audits_only_its_own_first_turn_script(tmp_path, monkeypatch) -> None:
    """§2: "Each arm audits only its **own** script."

    The cheap wrong implementation is to audit whatever script is lying around
    — the control arm's, or the `advise` arm's from the run before — and feed
    the findings into this arm. That would turn three comparable conditions
    into one three-step condition, and the `advise+audit` column would then be
    measuring the control arm's mistakes. Here the control arm's source is put
    on disk first, with its own sentinel, and the test asserts that what
    reached `mlcompass audit` is this arm's own first turn and nothing else.
    """
    audited: list[str] = []
    audited_paths: list[str] = []

    def fake_audit(script_path, timeout):  # noqa: ANN001
        audited.append(Path(script_path).read_text(encoding="utf-8"))
        audited_paths.append(str(script_path))
        return {
            "audit_stdout": AUDIT,
            "audit_status": "ok",
            "audit_stderr": "",
            "audit_command": ["mlcompass", "audit", str(script_path)],
        }

    replies = iter(
        [
            f"```python\n{FIRST_TURN_SCRIPT}```",
            f"```python\n{REVISED_SCRIPT}```",
        ]
    )

    def fake_call_model(member, messages, timeout):  # noqa: ANN001
        return {
            "ok": True,
            "text": next(replies),
            "error": "",
            "input_tokens": 10,
            "output_tokens": 20,
            "finish_reason": "stop",
            "seconds": 0.1,
            "temperature_requested": run_ab.TEMPERATURE,
            "temperature_status": "sent",
            "temperature_used": run_ab.TEMPERATURE,
            "temperature_rejection": "",
        }

    monkeypatch.setattr(run_ab, "mlcompass_audit", fake_audit)
    monkeypatch.setattr(run_ab, "call_model", fake_call_model)
    monkeypatch.setattr(
        run_ab,
        "score_holdout",
        lambda *a, **k: {
            "status": "unscorable",
            "metric": "",
            "score": "",
            "artifact": "",
            "attempts": [],
            "reason": "unit test does not score",
        },
    )

    run_dir = tmp_path / "run"
    workspace = tmp_path / "ws"
    # A neighbouring arm's output, already on disk where a sloppy audit could
    # pick it up.
    (tmp_path / "control_arm_emitted.py").write_text(CONTROL_ARM_SCRIPT, encoding="utf-8")

    outcome = run_ab.run_cell(
        dataset=DATASET,
        arm="advise+audit",
        panel_id="unit",
        member=MEMBER,
        split=_split_fixture(tmp_path),
        findings={"advise_stdout": ADVISE, "advise_status": "ok"},
        run_dir=run_dir,
        workspace=workspace,
        llm_timeout=5,
        script_timeout=30,
        score_timeout=5,
    )

    assert len(audited) == 1, "the revision round must audit exactly once"
    assert "FIRST-TURN-SENTINEL" in audited[0]
    assert "CONTROL-ARM-SENTINEL" not in audited[0]
    assert "REVISED-SENTINEL" not in audited[0]

    # The revised script is the one that gets scored (§2: "Scoring uses the
    # revised script"), and the pre-revision script is preserved beside it.
    assert (run_dir / "emitted.py").read_text(encoding="utf-8") == REVISED_SCRIPT
    assert (run_dir / "emitted_pre_revision.py").read_text(encoding="utf-8") == FIRST_TURN_SCRIPT
    assert outcome["defects_pre_revision"] is not None
    assert outcome["defects"] is not None


def test_the_audited_path_does_not_name_the_arm_the_run_or_the_model(tmp_path, monkeypatch) -> None:
    """Found by the first live `advise+audit` run, not by reading the code.

    `mlcompass audit` prints the path of the script it read, and that line goes
    into the revision prompt verbatim. When the audited file lived under
    `runs/<experiment>-<dataset>-advise+audit-<panel>-r1/`, the model was shown
    its own arm name, the dataset id, the panel id and the run id — none of
    which came from mlcompass, and all of which tell it that it is one cell of
    a benchmark. §2 allows the arms to differ only by what mlcompass produced,
    so the file is handed over under a neutral name outside the run directory.
    The bytes are preserved separately, and `run.json` records their digest so
    a reader can still confirm which script was audited.
    """
    audited_paths: list[str] = []

    def fake_audit(script_path, timeout):  # noqa: ANN001
        audited_paths.append(str(script_path))
        return {
            "audit_stdout": "",
            "audit_status": "ok",
            "audit_stderr": "",
            "audit_command": ["mlcompass", "audit", str(script_path)],
        }

    replies = iter([f"```python\n{FIRST_TURN_SCRIPT}```", f"```python\n{REVISED_SCRIPT}```"])
    monkeypatch.setattr(run_ab, "mlcompass_audit", fake_audit)
    monkeypatch.setattr(
        run_ab,
        "call_model",
        lambda member, messages, timeout: {
            "ok": True,
            "text": next(replies),
            "error": "",
            "input_tokens": 1,
            "output_tokens": 1,
            "finish_reason": "stop",
            "seconds": 0.1,
            "temperature_requested": run_ab.TEMPERATURE,
            "temperature_status": "sent",
            "temperature_used": run_ab.TEMPERATURE,
            "temperature_rejection": "",
        },
    )
    monkeypatch.setattr(
        run_ab,
        "score_holdout",
        lambda *a, **k: {
            "status": "unscorable",
            "metric": "",
            "score": "",
            "artifact": "",
            "attempts": [],
            "reason": "unit test does not score",
        },
    )

    run_dir = tmp_path / "ab-20260915-abc123-9999-advise+audit-ollama-qwen2.5-7b-r1"
    outcome = run_ab.run_cell(
        dataset=DATASET,
        arm="advise+audit",
        panel_id="ollama-qwen2.5-7b",
        member=MEMBER,
        split=_split_fixture(tmp_path),
        findings={"advise_stdout": ADVISE, "advise_status": "ok"},
        run_dir=run_dir,
        workspace=tmp_path / "ws",
        llm_timeout=5,
        script_timeout=30,
        score_timeout=5,
    )

    assert len(audited_paths) == 1
    handed_over = audited_paths[0]
    for leak in ("advise+audit", "ab-20260915-abc123", "ollama-qwen2.5-7b", "9999", "unit-1"):
        assert leak not in handed_over, f"the audited path leaks {leak!r}: {handed_over}"

    # Neutrality is not allowed to cost traceability: the digest in the run
    # record ties the audited bytes to the preserved pre-revision script.
    digest = hashlib.sha256(FIRST_TURN_SCRIPT.encode("utf-8")).hexdigest()
    assert outcome["audit"]["audited_sha256"] == digest
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["audit"]["audited_sha256"] == digest


def test_a_control_arm_run_never_reaches_the_auditor(tmp_path, monkeypatch) -> None:
    """The other half of arm independence: the two-arm arms run one turn only."""
    audited: list[str] = []
    monkeypatch.setattr(
        run_ab, "mlcompass_audit", lambda script_path, timeout: audited.append(script_path)
    )
    monkeypatch.setattr(
        run_ab,
        "call_model",
        lambda member, messages, timeout: {
            "ok": True,
            "text": f"```python\n{CONTROL_ARM_SCRIPT}```",
            "error": "",
            "input_tokens": 1,
            "output_tokens": 1,
            "finish_reason": "stop",
            "seconds": 0.1,
            "temperature_requested": run_ab.TEMPERATURE,
            "temperature_status": "sent",
            "temperature_used": run_ab.TEMPERATURE,
            "temperature_rejection": "",
        },
    )
    monkeypatch.setattr(
        run_ab,
        "score_holdout",
        lambda *a, **k: {
            "status": "unscorable",
            "metric": "",
            "score": "",
            "artifact": "",
            "attempts": [],
            "reason": "unit test does not score",
        },
    )
    outcome = run_ab.run_cell(
        dataset=DATASET,
        arm="control",
        panel_id="unit",
        member=MEMBER,
        split=_split_fixture(tmp_path),
        findings=None,
        run_dir=tmp_path / "run",
        workspace=tmp_path / "ws",
        llm_timeout=5,
        script_timeout=30,
        score_timeout=5,
    )
    assert audited == []
    assert outcome["defects_pre_revision"] is None
    # And it records no hash for a prompt it never built. The control arm has
    # no `mlcompass advise` output, so the "advise prompt" it could compute is
    # control-plus-empty-block — a string nothing ever sent. Recording its
    # digest under the same key the two advise arms use puts a third value in
    # a column whose whole purpose is to let a reader confirm those two agree.
    record = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert "control_prompt_sha256" in record
    assert "advise_prompt_sha256" not in record
    assert record["prompt_sha256"] == record["control_prompt_sha256"]


def test_the_advised_csv_path_does_not_name_the_experiment_or_the_benchmark(
    tmp_path, monkeypatch
) -> None:
    """The same leak as the audited path, one command upstream.

    `mlcompass advise` prints the path of the CSV it read, and that line goes
    into both advise arms' prompt verbatim. Reading it straight out of
    `benchmark/runs/<experiment_id>/_splits/` therefore told the model it was a
    cell of a benchmark — something the control arm is never told, and
    something mlcompass did not find in the data. §2 lets the arms differ by
    what mlcompass produced about the dataset; "you are run 3 of an
    experiment" is not that. The bytes advise reads are unchanged.
    """
    seen: list[list[str]] = []
    handed_bytes: list[str] = []

    def fake_run(command, timeout):  # noqa: ANN001
        seen.append(list(command))
        # Read it here: the throwaway directory is deleted before the call
        # returns, which is the point of it being throwaway.
        handed_bytes.append(Path(command[-3]).read_text(encoding="utf-8"))
        return "⚠ Warnings\n  • 142 exact duplicate row(s).\n", "ok", ""

    monkeypatch.setattr(run_ab, "_run_mlcompass", fake_run)

    split_dir = tmp_path / "runs" / "ab-20260915-a736b4" / "_splits" / "openml-1464-seed20260915"
    split_dir.mkdir(parents=True)
    train = split_dir / "train.csv"
    train.write_text("V1,Class\n1,0\n2,1\n1,0\n", encoding="utf-8")

    findings = run_ab.mlcompass_findings(train, "Class", 30)

    assert len(seen) == 1
    handed_over = seen[0][-3]  # the positional CSV argument
    for leak in ("ab-20260915-a736b4", "_splits", "openml-1464", "runs"):
        assert leak not in handed_over, f"the advised path leaks {leak!r}: {handed_over}"
    # Same bytes, so advise is describing the same data it always was.
    assert handed_bytes == [train.read_text(encoding="utf-8")]
    assert findings["advise_source_sha256"] == hashlib.sha256(train.read_bytes()).hexdigest()


def test_both_advise_arms_record_the_same_first_turn_digest(tmp_path, monkeypatch) -> None:
    """The comparability claim, checkable from the evidence without rebuilding a prompt.

    §2 makes `advise` and `advise+audit` comparable only if they open from the
    identical turn. `run.json` therefore has to carry a digest that is equal
    across the two and visibly so — that is what a reader of `runs/` checks
    when they do not trust the harness.
    """

    def fake_call(member, messages, timeout):  # noqa: ANN001
        return {
            "ok": True,
            "text": f"```python\n{FIRST_TURN_SCRIPT}```",
            "error": "",
            "input_tokens": 1,
            "output_tokens": 1,
            "finish_reason": "stop",
            "seconds": 0.1,
            "temperature_requested": run_ab.TEMPERATURE,
            "temperature_status": "sent",
            "temperature_used": run_ab.TEMPERATURE,
            "temperature_rejection": "",
        }

    monkeypatch.setattr(run_ab, "call_model", fake_call)
    monkeypatch.setattr(
        run_ab,
        "mlcompass_audit",
        lambda script_path, timeout: {
            "audit_stdout": AUDIT,
            "audit_status": "ok",
            "audit_stderr": "",
            "audit_command": [],
        },
    )
    monkeypatch.setattr(
        run_ab,
        "score_holdout",
        lambda *a, **k: {
            "status": "unscorable",
            "metric": "",
            "score": "",
            "artifact": "",
            "attempts": [],
            "reason": "unit test does not score",
        },
    )

    digests = {}
    # One workspace path, shared by the arms of a cell and cleared before each
    # run, exactly as `main` does it. The path is inside the prompt, so a
    # per-arm workspace would put a second difference between the arms — the
    # regression `test_a_per_arm_workspace_path_would_break_the_guarantee`
    # pins.
    workspace = tmp_path / "ws"
    for arm in ("advise", "advise+audit"):
        run_dir = tmp_path / arm.replace("+", "-")
        run_ab.run_cell(
            dataset=DATASET,
            arm=arm,
            panel_id="unit",
            member=MEMBER,
            split=_split_fixture(tmp_path),
            findings={"advise_stdout": ADVISE, "advise_status": "ok"},
            run_dir=run_dir,
            workspace=workspace,
            llm_timeout=5,
            script_timeout=30,
            score_timeout=5,
        )
        digests[arm] = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert digests["advise"]["prompt_sha256"] == digests["advise+audit"]["prompt_sha256"]
    assert (
        digests["advise"]["advise_prompt_sha256"]
        == (digests["advise+audit"]["advise_prompt_sha256"])
    )
    # The revision turn is the third arm's alone, and has its own digest.
    assert "revision_prompt_sha256" in digests["advise+audit"]
    assert "revision_prompt_sha256" not in digests["advise"]


# --------------------------------------------------------------------------- #
# A/B 1.1 §9 A2 — split geometry frozen by the protocol, not by the harness    #
# --------------------------------------------------------------------------- #

PROTOCOL_PATH = ROOT / "benchmark" / "ab_protocol.md"


def test_the_split_geometry_the_harness_uses_is_the_one_the_protocol_freezes() -> None:
    """A2: 0.25 and stratified-for-classification are §3's now, not the harness's.

    Under 1.0 these were the harness's own defaults, documented as "not fixed
    by ab_protocol.md". §9 A2 froze them in the protocol text, so the harness
    has to agree with that text rather than carry a number someone retyped.
    A4 added a third value to the same paragraph, and it is checked the same
    way — see the A4 section below for why that matters more than the others.
    """
    assert run_ab.HOLDOUT_FRACTION == 0.25
    assert run_ab.STRATIFY_CLASSIFICATION is True
    assert run_ab.GROUP_ON_DUPLICATES is True
    run_ab.verify_protocol_constants()  # raises if the harness and §3/§5 disagree


def test_a_harness_that_drifts_from_the_protocols_split_fraction_is_caught() -> None:
    text = PROTOCOL_PATH.read_text(encoding="utf-8").replace(
        "holdout fraction **0.25**", "holdout fraction **0.30**"
    )
    with pytest.raises(run_ab.ProtocolDrift) as excinfo:
        run_ab.verify_protocol_constants(text=text)
    assert "0.3" in str(excinfo.value)


def test_a_harness_that_drops_stratification_is_caught() -> None:
    with pytest.raises(run_ab.ProtocolDrift):
        run_ab.verify_protocol_constants(stratify=False)


def test_no_surface_of_the_harness_still_calls_the_geometry_its_own_choice() -> None:
    """Under 1.0 the docstring, the comment and the frozen plan all said §3 did
    not fix the fraction and the harness had chosen it. A2 makes that sentence
    false wherever it survives, and a frozen plan that repeats it would
    misdescribe the run it governs.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    for stale in (
        "not fixed by ab_protocol.md",
        "it does not fix the fraction",
        "Both are chosen here",
    ):
        assert stale not in source, stale


# --------------------------------------------------------------------------- #
# A/B 1.1 §9 A3 — temperature frozen, sent explicitly, recorded                #
# --------------------------------------------------------------------------- #

OK_RESPONSE = {
    "choices": [{"message": {"content": "```python\nprint(1)\n```"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 22},
}


def test_the_temperature_the_harness_sends_is_the_one_the_protocol_freezes() -> None:
    assert run_ab.TEMPERATURE == 1.0
    with pytest.raises(run_ab.ProtocolDrift):
        run_ab.verify_protocol_constants(temperature=0.0)


def test_every_model_call_sends_the_temperature_explicitly(monkeypatch) -> None:
    """§5: "sent explicitly rather than left to the provider default"."""
    seen: list[dict] = []

    def fake_post(base_url, key, payload, timeout):  # noqa: ANN001
        seen.append(dict(payload))
        return True, OK_RESPONSE, 0.1

    monkeypatch.setattr(run_ab, "_post", fake_post)
    result = run_ab.call_model(MEMBER, [{"role": "user", "content": "hi"}], 10)

    assert seen, "no request was made"
    assert seen[0]["temperature"] == 1.0
    assert result["temperature_status"] == "sent"
    assert result["temperature_used"] == 1.0


def test_a_provider_that_rejects_the_temperature_is_recorded_not_silently_defaulted(
    monkeypatch,
) -> None:
    """§5/A3: a lane at an unknown temperature is not comparable, and must say so.

    The failure mode this guards is the quiet one: drop the parameter, get a
    200, write the row, and leave a reader of `ab_results.csv` believing every
    lane ran at 1.0. The run is allowed to continue — the evidence is worth
    keeping — but `temperature_used` must not claim a number the provider
    never honoured.
    """
    attempts: list[dict] = []

    def fake_post(base_url, key, payload, timeout):  # noqa: ANN001
        attempts.append(dict(payload))
        if "temperature" in payload:
            return (
                False,
                "HTTP 400: Unsupported value: 'temperature' does not support 1.0 "
                "with this model. Only the default (1) is supported.",
                0.1,
            )
        return True, OK_RESPONSE, 0.1

    monkeypatch.setattr(run_ab, "_post", fake_post)
    result = run_ab.call_model(MEMBER, [{"role": "user", "content": "hi"}], 10)

    assert result["ok"] is True
    assert result["temperature_status"] == "rejected"
    assert result["temperature_used"] == "unknown"
    assert "temperature" in result["temperature_rejection"].lower()
    assert any("temperature" in a for a in attempts), "the parameter was never even attempted"
    assert not all("temperature" in a for a in attempts)


def test_a_failure_that_does_not_name_the_temperature_is_not_read_as_a_rejection(
    monkeypatch,
) -> None:
    """Only an attributable rejection licenses dropping the parameter.

    A generic 500 is not evidence that the provider dislikes `temperature`,
    and retrying without it would be the harness inventing an explanation and
    then recording it as fact.
    """
    monkeypatch.setattr(
        run_ab, "_post", lambda *a, **k: (False, "HTTP 503: upstream unavailable", 0.1)
    )
    result = run_ab.call_model(MEMBER, [{"role": "user", "content": "hi"}], 10)
    assert result["ok"] is False
    assert result["temperature_status"] == "failed"
    assert result["temperature_used"] == "unknown"


# --------------------------------------------------------------------------- #
# ab_results.csv — the columns A1 and A3 add                                   #
# --------------------------------------------------------------------------- #


def test_the_results_schema_carries_temperature_and_both_revision_defect_counts() -> None:
    for column in (
        "temperature",
        "temperature_status",
        "defect_count_pre_revision",
        "defect_count_post_revision",
    ):
        assert column in run_ab.RESULT_COLUMNS, column


def test_appending_a_1_1_row_to_a_1_0_results_file_is_refused(tmp_path, monkeypatch) -> None:
    """A 1.0 header and a 1.1 row do not line up, and DictWriter will not say so.

    `csv.DictWriter` writes values in *its* field order into a file whose
    header is someone else's. Every subsequent row would be silently shifted,
    and the corruption would only surface when a reader wondered why an arm
    column held a temperature.
    """
    stale = tmp_path / "ab_results.csv"
    stale.write_text("run_id,experiment_id,arm\nr1,e1,treatment\n", encoding="utf-8")
    monkeypatch.setattr(run_ab, "AB_RESULTS", stale)
    with pytest.raises(run_ab.ResultSchemaMismatch):
        run_ab.append_result({"run_id": "r2", "arm": "advise+audit"})


# --------------------------------------------------------------------------- #
# A/B 1.2 §9 A4 — the split is on duplicate groups                             #
# --------------------------------------------------------------------------- #
#
# A2 froze the fraction and the stratification and stopped there, and the gap
# it left was not cosmetic. A plain stratified split of OpenML 1464 put 71 of
# 187 holdout rows verbatim into train, so `holdout_score` — the measure §1
# calls the one that cannot be argued with — was paying scripts to leak. The
# seven validation runs under 1.1 came out perfectly inverted: every 0-defect
# script scored 0.5552, every 1-defect script scored 0.6835, the defect being
# failure to de-duplicate.
#
# So these tests do not check that the harness calls a grouping function. They
# check the two properties A4 exists to hold apart, which a grouping bug would
# break in opposite directions:
#
#   - no holdout row appears verbatim in train (the leak is gone), and
#   - duplicates survive *inside* train (the defect is still reachable).
#
# De-duplicating the frame before splitting would satisfy the first and
# destroy the second, and `leak_duplicate_rows` would stop being something a
# script can fail at — the benchmark would then measure nothing where it used
# to measure the wrong thing.

SEED = 20260915

# ab_protocol.md §3/§9 A4 states these four numbers for 1464 at fraction 0.25,
# seed 20260915, stratified. They are in the document, so they are checked
# against the code rather than trusted — the same discipline A2 applied to the
# fraction, applied to what the fraction now produces.
PROTOCOL_1464 = {
    "train_rows": 524,
    "holdout_rows": 224,
    "holdout_rows_present_in_train": 0,
    "duplicate_rows": 125,
}


def _pinned_datasets() -> list[dict]:
    """The three datasets `ground_truth.json` pins, as `prepare_split` takes them."""
    return run_ab._load_ground_truth()["datasets"]


def _dataset(dataset_id: str) -> dict:
    return next(d for d in _pinned_datasets() if d["dataset_id"] == dataset_id)


def _rows(path) -> list[tuple]:
    import pandas as pd

    frame = pd.read_csv(path)
    return [tuple(r) for r in frame.itertuples(index=False, name=None)]


def _contamination(split: dict) -> int:
    """Holdout rows that appear verbatim in train, counted from the written files.

    Deliberately recomputed from the two CSVs rather than read out of the
    split record: the record is the harness's own claim, and this is the test
    that decides whether the claim is true.
    """
    train = set(_rows(split["train_csv"]))
    return sum(1 for row in _rows(split["holdout_csv"]) if row in train)


_pinned = pytest.mark.skipif(
    not (ROOT / "benchmark" / "data").is_dir(),
    reason="pinned CSVs absent; run `python benchmark/fetch_datasets.py`",
)


@_pinned
@pytest.mark.parametrize("dataset_id", ["openml-1464", "openml-1480", "openml-44031"])
def test_no_holdout_row_appears_verbatim_in_train(dataset_id: str, tmp_path) -> None:
    """A4's first half, on every pinned dataset.

    Stated as a count rather than as a property of the implementation: a
    harness that stopped grouping, grouped on the wrong key, or grouped and
    then reshuffled rows between the sides would all fail here, and none of
    them would fail a test that asserted a function was called.
    """
    split = run_ab.prepare_split(_dataset(dataset_id), SEED, tmp_path / dataset_id)
    leaked = _contamination(split)
    assert leaked == 0, (
        f"{dataset_id}: {leaked} of {split['holdout_rows']} holdout rows appear "
        "verbatim in train. The holdout is then partly memorised, and a script "
        "that de-duplicates is scored on rows it was denied while one that does "
        "not is scored on rows it kept — which is §9 A4's whole subject."
    )
    # And the harness's own record of the same number agrees, so a reader of
    # `runs/<id>/run.json` sees the checked quantity rather than an assurance.
    assert split["holdout_rows_present_in_train"] == 0


@_pinned
def test_the_1464_split_matches_the_numbers_the_protocol_states(tmp_path) -> None:
    """§3 and §9 A4 print four numbers for this split. They are the code's too.

    A2's lesson generalises: a number written into a document and separately
    into a harness is a number that will eventually differ. §3 says grouping
    leaves 125 duplicates inside train, and that claim is load-bearing — it is
    the sentence saying the defect survives the remedy.
    """
    split = run_ab.prepare_split(_dataset("openml-1464"), SEED, tmp_path / "1464")
    measured = {k: split[k] for k in PROTOCOL_1464}
    assert measured == PROTOCOL_1464, (
        "the 1464 split no longer produces the geometry ab_protocol.md §3 "
        f"describes: {measured} against the document's {PROTOCOL_1464}."
    )


@_pinned
@pytest.mark.parametrize("dataset_id", ["openml-1464", "openml-1480", "openml-44031"])
def test_duplicates_survive_inside_train_where_the_source_has_them(
    dataset_id: str, tmp_path
) -> None:
    """A4's second half: the remedy must not delete the defect.

    mlcompass's own warning names two remedies — de-duplicate before
    splitting, or split on a group key — and A4 takes the second precisely
    because the first would make `leak_duplicate_rows` unfireable. A harness
    that handed the arms a de-duplicated `train.csv` would pass every leak
    test above and quietly turn a six-item checklist into a five-item one.

    Stated conditionally on the source, so it cannot be satisfied by a harness
    that injects duplicates: where the pinned file has none, train has none.
    """
    import pandas as pd

    dataset = _dataset(dataset_id)
    source = pd.read_csv(run_ab.BENCH / dataset["csv_path"])
    source_duplicates = int(source.duplicated().sum())
    split = run_ab.prepare_split(dataset, SEED, tmp_path / dataset_id)

    assert split["source_duplicate_rows"] == source_duplicates
    if source_duplicates == 0:
        assert split["duplicate_rows"] == 0
        return

    assert split["duplicate_rows"] > 0, (
        f"{dataset_id}: the pinned file has {source_duplicates} duplicate rows and "
        "train has none. Grouping is supposed to keep them on one side, not "
        "remove them — `leak_duplicate_rows` is a defect a script commits "
        "against the data it is given, and there is nothing left to commit."
    )
    # Reachable, not merely present: the checklist fires on a script that never
    # calls `drop_duplicates`, against this split's own measured facts.
    never_dedupes = run_ab.check_defects(
        LIVE_CONTROL,
        target=dataset["target"],
        duplicate_rows=split["duplicate_rows"],
        minority_fraction=split["minority_fraction"],
        target_is_last_column=split["target_is_last_column"],
    )
    assert never_dedupes["flags"]["leak_duplicate_rows"] is True


@_pinned
@pytest.mark.parametrize("dataset_id", ["openml-1464", "openml-1480"])
def test_stratification_still_holds_after_grouping(dataset_id: str, tmp_path) -> None:
    """Grouping changes what is being stratified, so the result is measured.

    `train_test_split(stratify=...)` now balances *groups*, and groups hold
    different numbers of rows, so an exactly balanced draw of groups does not
    give an exactly balanced draw of rows. The guarantee A2 bought — that a
    class cannot go missing from the holdout, where ROC AUC is undefined — has
    to be re-established at the row level rather than inherited.

    Tolerance is 5 percentage points, chosen against the reason §3 gives for
    stratifying at all: 1464's minority class is 23.8%, and the failure being
    prevented is a holdout with one class in it, not a holdout a point or two
    off. On 1464 the drift is 1.2pp in train and 2.8pp in the holdout.
    """
    import pandas as pd

    dataset = _dataset(dataset_id)
    target = dataset["target"]
    source = pd.read_csv(run_ab.BENCH / dataset["csv_path"])
    split = run_ab.prepare_split(dataset, SEED, tmp_path / dataset_id)
    train = pd.read_csv(split["train_csv"])
    holdout = pd.read_csv(split["holdout_csv"])

    classes = set(source[target].unique())
    assert set(train[target].unique()) == classes
    assert set(holdout[target].unique()) == classes, (
        f"{dataset_id}: a class is missing from the holdout. §3 stratifies to "
        "prevent exactly this — ROC AUC is undefined on a single-class holdout "
        "and the run is wasted for a reason unrelated to the arms."
    )

    expected = source[target].value_counts(normalize=True)
    for label, frame in (("train", train), ("holdout", holdout)):
        share = frame[target].value_counts(normalize=True)
        drift = (share - expected).abs().max()
        assert drift <= 0.05, (
            f"{dataset_id} {label}: class balance drifted {drift:.4f} from the "
            "source. Grouping stratifies groups; the row-level balance that "
            "follows is what the score is actually computed on."
        )

    # The mechanism, so a failure above can be read: the group-level draw is
    # tight, and any row-level drift comes from groups holding different
    # numbers of rows rather than from stratification having been dropped.
    assert split["stratified"] is True
    assert abs(split["holdout_groups"] / split["duplicate_groups"] - 0.25) < 0.01


@_pinned
def test_a_regression_to_the_ungrouped_split_is_caught_by_the_contamination_count(
    tmp_path,
) -> None:
    """The failure this whole amendment is about, reproduced beside the fix.

    The ungrouped split is built here with the same fraction, the same seed and
    the same stratification, so the only difference between the two is the one
    A4 introduces. It contaminates; the harness's does not. Written this way
    the test is not asserting that some function is called — it is asserting
    that the harness sits on the correct side of a measured difference, which
    is what would still be true if the implementation were replaced.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    dataset = _dataset("openml-1464")
    frame = pd.read_csv(run_ab.BENCH / dataset["csv_path"])
    ungrouped_train, ungrouped_holdout = train_test_split(
        frame,
        test_size=run_ab.HOLDOUT_FRACTION,
        random_state=SEED,
        shuffle=True,
        stratify=frame[dataset["target"]],
    )
    known = {tuple(r) for r in ungrouped_train.itertuples(index=False, name=None)}
    ungrouped_leak = sum(
        1 for r in ungrouped_holdout.itertuples(index=False, name=None) if tuple(r) in known
    )
    assert ungrouped_leak == 71, (
        "the ungrouped baseline no longer leaks 71 rows, so this test is no "
        f"longer reproducing what §9 A4 describes (got {ungrouped_leak})."
    )

    split = run_ab.prepare_split(dataset, SEED, tmp_path / "1464")
    assert _contamination(split) == 0


def test_a_harness_that_stops_grouping_is_caught() -> None:
    """`verify_protocol_constants` covers A4 the way it covers the fraction.

    This is the check that matters most of the three. A wrong fraction makes a
    holdout the wrong size, which is visible in any run record; a harness that
    stopped grouping would go on producing plausible scores, silently inflated
    for exactly the scripts that leak, which is what happened. So a harness
    disagreeing with §3 about grouping aborts before it splits, exactly as one
    disagreeing about 0.25 does.
    """
    with pytest.raises(run_ab.ProtocolDrift):
        run_ab.verify_protocol_constants(grouping=False)


def test_a_protocol_that_no_longer_asks_for_grouping_is_caught() -> None:
    """The check reads the document, so it fails from either side.

    Both directions are drift and both stop the run: the point of reading §3
    back out of the file is that neither the harness nor the protocol can move
    without the other.
    """
    text = PROTOCOL_PATH.read_text(encoding="utf-8").replace(
        "**split on duplicate groups**", "split however you like"
    )
    with pytest.raises(run_ab.ProtocolDrift) as excinfo:
        run_ab.verify_protocol_constants(text=text)
    assert "group" in str(excinfo.value).lower()


def test_the_harness_declares_the_protocol_version_that_carries_a4_and_a5() -> None:
    """The version literal moves with the amendments; A4 must still be in the text.

    1.2 carried A4 and 1.3 carries A5. The version is asserted against the
    document rather than against a remembered number, and A4's own paragraph is
    asserted to still be there — a version bump that quietly dropped an
    amendment would otherwise pass here.
    """
    document = PROTOCOL_PATH.read_text(encoding="utf-8")
    assert run_ab.AB_PROTOCOL_VERSION == "A/B 1.3"
    assert "Protocol version: A/B 1.3" in document
    assert "**A4 — split on duplicate groups.**" in document
    assert "**A5 — state the execution environment in the prompt.**" in document


# --------------------------------------------------------------------------- #
# A/B 1.3 §9 A5 — the execution environment, stated in the prompt and measured  #
# --------------------------------------------------------------------------- #
#
# A validation run's `advise` arm died on `import imblearn`. That is a
# reasonable library to reach for and simply absent here, so `runs_at_all` —
# one of §1's three outcomes — was partly reading this machine's `pip list`.
# The confound is directional: advise output recommends handling class
# imbalance, which points at exactly the resampling libraries most likely to be
# missing, so an unstated environment penalises the treatment arms for taking
# the intervention's advice.
#
# The fix is a package list in the prompt, identical across the arms. What
# these tests are about is the word *measured*: a hand-written list drifts from
# the environment the moment anything is installed or removed, and then the
# prompt is guessing with more confidence than before. So none of them assert
# that the list contains a particular package. They assert that membership is a
# function of what this interpreter can actually import.


def _package_lines(rendered: str) -> list[str]:
    return [line.strip() for line in rendered.splitlines() if line.strip()]


def _package_names(rendered: str) -> list[str]:
    """The import names the rendered list offers, in the order it offers them."""
    return [line.lstrip("- ").split()[0] for line in _package_lines(rendered)]


def test_every_package_the_prompt_names_can_actually_be_imported() -> None:
    """§2: the list "names what is actually importable rather than what we would
    like to be".

    Checked by importing each name in a subprocess of the interpreter that runs
    the emitted script, which is the only authority on the question. A list
    that had been typed out and left to rot would fail here on its first stale
    entry — which is the failure A5 exists to prevent, one level up.
    """
    import subprocess

    names = _package_names(run_ab.environment_packages())
    assert names, "the list is empty; see test_an_empty_package_list_is_refused"
    probe = (
        "import importlib, sys\n"
        f"names = {names!r}\n"
        "bad = []\n"
        "for n in names:\n"
        "    try:\n"
        "        importlib.import_module(n)\n"
        "    except Exception as exc:\n"
        "        bad.append(f'{n}: {exc}')\n"
        "print('|'.join(bad))\n"
    )
    proc = subprocess.run(
        [run_ab.SCRIPT_INTERPRETER, "-X", "utf8", "-c", probe],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", (
        "the prompt names packages that the interpreter running the emitted "
        f"script cannot import: {proc.stdout.strip()}"
    )


def test_a_roster_package_appears_exactly_when_this_environment_has_it() -> None:
    """Membership is decided by the environment, one roster entry at a time.

    The roster is the harness's question — which packages are worth asking
    about — and the environment is the answer. This asserts the answer is never
    overridden in either direction: nothing present is withheld, nothing absent
    is promised.
    """
    import importlib.util

    named = set(_package_names(run_ab.environment_packages()))
    for candidate in run_ab.PACKAGE_ROSTER:
        try:
            present = importlib.util.find_spec(candidate) is not None
        except (ImportError, ValueError):
            present = False
        assert (candidate in named) is present, (
            f"{candidate}: importable={present} but named in the prompt="
            f"{candidate in named}. The list must follow the environment."
        )


def test_the_list_is_read_from_the_environment_rather_than_written_down(monkeypatch) -> None:
    """The test the task asks for: swap the environment, and the list follows.

    A literal would be indifferent to this. The probe is replaced with one that
    reports a different world, and the rendered block has to describe that
    world instead — which is only possible if nothing about the list is
    hard-coded.
    """
    invented = {"xgboost", "torch"}
    monkeypatch.setattr(run_ab, "_importable", lambda name: name in invented)
    monkeypatch.setattr(run_ab, "_distribution_version", lambda name: "")

    rendered = run_ab.environment_packages()
    assert set(_package_names(rendered)) == invented
    # And pandas — importable in the real environment, absent from the invented
    # one — is gone, so the render cannot be carrying a baked-in core list.
    assert "pandas" not in rendered


def test_the_library_that_killed_a_validation_run_is_asked_about() -> None:
    """A5's own case, pinned.

    `imblearn` has to be *on the roster*, so the prompt answers the question
    rather than leaving it open; whether it is *in the list* is the
    environment's business. Pinned as a conditional rather than as "imblearn is
    absent", because installing imbalanced-learn tomorrow should change the
    prompt, not break this test.
    """
    import importlib.util

    assert "imblearn" in run_ab.PACKAGE_ROSTER
    try:
        present = importlib.util.find_spec("imblearn") is not None
    except (ImportError, ValueError):
        present = False
    control, _advise = run_ab.build_prompts(
        csv_path=CELL["csv_path"],
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout="",
    )
    assert ("imblearn" in control) is present


def test_both_arms_are_handed_the_identical_package_list() -> None:
    """§2: "The same list goes to every arm."

    Structural, like the rest of §2's single-difference machinery: the list
    lives in the control prompt, and the advise prompt is control plus a block,
    so the two cannot hold different lists. Asserted anyway, because that is
    the property the experiment depends on rather than the implementation that
    currently provides it.
    """
    packages = run_ab.environment_packages()
    control, advise = run_ab.build_prompts(
        csv_path=CELL["csv_path"],
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout=ADVISE,
        packages=packages,
    )
    assert packages in control
    assert advise.count(packages) == 1
    assert control.count(packages) == 1


def test_the_control_prompt_never_names_the_tool_under_study() -> None:
    """Why the list is curated rather than a `pip list` dump.

    `mlcompass` is installed in this environment. A complete dump would
    therefore put the intervention's own name into the control arm's prompt —
    telling the arm that is defined by *not* being given mlcompass that a thing
    called mlcompass is sitting in its environment. §2 allows the arms to
    differ only by mlcompass's output; it does not allow the control arm to be
    told the tool exists.
    """
    control, _advise = run_ab.build_prompts(
        csv_path=CELL["csv_path"],
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout="",
    )
    assert "mlcompass" not in control.lower()


def test_a_roster_naming_the_tool_under_study_is_refused() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        run_ab.environment_packages(roster=("numpy", "mlcompass"))
    assert "mlcompass" in str(excinfo.value)


def test_the_package_paragraph_states_a_fact_and_gives_no_instruction() -> None:
    """The same discipline §2 imposes on the mlcompass block, applied here.

    An environment statement is allowed to say what is installed. It is not
    allowed to become a nudge — "you should use scikit-learn", "remember to
    handle imbalance" — because that would be the harness adding advice of its
    own to every arm, and the advice would land differently on arms that were
    already pointed at the same subject.
    """
    control, _advise = run_ab.build_prompts(
        csv_path=CELL["csv_path"],
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout="",
    )
    lowered = control.lower()
    for nudge in ("should", "recommend", "make sure", "remember", "be sure to", "avoid"):
        assert nudge not in lowered, nudge


def test_the_prompt_names_the_interpreter_that_will_run_the_script() -> None:
    import platform

    control, _advise = run_ab.build_prompts(
        csv_path=CELL["csv_path"],
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout="",
    )
    assert platform.python_version() in control
    assert sys.executable == run_ab.SCRIPT_INTERPRETER


def test_the_emitted_script_is_executed_by_the_interpreter_the_prompt_describes(
    tmp_path, monkeypatch
) -> None:
    """The claim the prompt makes has to be the one the harness honours.

    The prompt says "the script will be run by Python X with these packages".
    If `run_cell` launched some other interpreter, that sentence would be false
    and the confound A5 closes would be open again under a different name.
    """
    import subprocess as real_subprocess
    from types import SimpleNamespace

    launched: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        launched.append(list(command))
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(
        run_ab,
        "subprocess",
        SimpleNamespace(run=fake_run, TimeoutExpired=real_subprocess.TimeoutExpired),
    )
    monkeypatch.setattr(
        run_ab,
        "call_model",
        lambda member, messages, timeout: {
            "ok": True,
            "text": f"```python\n{CONTROL_ARM_SCRIPT}```",
            "error": "",
            "input_tokens": 1,
            "output_tokens": 1,
            "finish_reason": "stop",
            "seconds": 0.1,
            "temperature_requested": run_ab.TEMPERATURE,
            "temperature_status": "sent",
            "temperature_used": run_ab.TEMPERATURE,
            "temperature_rejection": "",
        },
    )
    monkeypatch.setattr(
        run_ab,
        "score_holdout",
        lambda *a, **k: {
            "status": "unscorable",
            "metric": "",
            "score": "",
            "artifact": "",
            "attempts": [],
            "reason": "unit test does not score",
        },
    )

    run_ab.run_cell(
        dataset=DATASET,
        arm="control",
        panel_id="unit",
        member=MEMBER,
        split=_split_fixture(tmp_path),
        findings=None,
        run_dir=tmp_path / "run",
        workspace=tmp_path / "ws",
        llm_timeout=5,
        script_timeout=30,
        score_timeout=5,
    )

    assert launched, "the emitted script was never executed"
    assert launched[0][0] == run_ab.SCRIPT_INTERPRETER
    assert launched[0][-1] == "emitted.py"


def test_an_empty_package_list_is_refused() -> None:
    """A prompt that promises a list and shows nothing is worse than no prompt.

    It reads as "nothing is installed", which is never true and which no arm
    could write a training script against. If the probe comes back empty the
    harness stops rather than sending it.
    """
    with pytest.raises(RuntimeError) as excinfo:
        run_ab.environment_packages(roster=("no_such_package_at_all_9c1f",))
    assert "no importable package" in str(excinfo.value).lower()


def test_the_package_list_is_deterministic() -> None:
    assert run_ab.environment_packages() == run_ab.environment_packages()


def test_a_harness_that_stops_stating_the_environment_is_caught() -> None:
    """A5 is checkable the way A2/A3/A4 are: against §2's own words.

    The failure being guarded is the quiet one again. A harness that dropped
    the package list would keep producing rows that look fine, with a share of
    the failures caused by an absent import rather than by the model — and the
    share would be larger on the arms that were told to handle imbalance.
    """
    with pytest.raises(run_ab.ProtocolDrift) as excinfo:
        run_ab.verify_protocol_constants(packages=False)
    assert "package" in str(excinfo.value).lower()


def test_a_protocol_that_no_longer_asks_for_a_package_list_is_caught() -> None:
    text = PROTOCOL_PATH.read_text(encoding="utf-8").replace(
        "the list of packages available in the execution environment,", "", 1
    )
    with pytest.raises(run_ab.ProtocolDrift) as excinfo:
        run_ab.verify_protocol_constants(text=text)
    assert "package" in str(excinfo.value).lower()
