"""The ledger must not lose entries when two processes write it.

`context.py` serialises writers inside one interpreter with a threading lock
and makes each write atomic with temp-then-rename. Neither helps across
processes, and the module's own comment said so: two `mlcompass` processes
writing the same project could interleave read-A / read-B / write-A / write-B,
and B would silently drop A's decision.

That is not a hypothetical configuration. mlcompass ships a CLI and an MCP
server over the same `.mlcompass/` directory, and an editor session running the
server while the developer runs a CLI command in a terminal is the ordinary way
to use the tool. An audit ledger that loses entries under ordinary use is worse
than no ledger, because it looks complete.

These tests spawn real subprocesses rather than threads. A threading test would
pass against the old code, because the old code already serialised threads --
it is precisely the cross-process case that needed evidence.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")

WRITER = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {src!r})
    from mlcompass.context import ProjectContext

    from pathlib import Path
    ctx = ProjectContext(Path({project!r}))
    for i in range({count}):
        ctx.append_decision(command={tag!r}, summary="entry %d" % i)
    """
)


def _spawn(project: Path, tag: str, count: int) -> subprocess.Popen:
    code = WRITER.format(src=SRC, project=str(project), tag=tag, count=count)
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    ProjectContext = pytest.importorskip("mlcompass.context").ProjectContext
    ProjectContext.init("locktest")
    return tmp_path / ".mlcompass"


def test_two_processes_writing_the_ledger_lose_nothing(project: Path) -> None:
    """The measurement this lock exists for.

    Four processes, twenty appends each. Every one of the eighty must survive.
    Under a per-process lock alone this loses roughly half.
    """
    per_process = 20
    tags = ["alpha", "beta", "gamma", "delta"]
    procs = [_spawn(project, tag, per_process) for tag in tags]
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err

    decisions = json.loads((project / "context.json").read_text(encoding="utf-8"))["decisions"]
    assert len(decisions) == len(tags) * per_process, (
        f"expected {len(tags) * per_process} decisions, found {len(decisions)}: "
        "a concurrent writer's update was overwritten"
    )
    for tag in tags:
        got = sum(1 for d in decisions if d["command"] == tag)
        assert got == per_process, f"process {tag} landed {got} of {per_process} entries"


def test_the_file_stays_parseable_throughout(project: Path) -> None:
    """A reader must never observe a torn file.

    Atomicity and exclusion are different properties, and this pins the one the
    lock does not provide: temp-then-rename means every state a reader can
    observe is either the whole old file or the whole new one.
    """
    procs = [_spawn(project, f"w{i}", 15) for i in range(3)]
    context_path = project / "context.json"
    torn = 0
    while any(p.poll() is None for p in procs):
        try:
            json.loads(context_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            torn += 1
        except PermissionError:
            pass  # Windows rename window; not a torn read
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
    assert torn == 0, f"observed {torn} torn or missing reads during concurrent writes"


def test_the_lock_is_reentrant_within_one_process(project: Path) -> None:
    """`append_decision` reads the context while holding the lock.

    A fresh cross-process lock per call would block against its own process
    here -- a self-deadlock that surfaces as a timeout blaming another process.
    One shared lock instance per path is what makes the nesting work, and this
    is the test that fails if that sharing is removed.
    """
    from mlcompass.context import ProjectContext

    ctx = ProjectContext(project)
    ctx.append_decision(command="outer", summary="one")
    ctx.write_context({"active_dataset": "x.csv"})
    ctx.append_decision(command="outer", summary="two")

    data = json.loads((project / "context.json").read_text(encoding="utf-8"))
    assert [d["summary"] for d in data["decisions"]] == ["one", "two"]
    assert data["active_dataset"] == "x.csv"


def test_a_missing_project_directory_does_not_become_a_lock_error(tmp_path: Path) -> None:
    """Locking must not invent a failure mode where there was none.

    With no project directory there is no file to race over and nowhere to put
    a lock file. The caller should get the error it would have got before this
    lock existed, not a FileNotFoundError about a `.lock` file it never asked
    for.
    """
    from mlcompass.context import ProjectContext

    ctx = ProjectContext(tmp_path / "absent" / ".mlcompass")
    with pytest.raises((FileNotFoundError, OSError)) as caught:
        ctx.append_decision(command="advise", summary="x")
    assert ".lock" not in str(caught.value)


def test_a_held_lock_times_out_with_a_usable_message(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error path the linter caught and the tests had not.

    A stale lock file from a process that died holding it would otherwise hang
    every later command with no explanation, so the wait is bounded. The
    message has to name the file to delete -- and it referenced a variable that
    no longer existed, so raising it at all would have produced a NameError
    instead of the diagnosis. Exercised here rather than reasoned about.
    """
    import threading
    import time

    import filelock

    from mlcompass import context as ctx_mod
    from mlcompass.context import ProjectContext

    monkeypatch.setattr(ctx_mod, "_FILE_LOCK_TIMEOUT_S", 0.2)
    held = ctx_mod._file_lock_for(project / "context.json")
    foreign = filelock.FileLock(held.lock_file, timeout=5)

    released = threading.Event()

    def hold() -> None:
        with foreign:
            released.wait(timeout=5)

    holder = threading.Thread(target=hold)
    holder.start()
    time.sleep(0.2)
    try:
        with pytest.raises(TimeoutError) as caught:
            ProjectContext(project).append_decision(command="x", summary="y")
        message = str(caught.value)
        assert "context.json.lock" in message, message
        assert "delete that file" in message, message
    finally:
        released.set()
        holder.join(timeout=5)
