"""The corpus counts of the paper describe the runs made up to 2026-09-30.

The registered follow-up runs of analysis plan A9 are reported on their own
(scripts/analyze_a9.py); unified_scoring.run_files leaves them out unless asked,
so adding them cannot move a corpus count such as the 19,132 responses.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import unified_scoring as u  # noqa: E402


def test_a9_runs_are_out_of_the_corpus_by_default():
    rels = [rel for _, rel, _ in u.run_files()]
    assert not any(p in u.A9_DIRS for rel in rels for p in rel)


def test_a9_runs_can_be_included():
    default = {"/".join(rel) for _, rel, _ in u.run_files()}
    everything = {"/".join(rel) for _, rel, _ in u.run_files(include_a9=True)}
    assert default <= everything
