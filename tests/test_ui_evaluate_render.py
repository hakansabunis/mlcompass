"""Renderer contract tests for ``mlcompass evaluate``.

The manuscript's Limitation 4 mitigation is a claim *about the renderer*:
that the validated fields are surfaced as the actionable output and that the
unverified free-text channel is visibly marked as such. A claim about a
renderer is discharged by a renderer test, not by a sentence. These are that
test.

Four properties are asserted here:

1. **Completeness reaches the user.** Proposition 2 says completeness admits
   no safe repair, so the only available remedy is to flag it. A flag that
   stops at a dict key is not a flag. ``omitted_critical_evidence`` and
   ``had_unrecoverable_violation`` must appear in the rendered panel — each
   one exactly when its flag is set, and the omission warning must name the
   column that went unaddressed.
2. **Inside the contract panel, verified content leads and prose is marked.**
   The panel carries both channels: entity and value blocks that passed Tier
   B, and a hypothesis the contract never inspected. Limitation 4's mitigation
   is precisely the claim that the verified channel is the actionable output
   and the free-text one is labelled — an ordering-and-labelling claim about
   this one function, asserted here on the contract panel alone.
3. **The verified/unverified partition is visible.** The contract-governed
   panel and the unguarded interpreter panel are both rendered by the CLI in
   the same terminal output. A reader must be able to tell which one carries
   a verification guarantee.
4. **The validated value channel is rendered at all.** ``claims`` is the
   output of Tier B check (2); it was absent from the panel before ``96f0071``
   and its absence is not detectable without a test.
"""

from __future__ import annotations

import io
import re
from typing import Any

from rich.console import Console

from mlcompass.ui.evaluate import (
    render_evaluation_interpretation,
    render_leakage_narration,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

# Rich draws panel borders with box-drawing characters and hard-wraps the body
# at the console width. Both would break naive substring assertions, so the
# rendered text is flattened before matching: box characters become spaces and
# runs of whitespace collapse to one. Rich wraps on word boundaries, so the
# flattened form is the original sentence.
_BOX_CHARS = frozenset(chr(c) for c in range(0x2500, 0x2580))


def _flatten(text: str) -> str:
    stripped = "".join(" " if ch in _BOX_CHARS else ch for ch in text)
    return " ".join(stripped.split())


def _render(fn: Any, payload: dict[str, Any], *, width: int = 160) -> str:
    console = Console(record=True, width=width, no_color=True, force_terminal=False)
    fn(console, payload)
    return _flatten(console.export_text())


def _render_cli_transcript(
    interpretation: dict[str, Any],
    narration: dict[str, Any],
    *,
    width: int = 160,
    styles: bool = False,
) -> str:
    """Both narrators into one console, in the order ``cli.py`` prints them.

    ``evaluate --llm`` runs the unguarded interpreter first and the contract
    panel second, so the unverified prose is what the reader meets first. Any
    claim about the panels being distinguishable has to hold in this combined
    transcript, not only when each is rendered alone.
    """
    console = Console(
        record=True,
        width=width,
        file=io.StringIO(),
        force_terminal=styles,
        legacy_windows=False,
        no_color=not styles,
    )
    render_evaluation_interpretation(console, interpretation)
    render_leakage_narration(console, narration)
    text = console.export_text(styles=styles)
    return text if styles else _flatten(text)


INTERPRETATION: dict[str, Any] = {
    "assessment": "Strong ranking performance; the threshold is too aggressive.",
    "strengths": ["AUC 0.94 — the model separates the classes well."],
    "weaknesses": ["Precision 0.62 at threshold 0.5."],
    "next_steps": ["Ship threshold 0.65."],
}


# A contract response in the exact shape ``investigate_leakage_bound`` returns.
CLEAN_NARRATION: dict[str, Any] = {
    "verdict": "leakage_likely",
    "confidence": "high",
    "evidence_cited": ["log_target_v2"],
    "primary_hypothesis": "log_target_v2 is a transformed copy of the target.",
    "recommended_checks": ["Inspect the feature pipeline for a leak."],
    "columns_referenced": ["log_target_v2"],
    "claims": [{"column": "log_target_v2", "statistic": "correlation", "value": 1.0}],
    "schema_rejections": 0,
    "rejection_kinds": [],
    "attempts_made": 1,
    "had_unrecoverable_violation": False,
    "omitted_critical_evidence": False,
    "critical_column": "log_target_v2",
    "evidence_bound": True,
}


def _narration(**overrides: Any) -> dict[str, Any]:
    out = dict(CLEAN_NARRATION)
    out.update(overrides)
    return out


# --------------------------------------------------------------------------- #
# Defect 1 — the Proposition-2 flags must reach the user                      #
# --------------------------------------------------------------------------- #


def test_persistent_omission_is_surfaced_to_the_user() -> None:
    """A flagged omission must be visible in the panel, not only in the dict.

    Proposition 2's whole consequence is that completeness can only be
    detected, re-prompted and *flagged*. If the flag never renders, the one
    channel the paper's own theory says the user must be warned about has no
    user-facing surface.
    """
    text = _render(
        render_leakage_narration,
        _narration(
            omitted_critical_evidence=True,
            critical_column="near_target_proxy",
            schema_rejections=3,
            attempts_made=3,
        ),
    )
    lowered = text.lower()
    assert "incomplete" in lowered, text
    assert "top-ranked candidate" in lowered, text
    # The warning must name the column, not merely describe its rank. A reader
    # who has to re-derive the anchor from the evidence panel is doing the
    # verifier's bookkeeping by hand.
    assert "near_target_proxy" in text, text
    # Omission is not a soundness failure: the strip notice must stay silent.
    assert "stripped" not in lowered, text


def test_omission_warning_survives_a_missing_anchor() -> None:
    """A payload without ``critical_column`` still warns, just without a name.

    The key is new; a persisted or third-party narration dict predating it
    must not silence the one warning Proposition 2 says is the only remedy.
    """
    payload = _narration(omitted_critical_evidence=True)
    del payload["critical_column"]
    lowered = _render(render_leakage_narration, payload).lower()
    assert "incomplete" in lowered, lowered
    assert "top-ranked candidate leak column." in lowered, lowered


def test_stripped_violation_is_surfaced_to_the_user() -> None:
    """A silently stripped phantom leaves the user no signal that it happened.

    Stripping *is* the repair, so the response is sound. But the user has no
    way to know the narrator misbehaved, and the telemetry story implies
    otherwise.
    """
    text = _render(
        render_leakage_narration,
        _narration(
            had_unrecoverable_violation=True,
            evidence_cited=[],
            claims=[],
            schema_rejections=3,
            attempts_made=3,
        ),
    )
    lowered = text.lower()
    # "stripped", not "not in the evidence": the prose label already contains
    # the latter ("names and numbers here may not be in the evidence"), so an
    # assertion on it would pass without the warning existing at all.
    assert "stripped" in lowered, text
    assert "did not survive the corrective retry" in lowered, text
    # A strip is not an omission: the completeness warning must stay silent.
    assert "incomplete" not in lowered, text


def test_clean_response_shows_neither_warning() -> None:
    """The warnings are conditional: a clean contract run must stay quiet."""
    text = _render(render_leakage_narration, _narration())
    lowered = text.lower()
    assert "incomplete" not in lowered, text
    assert "stripped" not in lowered, text


def test_contract_warnings_precede_the_findings_they_qualify() -> None:
    """Both flags describe the findings below them, so they render above them.

    A completeness warning printed *after* the evidence it qualifies is read
    only by someone who already read to the bottom of the panel.
    """
    text = _render(
        render_leakage_narration,
        _narration(omitted_critical_evidence=True, had_unrecoverable_violation=True),
    ).lower()
    for warning in ("incomplete", "stripped"):
        assert text.find(warning) != -1, text
        assert text.find(warning) < text.find("evidence cited"), (warning, text)


# --------------------------------------------------------------------------- #
# Defect 2 — inside the contract panel, verified content leads                 #
# --------------------------------------------------------------------------- #
#
# The panel mixes two channels with different warranties: the entity and value
# blocks passed Tier B, the hypothesis and the recommended checks are prose the
# contract never inspected. Limitation 4's mitigation is the claim that the
# verified channel is the actionable output and the free-text one is marked as
# such. That is a statement about ordering and labelling in this function, and
# it holds or fails independently of whether the two *panels* are
# distinguishable, so it is asserted here on the contract panel alone.


def test_verified_blocks_precede_the_model_prose() -> None:
    """Verified findings render above the unverified hypothesis, not below it.

    Leading with the prose puts the one unwarranted sentence at the top of the
    panel and the verified findings beneath it — the reader's first impression
    would come from the channel with no guarantee behind it.
    """
    text = _render(render_leakage_narration, _narration()).lower()

    cited_at = text.find("evidence cited")
    claims_at = text.find("claims")
    hypothesis_at = text.find("hypothesis")
    for name, index in (("evidence cited", cited_at), ("claims", claims_at)):
        assert index != -1, (name, text)
    assert hypothesis_at != -1, text
    assert cited_at < hypothesis_at, text
    assert claims_at < hypothesis_at, text

    # And the last verification label sits above the prose, so no "verified"
    # marker can be read as covering the sentence underneath it.
    assert text.rfind("verified against the evidence") < hypothesis_at, text


def test_contract_panel_prose_is_labelled_not_verified() -> None:
    """The hypothesis block carries its disclaimer on its own header.

    Not in the panel somewhere: on the block, so a reader skimming to the
    prose meets the label without having to scan back up.
    """
    text = _render(render_leakage_narration, _narration()).lower()
    hypothesis_at = text.find("hypothesis")
    assert hypothesis_at != -1, text
    header = text[hypothesis_at : hypothesis_at + 120]
    assert "not verified" in header, header
    assert "model prose" in header, header


def test_recommended_checks_are_not_labelled_verified() -> None:
    """The check list is passed through as written and must not claim otherwise."""
    text = _render(render_leakage_narration, _narration()).lower()
    checks_at = text.find("recommended manual checks")
    assert checks_at != -1, text
    header = text[checks_at : checks_at + 80]
    assert "not verified" in header, header


# --------------------------------------------------------------------------- #
# Defect 3 — the guarded path is not the only path, so say which is which      #
# --------------------------------------------------------------------------- #
#
# ``interpret_evaluation`` is a second narrator over the same evaluation dict
# (which carries ``leakage_investigation``). It has no enum, no verification
# and no contract — only a ``required_keys`` shape check — and the CLI renders
# it *above* the contract panel. It is a shipped feature and cannot be deleted,
# so the remedy available in the renderer is to make the partition legible: the
# reader must be able to tell, without reading the source, which narrator is
# covered by a guarantee and which is not.


def test_unguarded_interpreter_is_labelled_unverified() -> None:
    """The interpreter panel must say, in words, that nothing verified it."""
    text = _render(render_evaluation_interpretation, INTERPRETATION).lower()
    assert "not verified" in text, text


def test_interpreter_does_not_borrow_the_contract_verification_language() -> None:
    """Only the contract panel may claim evidence verification.

    The interpreter is handed the same evidence and comments on it. If it
    renders in the same register as the verified panel, the guarantee has been
    laundered onto output that carries none.
    """
    text = _render(render_evaluation_interpretation, INTERPRETATION).lower()
    assert "verified against the evidence" not in text, text


def test_the_two_narrators_are_distinguishable_in_one_transcript() -> None:
    """In the combined CLI output, each panel states its own status.

    The unverified panel comes first, exactly as ``cli.py`` prints it, so the
    unverified marker must precede the verified one — the reader meets the
    disclaimer before the prose it applies to.
    """
    text = _render_cli_transcript(INTERPRETATION, _narration()).lower()

    unverified_at = text.find("not verified")
    verified_at = text.find("verified against the evidence")
    assert unverified_at != -1, text
    assert verified_at != -1, text
    assert unverified_at < verified_at, text

    # And the two panels carry different titles, so they cannot be read as
    # two halves of one report.
    assert "assessment" in text
    assert "leakage investigator" in text


def _border_colour(fn: Any, payload: dict[str, Any]) -> str:
    """The ANSI colour of a panel's top border.

    Rendered alone, so the other panel's border cannot bleed into the slice.
    The first styled run on the first non-blank line is the top border.
    """
    console = Console(
        record=True, width=160, file=io.StringIO(), force_terminal=True, legacy_windows=False
    )
    fn(console, payload)
    styled = console.export_text(styles=True)
    first_line = next(line for line in styled.splitlines() if line.strip())
    match = re.search(r"\x1b\[([0-9;]*)m", first_line)
    return match.group(1) if match else ""


def test_the_two_panels_do_not_share_a_border_colour() -> None:
    """The distinction is carried visually as well as lexically.

    Green is this renderer's success colour ("✓ No evaluation warnings",
    "✓ Strengths"). An unverified narrator drawn in the success colour, above
    the verified panel, inverts the signal.
    """
    interpreter = _border_colour(render_evaluation_interpretation, INTERPRETATION)
    contract = _border_colour(render_leakage_narration, _narration())
    assert interpreter != contract, (interpreter, contract)
    assert interpreter != "32", interpreter  # 32 = green = this renderer's "pass"


# --------------------------------------------------------------------------- #
# Defect 4 — the validated value channel must actually be on screen            #
# --------------------------------------------------------------------------- #
#
# At ``96f0071^`` the string "claims" did not occur anywhere in this module:
# Tier B check (2) ran, produced a validated claim set, and the renderer threw
# it away. That is invisible without a test, which is why there is one now.


def test_validated_claims_are_rendered() -> None:
    """``claims`` is the output of Tier B (2) and must reach the panel.

    Column, statistic and value are each asserted: a renderer that printed
    only the column names would satisfy a laxer test while still dropping the
    verified number, which is the whole point of the channel.
    """
    text = _render(
        render_leakage_narration,
        _narration(
            claims=[
                {"column": "log_target_v2", "statistic": "correlation", "value": 1.0},
                {"column": "near_target_proxy", "statistic": "correlation", "value": 0.95},
            ]
        ),
    )
    assert "Claims" in text, text
    for column, value in (("log_target_v2", "1.0"), ("near_target_proxy", "0.95")):
        assert column in text, text
        assert value in text, text
    assert "correlation" in text, text


def test_rendered_claims_are_marked_verified() -> None:
    """The claim block must carry its verification status, like the entity block.

    Without the label the numbers read as prose, and the reader cannot tell
    the verified value channel from the unverified hypothesis beneath it.
    """
    text = _render(render_leakage_narration, _narration()).lower()
    claims_at = text.find("claims")
    assert claims_at != -1, text
    assert "verified against the evidence" in text[claims_at : claims_at + 120], text


def test_claims_block_is_absent_when_no_claims_were_made() -> None:
    """A narrator that made no quantitative claim gets no empty Claims header."""
    text = _render(render_leakage_narration, _narration(claims=[]))
    assert "Claims" not in text, text
