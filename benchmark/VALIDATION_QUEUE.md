# Validation queue

Six pieces of work an external reviewer asked for. None is a wording change;
each one needs something run. Ordered by how much of the paper's argument
rests on it.

| # | Work | Answers | Status |
|---|---|---|---|
| 1 | Validate the final defect scorer | "Why should I trust rescored-v2, after Section 10?" | **1a done; 1b built, not run** |
| 2 | Generic-checklist fifth arm | Is it the specific evidence, or just naming the rubric in the prompt? | built, plan frozen, needs API keys |
| 3 | Guardrails dynamic allowed-values baseline | Is the 35 % a strawman? | **built and tested, needs a battery** |
| 4 | Template-only, LLM-free baseline | If the admissible set is finite, why an LLM at all? | queued |
| 5 | Evidence-domain scalability, 10 / 100 / 500 / 1000 | Does Tier A survive a wide frame? | queued |
| 6 | Tier B mutation / adversarial tests | Tier A caught everything; what is Tier B for? | queued |

---

## 1. Validate the final defect scorer

The threat. Section 10 reports that `check_defects` flagged correct code, that
the fix for one defect introduced another of the same shape, and that four
consecutive corrections moved results toward our hypothesis. A reader is
entitled to ask what makes the current version different from the three that
were wrong. "We looked at it again" is not an answer.

Two things are being built, and they answer different halves of the question.

**1a — differential audit (we can finish this ourselves).** A second scorer,
`independent_defect_scorer.py`, decides the same six rules by a different
method: `ast` over the parsed script rather than regular expressions over its
text. Two rules are deliberately inverted. `check_defects` scores
`target_in_features` and `no_validation` by the *absence* of every enumerated
correct form, which is the shape Section 10 names as the reason the defect
existed: every form nobody thought of becomes a false positive on correct
code. The second scorer flags them only on *positive evidence of the defect*.
Where the two disagree is where the enumeration is incomplete, and that set is
the finding.

This is not an independent replication. Both scorers are ours, written in the
same week, by people who have read each other's. It is a differential audit,
and the paper has to call it that.

**1b — blind human rating (needs two people who are not us).** A stratified
sample of the emitted scripts with arm, model and dataset stripped and the
order randomised, rated against the six rules by two raters who do not know
which arm produced what. Yields checklist precision and recall against a human
reference, and Cohen's $\kappa$ between the raters. The apparatus is built and
the sample is frozen; the ratings are not something a machine can supply, and
the paper should not pretend otherwise.

Deliverables: `benchmark/independent_defect_scorer.py`,
`benchmark/audit_defect_scorers.py`, `benchmark/blind_review/`, and a
subsection in Section 10 reporting whatever the audit actually found.

### What 1a found

Both scorers over 143 scripts, 858 rule-instances. Five rules agreed on every
script ($\kappa = 1.000$). `target_in_features` disagreed twice, and both times
the regex scorer was wrong:

    def train_and_save_model(csv_path, target_column="price"):
        X = df.drop(columns=[target_column])

The target is dropped on line 2. `_target_tokens` harvested target-holding
variables with a line-anchored regex that finds `TARGET = "price"` and not a
parameter default. Both runs exited 0 and were published carrying the defect.

Fixed by parsing (amendment A16), pinned by three tests, rescored as `v4`:

| arm | v3 | v4 |
|---|---|---|
| `control` | 33 | **31** |
| `control+revise` | 35 | 35 |
| `advise` | 24 | 24 |
| `advise+audit` | 8 | 8 |

**This is the first correction in the study to move a result against our own
hypothesis.** The effect shrinks from a 25-defect fall to a 23-defect one.
Section 12 of the manuscript had committed in writing to reporting exactly such
a correction if it arrived.

Re-running the audit after the fix gives 858/858, which is a statement about
the repair rather than new evidence.

### What 1b needs

Two people who are not the authors, about two hours each.
`benchmark/blind_review/README.md` is written for them. Building the sample
surfaced its own problem worth recording: two scripts named their own treatment
in a comment (`# Remove duplicate rows as advised`, `# ... for audit
compliance`), both from treatment arms, so the leak had a direction. Treatment
vocabulary is now redacted and the generator refuses to write a sample with a
surviving tell.

---

## 3. Guardrails dynamic allowed-values baseline

**Built and tested; not yet run as a battery.**

The paper reports Guardrails AI at 35.0 % and concedes, in the results section,
that this is "a claim about a *default configuration*, not about the toolkit's
capability". A reviewer is entitled to push harder: if a competent user could
take Guardrails' own acceptable-values validator and populate it from the
evidence, then the 35 % measures nobody's real configuration and the comparison
is a strawman.

`guardrails_choices` is that user. Same Guard, same open schema, same reask
budget of two, same transport, same scorer. One generic membership check on
`columns_referenced`, with the list built from `E` at call time. No value
tolerance, no anchor requirement, no bespoke logic — strictly weaker than
Tier B by construction, which is the point.

The three arms now separate three different things:

    guardrails_stock    structure only            what the toolkit gives you
    guardrails_choices  + evidence-bound choices  what a competent user builds
    guardrails_tierb    + the three Tier B checks a contract

**A fact that sharpens the concession.** The paper says Guardrails "ships a hub
validator taking a list of acceptable values". Checked against the pinned
version, that is too generous to us in one direction and too harsh in another:
a fresh `pip install guardrails-ai==0.11.0` registers **zero** validators.
`validators_registry` is empty, `guardrails.hub` exports nothing, and each
validator is a separate `guardrails hub install` fetching a manifest from
`hub.api.guardrailsai.com`. So "out of the box" is a stronger claim than we
made — out of the box there is no faithfulness validator to configure — while
"a competent user could build one" remains true and is exactly what this arm
measures.

Because the hub install needs network access this environment does not have,
the arm reimplements the documented `ValidChoices` semantics locally: one
membership test, short enough to audit by reading. That also keeps the arm
runnable from the replication package alone.

Eight tests in `tests/test_guardrails_choices_arm.py` run offline against a
stub transport and pin both halves of the claim: that the check actually fires
on an out-of-evidence name (an arm that passed everything would reach 0/200 for
the wrong reason), and that it lets through the value error and the missing
anchor that Tier B catches.

**What the outcome would mean, recorded before running it.** If
`guardrails_choices` reaches 0/200, the paper's claim gets both narrower and
stronger: the mechanism is framework-agnostic, evidence binding at call time is
what matters, and Guardrails could have done it — the default simply does not.
If it lands between 35 % and 0, the generic check is insufficient and the three
Tier B channels are doing work a choice list cannot. Either result is
publishable and we say so now.

**To run it:** the arm needs `pip install '.[baselines]'` and a provider key,
and slots into the existing battery as a seventh layer alongside
`guardrails_stock` and `guardrails_tierb`.

---

## Results, 2026-09-18

Both batteries ran. Item 3 narrows a claim; item 2 breaks one.

### Item 3 — `A-GR-CHOICES` reaches 0/200

DeepSeek `deepseek-chat`, N=200 per arm, three days after the manuscript's run:

| arm | entity | 95 % CI | k/N | Tier B catches |
|---|---|---|---|---|
| `A-L1` | 43.0 % | [36.3, 49.9] | 86/200 | 0 |
| `A-GR-STOCK` | 43.5 % | [36.8, 50.4] | 87/200 | 0 |
| **`A-GR-CHOICES`** | **0.0 %** | [0.0, 1.9] | **0/200** | **88** |
| `A-GR-OURS` | 0.0 % | [0.0, 1.9] | 0/200 | 83 |
| `A-CONTRACT` | 0.0 % | [0.0, 1.9] | 0/200 | 0 |
| `A-STRESS` | 0.0 % | [0.0, 1.9] | 0/200 | 23 |
| `A-STRICT-STATIC` | 7.0 % | [4.2, 11.4] | 14/200 | 0 |
| `A-STATIC-ENUM-STALE` | 0.0 % | [0.0, 1.9] | 0/200 | 0 |

The toolkit's own generic acceptable-values check, populated from `E` at call
time and carrying none of our three channels, reaches the same 0/200 the
contract does. The outcome recorded before the run said this would make the
claim "both narrower and stronger", and it does. The contribution is not that
our validator beats theirs. It is that **binding the admissible set to the
evidence at call time is what does the work, and nothing in the toolkit's
default configuration does that.** §7 has to be rewritten to say so.

Two things worth keeping from the same battery. `A-GR-STOCK` moved from
35.0 % to 43.5 % in three days on the same model name, which is a third data
point for the paper's own thesis that a rate is a property of a model version
at a date. And on local `qwen2.5:7b` (N=20) every arm is 0/20 except
`A-STRICT-STATIC` at 2/20 — a third provider, a third null.

### Item 2 — the rubric arm breaks §9's causal claim

See `A17_RESULT.md` for the full analysis. In short: an arm told the six scored
concerns and nothing about its own script ends at **1 flagged defect over 32
scored runs**, better than `advise+audit`'s 8. A17 recorded before the run that
a result near 8 would force a rewrite; it landed at 1.

The arm pays for it somewhere the checklist cannot see. It executes 17/36
(47 %) against 78–83 % for every other arm, so on the joint outcome a
practitioner gets — runs *and* carries no flagged defect — `advise+audit` is
still first at 64 % [48, 78] against the rubric arm's 44 % [30, 60], with
overlapping intervals.

And it explains a claim rather than removing it: `control+revise` repaired 0
of 33 defects, the rubric arm repaired 21 of 25 and regressed none. A model
asked to review its own script with nothing named fixes nothing; a model told
what to look for fixes most of it, whether or not anyone tells it what is
actually wrong.

### Both manuscripts revised, 2026-09-18

`paper/emse_latex/main.tex` (49 pp.) and `paper/tse_latex/main.tex` (16 pp.,
body ends on 15) now carry both results. What changed, in the order a reader
meets it:

- **Abstract, contributions, conclusion.** The Guardrails claim narrows from
  "our validator" to "the binding". The downstream claim withdraws the
  31 → 8 attribution and states what survives: the joint outcome, weakly.
- **§7.** New heading, the `A-GR-CHOICES` passage, the empty-registry fact,
  and the three-days-later replication as its own labelled subsection.
- **§8.** A new table separating syntactic / type / **evidence** validity,
  with the arm that measures each. The two failure modes of §6 sit one level
  apart, which is why a blended rate hid them.
- **§9.** The fifth arm, `tab:rubric` with the joint-outcome column, and the
  Goodhart paragraph.
- **§12.** The overlap paragraph stops calling itself "weakened but not
  removed"; the checklist-is-a-proxy paragraph gains the 47 % execution rate;
  the corrections paragraph now reads five-for and two-against; a new
  paragraph classifies the seventeen amendments.
- **Figure 1** is regenerated from a committed generator
  (`paper/make_fig1_contract.py`, which did not previously exist) with three
  scope annotations: the producer is trusted and never verified, unverified
  prose still reaches the user, and a band reading GUARANTEED response ↔ E /
  NOT GUARANTEED E ↔ ground truth.
- Deployment language moves from "production" to "shipped", with the reason
  stated: we have no telemetry to back a production claim.
- Temperature 1.0 is justified in Methods, including the admission that it
  cuts against the baselines.

Both build with zero undefined references. EMSE has zero overfull boxes; TSE
has one 1.8 pt overfull vbox.

### Remaining

Re-ordered after the external review of 2026-09-18, which ranks the top three
as rubric arm (done), blind human audit, and a second evidence-closed task.

| # | Work | Status |
|---|---|---|
| 7 | **A second evidence-closed task** — a different deterministic producer (schema profiler, missing-value diagnostics, fairness report, static analyser) narrated under the same contract | **queued, and the largest gap.** RQ1–RQ4 measure one evidence shape while the paper defines a class. One second task retires "you define a general class and evaluate one instance" |
| 1b | Blind human rating of the checklist | apparatus frozen, needs two raters who are not us |
| 5 | Evidence-domain scalability, 10 / 100 / 500 / 1000 | queued. §12 concedes Tier A is untested above ten columns; the measurement is cheap (schema bytes, latency, API acceptance, tokens, first-pass compliance) |
| 8 | **Temperature sensitivity, T ∈ {0, 0.2, 1.0}** | queued. Methods now justifies T=1.0 and admits the baselines are conditioned on it. A supplementary sweep would close it |
| 4 | Template-only, LLM-free baseline | queued |
| 6 | Tier B mutation / adversarial tests | queued |

Two of 7, 1b and 5 would, by the reviewer's own account, move the paper from
borderline to the accept side.
