# Validation queue

Six pieces of work an external reviewer asked for. None is a wording change;
each one needs something run. Ordered by how much of the paper's argument
rests on it.

| # | Work | Answers | Status |
|---|---|---|---|
| 1 | Validate the final defect scorer | "Why should I trust rescored-v2, after Section 10?" | **1a done; 1b built, not run** |
| 2 | Generic-checklist fifth arm | Is it the specific evidence, or just naming the rubric in the prompt? | **next** |
| 3 | Guardrails dynamic allowed-values baseline | Is the 35 % a strawman? | queued |
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
