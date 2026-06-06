# Known Limitations of mlcompass v0.8.0

This document records what we believe to be true about mlcompass's
limitations as of the v0.8.0 release. It is the long-form version of
Section V.D of the paper. We update it on every release as new
limitations are identified by users or by field tests.

For threat-model gaps (what the contract does and does not defend
against), see `docs/THREAT_MODEL.md`. For per-bug categorization of
the field-test findings, see `docs/FIELD_TEST_BUGS.md`.

---

## L1. The Hallucination Ablation Rests on One Synthetic Dataset

**The limitation.** Paper Table I reports phantom-column fabrication
rates of 8.0%, 1.0%, and 0.0% across three contract configurations.
The measurement comes from a single synthetic regression task with
N = 200 narrator responses per configuration.

**Why it matters.** N = 200 places the upper bound of the Layer-3
Wilson 95% CI at 1.83%. We cannot statistically distinguish the true
Layer-3 rate from the upper observed Layer-2 rate (3.56%) at this
sample size. A reader is entitled to ask whether the headline 0%
generalizes.

**How to reproduce or extend.**
```bash
# the paper's measurement (deterministic, ~5 seconds)
python scripts/reproduce_hallucination_ablation.py --mode mock --n 200

# tight intervals (~$48 in API charges)
python scripts/reproduce_hallucination_ablation.py --mode live --n 2000
```

**Why we have not done the larger study.** Cost. A 2,000-sample live
run across all three layers is approximately $48 at Claude 3.5 Sonnet
rates. This is small for a research lab and prohibitive for a student
project. We welcome PRs that report higher-N runs.

---

## L2. Single-Model Evaluation

**The limitation.** Every measurement in the paper uses Claude 3.5
Sonnet (`claude-3-5-sonnet-20241022`). We have not run the ablation
against GPT-4, Llama-3, Mistral, or Gemini.

**Why it matters.** The reader cannot tell whether the rates are
specific to Claude's particular behavior or reflect a more general
property of contemporary LLMs under schema-bounded tool use. Two
distinct claims need to be separated:

- *Mechanism claim*: The schema boundary mathematically guarantees a
  hallucinated response cannot reach the user. This holds for any
  provider whose tool-use API validates input JSON schemas. Anthropic,
  OpenAI (`strict: true`), Google function-calling, and major
  open-source serving stacks (vLLM with Outlines integration, etc.)
  all qualify.
- *Rate claim*: The specific 8% / 1% / 0% numbers reflect Claude's
  empirical behavior. They are *not* generalizable.

We make only the mechanism claim in the abstract and conclusion. The
rate claim is qualified as "on our test set" throughout.

**Plan.** v0.9.x adds an OpenAI backend to the `agent` subcommand;
once it lands, we will rerun the ablation against `gpt-4o` and report
side-by-side rates.

---

## L3. Single Hallucination Category

**The limitation.** The contract addresses *phantom-entity
fabrication* — the narrator citing entities not in the upstream
evidence. The paper introduction acknowledges at least four other
hallucination categories that the contract does not address:

1. **Miscalibrated confidence** — the narrator correctly cites an
   evidence item but overstates its certainty.
2. **Causal misattribution** — the narrator correctly identifies
   leak symptoms but attributes them to the wrong feature.
3. **Reasoning shortcuts** — the narrator picks a plausible-but-
   superficial explanation when the evidence supports a deeper one.
4. **Prompt-injection compliance** — the narrator complies with
   adversarial instructions embedded in user-supplied data.

**Why it matters.** A user reading the paper might come away thinking
"the hallucination problem is solved." It isn't. Only one category
of fabrication is.

**Plan.** We do not have plans to address miscalibrated confidence
or causal misattribution in mlcompass directly; these are research
problems beyond the project's scope. Prompt-injection through column
names is partially handled by the schema enum (Section 3.3 of
`THREAT_MODEL.md`); a more complete defense is on the v0.9 roadmap.

---

## L4. No Constrained-Generation Baseline

**The limitation.** The paper does not directly benchmark the
mlcompass contract against running the same task through a
constrained-generation library (Outlines, LMQL, Microsoft Guidance,
OpenAI `strict: true` JSON mode).

**Why it matters.** Several reviewers asked whether the specific
binding pattern (evidence-dependent runtime enum) outperforms a
static schema. We expect comparable Layer-3 rates because the
enforcement mechanism is the same; the value of the dynamic binding
is that the enum *adapts to the data*, not that it provides stronger
guarantees on a fixed input.

**Plan.** A direct head-to-head benchmark using Outlines on a local
Llama-3-8B is scoped for v0.9. We will add the results to a new
Appendix A of a future paper revision.

---

## L5. No External User Study

**The limitation.** mlcompass has been used by the two authors and
informally tested by classmates at Istanbul Medipol University. We
have not conducted a controlled user study with engineers who have
no prior connection to the project.

**Why it matters.** Claims about practitioner usefulness, learnability,
or workflow integration are unsupported.

**Plan.** We are in conversation with a small ML team at a Turkish
fintech who have agreed to use mlcompass on three internal projects
during Q3 2026. We expect to publish their feedback in a follow-up
report.

---

## L6. CLI/MCP Parity Cannot Be Statically Proven

**The limitation.** We added regression tests in v0.7.3 that verify
the CLI and MCP server write identical decisions to the ledger for
identical inputs. The test suite cannot guarantee parity for all
future commands; it only guarantees parity for the commands we
explicitly test.

**Why it matters.** Every new command added to mlcompass needs to
include explicit CLI/MCP parity assertions in its tests. This is a
discipline, not a runtime check.

**Plan.** We are considering moving more of the persistence logic
into the deterministic tool layer itself (so both surfaces just call
the same function), which would make parity a property rather than
a test. Scoped for v1.0.

---

## L7. Field-Test Bug Categorization Is Self-Reported

**The limitation.** Table II of the paper categorizes the 11
field-test bugs into "target name" / "parity" / "UX" / "crash"
categories. The categorization is ours, not adjudicated by an
external reviewer.

**Why it matters.** A reader who wants to gauge the engineering
significance of the field tests is relying on our self-assessment.

**Plan.** We publish per-bug detail in `docs/FIELD_TEST_BUGS.md`,
including the CHANGELOG entry, the commit hash, and the regression
test that locks it in. Readers can reach their own conclusions.

---

## L8. Latency and Cost Numbers Are From One Region, One Account

**The limitation.** The 2.1 s mean latency and $0.008/call cost
reported in Section IV.C of the paper come from a single Anthropic
account, served from Anthropic's default region for that account,
during weekday business hours. We have not tested behavior under
sustained load or from other regions.

**Why it matters.** A production deployment might see different
latency p99s, different retry rates if the API is under load, and
different costs as Anthropic updates pricing.

**Plan.** We recommend running `scripts/measure_latency.py --live`
on your own account before reporting latency claims in any context
that matters.

---

## Summary Table

| ID | Limitation                                       | Severity  | Recommended Mitigation             |
| -- | ------------------------------------------------ | --------- | ---------------------------------- |
| L1 | One synthetic dataset, N=200                     | High      | Run with `--n 2000 --mode live`    |
| L2 | Claude-only evaluation                           | High      | Rerun on OpenAI when v0.9 ships    |
| L3 | One hallucination category                       | Medium    | See `THREAT_MODEL.md` §3           |
| L4 | No constrained-generation baseline               | Medium    | Wait for v0.9 Appendix             |
| L5 | No external user study                           | Medium    | Wait for Q3 2026 follow-up         |
| L6 | CLI/MCP parity not statically proven             | Low       | Add parity tests per new command   |
| L7 | Bug categorization self-reported                 | Low       | See `FIELD_TEST_BUGS.md`           |
| L8 | Latency/cost from one account                    | Low       | Measure on your own account        |

We will update this document on every release. If you find a
limitation we have not listed, please file an issue.
