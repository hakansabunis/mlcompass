# Where this stopped — 2026-09-24

Submission target is **IEEE TSE**, `paper/tse_latex/`. The Springer long form in
`paper/emse_latex/` is frozen and unmaintained (see its `UNMAINTAINED.md`).

Current build: 20 pages, body through 18, no dangling references, one 1.5 pt
overfull vbox. Last commit before this note: `f8a7ba4`.

---

## 1. Finish the free-text battery — run this first

It was running when the machine was shut down. Every response is written as it
arrives, so nothing paid for is lost; resume picks up exactly where it stopped.

```bash
source scripts/load_keys.sh
python -X utf8 -u scripts/reproduce_hallucination_ablation.py \
  --mode live --provider deepseek --n 200 --only-baselines \
  --resume --log-dir scripts/runs/2026-09-19_freetext
```

State at shutdown: `layer1` 200/200, `layer3_bare` ~70/200, six arms not
started. Roughly 1,700 calls and under a dollar on `deepseek-chat`.

**Before trusting it, check the prose is really there** — this battery has
already been lost once to an empty field:

```bash
python -X utf8 -c "
import json,pathlib
for f in sorted(pathlib.Path('scripts/runs/2026-09-19_freetext').glob('*.jsonl')):
    L=[json.loads(l) for l in f.read_text(encoding='utf-8').splitlines() if l.strip()]
    e=sum(1 for r in L if not (r.get('narration') or '').strip())
    print(f.name.split('_synthetic_')[1].rsplit('_n',1)[0], len(L), 'empty:', e)"
```

Every arm must show `empty: 0`. The two bugs that emptied it are fixed —
`_normalize` read only `narration`, and `_from_contract_result` started from
`_normalize({})` and never copied `primary_hypothesis` across — and a smoke test
confirmed every arm records prose. But check.

## 2. Measure displacement

```bash
python -X utf8 scripts/measure_freetext_displacement.py --runs scripts/runs \
  --json benchmark/freetext_displacement.json
```

The question: when Tier B strips a claim from the structured fields, does the
model put it in the prose instead? Displacement predicts the arms that **strip**
(`A-CONTRACT`, `A-STRESS`, `A-GR-OURS`, `A-GR-CHOICES`) carry *more* ungrounded
prose than the arms that do not (`A-L1`, `A-GR-STOCK`, `A-STRICT-STATIC`,
`A-STATIC-ENUM-STALE`). Schema-only arms count as non-stripping on purpose:
the hypothesis is about deletion, not about lowering a rate.

Report whichever way it comes out. What it cannot see, and must say so: a
sentence naming only real columns and real numbers can still assert a causal
story the evidence does not support. That half needs human raters.

## 3. Write it into the manuscript

Replace the §12 paragraph that currently says the free-text channel could not
be measured because the narration was never recorded. It becomes either
"measured, displacement not observed at the scale this design sees" or
"measured, and here is how much". Keep the instrument-fault history — the field
was dropped, then emptied on the enforced arms — because both are real.

Then recompile and re-run the table verification (`scratchpad/verify_tables.py`
and `verify_rq5c.py` pattern: recompute every number from the run records, do
not read them back from the text).

---

## 4. Remaining points from the senior review

Writing-level points 1, 2, 3, 4, 5, 8, 10, 11, 13, 14 are **done** (commit
`f8a7ba4`). What is left needs runs or people:

| # | What | Cost | Note |
|---|---|---|---|
| 6 | Wider baseline matrix: prompt-only, schema-only, static enum, dynamic enum, validation-only, validation+retry, full — each with first-pass rate, retries, latency | ~$1 | Most arms exist; the gap is a clean prompt-only and a validation-without-retry arm |
| 7 | Model × mechanism matrix | ~$2–3 | Needs a model that actually fails. `gpt-5.4-mini` and local `qwen2.5:7b` both score 0 unenforced, so they add nothing to a comparison |
| 9 | Adversarial evidence benchmark — near-identical names across fields, nested evidence, duplicates | ~$1 + design | Strongest remaining experiment. The profile task already shows the mechanism: `cat_feature_07` misfiled onto `num_feature_07` by shared suffix. Build the benchmark to provoke that on purpose |
| 12 | Independent human gold labels for the validator | people | Apparatus is ready in `benchmark/blind_review/`, including `rate.html`. Needs two people who are not authors. The Gemini-agent sheets in `pilot_llm/` do not count and must not be reported |
| 16 | More evidence topologies | — | Partly done: RQ6 is a second shape, `--scale-table` measures 10–1000 columns. A nested-evidence shape would close it |

Also open, not from the senior list:

- ~~**τ = 0.005 sensitivity** (Stanford Q3).~~ **Done** (`3025aa8`,
  `scripts/tau_sensitivity.py`): counts identical at every τ from 5e-4 to 0.05;
  in the paper's Measures subsection.
- **The unexplained entity catch** in `P-CONTRACT`: one entity violation in the
  arm whose schema carries the enum. The profile harness now records every
  attempt, so one re-run of that arm would say whether the provider ignored its
  enum.

## 5. Submission items — the user's side

- **arXiv preprint.** The repository is public and the paper unpublished;
  a dated priority record matters. IEEE allows preprints alongside TSE.
- **GitHub release → Zenodo DOI.** The paper promises it. The integration does
  nothing until a release is actually cut. Once there is a DOI, add it to
  `CITATION.cff` and the Data Availability section.
- **Page count.** 18 pages of body. Agreed 17 was fine; the formal definition
  and the retry table added one. Revisit if the advisor objects.

---

## Settled — do not reopen

- **Blind audit (#1b / senior #12):** not in the paper. The Gemini sheets are
  filed honestly under `pilot_llm/`.
- **Licence:** stays MIT. Fourteen versions already shipped under it; relicensing
  would not protect them.
- **"main2 was worse":** it was not. The Stanford review labelled main2 described
  main1 (seven faults, rubric pending, attribution intact). Both reviews
  recommended acceptance. The current build supersedes both.
- **0/200:** never presented as a rate of zero. Enforced arms read "none observed
  in 200; below 1.9 % at 95 % confidence", and the guarantee is stated as a
  property of the verifier, not of the sample.

## Instrument faults so far: eleven

Plus two found while running the free-text battery (the narration field empty
on enforced arms, twice, for two different reasons). Whether those count as the
twelfth and thirteenth depends on whether any number from them is reported —
none has been. Decide when writing §12.
