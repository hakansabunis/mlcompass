# mlcompass Presentation — English Speaker Notes

**13 slides · ~12-13 minutes of speaking + ~2 minutes Q&A**

Each slide has 45-60 seconds of speech. **Bold** marks emphasis. *Italics*
are stage directions (not spoken). Memorization tip: each slide has a
**single core message** in the first paragraph — if you forget the rest,
say that one and move on.

---

## 🎬 Slide 1 — Title (~30 seconds)

Good morning, everyone. I am Hakan Sabuniş, and this is my partner Yusuf
Ünlü. We are from Istanbul Medipol University, Computer Engineering, and
today we are presenting our capstone project.

The project is called **mlcompass**. It is an open-source command-line
assistant for machine-learning engineers. It is live on the Python
Package Index, **anyone can install it with `pip install mlcompass`**, and
it is published under the MIT license.

In the next twelve minutes I will walk you through: the problem we
identified, the architecture we designed, our main technical contribution
which is the **anti-hallucination contract**, our experimental results,
the challenges we faced, and a live demo.

*[Pause for 3 seconds after this slide. Let the audience read.]*

---

## 🎬 Slide 2 — The Problem (~60 seconds)

The current machine-learning ecosystem is fragmented. A data scientist
opens **pandas-profiling** to look at the data. They use **Weights and
Biases** or **TensorBoard** to track training. They ask **Copilot** or
**Cursor** to write code. **None of these tools give advice.**

Pandas-profiling shows the data distribution but doesn't know which
column is your target. Weights and Biases draws a loss curve but never
tells you "this looks like overfitting." Copilot generates code, but if
you write `Adam(momentum=0.9)` — which is **invalid because Adam doesn't
accept a momentum parameter** — it won't warn you.

The newer LLM-based assistants are even more dangerous. They **hallucinate**.
They invent column names that do not exist. They propose fixes for
problems that are not real. In 2025 alone, researchers documented over
**146,000 hallucinated citations** across arXiv, bioRxiv, and SSRN.
**Confident wrong answers are worse than no answer at all** — because the
practitioner trusts them.

This is the gap we set out to fill with mlcompass.

*[Vurgu: gesture to the 146,932 number on the screen.]*

---

## 🎬 Slide 3 — mlcompass in 30 Seconds (~50 seconds)

If I had to describe mlcompass in one sentence: **it is an open-source CLI
that follows you through the whole ML pipeline, and is allowed to explain
but never to invent.**

We have **eleven commands**, covering every stage of the pipeline. `init`
to start a new project, `advise` to look at your data, `audit` to review
your training script, `watch` to monitor the training log, `compare` for
two runs, `evaluate` for results, `deploy` for production readiness. Plus
`monitor`, `optimize`, `status`, and `agent`.

These eleven tools are reachable through **four different surfaces**: the
terminal CLI, an MCP server that plugs into Claude Desktop, Cursor, and
Claude Code, an autonomous self-driving agent, and Claude Code slash
commands.

And there is a small `.mlcompass/` folder in your project that **remembers
every decision** — just like git's `.git/` folder. So when you come back
tomorrow, the assistant knows what you decided yesterday.

---

## 🎬 Slide 4 — System Architecture (~60 seconds)

This is the system architecture. It has three layers.

At the **top**, three user-facing surfaces. On the left, the **CLI** —
plain terminal commands. In the middle, the **MCP Server** that connects
to chat assistants like Claude Desktop. On the right, the **self-driving
agent** that runs the whole pipeline autonomously.

In the **middle**, what we call the two-layer brain. This is the most
important design decision in the project. The upper part is **deterministic
Python code** that produces structured findings. The lower part is an
**optional Claude narrator** that explains those findings in natural
language.

At the **bottom**, the persistent project context folder.

The reason we split the brain into two layers is this: **the source of
hallucination is mixing fact-extraction with fact-narration in one model**.
If you let an LLM both analyze the data and explain it in the same step,
it will invent things. So we separated them. Pandas reads the data. Claude
only explains what pandas found.

*[Use right hand to point to each layer as you describe it.]*

---

## 🎬 Slide 5 — The Two-Layer Brain (~55 seconds)

Let me zoom into these two layers.

On the **left**, the **deterministic tool layer**. Pure Python. It uses
pandas to read CSVs, Python's `ast` module to parse scripts, NumPy and
SciPy to compute correlations. The output of this layer is always a
**structured dictionary** — never free text. **And because it does not
generate text, it physically cannot hallucinate.**

On the **right**, the **optional Claude narrator**. This layer receives
the dictionary from below and explains it in natural language. It only
runs when the user passes the `--llm` flag. So even **a user without an
Anthropic API key** can use mlcompass — they just see the deterministic
analysis, which is already useful on its own.

The contract we will describe on the next slide ensures that **the
narrator can only mention things the lower layer actually found**.

---

## 🎬 Slide 6 — The Anti-Hallucination Contract (~70 seconds)

This is the **main technical contribution** of the project. The
anti-hallucination contract has **three layers**. Each one alone is weak.
Together they are strict.

**Layer One — Deterministic evidence**. The module
`tools/leakage.py` reads the predictions table and emits a structured
dictionary. Inside: candidate leak columns, top correlations, perfect-match
rate. All computed with pure pandas. No LLM.

**Layer Two — Strict prompt**. The narrator agent receives the evidence
with three explicit instructions: cite only items present in the evidence,
say `cannot_determine` when unsure, and **do not propose code patches** —
only suggest manual checks.

**Layer Three — Runtime schema boundary**. This is where it gets
interesting. The agent emits its answer through a Claude tool whose JSON
input schema is **generated at call time from the evidence dictionary**.
The `columns_referenced` parameter is restricted, via JSON-schema enum, to
exactly the columns that the evidence contains. **If the agent emits a
column outside that set, the Anthropic SDK throws a validation error and
the agent retries.**

Our slogan from this experience is: **prompts are advisory; schemas are
enforcement**. Instead of politely asking the model not to hallucinate, we
make it physically impossible.

*[Pause 2 seconds after the slogan. Let it land.]*

---

## 🎬 Slide 7 — Worked Example: Ames House Prices (~55 seconds)

Let me show this concretely. A user is training a model to predict house
prices on the Ames Housing dataset. They accidentally include a feature
called `log_price` — which is the logarithm of the target. **This is
direct data leakage.** The model's R-squared comes out at 1.000.

**Layer One** fires. It computes Pearson correlation at 0.94 and Spearman
correlation at 1.00. The reason we also compute Spearman is that monotone
transformations like log break Pearson but preserve Spearman. Without
Spearman, this leak would have slipped past us. `log_price` is flagged as
a candidate.

**Layer Two** narrates: *"The column `log_price` has Spearman correlation
of 1.00 with the target. This is almost certainly a transformed leak.
Recommend manually checking the feature pipeline."*

**Layer Three** validates: `log_price` is in the evidence enum. Response
ships. **If the LLM had tried to say `revenue` is also a leak — and
`revenue` is not in the dataset — the SDK would reject the response and
force a retry.** That is the power of Layer 3.

---

## 🎬 Slide 8 — Results: Wilson 95% CI Ablation (~65 seconds)

Now the experimental results. Two hundred narrator responses per
configuration, Claude 3.5 Sonnet, on a synthetic regression test set.

**First row:** Layer 1 only — no prompt, no schema. Hallucination rate is
**eight percent**. Wilson 95 percent confidence interval is roughly five
to twelve point seven percent. So the hallucination is statistically real,
not noise.

**Second row:** Layer 1 plus Layer 2 — we add the strict prompt. The rate
drops to **one percent**. That is an **eight-fold improvement** from prompt
engineering. But it is not zero.

**Third row:** All three layers — schema enforcement on. **Observed rate is
zero**. Across two hundred samples, not a single hallucination.

**An important honesty point:** at N equals 200, the confidence intervals
for Layer 2 and Layer 3 overlap. So statistically we cannot prove the
underlying rate is different. But the value of Layer 3 is not in the
average — it is in the **worst-case guarantee**. The schema rejects any
hallucinated response, no matter what.

The takeaway: **prompt engineering improves, but cannot guarantee zero. For
guarantees, you need runtime enforcement.**

---

## 🎬 Slide 9 — Field Tests (~55 seconds)

We did not just stop at synthetic data. **Every release was tested on a
real Kaggle dataset before shipping.** Five Kaggle field tests caught
**eleven real bugs**.

Test 1: **Telco Churn** — baseline, zero bugs as expected.

Test 2: **Ames House Prices** — three bugs in thirty seconds. The target
name `SalePrice` was not in our list. The `YearBuilt` column was being
mistaken for an ID column. `PoolQC` at 96 percent NaN was crashing IQR
computation.

Test 3: **Titanic** — missing `Survived` from the binary target list.

Test 4: **Penguins** — multiclass was being mis-classified as binary. And
more importantly, when we ran the pipeline through Claude Code's MCP
integration, **the project ledger was empty** afterward. The CLI was
writing to the ledger; the MCP wrappers were silently skipping it. **Two
surfaces over one capability had drifted apart.**

Test 5: **Insurance Charges** — finance-related target names missing,
plus another MCP state-parity bug.

The lesson: **none of these bugs were predicted by our pre-existing unit
test suite**. Real datasets exercise production code paths in ways
synthetic tests do not.

---

## 🎬 Slide 10 — The 11 Slash Commands & Live Demo (~95 seconds)

Before I show the demo, look at this slide. **Eleven Claude Code slash
commands**, each one mapping to a mlcompass tool. They are installed with a
**single command** — `mlcompass install-slash-commands` — no JSON config,
no manual editing of any setup file.

*[Walk the audience briefly through the four highlighted cards: init,
advise, evaluate, status — these are the ones we will use in the demo.
Then point at the two gray dashed cards: monitor and optimize fall back to
the CLI because they are not in the MCP server yet.]*

Now let me switch to a terminal and show you mlcompass live, using these
slash commands from inside Claude Code.

*[Switch to Claude Code. Make sure it's full-screen, font size is large.]*

The first command initializes a new project:

`/mlc-init insurance-demo`

This creates a `.mlcompass/` folder, just like `git init`. Notice — I am
**not** typing a Python command. I am talking to Claude Code in its own
slash menu, and **Claude calls mlcompass behind the scenes.**

Second, we analyze the dataset:

`/mlc-advise demo-data/insurance.csv`

You can see — it correctly infers the target column as `charges`. It warns
about smoker imbalance. It recommends models. All of this analysis is
**deterministic Python under the hood** — Claude is only narrating what
the tool returned.

Third, the highlight. I have a predictions file where I deliberately added
a leak. Watch what happens:

`/mlc-evaluate demo-data/predictions_with_leak.csv`

**R-squared 1.000 detected — and the leakage panel automatically fires.**
The deterministic layer flags `log_charges_leak`. The narrator explains it
under our schema-bounded contract — meaning **Claude can only cite columns
the evidence actually contains.** If it tried to invent a column called
`revenue`, the schema enum would reject it.

Finally, status:

`/mlc-status`

It summarizes every decision we made in this session — across both CLI
and MCP surfaces.

*[Switch back to slides. Total demo time: about 90 seconds. Don't go over.]*

*[Backup: if the live demo fails for any reason, say: "I have a
pre-recorded video for backup" and play the MP4.]*

---

## 🎬 Slide 11 — Challenges We Faced (~55 seconds)

Four challenges, four lessons.

**First: prompt engineering was not enough.** We spent a week trying to
make the LLM stop hallucinating columns through prompts alone. We got it
from 8 percent down to 1 percent. Still wrong 1 percent of the time. The
fix was the schema layer, not the prompt layer.

**Second: CLI and MCP server drifted apart.** Same Python functions,
different observable behavior. The CLI was writing to the ledger, the MCP
server was bypassing that step. We shipped a shared helper and added
explicit parity tests in v0.7.2 and v0.7.3.

**Third: synthetic unit tests missed real bugs.** Eleven bugs from five
Kaggle datasets — zero predicted by our test suite. We moved to a
field-test-then-patch release cadence.

**Fourth: reviewer feedback.** Three peer reviewers gave us critique on
our paper. We responded with v0.8.1 — Wilson confidence intervals,
reproducibility scripts, threat model documentation, limitations document.
**Taking critique seriously, at the code level, is far stronger than
taking it at the paper level alone.**

---

## 🎬 Slide 12 — Limitations and Future Work (~50 seconds)

We are honest about what we did not measure.

On the left: **cross-model generalization** — all our experiments use
Claude 3.5 Sonnet only. We do not know how GPT-4 or Llama-3 would behave
under the same contract. **Single synthetic dataset** with N=200 — the
headline ablation is one task only. **No constrained-generation baseline**
— we do not benchmark against Outlines or OpenAI's strict mode. **No
external user study** — only the two authors and our classmates have used
the tool. And **only one hallucination category** — phantom-entity
fabrication. Other categories like miscalibrated confidence are out of
scope.

On the right: our plan for **v0.9 and v1.0**. Add an OpenAI backend to
repeat the ablation on GPT-4 with strict mode. Benchmark against Outlines
on Llama-3-8B. Scale to N=2000 with the reproducibility script we ship.
Build a plug-in system for domain-specific analyzers.

**Listing your limitations honestly is a sign of academic maturity, not
weakness.**

---

## 🎬 Slide 13 — Thank You and Q&A (~30 seconds)

Thank you for your attention.

All source code is on **github.com/hakansabunis/mlcompass** under the MIT
license. You can install it right now with `pip install mlcompass`. The
paper is in the `paper/` folder of the repository.

I am happy to take questions.

*[Wait for questions. Common questions and pre-prepared answers below.]*

---

## 🎯 Anticipated Q&A — Pre-Prepared Answers

### Q: "Why Claude and not GPT-4?"

**A:** *"Two reasons. First, Anthropic's tool-use API has the strictest
schema enforcement we found — OpenAI's strict mode is comparable but newer.
Second, the mechanism is generic. Any provider that validates JSON-schema
input on tool calls will work. We plan to add an OpenAI backend in v0.9
and re-run the ablation to validate that the rate numbers transfer."*

### Q: "N=200 seems small."

**A:** *"You are right — we acknowledge this in the paper, Section 5.D.
The Wilson 95 percent confidence intervals for Layer 2 and Layer 3
overlap at this sample size. Our reproducibility script
`scripts/reproduce_hallucination_ablation.py` is designed to scale to
N=2000, which would cost roughly $48 in API charges. We did not have the
budget for that during the project, but the path is clear and the
artifact is ready."*

### Q: "What is the difference from constrained generation libraries like
Outlines or LMQL?"

**A:** *"The mechanism is the same — runtime schema enforcement. What
distinguishes our setup is that **the schema's enum domain is generated at
call time from the deterministic upstream evidence**, not declared
statically. The constraint adapts to the data. We do not claim a novel
mechanism — we claim a novel binding pattern. Section II of the paper
discusses this explicitly."*

### Q: "What was the hardest moment of the project?"

**A:** *"Field Test number four — the CLI and MCP parity bug. Two hours
of debugging before we found that one surface was silently skipping a
ledger-write step. But that two hours produced the most generalizable
finding of the project: surface parity is engineering work, not a
freebie. We documented it in Section 3.D of the paper as a first-class
methodological lesson."*

### Q: "How would you extend this to other tasks?"

**A:** *"The same three-layer pattern generalizes to any task where an
LLM narrates structured evidence. We are planning to extend it to the
other narrators in mlcompass — for audit, watch, compare, monitor, and
optimize. Beyond ML, we believe it applies to medical diagnostic
explanation, legal case analysis, and scientific paper summarization."*

### Q: "Can a non-developer use it?"

**A:** *"Yes — through the MCP server. The user installs Claude Desktop,
runs `mlcompass install-slash-commands`, and then types something like
`/mlc-advise data.csv` in Claude Code. No Python knowledge required. The
slash commands are the most user-friendly entry point."*

### Q: "Did you find any bugs in Claude itself during this work?"

**A:** *"We did notice that the strict prompt alone reduced hallucination
8x, which suggests Claude is responsive to clear constraints. But the
1 percent residual rate is consistent with what other researchers have
reported. The whole motivation for Layer 3 was to bridge that residual
gap — not to fix Claude, but to enforce around it."*

---

## 📝 Presentation Pacing Tips

1. **Speed:** Each slide is 45-60 seconds. Total speaking time
   12-13 minutes. Q&A 2-3 minutes.

2. **Pauses:** After every emphatic sentence ("prompts are advisory,
   schemas are enforcement", "146,932 hallucinated citations") **pause
   3 seconds**. Let it land.

3. **Body language:** Stand on the left of the screen, gesture to the
   slide with your right hand. Don't read from the laptop screen — look
   at the audience.

4. **Demo backup:** Have an MP4 recording of the demo on a USB drive.
   If the live terminal fails, say: *"I have a pre-recorded video as
   backup, let me play that"* — no panic, sound natural.

5. **Numbers:** Memorize **8 percent → 1 percent → 0 percent** and
   **eleven real bugs caught** and **146,000 hallucinated citations**.
   These three numbers are your anchors. If you forget anything else,
   these three carry the talk.

6. **Mixing English and technical jargon:** Don't translate jargon —
   say "hallucination", "schema", "Wilson confidence interval", "MCP
   server" as is. Translating breaks rhythm.

7. **Q&A strategy:** If you don't know an answer, say: *"That's a great
   question. We discuss this in Section 5.D of the paper as a limitation
   — we don't have a definitive answer yet, but our current thinking
   is..."* — honesty beats guessing.

Good luck. You've got this. 🚀
