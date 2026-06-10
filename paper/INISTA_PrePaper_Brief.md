# INISTA 2026 — Pre-Paper Brief

**Paper:** *An Evidence-Bound Runtime Schema for Reliable LLM Narration of Machine-Learning Pipeline Evidence*
**Authors:** Hakan Sabuniş¹, Yusuf Ünlü¹, Selim Akyokuş² — Istanbul Medipol University
**Target:** IEEE INISTA 2026, Main Track — **Agentic AI** (secondary: Generative AI)
**Constraints:** ≤ 6 pages incl. references, IEEE conf. template, English. No product/trademark names in body.

> This document is the *pre-paper work* the team agreed to do **before** drafting: (1) a hallucination-free,
> web-verified reference set; (2) the research-gap statement; (3) the mlcompass positioning. The paper is
> written *from* this brief, then sent through five simulated INISTA reviewers and revised.

---

## 1. Verified reference set (every entry web-checked — 0 fabricated)

> **Why this matters:** the previous capstone draft carried `[1] T. Zhao et al., "Hallucinated citations…,"
> arXiv:2605.07723, 2026` — a **fabricated citation** that leaked in from a tooling narrative. It is removed.
> Ironically, a paper about preventing fabrication must not contain a fabricated reference. Every reference
> below was confirmed via web search against arXiv / ACL Anthology / ACM DL / proceedings pages.

### A. LLM hallucination & faithfulness (frames the problem)
| # | Citation | Verified locator |
|---|----------|------------------|
| H1 | Z. Ji *et al.*, "Survey of Hallucination in Natural Language Generation," *ACM Comput. Surv.*, vol. 55, no. 12, Art. 248, 2023. | DOI 10.1145/3571730 ✓ |
| H2 | L. Huang *et al.*, "A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions," arXiv:2311.05232, 2023. | arXiv:2311.05232 ✓ |

### B. LLM agents & tool use (the agentic context)
| # | Citation | Verified locator |
|---|----------|------------------|
| A1 | S. Yao *et al.*, "ReAct: Synergizing Reasoning and Acting in Language Models," in *Proc. ICLR*, 2023. | arXiv:2210.03629 ✓ |
| A2 | T. Schick *et al.*, "Toolformer: Language Models Can Teach Themselves to Use Tools," in *Proc. NeurIPS*, 2023. | arXiv:2302.04761 ✓ |
| A3 | Model Context Protocol specification, 2024. [Online]. | modelcontextprotocol.io ✓ (open JSON-RPC standard) |

### C. Constrained / structured generation (the closest prior art — our differentiator)
| # | Citation | Verified locator |
|---|----------|------------------|
| C1 | B. T. Willard and R. Louf, "Efficient Guided Generation for Large Language Models," arXiv:2307.09702, 2023. *(Outlines)* | arXiv:2307.09702 ✓ |
| C2 | L. Beurer-Kellner, M. Fischer, and M. Vechev, "Prompting Is Programming: A Query Language for Large Language Models," *Proc. ACM Program. Lang.*, vol. 7, no. PLDI, pp. 1946–1969, 2023. *(LMQL)* | DOI 10.1145/3591300 ✓ |
| C3 | S. Geng *et al.*, "Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning," in *Proc. EMNLP*, 2023, pp. 10932–10952. | aclanthology 2023.emnlp-main.674 ✓ |
| C4 | "Guidance: A guidance language for controlling large language models," 2023. [Online]. | github.com/guidance-ai/guidance ✓ |
| C5 | "Introducing Structured Outputs in the API," 2024. [Online]. | (constrained JSON-schema decoding) ✓ |

### D. LLM code assistants & their reliability (motivates the risk)
| # | Citation | Verified locator |
|---|----------|------------------|
| R1 | M. Chen *et al.*, "Evaluating Large Language Models Trained on Code," arXiv:2107.03374, 2021. | arXiv:2107.03374 ✓ |
| R2 | C. E. Jimenez *et al.*, "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?," in *Proc. ICLR*, 2024. | arXiv:2310.06770 ✓ |

### E. AutoML & ML pipeline tooling (the baseline ecosystem)
| # | Citation | Verified locator |
|---|----------|------------------|
| T1 | M. Feurer *et al.*, "Efficient and Robust Automated Machine Learning," in *Proc. NeurIPS*, 2015, pp. 2962–2970. *(auto-sklearn)* | papers.nips.cc/paper/5872 ✓ |
| T2 | "ydata-profiling: Automated EDA reports from pandas DataFrames." [Online]. | github.com/ydataai/ydata-profiling ✓ |
| T3 | M. Zaharia *et al.*, "Accelerating the Machine Learning Lifecycle with MLflow," *IEEE Data Eng. Bull.*, vol. 41, no. 4, pp. 39–45, 2018. | IEEE Data Eng. Bull. ✓ |
| T4 | M. Abadi *et al.*, "TensorFlow: A System for Large-Scale Machine Learning," in *Proc. USENIX OSDI*, 2016, pp. 265–283. *(TensorBoard)* | USENIX OSDI ✓ |

### F. Data leakage (the demonstrator task)
| # | Citation | Verified locator |
|---|----------|------------------|
| L1 | S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in Data Mining: Formulation, Detection, and Avoidance," *ACM Trans. Knowl. Discovery Data*, vol. 6, no. 4, Art. 15, 2012. | DOI 10.1145/2382577.2382579 ✓ |

**Core set = 16 references.** Optional add if space allows: Ratner *et al.*, "Snorkel," *PVLDB* 11(3):269–282, 2017 (DOI 10.14778/3157794.3157797 ✓) and a Weights & Biases tooling cite. Target ~16–18 total for a 6-page IEEE paper.

---

## 2. The research gap (one paragraph + the precise sentence)

**Context.** LLM agents are now routinely placed as an *advisory layer* over deterministic tools: they call a
tool, receive structured output, and narrate it for a human (ReAct [A1], Toolformer [A2], MCP [A3]). When the
narrated object is *structured evidence* — a dictionary of measured facts — the agent can introduce
**phantom-entity fabrication**: it cites a column, a defect, or a fix that is *not present* in the upstream
evidence. This is an intrinsic/extrinsic faithfulness failure in the taxonomy of Ji *et al.* [H1] and Huang
*et al.* [H2], and in an ML-engineering setting it is actively dangerous — a confidently fabricated diagnosis
is worse than silence because the practitioner trusts the tool.

**Closest prior art and what it does *not* do.** Constrained / structured generation — Outlines [C1], LMQL
[C2], grammar-constrained decoding [C3], Guidance [C4], strict JSON-schema decoding [C5] — and the
schema-validated tool-use APIs that ship with every major LLM provider all enforce a constraint that is
**declared statically at authoring time**: a regular language, a grammar, or a fixed JSON schema. A static
schema can guarantee that an output *is a string in field `columns_referenced`*; it **cannot** know that, for
*this specific dataset at this specific call*, the only legitimate strings are `{age, bmi, log_charges_leak}`.
A fabricated-but-well-typed column name therefore passes every static check.

> **Gap sentence (verbatim for the Introduction):** *No existing constrained-generation or tool-use mechanism
> binds the constraint's value domain to the deterministic upstream evidence at call time; consequently a
> fabricated entity that is type-valid but evidence-absent is not rejected.*

**Our move.** mlcompass closes exactly this gap with an **evidence-bound runtime schema**: at call time, the
`enum` domain of the narrator's tool-input schema is *generated from the deterministic evidence dictionary*,
so the agent literally cannot emit a column the upstream layer did not observe — any such emission is rejected
at the API boundary and the agent retries. This converts a *probabilistic, prompt-level* mitigation into a
*structural, worst-case* guarantee. The mechanism (schema-validated tool use) is standard; **the novelty is
binding the enum domain to runtime-computed evidence rather than to a static type.**

---

## 3. mlcompass positioning (what it is, why we built it)

**What it is.** An open-source (MIT) command-line assistant that accompanies an engineer across the *whole* ML
pipeline — dataset advice, script audit, live-training watch, evaluation, drift monitoring, deployment checks —
where existing tools each cover one slice (ydata-profiling [T2] for EDA, MLflow [T3] / TensorBoard [T4] for
tracking, auto-sklearn [T1] for model search) and none carry state across stages or narrate their own evidence
safely. It is built as **two strictly separated layers**: a deterministic pure-Python *evidence* layer (output
is always a structured dictionary, never free text → fully offline) and an *optional* LLM *narrator* layer that
explains that dictionary in natural language. The capability set is exposed through four surfaces — CLI, an MCP
server [A3], an autonomous agent, and coding-assistant slash commands.

**Why we built it / why it earns a venue.** It is the concrete vehicle for the evidence-bound contract above,
demonstrated on a high-stakes, easy-to-miss task: **automatic data-leakage investigation** [L1]. The narrator
is triggered only when a deterministic metric looks impossibly good (e.g., R² > 0.999); a fabricating advisor
at that exact moment would send the engineer chasing a nonexistent cause. We measure the fabrication rate under
three contract configurations and show it falls 8.0% → 1.0% → 0.0% as the evidence-bound schema is engaged
(Wilson 95% CIs reported; honest note that the L2/L3 separation is not statistically clean at N=200, but L3
gives a *worst-case* boundary regardless of the underlying rate).

**INISTA framing — Agentic AI.** This is an *agent-reliability* result: a structural safety boundary for
tool-using LLM agents that narrate deterministic evidence. It generalizes beyond ML tooling to **any agentic
system that explains structured outputs to a human** — diagnostics, expert systems, intelligent healthcare
assistants — which is squarely within INISTA's "Agentic AI / Generative AI / Intelligent Agents" scope.

---

## 4. De-trademark plan (no product/brand names in the body)

| In current draft | INISTA replacement |
|------------------|--------------------|
| "Claude-based narrator", "an optional Claude narrator" | "an optional LLM narrator (a commercial instruction-tuned model accessed via a JSON-schema-constrained tool-use API)" |
| "Claude 3.5 Sonnet, temperature 0.7, `claude-3-5-sonnet-20241022`" | "a widely used commercial LLM with strict-schema tool use, temperature 0.7; exact model version, prompts, and seeds are pinned in the public repository for reproducibility" |
| "Claude Desktop, Cursor, Continue, Claude Code" (MCP clients) | "MCP-compatible LLM clients (popular code editors and desktop assistants)" |
| "Claude Code slash commands (`mlc-*`)" | "slash-command integration for MCP-aware coding assistants" |
| "Anthropic SDK throws `ToolInputValidationError`" | "the provider SDK raises a tool-input validation error" |
| "standard Anthropic tool-use schema enforcement" | "standard schema-validated tool-use (as offered by major LLM providers)" |

> Related-work citations of OpenAI structured outputs [C5], Microsoft Guidance [C4], and the MCP standard [A3]
> stay — these are legitimate scholarly comparisons, not advertising. The change is that **no single product is
> named as "the engine" of mlcompass**; the engine is described generically and reproducibly.

---

## 5. Section budget for 6 pages (IEEE two-column)

| Section | Target |
|---------|--------|
| Abstract + Index Terms | ~180 words |
| I. Introduction (gap + 3 contributions) | ~0.7 col |
| II. Related Work (5 buckets: hallucination, agents/tool-use, constrained gen, code-assistant reliability, ML tooling+leakage) | ~1.0 col |
| III. Methodology (threat model, architecture+Fig.1, evidence-bound contract L1/L2/L3, cross-surface parity) | ~2 col |
| IV. Experiments (setup, metrics, Table I ablation, latency/cost, Table II field bugs) | ~1.5 col |
| V. Discussion & Error Analysis (prompts vs schemas, real-data bugs, parity, limitations) | ~1 col |
| VI. Conclusion | ~0.4 col |
| References (16–18) | ~0.8 col |

**Next step after sign-off:** write the full paper from this brief → run 5 INISTA reviewers → revise.
