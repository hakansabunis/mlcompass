# An Evidence-Bound Runtime Schema for Claim-Faithful LLM Narration of Machine-Learning Pipeline Evidence

**Hakan Sabuniş\*, Yusuf Ünlü\*, Selim Akyokuş†**
*\*Dept. of Computer Engineering, †Dept. of Computer Engineering (Advisor), Istanbul Medipol University, Istanbul, Türkiye*
*hakan.sabunis@std.medipol.edu.tr, yusuf.unlu@std.medipol.edu.tr, sakyokus@medipol.edu.tr*

## Abstract

Large language model (LLM) agents are increasingly placed as an advisory layer over deterministic software tools: the agent invokes a tool, receives structured output, and narrates it in natural language for a human operator. We address a reliability failure of this pattern, *phantom-entity fabrication*, in which the narrator cites entities — column names, defects, fixes, or numbers — that are absent from, or contradict, the upstream deterministic evidence. We observe that such narration tasks are *evidence-closed*: the set of citable entities and verifiable quantities is finite and machine-enumerable at call time, which makes faithfulness checking *decidable* — so fabrication can be **prevented by construction** rather than statistically mitigated. We present a two-tier *evidence-bound runtime schema* that exploits this: (i) the `enum` domains of the narrator's tool-input fields are generated from the deterministic evidence dictionary at call time, and (ii) the returned answer is deterministically verified on three channels — *entity soundness* (every cited column exists in the evidence), *value soundness* (every quantitative claim, emitted as a structured `{column, statistic, value}` triple, matches the measured value), and *completeness* (a committed verdict addresses the top-ranked candidate) — with corrective retry and residual stripping. Tier (ii) is provider-independent: the guarantee holds even where the provider does not enforce schemas during decoding. We instantiate the contract in **mlcompass**, an open-source ML-pipeline assistant, in the shipped narrator that investigates suspiciously perfect metrics (a classic symptom of data leakage). In a live three-channel ablation (N = 200 per configuration, Wilson 95% CIs), the bare narrator fabricated out-of-evidence entities in 15.0% of responses — and 56.5% under a second bare-prompt variant, a near-fourfold swing from one innocuous prompt change — while the strict prompt and the full contract eliminated every observed violation on all three channels (0/200 each). We frame the result honestly: observed rates are volatile across prompts and models; the contribution is the categorical guarantee, which is not.

**Index Terms:** Agentic AI, large language models, hallucination mitigation, constrained generation, claim verification, tool use, data leakage.

---

## I. Introduction

Modern machine-learning engineering is supported by a fragmented ecosystem of single-purpose tools: profiling libraries for dataset inspection [13], experiment trackers for training [14], [15], automated model search for selection [16], and container tooling for deployment. Each covers one slice of the pipeline; none gives end-to-end advice, and none carries state across stages. A natural response is to place a large language model (LLM) on top as an advisory layer — an *agent* that calls these tools, reads their structured output, and explains it to the engineer. This is the agentic pattern formalized by ReAct [5] and Toolformer [6] and operationalized by tool-integration standards [7].

The pattern has a well-documented hazard: LLMs hallucinate [1], [2]. General-purpose code assistants built on the same foundation models [3] still fail on real software tasks at substantial rates [4]. In an advisory ML setting the danger is specific and acute. When the narrated object is *structured evidence* — a dictionary of measured facts produced by a deterministic tool — the agent can introduce what we call **phantom-entity fabrication**: it names a column, a defect, a remedy, or a number that is *not present in* (or contradicts) that evidence. This is a closed-world *extrinsic* faithfulness failure in the taxonomy of Ji *et al.* [1]: the cited material has no support in a source that is fully enumerable. A fabricated diagnosis delivered with fluent confidence is worse than no diagnosis at all, because the practitioner acts on it.

Our starting observation is that this setting has a special structure the general hallucination problem lacks. **Definition (evidence-closed narration).** *A narration task is evidence-closed if the set of citable entities and verifiable quantities is finite and machine-enumerable at the moment of the narration call.* For evidence-closed tasks, faithfulness checking is *decidable*: entity citations reduce to set membership, quantitative claims to numeric comparison, and coverage of critical evidence to anchor checking. Decidability changes the goal from *mitigation* (lowering a fabrication probability) to *prevention* (making the unfaithful output unrepresentable or unreturnable).

A reasonable starting point is *constrained generation*: Outlines [8], LMQL [9], grammar-constrained decoding [10], and Guidance [11] enforce a grammar or schema during decoding, and provider tool-use APIs validate calls against a declared JSON schema [12]. These mechanisms can be driven by *input-dependent* grammars [10] and per-request schemas; the capability to vary a constraint with the input exists. Two gaps remain. First, decoding-time enforcement requires access to the model's token distribution, which commercial APIs do not expose. Second, even a per-call schema offers no guarantee unless the provider's enforcement is trusted — which, for a safety claim, it should not be.

We close these gaps with a two-tier **evidence-bound runtime schema**. *Tier A (constrained generation):* the `enum` domains of the narrator's tool-input fields are generated at call time from the deterministic evidence dictionary. *Tier B (deterministic verification):* the returned answer is checked, in plain code, on three channels — entity soundness, value soundness of structured `{column, statistic, value}` claims (within a small tolerance so honest rounding passes), and completeness with respect to the top-ranked candidate. Violations trigger a corrective retry; after the budget, unsound entities and claims are stripped, and persistent omissions are flagged. The guarantee — *every entity and every number the user sees is present in, and equal to, the deterministic evidence* — holds independent of the provider. We are explicit that the verification step is, in isolation, simple; the contribution is recognizing the decidable class, binding the constrained-generation domain to the evidence, and combining both into a measured, provider-independent contract.

We instantiate the contract in **mlcompass**, an open-source (MIT) ML-pipeline assistant, in the production narrator triggered when a deterministic metric looks impossibly good (e.g., R² > 0.999) — the canonical signature of data leakage [17].

The paper makes three contributions:

1. We characterize *phantom-entity fabrication* and define the *evidence-closed* class of narration tasks for which faithfulness is decidable and fabrication is preventable by construction (Sec. I, III-C).
2. We propose the two-tier evidence-bound contract covering entity soundness, value soundness, and completeness, and measure it in a live three-channel ablation (N = 200, Wilson 95% CIs) — including a prompt-sensitivity finding: two bare-prompt variants on the same model and task fabricated at 56.5% and 15.0%, a near-fourfold swing that argues directly for structural rather than prompt-level defenses (Sec. IV).
3. We report engineering findings from shipping the contract inside a real multi-surface tool, including a cross-surface state-parity hazard (Sec. III-D, V-C).

Section II reviews related work, Section III describes the system and contract, Section IV reports experiments, Section V discusses lessons and limitations, and Section VI concludes.

---

## II. Related Work

**Hallucination and faithfulness in generation.** Ji *et al.* [1] survey hallucination across generation tasks and distinguish intrinsic from extrinsic faithfulness failures; Huang *et al.* [2] give an LLM-specific taxonomy. Phantom-entity fabrication is an extrinsic failure with special structure: the "source" is a finite, machine-enumerable evidence dictionary, making the faithfulness check decidable rather than open-world. Our work eliminates this decidable slice at the tool boundary.

**LLM agents and tool use.** ReAct [5] interleaves reasoning with tool actions, and Toolformer [6] shows models can learn when and how to call external APIs. Tool-integration standards [7] let LLM clients discover and call third-party tools over a uniform protocol. These establish *how* agents act; they do not constrain *what* an agent may say about a tool's output. Our contract sits at that under-specified boundary.

**Constrained and structured generation — the closest prior art.** Outlines [8] reformulates guided decoding as transitions over a finite-state machine. LMQL [9] adds declarative constraints over a typed grammar. Grammar-constrained decoding [10] guarantees grammatical validity without fine-tuning and *explicitly supports input-dependent grammars*. Guidance [11] enforces JSON-schema and regex constraints at the token level, and provider tool-use APIs validate calls against a declared JSON schema [12]. We do not claim that varying a constraint with the input is new — [10] already does. Our setup differs in two ways: the value domain is bound specifically to a *deterministic evidence producer*, so the constraint tracks ground-truth observations rather than task structure; and enforcement is not delegated to the decoder or provider — a deterministic post-verification makes the guarantee hold even if the provider admits an out-of-enum token, which matters because decoding-time enforcement is unavailable behind commercial APIs. The combination — evidence-bound domains, claim-level value verification, completeness checking, and provider-independent enforcement, framed around a decidable faithfulness class — is, to our knowledge, not described as a contract in this literature.

**Code assistants and their reliability.** Foundation models trained on code [3] power widely-used assistants, yet they resolve only a fraction of real-world software issues [4] and, in our informal pre-project trials, silently accepted training scripts with semantically invalid configurations (e.g., a `momentum` argument to an optimizer that does not accept one). General-purpose assistants target syntactic correctness; ML-pipeline tooling must target domain-specific semantic correctness — without fabricating.

**ML pipeline tooling.** Profiling libraries [13] produce descriptive statistics but have no notion of a target column. Experiment trackers [14], [15] record per-run metrics without interpreting them. Automated model search [16] selects estimators without narrating *why*. mlcompass adds a state-carrying advisory layer over the same artifacts; the contribution of *this paper* is the reliability contract that makes its narrator safe, not the tool surface.

---

## III. Methodology

### A. Threat Model

We assume an honest operator and a well-behaved LLM provider — no adversarial weight poisoning, no malicious operator. The risk we address is the model's tendency to fabricate or misreport entities and quantities under output-distribution pressure. We explicitly do **not** address: (i) adversarial column names in user data attempting prompt injection; (ii) provider outages; or (iii) miscalibrated confidence on correctly-cited, correctly-valued evidence. Residual risks are discussed in Sec. V-E.

### B. System Architecture

mlcompass is organized into three layers (Fig. 1). **Surfaces:** eleven pipeline tools are exposed through a CLI, a tool-integration server [7] for compatible LLM clients, an autonomous agent, and slash commands for coding assistants — all routing to one deterministic backend. **Two-layer brain:** the deterministic evidence layer is pure Python and always returns a structured dictionary (never free text), so every command has a fully offline path; the optional LLM narrator receives that dictionary and explains it, running only when requested. **Persistent context:** a `.mlcompass/` folder persists project metadata, a chronological decision log, and dataset/run records across invocations.

**[INSERT FIGURE 1 HERE]**

*Fig. 1. System architecture. Four surfaces sit on a two-layer brain — a deterministic tool layer plus an optional LLM narrator — over a persistent project-context folder. The evidence-bound schema is generated at the tool→narrator boundary, and the narrator's entities and claims are verified against the evidence before they reach the user.*

### C. The Evidence-Bound Contract

We describe the contract through the automatic leakage investigation [17] triggered when an evaluation observes a suspicious metric (AUC > 0.995, accuracy > 0.99, or R² > 0.999). The investigation is evidence-closed: the citable world is exactly the evidence dictionary *E*.

**Layer 1 — Deterministic evidence producer.** A pure-Python module reads the predictions table and emits *E* containing (i) the suspicious metric; (ii) per-feature correlations against the target *y*, reported as max(|ρ_P|, |ρ_S|) with ρ_P, ρ_S the Pearson and Spearman correlations; (iii) a candidate-leak set *C* = { *c* : max(|ρ_P(c, y)|, |ρ_S(c, y)|) ≥ 0.99 }, ranked by strength; and (iv) the perfect-match rate P(ŷ = y). Spearman is reported alongside Pearson because monotone target transforms (log y, √y, αy + β) weaken ρ_P while leaving ρ_S = 1.0; without it, a log-target leak at ρ_P ≈ 0.94 would slip past a Pearson-only filter.

**Layer 2 — Strict-prompted narrator.** The narrator receives *E* with a system prompt: cite only items present in *E*; report every correlation number as a structured claim copied exactly from *E*; address the top-ranked candidate when committing to a verdict; emit `cannot_determine` rather than guess; propose only manual checks.

**Layer 3 — Evidence-bound runtime schema (two tiers).** The narrator answers through a `submit_investigation` tool.

*Tier A — runtime enums.* The tool's `columns_referenced` items and `claims[].column` field are restricted by JSON-schema `enum`s *generated at call time* from *E*: the domain is *C* ∪ {features reported in the correlation pass}. This biases the decoder toward valid citations and is the constrained-generation half of the contract.

*Tier B — deterministic verification.* When the tool input returns, plain code checks three properties — *not* trusting the provider's enforcement:

1. **Entity soundness:** every column in `columns_referenced` and in claims is in the evidence set.
2. **Value soundness:** every claim `{column, statistic, value}` matches the measured value within a tolerance of 0.005 (so a value quoted to two decimals passes; a genuine misquote fails).
3. **Completeness:** if the verdict is anything other than `cannot_determine`, the response must address the top-ranked candidate column (in citations or claims). An explicit abstention is not an omission.

On any violation the contract issues a corrective retry naming the offending items, up to two attempts. After the budget, unsound entities and claims are *stripped* before the response reaches the user; a persistent omission is *flagged* (an omission cannot be stripped). The resulting invariant: *for any response the contract returns, every cited entity and every quantitative claim is present in, and equal to, the deterministic evidence — by construction, independent of the provider.* We are explicit that each individual check is simple — set membership, numeric comparison, anchor coverage; that simplicity is exactly what decidability buys, and it is why the guarantee can be absolute where decoding-time methods [8], [10] cannot even be applied (no logit access behind commercial APIs).

### D. Cross-Surface State Parity

A practical hazard, independent of the contract, is that one capability shipped through two surfaces can silently diverge. In mlcompass the CLI wrapped every command in a ledger-write step that the independently-written tool-integration server bypassed, so a full pipeline run through the server left the status query empty. We fixed this with a single shared persistence helper both surfaces call, plus dedicated parity regression tests. The lesson generalizes: when one capability ships through several interfaces, parity must be enforced through shared code and tested as a first-class invariant.

---

## IV. Experiments

### A. Setup

**Hallucination test set.** We construct a synthetic regression frame (1,000 rows, 12 features) with a monotone transformed leak `log_target_v2 = log(y) + noise(σ = 0.01)` and a high-correlation distractor `near_target_proxy = y + noise(σ = 0.5)`, train a model that essentially copies the leak, and run the *shipped* `detect_leakage` to obtain the evidence dictionary — so the experiment narrates the exact evidence shape the product surfaces (10 evidence columns; anchor `log_target_v2`). For each configuration we sample 200 narrator responses (a widely used commercial instruction-tuned LLM accessed through a JSON-schema tool-use API at provider-default sampling settings; the exact model version, all prompts, seeds, and the reproduction script are pinned in the public repository). The Layer-3 configuration calls the *shipped* `investigate_leakage_bound`, so the evaluated mechanism is the deployed mechanism. Notably, the measured endpoint does not strictly enforce schema enums during decoding, so Tier B is the operative enforcement — making the run a direct test of the provider-independent guarantee.

**Configurations.** L1: a bare prompt that requests the structured fields but states no faithfulness rules, with an unconstrained (open) tool schema and no verification. L2: the strict prompt of Sec. III-C, still unenforced. L3: the full contract. We additionally report an earlier live run whose bare prompt did *not* request structured claims; both raw records are in the repository.

**Field test set.** Five public tabular datasets spanning binary classification, regression, categorical-feature classification, three-class classification, and a target-naming corner case; for each we ran the full pipeline and logged every unexpected behavior.

### B. Evaluation Metrics

Per response that reaches the user: **entity-fabrication** — references ≥ 1 column (citation or claim) outside *E*; **value-fabrication** — carries ≥ 1 claim whose column is in *E* but whose value differs from the measured one by > 0.005; **critical omission** — commits to a verdict yet never references the top-ranked candidate. We report rates with Wilson 95% CIs (well-behaved near 0) and raw k/N.

### C. Results

**Three-channel ablation.** Table I reports the entity channel, measured live. The bare narrator fabricates out-of-evidence entities in 15.0% of responses (30/200). The strict prompt eliminates every observed violation (0/200), as does the full contract (0/200). On the two new channels the observed rates were zero in *every* configuration — no value misquote and no critical omission in 1,800 total scored responses (0/200 per cell; Wilson upper bound 1.88% each): when this model emits structured claims, it copies the values accurately on this task, and the anchor is salient enough that it is always addressed.

**TABLE I. USER-FACING ENTITY-FABRICATION RATE BY CONTRACT CONFIGURATION (LIVE RUN, N = 200, WILSON 95% CI)**

| Configuration | Rate | 95% CI | k / N |
| --- | :---: | :---: | :---: |
| L1: bare prompt (open schema, no verification) | 15.0% | [10.71, 20.61] | 30 / 200 |
| L2: + strict prompt | 0.0% | [0.00, 1.88] | 0 / 200 |
| L3: + evidence-bound contract (Tier A + B) | **0.0%** | [0.00, 1.88] | 0 / 200 |

**The prompt-sensitivity finding.** An earlier live run on the *same model, task, and seed*, whose bare prompt did not request structured claims, measured 56.5% (113/200) entity fabrication. Adding one sentence — a request to report numbers as structured claims — dropped the bare rate to 15.0%: a near-fourfold swing from an innocuous prompt change, with no enforcement involved. Two lessons follow. First, even *requesting* structure (without enforcing it) grounds the narrator substantially, consistent with the constrained-format literature. Second, and more importantly, prompt-level fabrication rates are volatile in both directions across prompts and models; a defense whose effectiveness swings 4× with one sentence is not a defense to certify. **We therefore lead with the guarantee, not the point estimates.** At N = 200 the strict prompt and the full contract are empirically indistinguishable (both 0/200, upper bound 1.88%); the contribution of Layer 3 is categorical — by Tier B, every returned response is claim-faithful by construction, at any underlying propensity, under any prompt.

**Overhead.** The contract adds provider calls only on violation: one call per investigation, plus at most two corrective retries. In the live Layer-3 run no violation occurred, so the contract added zero overhead calls; the verification itself is set membership and numeric comparison. Worst case is three calls per investigation.

**Field-test bug categorization.** Five field tests surfaced eleven distinct issues (Table II): five trivial canonical-target-name gaps, three CLI/server state-parity defects (Sec. III-D), and three UX confusions. Each was reproduced as a failing regression test before being fixed.

**TABLE II. FIELD TESTS, ISSUES SURFACED, AND CATEGORIES**

| # | Dataset domain | Patch | Issues | Categories |
| :---: | --- | :---: | :---: | --- |
| 1 | Binary churn | — | 0 | baseline; none surfaced |
| 2 | House-price regression | v0.7.0 | 3 | 1 target name, 1 UX, 1 crash |
| 3 | Categorical-feature binary | v0.7.1 | 1 | 1 target name |
| 4 | Three-class classification | v0.7.2 | 3 | 1 multiclass, 1 parity, 1 UX |
| 5 | Insurance-charge regression | v0.7.3 | 4 | 3 target names, 1 parity |

**Release artifact.** The release ships 11 CLI commands, 8 protocol-exposed tools, and 11 slash commands; the regression suite includes 20 dedicated end-to-end tests of the contract (enum binding, value tolerance, misquote retry/strip, omission retry/flag, abstention exemption, both provider tool formats) with mock clients, and all quality gates (`ruff`, `mypy --strict`) report clean.

---

## V. Discussion and Error Analysis

### A. Prompts Are Advisory, Schemas Are Enforcement

The measured ablation sharpens the principle. Across two bare-prompt variants the entity-fabrication rate swung from 56.5% to 15.0% — prompt efficacy is an empirical property of a specific model–prompt–task triple, exquisitely sensitive to wording, and nothing in a prompt prevents silent regression under a model update. The strict prompt then drove every observed rate to zero — and that is still not certifiable, because a 0/200 leaves a 1.88% upper bound and no structural reason to hold. The evidence-bound contract provides what no prompt can: a worst-case guarantee, enforced by Tier B's deterministic verification, that holds at any propensity. *Prompts are advisory; schemas, with verification, are enforcement.*

### B. Negative Results Are Boundaries, Not Failures

We measured no value misquotes and no critical omissions anywhere. We report this plainly rather than dramatizing those channels: on this task the numbers sit verbatim in the context (copying is easy) and the anchor is the most salient evidence item. The channels may become live under longer contexts, derived statistics, or busier evidence; the contract closes them *pre-emptively* at the cost of a set lookup and a float comparison. Verifying cheap invariants before they are observed failing is, we argue, the right default for evidence-closed narrators.

### C. Cross-Surface Parity Is Engineering Work

The state-parity gap (Sec. III-D) is the most transferable engineering finding: two surfaces invoking the same deterministic functions diverged because one performed a ledger-write step the other skipped. Any system shipping one capability through several interfaces must enforce parity through shared code and test it as an invariant.

### D. Beyond ML: Where the Contract Transposes

The contract requires three invariants: a *deterministic evidence producer*, a *finite enumerable set of observed entities and values*, and a *runtime enum bound to that set with independent verification*. Wherever those hold — i.e., wherever narration is evidence-closed — the same guarantee applies. A clinical-decision narrator over a deterministic lab-analysis tool could bind its citations to the patient's actually-measured abnormal results and verify cited values against the measurements; an expert-system narrator could bind citations to fired rule IDs. We have not evaluated these domains — we offer them as the boundary of the claim. The contribution is a reliability primitive for tool-using agents that narrate structured evidence, of which ML-pipeline diagnosis is one instance.

### E. Limitations and Residual Risks

1. **Single-provider evaluation.** All measurements use one commercial LLM; rates should not be generalized across models — indeed, our own prompt-sensitivity finding argues they do not even generalize across prompts.
2. **Single synthetic dataset, N = 200.** A larger study across diverse leak patterns is needed for tight CIs. We did not log the composition of L1 entity violations (invented names vs. real-but-out-of-evidence columns) or per-response claim counts.
3. **Channels not covered.** Miscalibrated confidence on correctly-cited, correctly-valued evidence; fabrication inside the free-text `narration` field (renderers surface the validated citation and claim sets as the actionable output); and prompt-injection compliance remain out of scope.
4. **Evidence-omission trade.** If the deterministic layer misses the true leak, the enum structurally bars the narrator from naming it — a precision-for-recall trade inherent to the binding.
5. **No head-to-head against a static-schema decoder**; we expect comparable enforcement where decoders are applicable, but the comparison has not been run.

---

## VI. Conclusion

We defined *evidence-closed narration* — tasks whose citable entities and verifiable quantities are finite and machine-enumerable at call time — and showed that for this class, faithfulness is decidable and fabrication preventable by construction. We presented a two-tier *evidence-bound runtime schema* realizing this: call-time enum binding (Tier A) plus deterministic verification of entity soundness, value soundness, and completeness with retry-strip-flag semantics (Tier B), provider-independent by design. In a live three-channel ablation (N = 200, Wilson 95% CIs), bare narrators fabricated entities at 56.5% and 15.0% across two prompt variants — a near-fourfold prompt sensitivity that is itself the argument for structural defenses — while the contract eliminated every observed violation and guarantees the worst case categorically. The contract ships in the production narrator of mlcompass, with its measurement harness and records public. Future work is a research program rather than a single study: scaling the measurement across providers, tasks, and leak patterns into a public narrator-fabrication benchmark; extending verified claims to derived and relational statistics; composing contracts across multi-step agent chains, where each step's enum derives from the *verified* output of the previous one; and applying the contract to other evidence-closed narrators inside and beyond ML tooling.

**Source code:** github.com/hakansabunis/mlcompass — **Package:** PyPI, MIT license.

---

## References

[1] Z. Ji *et al.*, "Survey of hallucination in natural language generation," *ACM Comput. Surv.*, vol. 55, no. 12, Art. 248, 2023.

[2] L. Huang *et al.*, "A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions," *arXiv:2311.05232*, 2023.

[3] M. Chen *et al.*, "Evaluating large language models trained on code," *arXiv:2107.03374*, 2021.

[4] C. E. Jimenez *et al.*, "SWE-bench: Can language models resolve real-world GitHub issues?" in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2024.

[5] S. Yao *et al.*, "ReAct: Synergizing reasoning and acting in language models," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2023.

[6] T. Schick *et al.*, "Toolformer: Language models can teach themselves to use tools," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2023.

[7] Model Context Protocol specification, 2024. [Online]. Available: https://modelcontextprotocol.io

[8] B. T. Willard and R. Louf, "Efficient guided generation for large language models," *arXiv:2307.09702*, 2023.

[9] L. Beurer-Kellner, M. Fischer, and M. Vechev, "Prompting is programming: A query language for large language models," *Proc. ACM Program. Lang.*, vol. 7, no. PLDI, pp. 1946–1969, 2023.

[10] S. Geng, M. Josifoski, M. Peyrard, and R. West, "Grammar-constrained decoding for structured NLP tasks without finetuning," in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, 2023, pp. 10932–10952.

[11] Guidance: A guidance language for controlling large language models, 2023. [Online]. Available: https://github.com/guidance-ai/guidance

[12] Structured outputs for schema-constrained tool use, 2024. [Online]. Available: https://platform.openai.com/docs/guides/structured-outputs

[13] ydata-profiling: Automated exploratory data analysis for pandas DataFrames. [Online]. Available: https://github.com/ydataai/ydata-profiling

[14] M. Zaharia *et al.*, "Accelerating the machine learning lifecycle with MLflow," *IEEE Data Eng. Bull.*, vol. 41, no. 4, pp. 39–45, 2018.

[15] M. Abadi *et al.*, "TensorFlow: A system for large-scale machine learning," in *Proc. USENIX Symp. Operating Syst. Design Implementation (OSDI)*, 2016, pp. 265–283.

[16] M. Feurer *et al.*, "Efficient and robust automated machine learning," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2015, pp. 2962–2970.

[17] S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," *ACM Trans. Knowl. Discovery Data*, vol. 6, no. 4, Art. 15, 2012.
