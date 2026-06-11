# An Evidence-Bound Runtime Schema for Claim-Faithful LLM Narration of Machine-Learning Pipeline Evidence

**Hakan Sabuniş\*, Yusuf Ünlü\*, Selim Akyokuş†**
*\*Dept. of Computer Engineering, †Dept. of Computer Engineering (Advisor), Istanbul Medipol University, Istanbul, Türkiye*
*hakan.sabunis@std.medipol.edu.tr, yusuf.unlu@std.medipol.edu.tr, sakyokus@medipol.edu.tr*

## Abstract

Large language model (LLM) agents increasingly narrate the structured output of deterministic tools for human operators, and they fabricate: they cite columns, defects, or numbers that are absent from, or contradict, the upstream evidence. We observe that such narration is *evidence-closed*: the citable entities and verifiable quantities are finite and machine-enumerable at call time, so faithfulness checking is decidable and fabrication can be prevented by construction rather than statistically mitigated. We present a two-tier *evidence-bound runtime schema*: tool-input `enum` domains are generated from the evidence at call time, and the returned answer is deterministically verified for entity soundness, claim-value soundness, and completeness, with corrective retry, stripping, and flagging, independent of provider schema enforcement. We record two load-bearing formal observations: verification is linear-time, and soundness is safely repairable by deletion while completeness is not. Live measurements (N = 200 per cell) make the case: across six rule-free paraphrases of one instruction, the bare narrator's fabrication rate on identical evidence spanned 1%–100%, while the contract held the user-facing rate at 0/200 on all three verified channels of a synthetic and a real-data leakage task; under stress the verifier caught 76 live violations, passed none to the user, and one corrective message cut repeat violations from 37.5% to 1.3%. The contract ships in the production narrator of mlcompass, an open-source ML-pipeline assistant.

**Index Terms:** Agentic AI, large language models, hallucination mitigation, constrained generation, claim verification, tool use, data leakage.

---

## I. Introduction

Modern machine-learning engineering is supported by a fragmented ecosystem of single-purpose tools: profiling libraries for dataset inspection [13], experiment trackers for training [14], [15], automated model search for selection [16]. Each covers one slice of the pipeline; none gives end-to-end advice. A natural response is to place a large language model (LLM) on top as an advisory layer: an *agent* that calls these tools, reads their structured output, and explains it to the engineer, in the agentic pattern formalized by ReAct [5] and Toolformer [6] and operationalized by tool-integration standards [7].

The pattern has a well-documented hazard: LLMs hallucinate [1], [2]. Code assistants built on the same foundation models [3] still fail on real software tasks at substantial rates [4]. In an advisory ML setting the danger is specific. When the narrated object is *structured evidence*, a dictionary of measured facts produced by a deterministic tool, the agent can introduce what we call **phantom-entity fabrication**: it names a column, a defect, a remedy, or a number that is *not present in* (or contradicts) that evidence. In the taxonomy of Ji *et al.* [1] this is a closed-world faithfulness failure (extrinsic when an entity is invented, intrinsic when a quantity contradicts the source) over a source that is fully enumerable. A fabricated diagnosis delivered with fluent confidence is worse than no diagnosis, because the practitioner acts on it.

Our starting observation is that this setting has structure the general hallucination problem lacks. **Definition (evidence-closed narration).** *A narration task is evidence-closed if the set of citable entities and verifiable quantities is finite and machine-enumerable at the moment of the narration call.* For evidence-closed tasks, faithfulness checking is *decidable*: entity citations reduce to set membership, quantitative claims to numeric comparison, coverage of critical evidence to anchor checking. Decidability changes the goal from *mitigation* (lowering a fabrication probability) to *prevention* (making unfaithful output unreturnable).

A natural starting point is *constrained generation*: Outlines [8], LMQL [9], grammar-constrained decoding [10], and Guidance [11] enforce a grammar or schema during decoding, and provider tool-use APIs validate calls against a declared JSON schema [12]. These mechanisms can be driven by *input-dependent* grammars [10] and per-request schemas. Two gaps remain. First, decoding-time enforcement requires access to the model's token distribution, which commercial APIs do not expose. Second, even a per-call schema offers no guarantee unless the provider's enforcement is trusted, which, for a safety claim, it should not be.

We close these gaps with a two-tier **evidence-bound runtime schema**. *Tier A (constrained generation):* the `enum` domains of the narrator's tool-input fields are generated at call time from the deterministic evidence dictionary. *Tier B (deterministic verification):* the returned answer is checked in plain code on three channels: entity soundness, value soundness of structured `{column, statistic, value}` claims (within a small tolerance so honest rounding passes), and completeness with respect to the top-ranked candidate. Violations trigger a corrective retry; after the budget, unsound entities and claims are stripped, and persistent omissions are flagged. The resulting guarantee holds independent of the provider: *every entity and every number in the validated citation and claim channels is present in, and equal to, the deterministic evidence*; free-text narration sits outside the verified channels, and renderers surface the validated fields as the actionable output. The verification step is deliberately simple; that simplicity is what decidability buys (Sec. III-C formalizes this), and it is why the guarantee can be absolute where decoding-time methods cannot even be applied.

We instantiate the contract in **mlcompass**, an open-source (MIT) ML-pipeline assistant, in the production narrator triggered when a deterministic metric looks impossibly good (e.g., R² > 0.999), the canonical signature of data leakage [17].

The paper makes three contributions:

1. We define the *evidence-closed* class of narration tasks and establish its two load-bearing formal properties: faithfulness verification is linear-time (Prop. 1), and soundness is safely repairable by deletion while completeness is not (Prop. 2); this is the structural reason the contract strips unsound content but can only flag omissions (Sec. III-C).
2. We measure the contract live on a synthetic and a real-data leakage task (N = 200 per cell, Wilson 95% CIs): bare narrators fabricate at 11.5% and 43.5%; the contract holds the user-facing rate at 0/200 on all three channels of both tasks; and in stress configurations the deterministic verifier demonstrably catches 80 live violations (76 on the real-data task) while passing none to the user; a pre-stated iid retry model is rejected in the informative direction: corrective feedback cuts repeat violations from 37.5% to 1.3% (Sec. IV).
3. We show with a six-way paraphrase sweep that the bare narrator's fabrication rate on identical evidence ranges from 1% to 100% under rule-free rewording; prompt-level defenses are not certifiable, which is the empirical case for structural enforcement (Sec. IV-C).

Section II reviews related work, Section III the system and contract, Section IV experiments, Section V discussion, Section VI conclusion.

---

## II. Related Work

**Hallucination and faithfulness.** Ji *et al.* [1] survey hallucination across generation tasks and distinguish intrinsic from extrinsic failures; Huang *et al.* [2] give an LLM-specific taxonomy. Phantom-entity fabrication is an extrinsic failure with special structure: the "source" is a finite, machine-enumerable evidence dictionary, making the faithfulness check decidable rather than open-world. Our work eliminates this decidable slice at the tool boundary.

**LLM agents and tool use.** ReAct [5] interleaves reasoning with tool actions; Toolformer [6] shows models can learn when and how to call external APIs; tool-integration standards [7] let LLM clients discover and call third-party tools over a uniform protocol. These establish *how* agents act; they do not constrain *what* an agent may say about a tool's output. Our contract sits at that under-specified boundary.

**Constrained and structured generation (closest prior art).** Outlines [8] reformulates guided decoding as transitions over a finite-state machine. LMQL [9] adds declarative constraints over a typed grammar. Grammar-constrained decoding [10] guarantees grammatical validity without fine-tuning and *explicitly supports input-dependent grammars*. Guidance [11] enforces JSON-schema and regex constraints at the token level, and provider tool-use APIs validate calls against a declared JSON schema [12]. We do not claim that varying a constraint with the input is new; [10] already does. Our setup differs in two ways: the value domain is bound specifically to a *deterministic evidence producer*, so the constraint tracks ground-truth observations rather than task structure; and enforcement is not delegated to the decoder or provider: deterministic post-verification makes the guarantee hold even if the provider admits an out-of-enum token, which matters because decoding-time enforcement is unavailable behind commercial APIs. The combination of evidence-bound domains, claim-level value verification, completeness checking, and provider-independent enforcement, framed around a decidable faithfulness class, is, to our knowledge, not described as a contract in this literature. A complementary practical line wraps LLM calls in output validators with automatic re-asking: Guardrails AI [18] and NeMo Guardrails [19] ship this validate-and-repair loop as general-purpose toolkits. Tier B shares the loop's shape; the difference is the validator itself: decidable faithfulness against a deterministic evidence producer, with the strip-vs-flag semantics that Prop. 2 shows is forced, rather than user-authored heuristic checks.

**Code assistants and their reliability.** Foundation models trained on code [3] power widely-used assistants, yet they resolve only a fraction of real-world software issues [4] and, in our informal pre-project trials, silently accepted semantically invalid training configurations. General-purpose assistants target syntactic correctness; ML-pipeline tooling must target domain-specific semantic correctness without fabricating.

**ML pipeline tooling.** Profiling libraries [13] have no notion of a target column; experiment trackers [14], [15] record metrics without interpreting them; automated model search [16] selects estimators without narrating *why*. mlcompass adds a state-carrying advisory layer over the same artifacts; the contribution of *this paper* is the reliability contract, not the tool surface.

---

## III. Methodology

### A. Threat Model

We assume an honest operator and a well-behaved LLM provider: no adversarial weight poisoning, no malicious operator. The risk addressed is the model's tendency to fabricate or misreport entities and quantities under output-distribution pressure. We do **not** address: (i) adversarial column names in user data attempting prompt injection; (ii) provider outages; (iii) miscalibrated confidence on correctly-cited, correctly-valued evidence. Residual risks: Sec. V-E.

### B. System Architecture

mlcompass is organized into three layers (Fig. 1): user-facing surfaces (a CLI, a tool-integration server [7], an autonomous agent, and editor slash commands) routing to one deterministic backend; a strictly split two-layer brain (a pure-Python deterministic evidence layer whose output is always a structured dictionary, plus an optional LLM narrator over it); and a persistent project-context folder. Because the evidence layer is deterministic and offline, every command works without the narrator; the narrator adds explanation, and the contract makes that explanation safe.

**[INSERT FIGURE 1 HERE]**

*Fig. 1. System architecture. Surfaces over a two-layer brain (deterministic tools + optional LLM narrator) over persistent context. The evidence-bound schema is generated at the tool→narrator boundary; the narrator's entities and claims are verified against the evidence before reaching the user.*

### C. The Evidence-Bound Contract

We describe the contract through the automatic leakage investigation [17] triggered when an evaluation observes a suspicious metric (AUC > 0.995, accuracy > 0.99, or R² > 0.999). The investigation is evidence-closed: the citable world is exactly the evidence dictionary *E*.

**Layer 1: Deterministic evidence producer.** A pure-Python module reads the predictions table and emits *E*: the suspicious metric; per-feature correlations against the target *y*, reported as max(|ρ_P|, |ρ_S|); a candidate-leak set *C* = { *c* : max(|ρ_P(c, y)|, |ρ_S(c, y)|) ≥ 0.99 }, ranked; and the perfect-match rate P(ŷ = y). Spearman is reported alongside Pearson because monotone target transforms (log y, √y, αy + β) weaken ρ_P while leaving ρ_S = 1.0; a log-target leak at ρ_P ≈ 0.94 would otherwise slip a Pearson-only filter.

**Layer 2: Strict-prompted narrator.** The narrator receives *E* with a system prompt: cite only items in *E*; report every correlation number as a structured claim copied exactly from *E*; address the top-ranked candidate when committing to a verdict; abstain (`cannot_determine`) rather than guess.

**Layer 3: Evidence-bound runtime schema (two tiers).** The narrator answers through a `submit_investigation` tool. *Tier A:* the tool's `columns_referenced` items and `claims[].column` field are restricted by JSON-schema `enum`s generated at call time from *E*. *Tier B:* when the tool input returns, plain code verifies, without trusting the provider, (1) **entity soundness**: every column cited or claimed is in the evidence set; (2) **value soundness**: every claim `{column, statistic, value}` matches the measured value within 0.005 (two-decimal rounding passes, misquotes fail); (3) **completeness**: a verdict other than `cannot_determine` must address the top-ranked candidate (explicit abstention is not an omission). Violations trigger a corrective retry naming the offending items (budget 2); residual unsound entities/claims are *stripped*; a persistent omission is *flagged*; completeness is then re-checked after stripping, since removing an unsound claim can itself orphan the anchor.

The asymmetric handling (strip vs. flag) is not an engineering preference but a structural necessity. We record it in two propositions; both are near-definitional, stated because the design leans on them:

> **Proposition 1 (linear-time verification).** *Let E define a finite entity set A_E (|A_E| = m) and a claim table V_E. Deciding faithfulness of a structured response r (entity membership, value tolerance, anchor coverage) takes O(|r|) expected time after O(m) preprocessing.* **Proof.** Hash A_E and V_E once; entity soundness is |r| membership tests, value soundness one lookup and one comparison per claim, completeness one membership test. ∎

> **Proposition 2 (repair asymmetry).** *Call a repair* safe *if it only deletes content: deletion never attributes new content to the narrator. Soundness is safe-repairable: the projection Π_E(r) that deletes out-of-evidence entities and out-of-tolerance claims lands in the sound set for every r and is the identity on sound responses. Completeness is not safe-repairable: it is an existential requirement, and any repair must add a reference the model did not produce; the repair itself would fabricate attribution. Deleting the committed verdict is no escape: the schema requires a verdict field, and substituting an abstention attributes an epistemic stance the model did not take, which is likewise unsafe.* **Consequence.** A contract can *enforce* soundness by construction (strip) but can only *detect, re-prompt, and flag* incompleteness. ∎

The resulting invariant: *for any response the contract returns, every cited entity and every quantitative claim is present in, and equal to, the deterministic evidence, by construction, independent of the provider.* The simplicity of each check (membership, comparison, anchor) is exactly what decidability buys, and why the guarantee can be absolute where decoding-time methods [8], [10] cannot even be applied behind commercial APIs.

---

## IV. Experiments

### A. Setup

**Tasks.** *Synthetic:* a 1,000 × 12 regression frame with a monotone transformed leak (`log_target_v2 = log(y) + 𝒩(0, 0.01²)`) and a high-correlation distractor; 10 evidence columns, anchor `log_target_v2`. *Real-data:* the public Insurance Charges dataset with an injected log-of-target leak and near-perfect predictions; external column names, distributions, and shape; 7 evidence columns, anchor `log_charges_leak`. In both, a model essentially copies the leak and the *shipped* `detect_leakage` produces the evidence, so the experiment narrates the exact evidence shape the product surfaces.

**Configurations (per task, N = 200 each).** *L1:* bare prompt (requests the structured fields, states no faithfulness rule), open tool schema, no verification. *L2:* the strict prompt, still unenforced. *L3:* the full shipped contract (`investigate_leakage_bound`). *STRESS:* the same shipped contract with the bare prompt and Tier A enums removed, so the narrator fabricates at its natural rate and every Tier B catch is observable. All runs use DeepSeek's `deepseek-chat` endpoint (OpenAI-compatible JSON-schema tool use, provider-default sampling; the name is a rolling model alias, so run dates and the harness commit pin the configuration). The provider does not document decode-time enum enforcement, and Tier B does not rely on it; our arms do not isolate Tier A's marginal contribution (Sec. V-E). Prompts, seeds, the harness, and all raw run records are in the public repository.

**Paraphrase sweep.** Six rule-free paraphrases of the bare instruction (terse, baseline, helpful, expert, mechanical, cautious), N = 100 each, same evidence and tool schema; only the wording varies.

### B. Metrics

Per response that reaches the user: **entity-fabrication** (≥ 1 column, cited or claimed, outside *E*); **value-fabrication** (≥ 1 claim whose column is in *E* but whose value misses by > 0.005); **critical omission** (committed verdict that never references the anchor). Wilson 95% CIs and raw k/N throughout. For contract arms we additionally report **Tier B catches**: responses with ≥ 1 deterministic rejection, and total rejections.

### C. Results

**TABLE I. USER-FACING ENTITY-FABRICATION RATE (LIVE, N = 200 PER CELL, WILSON 95% CI IN TEXT)**

| Configuration | Synthetic | Real-data (Insurance) |
| --- | :---: | :---: |
| L1: bare prompt, no enforcement | 11.5% (23/200) | 43.5% (87/200) |
| L2: + strict prompt | 0.0% (0/200) | 0.0% (0/200) |
| L3: full contract | 0.0% (0/200) | 0.0% (0/200) |
| STRESS: bare prompt, Tier B only | **0.0% (0/200)** | **0.0% (0/200)** |

CIs: L1 synthetic [7.79, 16.66], L1 real-data [36.82, 50.43]; all 0/200 cells [0.00, 1.88]. On the two further channels (value misquotes and critical omissions), observed violations were **zero in every cell of both tasks** (0/200 each): when this model emits structured claims it copies values accurately here, and the anchor is salient. We report these as boundaries, not victories (Sec. V-C).

**The paraphrase sweep: prompt wording is not a defense.** Table II gives the entity-fabrication rate of six rule-free paraphrases of the same instruction on identical evidence and schema. All six variants were authored before any measurement and none was discarded. The rate spans **1% to 100%**: a terse instruction nearly never fabricates, a "mechanical" field-listing one always does, and a helpful-assistant framing fabricates in 93% of responses. No obvious wording feature predicts the rate. A defense whose effectiveness moves two orders of magnitude under innocuous rewording cannot be certified; this is the empirical case for structural enforcement.

**TABLE II. BARE-PROMPT PARAPHRASE SWEEP (SYNTHETIC, N = 100 PER VARIANT)**

| Variant | Entity-fab | 95% CI | k / N |
| --- | :---: | :---: | :---: |
| terse | 1.0% | [0.18, 5.45] | 1 / 100 |
| baseline | 7.0% | [3.43, 13.75] | 7 / 100 |
| cautious | 45.0% | [35.61, 54.76] | 45 / 100 |
| expert | 90.0% | [82.56, 94.48] | 90 / 100 |
| helpful | 93.0% | [86.25, 96.57] | 93 / 100 |
| mechanical | 100.0% | [96.30, 100.00] | 100 / 100 |

**The verifier demonstrably fires.** In the STRESS configuration on the real-data task, Tier B caught violations in **75 of 200 responses (76 total catches)** and passed **zero** to the user; on the synthetic task, 4 catches, zero passed. A pre-stated iid retry model (E[calls] = 1 + q + q², residual-strip probability q³) predicts ≈ 114 total catches at the observed first-attempt rate q = 0.375; the measured 76 rejects the iid assumption in the informative direction: retries are not independent draws, because the corrective message changes behavior. After one corrective message naming the violation, only **1 of 75** responses violated again: the conditional repeat rate fell from 37.5% to 1.3% (Wilson 95% CI [0.2%, 7.2%]; a ~28× drop in point estimate, ≥ ~5× at the bound), and no response reached the strip fallback. The stress configuration is not an artificial weakening: the sweep shows that wordings fabricating at up to 100% arise from innocuous paraphrases, so a rule-free narrator is an empirically occurring operating point; the stress arm instantiates that point and shows the contract absorbing it. Precise, deterministic feedback is itself a strong mitigation; the verifier is what makes the feedback precise. Expected overhead is E[calls] ≈ 1.38 even in the stress configuration and ≈ 1.0 in production (zero violations observed under the strict prompt).

**Overhead.** The contract adds provider calls only on violation: one call per investigation plus at most two corrective retries; the verification itself is set membership and numeric comparison (Prop. 1). Worst case three calls; measured production overhead zero.

**Artifact.** The contract ships in the production narrator; 22 end-to-end contract tests (enum binding, value tolerance, misquote retry/strip, omission retry/flag, abstention exemption, stress configuration, both provider tool formats) run with mock clients, and the quality gates (`ruff`, `mypy --strict`) are clean.

---

## V. Discussion and Error Analysis

### A. Prompts Are Advisory, Schemas Are Enforcement

The sweep makes the principle quantitative: the same model on the same evidence fabricates between 1% and 100% of the time depending on innocuous wording, and the strict prompt's observed zeros (0/200 per task) still leave a 1.88% Wilson upper bound with no structural reason to hold under the next model update or the next paraphrase. The contract provides what no prompt can: a worst-case guarantee at any propensity, under any wording. *Prompts are advisory; schemas, with verification, are enforcement.*

### B. Detection Enables Correction

The stress data reveal a second-order benefit: the deterministic verifier does not merely block unfaithful responses; it also names the violation, and that named feedback cut the repeat-violation rate from 37.5% to 1.3% (Sec. IV-C). Vague "be accurate" guidance is weak (the *cautious* paraphrase fabricated at 45%); precise, machine-generated violation reports are strong. This suggests the right division of labor in narrator design: let the verifier, not the prompt author, do the correcting.

### C. Negative Results Are Boundaries, Not Failures

We measured no value misquotes and no critical omissions anywhere: the numbers sit verbatim in context (copying is easy) and the anchor is the most salient item. The channels may activate under longer contexts, derived statistics, or busier evidence; the contract closes them pre-emptively at the cost of a set lookup and a float comparison (Prop. 1). Verifying cheap invariants before they are observed failing is, we argue, the right default for evidence-closed narrators.

### D. Beyond ML: Where the Contract Transposes

The contract needs three invariants: a deterministic evidence producer, a finite enumerable set of observed entities and values, and a runtime enum bound to that set with independent verification. Wherever narration is evidence-closed, the same guarantee applies: a clinical-decision narrator could bind citations to the patient's actually-measured abnormal results and verify cited values against the measurements; an expert-system narrator could bind citations to fired rule IDs. We have not evaluated these domains; they mark the boundary of the claim. A related engineering lesson from shipping the system: when one capability ships through several interfaces (CLI and tool-server in our case), behavioral parity must be enforced through shared code paths and tested as an invariant; we hit and fixed exactly such a divergence.

### E. Limitations and Residual Risks

1. **Single provider.** All measurements use one commercial LLM; the rates should not be generalized; our own sweep shows they do not even generalize across paraphrases.
2. **Two tasks, one leak pattern.** Both tasks inject a log-of-target leak; diversity of leak patterns and evidence shapes remains future work. We did not log the composition of L1 violations, per-response claim counts, or per-arm abstention rates; an arm that drove the narrator into abstention would trivially score zero on all channels, though L1's high rates show the task elicits committed, citing behavior.
3. **Stress-arm confound.** The stress arm reuses the shipped contract's user message, which references a contract; its first-attempt propensity is therefore not exactly comparable to L1's (synthetic: 2% vs. 11.5%; real data: 37.5% vs. 43.5%). Catches are reported as demonstrations of the verifier firing, not as estimates of the bare rate.
4. **Channels not covered.** Miscalibrated confidence on correctly-cited, correctly-valued evidence; fabrication inside the free-text narration field (renderers surface the validated citation and claim sets as the actionable output); under enforcement pressure, fabrication may also migrate toward this unverified channel, and we did not measure such displacement; prompt-injection compliance. If the deterministic layer misses the true leak, the enum structurally bars the narrator from naming it, a precision-for-recall trade inherent to the binding.
5. **Tier A unmeasured in isolation**, and no head-to-head against a static-schema decoder where decoders are applicable, nor against validate-and-reask toolkits [18], [19]; on the tested endpoint Tier A acts, at most, as schema-as-prompt steering.

---

## VI. Conclusion

We defined *evidence-closed narration*, the class of tasks whose citable entities and verifiable quantities are finite and machine-enumerable at call time, and showed that for this class faithfulness is decidable (Prop. 1), soundness is enforceable by safe deletion while completeness is only detectable (Prop. 2), and fabrication is therefore preventable by construction. The two-tier evidence-bound contract realizing this ships in the production narrator of mlcompass and was measured live on a synthetic and a real-data leakage task: bare narrators fabricated at 11.5% and 43.5% (and anywhere from 1% to 100% across six rule-free paraphrases of the same instruction), while the contract held the user-facing rate at 0/200 on every channel of both tasks, and its deterministic verifier demonstrably caught 80 live violations in the stress configurations (76 on the real-data task), with named corrective feedback cutting repeat violations from 37.5% to 1.3%. We frame the contribution honestly: not a new decoding algorithm, but a decidable-class framing, a provider-independent contract with two formal observations, and first measurements, including the paraphrase sweep that, we believe, is the clearest quantitative argument yet that prompt-level faithfulness cannot be certified. Future work is a research program: a public narrator-fabrication benchmark across providers, tasks, and leak patterns; verified claims over derived and relational statistics; contract composition across multi-step agent chains, where each step's enum derives from the *verified* output of the previous one; and transposition to other evidence-closed narrators beyond ML tooling.

**Source code:** github.com/hakansabunis/mlcompass — **Package:** PyPI, MIT license. All prompts, seeds, the measurement harness, and raw live-run records are in the repository.

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

[18] Guardrails AI: Adding guardrails to large language models. [Online]. Available: https://github.com/guardrails-ai/guardrails

[19] T. Rebedea, R. Dinu, M. N. Sreedhar, C. Parisien, and J. Cohen, "NeMo Guardrails: A toolkit for controllable and safe LLM applications with programmable rails," in *Proc. EMNLP: System Demonstrations*, 2023, pp. 431–445.
