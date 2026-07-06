# Verified reference pool — ESWA manuscript expansion (7 Jul 2026)

Every entry below was verified by an agent that FETCHED the listed URL and
confirmed the exact title/authors (zero-fabricated-reference discipline;
same standard as the 19 INISTA references). `relevance` says where in the
manuscript the reference belongs. Compiled from workflow wwypwpj9g.

## Hallucination detection and faithfulness evaluation methods (Sec 2 related work + LLM-judge methodology for the displacement experiment)

- **[selfcheckgpt]** P. Manakul, A. Liusie, and M. J. F. Gales, "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models," Proc. EMNLP, 2023, pp. 9004-9017. 10.18653/v1/2023.emnlp-main.557
  - verified: https://aclanthology.org/2023.emnlp-main.557/
  - relevance: Sec 2 anchor for post-hoc sampling-based hallucination detection, against which we contrast our schema's a-priori prevention of fabricated pipeline evidence.
- **[factscore]** S. Min et al., "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation," Proc. EMNLP, 2023, pp. 12076-12100. 10.18653/v1/2023.emnlp-main.741
  - verified: https://aclanthology.org/2023.emnlp-main.741/
  - relevance: Sec 2 and evaluation design: its atomic-fact decomposition against a knowledge source directly parallels our claim-level binding of narrative statements to deterministic evidence records.
- **[factool]** I.-C. Chern et al., "FacTool: Factuality Detection in Generative AI -- A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios," arXiv preprint, 2023. arXiv:2307.13528
  - verified: https://arxiv.org/abs/2307.13528
  - relevance: Sec 2 related work on tool-augmented factuality checking, the closest detection-side analogue to our tool-grounded evidence channel, but applied after generation rather than as a runtime constraint.
- **[halueval]** J. Li, X. Cheng, X. Zhao, J.-Y. Nie, and J.-R. Wen, "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models," Proc. EMNLP, 2023, pp. 6449-6464. 10.18653/v1/2023.emnlp-main.397
  - verified: https://aclanthology.org/2023.emnlp-main.397/
  - relevance: Sec 2 benchmark evidence that LLMs hallucinate at scale even on recognition tasks, motivating why pipeline-evidence narration needs structural rather than benchmark-driven safeguards.
- **[truthfulqa]** S. Lin, J. Hilton, and O. Evans, "TruthfulQA: Measuring How Models Mimic Human Falsehoods," Proc. ACL, 2022, pp. 3214-3252. 10.18653/v1/2022.acl-long.229
  - verified: https://aclanthology.org/2022.acl-long.229/
  - relevance: Sec 2 foundational truthfulness benchmark showing that scaling alone does not yield truthful generation, supporting our argument that fabrication must be prevented by the runtime contract.
- **[summac]** P. Laban, T. Schnabel, P. N. Bennett, and M. A. Hearst, "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization," Trans. Assoc. Comput. Linguistics, vol. 10, 2022, pp. 163-177. 10.1162/tacl_a_00453
  - verified: https://aclanthology.org/2022.tacl-1.10/
  - relevance: Sec 2 summarization-faithfulness line (NLI-based consistency detection), the task family most analogous to narrating a fixed evidence source, and a contrast point for our schema-level grounding.
- **[mtbench]** L. Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," Proc. NeurIPS Datasets and Benchmarks Track, 2023. arXiv:2306.05685
  - verified: https://arxiv.org/abs/2306.05685
  - relevance: Methodological basis for the LLM-as-judge protocol in our displacement experiment, establishing judge-human agreement levels and known judge biases we control for.
- **[llm_unfair_eval]** P. Wang et al., "Large Language Models are not Fair Evaluators," Proc. ACL, 2024, pp. 9440-9450. 10.18653/v1/2024.acl-long.511
  - verified: https://aclanthology.org/2024.acl-long.511/
  - relevance: Cited in the displacement-experiment methodology as the key critique of LLM-judge validity (positional bias), justifying our calibration and ordering controls.

## Constrained decoding / grammar-guided generation / structured outputs (beyond Outlines, LMQL, Geng GCD 2023, Guidance, OpenAI structured outputs)

- **[xgrammar]** Y. Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models," Proc. MLSys, 2025. arXiv:2411.15100
  - verified: https://arxiv.org/abs/2411.15100
  - relevance: Related work on constrained-decoding engines: state-of-the-art context-free-grammar token masking shows schema enforcement is now near-zero-overhead at serving time, strengthening the feasibility argument for our evidence-bound runtime schema.
- **[sglang]** L. Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," Adv. Neural Inf. Process. Syst. 37 (NeurIPS), 2024. arXiv:2312.07104
  - verified: https://papers.nips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html
  - relevance: Related work on structured LM programs: its compressed finite-state-machine decoding for JSON outputs is the serving-stack counterpart to our schema-level binding of narration to deterministic pipeline evidence.
- **[domino]** L. Beurer-Kellner, M. Fischer, and M. Vechev, "Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation," Proc. ICML, PMLR vol. 235, 2024, pp. 3658-3673. arXiv:2403.06988
  - verified: https://proceedings.mlr.press/v235/beurer-kellner24a.html
  - relevance: Related work on constrained decoding: DOMINO's minimally-invasive, subword-aligned constraining addresses the accuracy distortions naive token masking causes, which we cite when arguing that our schema constrains content (evidence bindings) rather than distorting the token distribution.
- **[gad]** K. Park, J. Wang, T. Berg-Kirkpatrick, N. Polikarpova, and L. D'Antoni, "Grammar-Aligned Decoding," Adv. Neural Inf. Process. Syst. 37 (NeurIPS), 2024. arXiv:2405.21047
  - verified: https://arxiv.org/abs/2405.21047
  - relevance: Related work on the theory of grammar-constrained decoding: proves that naive GCD skews the LM's distribution and proposes distribution-faithful sampling, motivating why our design validates evidence bindings post-hoc at the schema layer instead of relying solely on mask-based enforcement.
- **[automata_constraints]** T. Koo, F. Liu, and L. He, "Automata-based constraints for language model decoding," Proc. Conf. on Language Modeling (COLM), 2024. arXiv:2407.08103
  - verified: https://arxiv.org/abs/2407.08103
  - relevance: Related work on formal foundations: casts regex/grammar constraints as automata operating over detokenized text, providing the formal-language grounding we cite for compiling our evidence-bound schema into decodable constraints.
- **[jsonschemabench]** S. Geng et al., "JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models," arXiv preprint, 2025. arXiv:2501.10868
  - verified: https://arxiv.org/abs/2501.10868
  - relevance: Evaluation section: a 10K real-world JSON-Schema benchmark of six constrained-decoding frameworks (efficiency, coverage, quality) that we cite as the closest existing evaluation methodology for schema-constrained generation, against which our evidence-faithfulness evaluation is differentiated.
- **[speak_freely]** Z. R. Tam, C.-K. Wu, Y.-L. Tsai, C.-Y. Lin, H.-y. Lee, and Y.-N. Chen, "Let Me Speak Freely? A Study On The Impact Of Format Restrictions On Large Language Model Performance," Proc. EMNLP Industry Track, 2024, pp. 1218-1236. 10.18653/v1/2024.emnlp-industry.91
  - verified: https://aclanthology.org/2024.emnlp-industry.91/
  - relevance: Limitations/discussion: documents reasoning degradation under format restrictions, the key counter-argument we address when claiming that binding narration to deterministic evidence does not sacrifice explanation quality.

## tool-use reliability, function-calling correctness, agent safety/guarding

- **[gorilla]** S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez, "Gorilla: Large Language Model Connected with Massive APIs," Advances in Neural Information Processing Systems 37 (NeurIPS), 2024. arXiv:2305.15334
  - verified: https://proceedings.neurips.cc/paper_files/paper/2024/hash/e4c61f578ff07830f5c37378dd3ecb0d-Abstract-Conference.html
  - relevance: Related-work anchor for tool-use reliability: documents hallucinated API calls and shows retrieval-grounding mitigates them, the tool-invocation analogue of our evidence-bound narration.
- **[toolllm]** Y. Qin et al., "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs," Proc. ICLR, 2024. arXiv:2307.16789
  - verified: https://arxiv.org/abs/2307.16789
  - relevance: Related work on function-calling at scale (introduces the ToolBench dataset and DFSDT evaluation), establishing tool-call correctness as an open reliability problem our schema addresses on the narration side.
- **[apibank]** M. Li, Y. Zhao, B. Yu, F. Song, H. Li, H. Yu, Z. Li, F. Huang, and Y. Li, "API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs," Proc. EMNLP, 2023, pp. 3102-3116. arXiv:2304.08244
  - verified: https://aclanthology.org/2023.emnlp-main.187/
  - relevance: Benchmark for whether/how/which API an LLM should call; supports our distinction that correct invocation does not guarantee faithful narration of returned evidence.
- **[bfcl]** S. G. Patil, H. Mao, F. Yan, C. C.-J. Ji, V. Suresh, I. Stoica, and J. E. Gonzalez, "The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models," Proc. ICML, PMLR vol. 267, 2025, pp. 48371-48392. PMLR 267:48371-48392 (proceedings.mlr.press/v267/patil25a.html)
  - verified: https://proceedings.mlr.press/v267/patil25a.html
  - relevance: Closest existing measurement of function-call schema conformance (AST-based validity checking, abstention, multi-step agentic settings); cite where we argue structural validity of calls is measured but faithfulness of evidence narration is not.
- **[taubench]** S. Yao, N. Shinn, P. Razavi, and K. Narasimhan, "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains," arXiv preprint, 2024. arXiv:2406.12045
  - verified: https://arxiv.org/abs/2406.12045
  - relevance: Agent-reliability evidence for our motivation section: state-of-the-art function-calling agents pass under 50% of rule-bound tasks and are inconsistent across trials (pass^k), motivating deterministic evidence binding over free narration.
- **[toolemu]** Y. Ruan, H. Dong, A. Wang, S. Pitis, Y. Zhou, J. Ba, Y. Dubois, C. J. Maddison, and T. Hashimoto, "Identifying the Risks of LM Agents with an LM-Emulated Sandbox," Proc. ICLR, 2024. arXiv:2309.15817
  - verified: https://arxiv.org/abs/2309.15817
  - relevance: Agent-safety related work: sandboxed emulation finds even the safest LM agent fails 23.9% of the time with high-stakes tools, complementing our runtime guarding of what agents report about tool results.
- **[agentharm]** M. Andriushchenko et al., "AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents," Proc. ICLR, 2025. arXiv:2410.09024
  - verified: https://arxiv.org/abs/2410.09024
  - relevance: Situates our guard schema within the agent-safety benchmark literature, showing tool-using agents comply with unsafe requests and need runtime-level guarding beyond model alignment.
- **[guardagent]** Z. Xiang, L. Zheng, Y. Li, J. Hong, Q. Li, H. Xie, J. Zhang, Z. Xiong, C. Xie, C. Yang, D. Song, and B. Li, "GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning," Proc. ICML, 2025. arXiv:2406.09187
  - verified: https://arxiv.org/abs/2406.09187
  - relevance: Closest guarding architecture to contrast with ours: an LLM guard agent enforces guard requests over a target agent's actions, whereas our schema enforces evidence-bound narration deterministically at runtime.

## data leakage + ML reproducibility

- **[kapoor2023leakage]** S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," Patterns, vol. 4, no. 9, art. 100804, 2023. 10.1016/j.patter.2023.100804
  - verified: https://pubmed.ncbi.nlm.nih.gov/37720327/
  - relevance: Anchor citation for our Related Work / problem statement: it documents leakage across 294 papers in 17 fields and gives the 8-type leakage taxonomy our data-leakage investigation task narrates against.
- **[neunhoeffer2019crossval]** M. Neunhoeffer and S. Sternberg, "How Cross-Validation Can Go Wrong and What to Do About It," Political Analysis, vol. 27, no. 1, pp. 101-106, 2019. 10.1017/pan.2018.39
  - verified: https://www.cambridge.org/core/product/identifier/S1047198718000396/type/journal_article
  - relevance: Case study on how CV done wrong (hyperparameter tuning inside the resampling loop) inflates accuracy; supports our two-real-case-studies section on leakage-driven over-optimism.
- **[muchlinski2016rf]** D. Muchlinski, D. Siroky, J. He, and M. Kocher, "Comparing Random Forest with Logistic Regression for Predicting Class-Imbalanced Civil War Onset Data," Political Analysis, vol. 24, no. 1, pp. 87-103, 2016. 10.1093/pan/mpv024
  - verified: https://www.cambridge.org/core/product/identifier/S1047198700012055/type/journal_article
  - relevance: The random-forest civil-war-onset paper whose headline result was later shown to stem from leakage; one of the concrete real-world case studies our runtime schema must be able to narrate faithfully.
- **[johnson1996bodyfat]** R. W. Johnson, "Fitting Percentage of Body Fat to Simple Body Measurements," Journal of Statistics Education, vol. 4, no. 1, 1996. 10.1080/10691898.1996.11910505
  - verified: https://jse.amstat.org/v4n1/datasets.johnson.html
  - relevance: Source of the classic body-fat regression dataset used as a leakage-teaching example (density-derived target vs. tape measurements); grounds our task's illustrative leakage scenario.
- **[tavallaee2009kdd99]** M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, "A Detailed Analysis of the KDD CUP 99 Data Set," Proc. IEEE Symp. Computational Intelligence for Security and Defense Applications (CISDA), 2009, pp. 1-6. 10.1109/CISDA.2009.5356528
  - verified: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/CISDA.2009.5356528
  - relevance: Canonical dissection of the KDD'99 intrusion-detection benchmark whose train/test redundancy inflates reported accuracy; cited as a benchmark-level leakage / reproducibility case study.
- **[roberts2021covid]** M. Roberts et al., "Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans," Nature Machine Intelligence, vol. 3, pp. 199-217, 2021. 10.1038/s42256-021-00307-0
  - verified: https://www.nature.com/articles/s42256-021-00307-0?error=cookies_not_supported&code=a92e7a25-61de-4015-909c-d8bf65806d42
  - relevance: Medical-ML leakage case (train/test contamination via overlapping public datasets, none of 62 models clinically usable); supports the medical-ML leakage extension the task requests.
- **[whalen2022genomics]** S. Whalen, J. Schreiber, W. S. Noble, and K. S. Pollard, "Navigating the pitfalls of applying machine learning in genomics," Nature Reviews Genetics, vol. 23, no. 3, pp. 169-181, 2022. 10.1038/s41576-021-00434-9
  - verified: https://pubmed.ncbi.nlm.nih.gov/34837041/
  - relevance: Domain-specific evidence that data structure induces leakage-like evaluation bias in genomics ML; broadens our reproducibility-crisis framing beyond tabular/vision cases.
- **[yang2022notebooks]** C. Yang, R. A. Brower-Sinning, G. A. Lewis, and C. Kastner, "Data Leakage in Notebooks: Static Detection and Better Processes," Proc. 37th IEEE/ACM Int. Conf. Automated Software Engineering (ASE), 2022, pp. 1-12. 10.1145/3551349.3556918
  - verified: https://arxiv.org/abs/2209.03345
  - relevance: Closest prior leakage-detection tooling work (static analysis of 100k+ notebooks for overlap/multi-test/preprocessing leakage); positions our evidence-bound runtime schema against static, non-narrating detectors.

## Benchmark-construction methodology and statistical practice

- **[wilson1927]** E. B. Wilson, "Probable Inference, the Law of Succession, and Statistical Inference," Journal of the American Statistical Association, vol. 22, no. 158, 1927, pp. 209-212. 10.1080/01621459.1927.10502953
  - verified: https://api.crossref.org/works/10.1080/01621459.1927.10502953
  - relevance: Cited in our evaluation section as the source of the Wilson score interval we use to report binomial confidence bounds on fabrication rates measured over small per-condition sample counts.
- **[vanmiltenburg2021prereg]** E. van Miltenburg, C. van der Lee, and E. Krahmer, "Preregistering NLP research," Proc. NAACL-HLT, 2021, pp. 613-623. 10.18653/v1/2021.naacl-main.51
  - verified: https://aclanthology.org/2021.naacl-main.51/
  - relevance: Supports our methodology section, where the benchmark's analysis plan (hypotheses, metrics, exclusion rules) is preregistered before any runs to prevent post-hoc flexibility in how fabrication is scored.
- **[helm2023]** P. Liang et al., "Holistic Evaluation of Language Models," Transactions on Machine Learning Research (TMLR), 2023. arXiv:2211.09110
  - verified: https://arxiv.org/abs/2211.09110
  - relevance: Belongs in related work on evaluation methodology as the exemplar of multi-metric, multi-scenario benchmark design that our fabrication benchmark's condition-by-model matrix follows.
- **[dodge2019showyourwork]** J. Dodge, S. Gururangan, D. Card, R. Schwartz, and N. A. Smith, "Show Your Work: Improved Reporting of Experimental Results," Proc. EMNLP-IJCNLP, 2019, pp. 2185-2194. 10.18653/v1/D19-1224
  - verified: https://aclanthology.org/D19-1224/
  - relevance: Grounds our reporting practice (compute budget, number of trials, and full per-run results rather than best-case numbers) in the experimental-reporting standard this paper established for NLP.
- **[bigbench2023]** A. Srivastava et al., "Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models," Transactions on Machine Learning Research (TMLR), 2023. arXiv:2206.04615
  - verified: https://arxiv.org/abs/2206.04615
  - relevance: Cited in the benchmark-construction discussion as the precedent for assembling a broad task battery with documented construction methodology, which our injector-by-dataset fabrication matrix scales down from.
- **[gebru2021datasheets]** T. Gebru, J. Morgenstern, B. Vecchione, J. W. Vaughan, H. Wallach, H. Daumé III, and K. Crawford, "Datasheets for Datasets," Communications of the ACM, 2021. arXiv:1803.09010
  - verified: https://arxiv.org/abs/1803.09010
  - relevance: Cited where we document the provenance, composition, and intended use of each benchmark dataset, following the datasheet standard for dataset documentation.
- **[sainz2023contamination]** O. Sainz, J. Campos, I. García-Ferrero, J. Etxaniz, O. Lopez de Lacalle, and E. Agirre, "NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark," Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 10776-10787. 10.18653/v1/2023.findings-emnlp.722
  - verified: https://aclanthology.org/2023.findings-emnlp.722/
  - relevance: Motivates the benchmark-validity subsection: because evaluated LLMs may have seen public datasets during training, we treat contamination as a threat to validity when constructing the fabrication benchmark.
- **[golchin2024timetravel]** S. Golchin and M. Surdeanu, "Time Travel in LLMs: Tracing Data Contamination in Large Language Models," Proc. ICLR, 2024 (Spotlight). arXiv:2308.08493
  - verified: https://arxiv.org/abs/2308.08493
  - relevance: Complements Sainz et al. in the threats-to-validity discussion by providing a concrete detection method we reference for checking whether benchmark instances were memorized by the models under evaluation.

**Total: 39 verified references** (+19 existing INISTA refs).
