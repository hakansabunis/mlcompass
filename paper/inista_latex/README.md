# INISTA 2026 — LaTeX submission bundle

This folder is a self-contained IEEE-conference LaTeX paper, ready for Overleaf.

## Files
- `main.tex` — the full paper (IEEEtran, `conference` mode). Bibliography is
  **embedded** (`thebibliography`), so it compiles with one click — no BibTeX
  step needed.
- `architecture.png` — Figure 1 (converted from `architecture.webp`).

## Compile on Overleaf (recommended)
1. Go to <https://overleaf.com> → **New Project → Upload Project**.
2. Zip this `inista_latex/` folder and upload it (or drag `main.tex` +
   `architecture.png` into a blank project).
3. Set the compiler to **pdfLaTeX** (Menu → Compiler). Click **Recompile**.
4. You should get a ~6-page two-column PDF.

## Compile locally (if you have a TeX distribution)
```
pdflatex main.tex
pdflatex main.tex      # run twice so \ref / \label resolve
```

## ✅ Table I is CERTIFIED — three-channel live run completed 2026-06-11
Table I carries **measured** figures from the three-channel live run (DeepSeek
OpenAI-compatible endpoint, `deepseek-chat`, N = 200 per layer, default seed):

| Layer | Entity-fab | Value-fab | Omission |
|---|---|---|---|
| L1 bare prompt | 15.0% (30/200) | 0.0% (0/200) | 0.0% (0/200) |
| L2 + strict prompt | 0.0% (0/200) | 0.0% (0/200) | 0.0% (0/200) |
| L3 + evidence-bound | 0.0% (0/200) | 0.0% (0/200) | 0.0% (0/200) |

Plus the prompt-sensitivity datum: the 2026-06-10 run's bare prompt (no
structured-claims request) measured **56.5%** entity fabrication — a near-4×
swing from one prompt sentence, reported in the paper as a finding.

Full records: `paper/ablation_live_deepseek_2026-06-11_three_channel.md` and
`paper/ablation_live_deepseek_2026-06-10.md`. To re-run:

```powershell
cd C:\Users\SABUNIS\OneDrive\Desktop\ml-copilot
$env:PYTHONPATH = "C:\Users\SABUNIS\OneDrive\Desktop\agentlite\src"
$env:DEEPSEEK_API_KEY = "sk-..."
python scripts/reproduce_hallucination_ablation.py --mode live --provider deepseek --n 200
```

The script builds the synthetic frame, runs the real `detect_leakage`, and the
Layer-3 column calls the **shipped** `investigate_leakage_bound` — so the table
reflects the deployed mechanism.

## Page-limit check
INISTA allows **6 pages including references**. After compiling, confirm the PDF
is ≤ 6 pages. If it runs slightly over, the usual trims are: shorten the
Related Work paragraphs, or move Limitations into a tighter single paragraph.

## Submission
- Venue: IEEE INISTA 2026, main track (suggested topic: **Agentic AI**).
- Submit at: <https://easychair.org/conferences/?conf=inista2026>
- Deadline: 15 June 2026 (AoE).
