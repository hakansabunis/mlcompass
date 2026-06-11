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

## ✅ All tables CERTIFIED — full measurement battery completed 2026-06-12
The paper's tables carry **measured** figures (DeepSeek OpenAI-compatible
endpoint, `deepseek-chat`, Wilson 95% CIs):

- **Table I** (entity-fab, N=200/cell): synthetic L1 11.5% → L2/L3/STRESS 0/200;
  real-data (Insurance) L1 43.5% → L2/L3/STRESS 0/200. Value-fab & omission:
  0/200 in every cell of both tasks.
- **Table II** (paraphrase sweep, 6 rule-free variants × N=100): **1% → 100%**
  (terse 1%, baseline 7%, cautious 45%, expert 90%, helpful 93%, mechanical 100%).
- **Tier B catches (STRESS)**: real-data 75/200 responses, 76 total catches,
  0 user-facing; repeat-violation rate 37.5% → 1.3% after one corrective
  message (~28×). Synthetic: 4 catches, 0 user-facing.

Full records: `paper/ablation_live_deepseek_2026-06-12_battery.md` (+ the
2026-06-10/-11 historical records). To re-run:

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
