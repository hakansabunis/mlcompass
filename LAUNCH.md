# v0.1.0 Launch Checklist

This file is the single source of truth for shipping mlcompass's first
public release. Tick items off as you go; everything you need (post
drafts, recording script, timing) is below.

> **Status as of this commit:**
> Build, tests, docs, demo datasets, and post drafts are ready.
> External steps (PyPI upload, demo recording, posting) require you.

---

## 0. Final pre-flight (do once, the day of launch)

- [ ] `git pull --rebase` to make sure local is fresh
- [ ] `.venv/Scripts/python -X utf8 -m pytest tests/` → 66 passing
- [ ] `.venv/Scripts/python -X utf8 -m build` → `dist/mlcompass-0.1.0-py3-none-any.whl` and `.tar.gz`
- [ ] `.venv/Scripts/python -X utf8 -m twine check dist/*` → both PASSED
- [ ] Skim `README.md`, `CHANGELOG.md`, `ARCHITECTURE.md` for typos
- [ ] Confirm `mlcompass --version` prints `0.1.0`
- [ ] Confirm `mlcompass advise examples/customer_churn.csv --no-llm` produces a clean output
- [ ] Tag the release: `git tag -a v0.1.0 -m "mlcompass 0.1.0 — advise mode"`
- [ ] Push the tag: `git push origin v0.1.0`

---

## 1. Record the demo (~30 min)

The hero asset for every post is a 30-second GIF of `mlcompass advise`
running on a real CSV. Use one of:

- **VHS** (recommended, deterministic): https://github.com/charmbracelet/vhs
- **asciinema + agg**: https://asciinema.org

### VHS script (save as `demo.tape` in repo root)

```vhs
Output demo.gif
Set FontSize 14
Set Width 1100
Set Height 700
Set Padding 20
Set Theme "Catppuccin Mocha"

Type "mlcompass init churn-demo"
Enter
Sleep 1s

Type "mlcompass advise examples/customer_churn.csv --no-llm"
Enter
Sleep 5s

# Optional: with real API key, drop --no-llm for the full advisor output
```

Run:
```bash
vhs demo.tape
```

Commit `demo.gif` to the repo root, embed in README under the hero line.

### Quick alternative (no VHS install)

Use Windows Terminal + ScreenToGif: record a single advise run for ~30s,
crop, export at 800px wide. Same end result.

---

## 2. PyPI publish (~10 min, requires PyPI account)

### One-time setup (if you don't have a token yet)

1. Go to https://pypi.org/manage/account/token/
2. Create an API token, scope = "entire account" (you can scope it down
   to `mlcompass` after the first upload)
3. Store the token in `~/.pypirc`:

   ```ini
   [pypi]
     username = __token__
     password = pypi-AgEIcHlwaS5vcmcCJG...
   ```

### Upload

```bash
.venv/Scripts/python.exe -m twine upload dist/*
```

Or for a dry run, use TestPyPI first:

```bash
.venv/Scripts/python.exe -m twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ mlcompass
```

### Verify

```bash
pip install --upgrade mlcompass
mlcompass --version   # → 0.1.0
mlcompass init smoke-test
```

---

## 3. GitHub release (~5 min)

```bash
gh release create v0.1.0 \
  --title "mlcompass 0.1.0 — advise mode" \
  --notes-file CHANGELOG.md \
  dist/mlcompass-0.1.0-py3-none-any.whl \
  dist/mlcompass-0.1.0.tar.gz
```

Or via the GitHub UI: New release → choose tag `v0.1.0` → paste the
"## [0.1.0]" section from CHANGELOG.md.

---

## 4. Show HN post (highest-leverage single action)

### Timing

Submit at **TR 16:00 on Sunday or Monday** = **SF 06:00**. This is the
window when HN's front page traffic peaks for west-coast US morning.

### Submission

- Title: `Show HN: mlcompass – an LLM agent that recommends models, features, and pitfalls from your dataset`
- URL: `https://github.com/hakansabunis/mlcompass`
- Text: paste the draft below

### Show HN text draft

```
Hi HN,

I built mlcompass, a small CLI agent that sits next to you through the
full ML pipeline. v0.1 ships the first stage — `advise`:

  $ mlcompass init churn-project
  $ mlcompass advise data.csv --target churn

It analyzes your CSV (schema, missingness, outliers, target detection,
task inference, class balance), then asks Claude to recommend top model
families, feature engineering hints, and pitfalls — as a structured JSON
that gets rendered in your terminal with rich.

Why I'm building it: I'm working on a Capstone flood-prediction model
and kept making the same setup mistakes across runs — wrong loss for
imbalanced data, missing seed, weird splits. Existing tools log;
nothing actually advises. So I started writing one.

Design notes:

- Built on agentlite (https://github.com/hakansabunis/agentlite), a
  2K-line Claude agent library I also wrote, so prompt caching, the
  permission system, and the sub-agent factory are first-class.
- The deterministic dataset analyzer is pure pandas, with the LLM only
  reasoning over its output — keeps `advise` fast and cheap.
- A persistent `.mlcompass/` directory (similar in spirit to .git/)
  carries context across commands so the planned `audit`, `watch`,
  `evaluate`, and `deploy` modes know what you already chose and why.

Roadmap is in CHANGELOG.md — Faz 2 adds the training watcher (the part
that originally motivated the project), Faz 3 adds post-training
evaluation, Faz 4 adds deployment checks.

Three small synthetic example datasets live in examples/ if you want to
try it without using your own data.

Tests: 66 passing. Apache 2.0. Feedback very welcome — especially on
the advisor prompt and the rendering, since those are the things I
expect to iterate on the most.

[demo GIF]
```

> Adjust the wording so it sounds like you — HN readers spot AI-written
> launch posts in 2 seconds and downvote them. Keep one or two "I"
> sentences that are unmistakably yours.

---

## 5. Reddit posts (within 24h after Show HN)

### r/MachineLearning (better for technical reception)

- Title: `[P] mlcompass — an LLM agent that recommends models + feature engineering from your dataset`
- Use [P] flair for "Project"
- Body: a tightened version of the Show HN text above, no marketing
  phrasing. Lead with what it does, follow with one screenshot of the
  advise output and the README link.

### r/LocalLLaMA (audience overlaps with Claude API users)

- Title: `Open-source ML pipeline assistant built on Claude (agentlite-py)`
- Body: lean into the agentlite + Claude angle; mention prompt caching
  and the permission system.

### r/Python

- Title: `mlcompass 0.1 — a CLI agent for tabular ML practitioners`
- Body: shorter, focus on the CLI ergonomics and the demo gif.

---

## 6. LinkedIn (Turkish-language post for local reach)

```
🚀 mlcompass v0.1 yayında.

ML projelerinde sürekli aynı hataları yaptığımı fark ettim:
yanlış metric, eksik seed, kötü split. Mevcut araçlar (W&B,
TensorBoard) sadece log tutuyor — gerçek öneri vermiyorlar.

Bu yüzden mlcompass'u yazdım: CSV'yi ver, hangi modelleri
denemen gerektiğini, hangi feature engineering'in fayda
sağlayacağını, hangi pitfall'lardan kaçınman gerektiğini
yapılandırılmış olarak söylesin.

Komut çok basit:

    pip install mlcompass
    mlcompass init my-project
    mlcompass advise data.csv

Altyapı olarak kendi yazdığım agentlite (Claude için minimal
agent kütüphanesi) üzerinde çalışıyor. Açık kaynak (MIT):

https://github.com/hakansabunis/mlcompass

Sonraki fazlar: eğitim sırasında canlı izleme + plateau/NaN
tespiti (v0.2), sonrası için değerlendirme + deployment
kontrolü.

#machinelearning #python #opensource #ai
```

---

## 7. Blog post (after launch, drives long-tail traffic)

Two drafts in `docs/blog/`:

- `2026-05-29-mlcompass-launch-en.md` — English
- `2026-05-29-mlcompass-launch-tr.md` — Turkish

Publish on dev.to or Hashnode (English) and Medium (Turkish) once Show
HN cools down.

---

## 8. Post-launch (first 48h)

- [ ] Respond to every GitHub issue in <24h
- [ ] Respond to every Show HN comment (HN values author engagement)
- [ ] If a real bug is reported, ship a 0.1.1 with the fix within 24h
- [ ] Track stars/installs daily for the first week
- [ ] Save screenshots of any kind feedback for capstone defense

---

## 9. Capstone defense angle

When you defend FloodGuard, here's the 3-sentence story:

> "I noticed that across my flood-prediction experiments I kept
> repeating the same setup mistakes — wrong metric for imbalanced
> data, missing seed, weird splits — so I built mlcompass, an LLM
> agent that flags these before they cost a training run. It runs
> on a 2K-line Claude agent library I also wrote (agentlite), and
> shipped publicly with X stars / Y installs as of today."

This works whether mlcompass gets 50 stars or 5000.
