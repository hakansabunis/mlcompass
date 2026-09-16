# Figure prompts

The manuscript has one figure and seven tables. Two of those tables carry the
paper's best results and are unreadable as tables — a reviewer called the
paraphrase decoupling "the paper's best result, currently third in the
contribution list" and buried.

**Ask for matplotlib code, not images.** A journal needs vector output at a
fixed column width, and the code belongs in the replication package next to
the data it plots. A PNG from a chat window is not submittable and cannot be
regenerated when a number changes — and numbers here have changed four times.

Constraints to repeat in every prompt:

- Springer `sn-jnl` single-column text width is **4.9 inches**. Set
  `figsize=(4.9, h)`.
- Must be legible in **grayscale** — print copies exist. Distinguish by
  hatch, marker shape and line style, never by colour alone.
- Font size 8–9pt, matching the body text. No title inside the figure; the
  caption carries it.
- Save as **PDF** (`fig.savefig(..., bbox_inches="tight")`), not PNG.
- No gridlines unless they carry information, no 3-D, no drop shadows, no
  legend box border.

---

## Figure 2 — the seven-arm comparison (highest value)

Currently Table 2. A forest plot makes the 35 % → 0/200 gap visible instantly
and shows the intervals, which the table states but does not convey.

> Write a self-contained Python matplotlib script that produces a horizontal
> forest plot (dot-and-interval) and saves it as `fig_arms.pdf`.
>
> Figure width exactly 4.9 inches, height about 3.0. Font size 8. Must be
> legible printed in grayscale.
>
> Seven arms, each a point estimate with a Wilson 95 % confidence interval,
> plotted as a filled circle with a horizontal line through it. X axis is
> "user-facing entity-fabrication rate (%)", from 0 to 55.
>
> Data (label, estimate, lower, upper, k/N):
> - "A-L1  bare prompt", 42.0, 35.37, 48.93, "84/200"
> - "A-GR-STOCK  Guardrails, stock", 35.0, 28.73, 41.84, "70/200"
> - "A-STRICT-STATIC  strict, no enum", 4.0, 2.04, 7.69, "8/200"
> - "A-GR-OURS  Guardrails' loop, our Tier B", 0.0, 0.0, 1.88, "0/200"
> - "A-CONTRACT  Tier A + Tier B", 0.0, 0.0, 1.88, "0/200"
> - "A-STRESS  Tier B alone", 0.0, 0.0, 1.88, "0/200"
>
> Order them top to bottom exactly as listed. Draw a thin horizontal rule
> between the third and fourth entries to separate the arms that leave
> fabrication from the arms that do not.
>
> Print the k/N string right-aligned at the far right of each row, outside the
> plotting area.
>
> The two Guardrails rows are the comparison the figure exists for: same
> toolkit, same reask budget, same model, different validator. Mark those two
> labels in bold.
>
> No title. No legend. No vertical gridlines.

---

## Figure 3 — the paraphrase decoupling (the paper's best result)

Currently Table 5, where nobody can see it. The point is that the blended rate
and the invention count move *independently* and invert at the extremes.

> Write a self-contained Python matplotlib script producing a two-panel figure
> saved as `fig_sweep.pdf`. Width exactly 4.9 inches, height about 4.2, two
> panels stacked vertically sharing the x axis. Font size 8. Grayscale-legible.
>
> Six prompt variants on the x axis, in this order (it is ascending blended
> rate, and that ordering is the point):
> terse, baseline, expert, cautious, helpful, mechanical
>
> TOP PANEL — stacked bars, per 100 responses:
> - misfiled: 4, 36, 51, 68, 71, 71
> - invented: 7, 0, 5, 2, 5, 1
> Use a light fill for misfiled and a dark or hatched fill for invented, so the
> invented segment is visible even though it is small. Y axis "responses per
> 100". Y limit 0 to 80.
>
> BOTTOM PANEL — a line with markers, same x order:
> - invented: 7, 0, 5, 2, 5, 1
> Y axis "invented, per 100". Y limit 0 to 8.
>
> The figure must make one thing unmistakable: the top panel rises left to
> right, and the bottom panel does not follow it. Annotate the leftmost bottom
> point ("terse", 7) with "lowest blended rate, most inventions" and the
> rightmost ("mechanical", 1) with "highest blended rate, almost none", using
> small arrows and 7pt text.
>
> No title, no gridlines. Legend only in the top panel, no frame.

---

## Figure 4 — where the downstream effect actually comes from

Currently Table 7. A stacked bar shows in one glance that two rules carry the
whole fall and one rule never fires — which is the honest framing the paper
argues for.

> Write a self-contained Python matplotlib script producing a stacked bar chart
> saved as `fig_defects.pdf`. Width exactly 4.9 inches, height about 3.0. Font
> size 8. Grayscale-legible: use distinct hatches, not colours.
>
> Four bars on the x axis, in this order:
> control, control+revise, advise, advise+audit
>
> Stacked segments per bar (six rules, bottom to top), values:
> - leak_duplicate_rows:        24, 24, 14, 8
> - no_validation:               7, 10,  9, 0
> - target_in_features:          2,  0,  0, 0
> - no_seed:                     0,  1,  0, 0
> - leak_fit_before_split:       0,  0,  1, 0
> - wrong_metric_for_imbalance:  0,  0,  0, 0
>
> Bar totals are 33, 35, 24, 8. Print each total above its bar in bold.
>
> Y axis "flagged defects, summed over the arm's runs", limit 0 to 40.
>
> `wrong_metric_for_imbalance` is zero in every bar. Keep it in the legend and
> append "(never fires)" to its label — the fact that a sixth of the instrument
> cannot fire on this data is a finding, not an omission.
>
> Legend to the right of the axes, no frame, one column.
>
> No title, no gridlines.

---

## Figure 1 — the architecture (redesign, lower priority)

The existing `contract_flow.pdf` is correct and plain. It is generated by
`../make_contract_figure.py`, which is already committed, so **regenerate it
from that script rather than redrawing from scratch** if all you want is
cosmetic improvement. Redraw only if you want a genuinely different layout.

If redrawing, the content it must carry — and the current version gets all of
this right, so do not lose any of it:

> A vertical flow with five stages and a trust boundary:
>
> 1. Deterministic evidence producer (pure Python, offline)
> 2. Evidence E — entities A_E, measured values V_E, ranked anchor a*
> 3. LLM narrator, any provider — answers through a `submit_investigation` tool
> 4. Tier B: deterministic verifier in plain code, three checks —
>    (1) entities subset of A_E, (2) |v − V_E| <= 0.005, (3) a* addressed
> 5. User
>
> Two labelled arrows that are the whole point:
> - Between stages 2 and 3, going down: "TIER A: enum(A_E) bound into the tool
>   schema at call time"
> - From stage 4 back up to stage 3: "violation: retry (<=2), naming the
>   offending items"
> - From stage 4 down to stage 5, after the budget: "strip unsound / flag
>   omission"
>
> A dashed line must separate stages 1–2 and 4–5 (deterministic, trusted) from
> stage 3 (the narrator, untrusted). The current figure omits this line and the
> caption promises it.
>
> The user box must read: "validated citations + verified claims, then free
> text marked NOT verified". Do not shorten this to "verified output only" —
> that overstates the guarantee and an earlier version of the figure did.
>
> Width 4.9 inches, grayscale, font size 8.

---

## After the code comes back

```
cd paper/emse_latex
python fig_arms.py && python fig_sweep.py && python fig_defects.py
tectonic -X compile main.tex --outdir .
```

Then tell me and I will wire them into the manuscript with captions and
`\ref`s, and decide which tables to drop — Figure 2 replaces Table 2, Figure 4
replaces Table 7, and Figure 3 sits beside Table 5 rather than replacing it,
because the table carries the confidence intervals.

Commit the generating scripts, not just the PDFs. Every other number in this
paper is reproducible from the repository and the figures should not be the
exception.
