"""Figure 2: the eight-arm comparison as a forest plot.

Numbers restated from Table 2 of the manuscript, which is itself produced by
scripts/reproduce_hallucination_ablation.py.  The degenerate
A-STATIC-ENUM-STALE arm is deliberately absent: it carries no comparison, and
plotting a zero next to three earned zeros would invite the wrong reading.
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})

# label, mechanism, estimate, lo, hi, k/N, bold
ARMS = [
    ("A-L1",            "no enforcement",                42.0, 35.4, 48.9, "84/200", False),
    ("A-GR-STOCK",      "Guardrails AI, stock",          35.0, 28.7, 41.8, "70/200", True),
    ("A-STRICT-STATIC", "provider strict, no enum",       4.0,  2.0,  7.7,  "8/200", False),
    ("A-GR-CHOICES",    "Guardrails' own choice check",   0.0,  0.0,  1.9,  "0/200", True),
    ("A-GR-OURS",       "Guardrails' loop, our Tier B",   0.0,  0.0,  1.9,  "0/200", True),
    ("A-CONTRACT",      "Tier A + Tier B",                0.0,  0.0,  1.9,  "0/200", False),
    ("A-STRESS",        "Tier B alone, no enum",          0.0,  0.0,  1.9,  "0/200", False),
]
SPLIT_AFTER = 3  # noqa: E262  # the rule separating "fabrication reaches the user" from "it does not"

fig, ax = plt.subplots(figsize=(4.9, 3.3))
ys = list(range(len(ARMS)))[::-1]

for y, (arm, mech, est, lo, hi, kn, bold) in zip(ys, ARMS):
    ax.plot([lo, hi], [y, y], color="black", lw=1.0, solid_capstyle="butt", zorder=2)
    for x in (lo, hi):
        ax.plot([x, x], [y - 0.13, y + 0.13], color="black", lw=0.8, zorder=2)
    ax.plot([est], [y], "o", ms=4.2, mfc="black" if bold else "white",
            mec="black", mew=1.0, zorder=3)
    ax.text(57.5, y, kn, ha="right", va="center", fontsize=7.5,
            family="DejaVu Sans Mono")

ax.axhline(len(ARMS) - SPLIT_AFTER - 0.5, color="black", lw=0.5, ls=(0, (4, 3)))

ax.set_yticks(ys)
ax.set_yticklabels([f"{a}\n{m}" for a, m, *_ in ARMS], fontsize=7.5)
for tick, (*_, bold) in zip(ax.get_yticklabels(), ARMS):
    if bold:
        tick.set_fontweight("bold")

ax.set_xlim(-1.5, 58)
ax.set_ylim(-0.6, len(ARMS) - 0.4)
ax.set_xticks([0, 10, 20, 30, 40, 50])
ax.set_xlabel("user-facing entity-fabrication rate (%)")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=3, width=0.6)

# The x axis stops at 50; the k/N column lives beyond it.
ax.spines["bottom"].set_bounds(0, 50)

out = pathlib.Path(__file__).with_name("fig2_arms.pdf")
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
print("wrote", out)
