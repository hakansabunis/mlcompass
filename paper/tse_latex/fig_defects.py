"""Figure 4: where the downstream effect comes from.

Regenerated rather than redrawn by hand, because the numbers moved: amendment
A16 removed both `target_in_features` flags and `control` fell 33 -> 31. The
supplied raster had the old totals baked in, and two cosmetic faults recorded
in FIGURE_PROMPTS.md -- a hatch shared between two rules, and the
`advise+audit` segment labels printed on top of each other.

Values restated from Table 8 of the manuscript, which is produced by
benchmark/rescore_ab.py under scoring version v4.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    }
)

ARMS = ["control", "control+revise", "advise", "advise+audit"]
# rule, per-arm counts, hatch -- every hatch distinct, which the raster's was not
RULES = [
    ("leak_duplicate_rows", [24, 24, 14, 8], "///"),
    ("no_validation", [7, 10, 9, 0], "..."),
    ("no_seed", [0, 1, 0, 0], "xxx"),
    ("leak_fit_before_split", [0, 0, 1, 0], "\\\\\\"),
    ("target_in_features", [0, 0, 0, 0], "|||"),
    ("wrong_metric_for_imbalance  (cannot fire)", [0, 0, 0, 0], "---"),
]

fig, ax = plt.subplots(figsize=(4.9, 2.9))
xs = range(len(ARMS))
bottom = [0] * len(ARMS)

for label, counts, hatch in RULES:
    ax.bar(
        xs,
        counts,
        bottom=bottom,
        width=0.62,
        facecolor="white",
        edgecolor="black",
        linewidth=0.6,
        hatch=hatch,
        label=label,
    )
    for x, (c, b) in enumerate(zip(counts, bottom, strict=True)):
        # Only label a segment tall enough to hold the text; the rest are read
        # off the table, which is what the caption points at.
        if c >= 4:
            # A white box behind the number: hatching runs straight through
            # glyphs otherwise and the count becomes guesswork in print.
            ax.text(
                x,
                b + c / 2,
                str(c),
                ha="center",
                va="center",
                fontsize=7.5,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2),
            )
    bottom = [b + c for b, c in zip(bottom, counts, strict=True)]

for x, total in zip(xs, bottom, strict=True):
    ax.text(x, total + 0.8, str(total), ha="center", va="bottom", fontweight="bold")

ax.set_xticks(list(xs))
ax.set_xticklabels(ARMS, fontsize=7.5)
ax.set_ylabel("flagged defects, summed over the arm's runs", fontsize=7.5)
ax.set_ylim(0, 40)
ax.set_xlim(-0.6, len(ARMS) - 0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=3, width=0.6)
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    frameon=False,
    fontsize=7,
    handlelength=1.8,
    labelspacing=0.6,
)

out = pathlib.Path(__file__).with_name("fig4_defects.pdf")
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
print("wrote", out)
