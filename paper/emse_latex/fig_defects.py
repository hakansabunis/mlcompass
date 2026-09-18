"""Figure 4: where the downstream effect comes from, and what it costs.

Regenerated because the arms moved twice. Amendment A16 removed both
`target_in_features` flags and `control` fell 33 -> 31. Amendment A17 then
added a fifth arm, `control+revise+rubric`, which reaches the best checklist
score in the study -- 1 flagged defect -- and is the worst arm on the only
outcome a practitioner receives.

Plotting that fifth bar on its own would say the opposite of what the study
found: a bar of height 1 beside `advise+audit`'s 8 reads as the winner. So the
figure carries the execution rate as a second series. The checklist total and
the share of scripts that actually run are the two halves of the same result,
and separating them into two figures would let a reader keep whichever half
suits them.

Values restated from Tables 8 and 9 of the manuscript, both produced by
re-running benchmark/run_ab.check_defects over the preserved emitted scripts.
Every rate is over the 36 planned runs of its arm.
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

ARMS = [
    "control",
    "control\n+revise",
    "advise",
    "advise\n+audit",
    "control+revise\n+rubric",
]
# rule, per-arm counts, hatch -- every hatch distinct, which the raster's was not
RULES = [
    ("leak_duplicate_rows", [24, 24, 14, 8, 1], "///"),
    ("no_validation", [7, 10, 9, 0, 0], "..."),
    ("no_seed", [0, 1, 0, 0, 0], "xxx"),
    ("leak_fit_before_split", [0, 0, 1, 0, 0], "\\\\\\"),
    ("target_in_features", [0, 0, 0, 0, 0], "|||"),
    ("wrong_metric_for_imbalance  (cannot fire)", [0, 0, 0, 0, 0], "---"),
]
# Share of each arm's 36 planned runs whose script executed.
RUNNABLE = [78, 81, 81, 83, 47]

fig, ax = plt.subplots(figsize=(5.2, 3.2))
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

# The second half of the result. Without it the fifth bar reads as the winner.
ax2 = ax.twinx()
ax2.plot(
    list(xs),
    RUNNABLE,
    marker="D",
    ms=5.0,
    mfc="white",
    mec="black",
    mew=1.1,
    color="black",
    lw=1.0,
    ls=(0, (2, 2)),
    zorder=5,
    label="executes (% of 36 runs)",
)
# Only the outlier is labelled. Labelling all five put the numbers on top of
# the bar-segment labels underneath them; the flat 78-83 run is in the caption
# and readable off the right axis.
ax2.annotate(
    f"{RUNNABLE[-1]}%",
    (len(RUNNABLE) - 1, RUNNABLE[-1]),
    textcoords="offset points",
    xytext=(0, -13),
    ha="center",
    fontsize=8.0,
    fontweight="bold",
    bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
    zorder=6,
)
ax2.set_ylim(0, 105)
ax2.set_ylabel("scripts that execute (%)", fontsize=7.5)
ax2.spines[["top"]].set_visible(False)
ax2.tick_params(length=3, width=0.6, labelsize=7.5)

ax.set_xticks(list(xs))
ax.set_xticklabels(ARMS, fontsize=7.4)
ax.set_ylabel("flagged defects, summed over the arm's runs", fontsize=7.5)
ax.set_ylim(0, 40)
ax.set_xlim(-0.6, len(ARMS) - 0.4)
ax.spines[["top"]].set_visible(False)
ax.tick_params(length=3, width=0.6)

handles, labels = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(
    handles + h2,
    labels + l2,
    loc="upper left",
    bbox_to_anchor=(1.13, 1.0),
    frameon=False,
    fontsize=7,
    handlelength=1.8,
    labelspacing=0.6,
)

out = pathlib.Path(__file__).with_name("fig4_defects.pdf")
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
print("wrote", out)
