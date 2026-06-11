"""Generate Fig. 1: the evidence-bound contract flow (vector PDF + PNG).

Grayscale, column-width, IEEE-print friendly. Replaces the generic product
architecture figure: the reviewers asked the figure to show the load-bearing
boundary — where the enums are generated from the evidence (Tier A) and where
the deterministic verification happens (Tier B).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).parent

FIG_W, FIG_H = 3.45, 3.55  # inches, single IEEE column

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

BOX = dict(boxstyle="round,pad=0.025,rounding_size=0.02", linewidth=0.9)


def box(x, y, w, h, lines, fc="#f5f5f5", bold_first=True, fs=6.8):
    ax.add_patch(
        FancyBboxPatch((x - w / 2, y - h / 2), w, h, fc=fc, ec="black", **BOX)
    )
    n = len(lines)
    for k, line in enumerate(lines):
        dy = (n - 1) / 2 - k
        ax.text(
            x,
            y + dy * 0.034,
            line,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold" if (k == 0 and bold_first) else "normal",
        )


def arrow(p, q, label=None, lx=0.0, ly=0.0, style="-|>", rad=0.0, fs=6.2):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle=style,
            mutation_scale=9,
            linewidth=0.9,
            color="black",
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        ax.text(
            (p[0] + q[0]) / 2 + lx,
            (p[1] + q[1]) / 2 + ly,
            label,
            ha="center",
            va="center",
            fontsize=fs,
            style="italic",
        )


# ---- boxes -----------------------------------------------------------------
box(0.5, 0.935, 0.78, 0.085, ["Deterministic evidence producer", "(pure Python, offline)"])
box(
    0.5,
    0.775,
    0.88,
    0.085,
    ["Evidence  $E$", "entities $A_E$   ·   measured values $V_E$   ·   anchor $a^*$"],
    fc="#e8e8e8",
)
box(0.5, 0.575, 0.78, 0.085, ["LLM narrator  (any provider)", "answers via submit_investigation tool"])
box(
    0.5,
    0.375,
    0.92,
    0.115,
    [
        "TIER B:  deterministic verifier (plain code)",
        "(1) entities $\\subseteq A_E$     (2) $|v - V_E| \\leq \\varepsilon$     (3) $a^*$ addressed",
    ],
    fc="#e8e8e8",
)
box(0.5, 0.075, 0.86, 0.085, ["User", "validated citations + verified claims only"])

# ---- arrows ----------------------------------------------------------------
arrow((0.5, 0.892), (0.5, 0.818))
arrow(
    (0.5, 0.732),
    (0.5, 0.618),
    label="TIER A:  enum($A_E$) bound into\ntool schema at call time",
    lx=-0.26,
    fs=6.0,
)
arrow((0.5, 0.532), (0.5, 0.433))
# pass path
arrow((0.42, 0.3175), (0.42, 0.118), label="✓ faithful", lx=-0.115, fs=6.2)
# retry loop (right side, curved up); label placed clear of the narrator box
arrow((0.72, 0.41), (0.72, 0.545), rad=-0.3)
ax.text(
    0.875,
    0.497,
    "✗ violation:\nretry (≤2),\nnames items",
    ha="center",
    va="center",
    fontsize=5.7,
    style="italic",
)
# residual path
arrow(
    (0.58, 0.3175),
    (0.58, 0.118),
    label="after budget:\nstrip unsound /\nflag omission",
    lx=0.155,
    fs=5.8,
)

fig.tight_layout(pad=0.15)
fig.savefig(ROOT / "inista_latex" / "contract_flow.pdf")
fig.savefig(ROOT / "contract_flow.png", dpi=300)
print("OK: contract_flow.pdf + contract_flow.png")
