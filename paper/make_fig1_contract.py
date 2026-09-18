"""Generate Fig. 1: the evidence-bound contract flow.

Both manuscripts shipped `fig1_contract.png` with no generator in the tree,
which for a paper about reproducibility is not a defensible state. This script
is that generator, and it adds the three annotations a reviewer asked for:

  * the evidence producer is *trusted and never verified* -- the guarantee
    starts below it, not above it;
  * unverified prose still reaches the user, so the guarantee covers the
    channels and not the screen;
  * the scope band under the flow: response <-> E is guaranteed,
    E <-> the world is not.

Those three are the fastest way for a reader to see where the claim stops,
and every one of them was already in the prose.

    python paper/make_fig1_contract.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).parent
OUT = [
    ROOT / "emse_latex" / "fig1_contract.png",
    ROOT / "tse_latex" / "fig1_contract.png",
]

fig, ax = plt.subplots(figsize=(6.6, 6.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# The flow occupies the left two-thirds; the right third is annotation margin.
# Keeping them disjoint is the whole reason the previous drawing was legible.
L, R = 0.035, 0.655
CX = (L + R) / 2
MARG = R + 0.022

BOXSTYLE = dict(boxstyle="square,pad=0.0", linewidth=1.15, zorder=3)


def box(top, h, lines, sizes, fc="white"):
    """Draw a box with its TOP edge at `top`; return (top, bottom)."""
    bot = top - h
    ax.add_patch(
        FancyBboxPatch((L, bot), R - L, h, fc=fc, ec="black", **BOXSTYLE)
    )
    n = len(lines)
    mid = (top + bot) / 2
    for k, (line, fs) in enumerate(zip(lines, sizes)):
        dy = ((n - 1) / 2 - k) * (h / (n + 0.7))
        ax.text(CX, mid + dy, line, ha="center", va="center", fontsize=fs, zorder=4)
    return top, bot


def arrow(y0, y1, x=CX):
    ax.add_patch(
        FancyArrowPatch(
            (x, y0), (x, y1),
            arrowstyle="-|>", mutation_scale=12, linewidth=1.15,
            color="black", zorder=2,
        )
    )


def note(x, y, s, fs=8.2, weight="normal", style="italic"):
    ax.text(x, y, s, ha="left", va="center", fontsize=fs,
            style=style, fontweight=weight, zorder=5)


H = 0.096          # box height
GAP = 0.078        # arrow run between boxes

# ---- the flow, top to bottom ------------------------------------------------ #
_, b1 = box(0.995, H, ["Deterministic evidence producer", "(pure Python, offline)"],
            [10.6, 9.8])
note(MARG, 0.946, "trusted,\nnever verified", fs=8.4, weight="bold", style="normal")
arrow(b1, b1 - GAP)

t2, b2 = box(b1 - GAP, H,
             ["Evidence  $E$",
              "entities $A_E$  ·  values $V_E$  ·  anchor $a^*$"],
             [10.6, 9.4])
arrow(b2, b2 - 0.108)
note(CX + 0.028, b2 - 0.058,
     "TIER A:  enum($A_{E,f}$) bound into\nthe tool schema at call time", fs=8.4)

t3, b3 = box(b2 - 0.108, H,
             ["LLM narrator — any provider",
              "answers via a submit_investigation tool"],
             [10.6, 9.4])
# untrusted boundary, around the narrator and nothing else
ax.add_patch(
    Rectangle((L - 0.022, b3 - 0.016), (R - L) + 0.044, H + 0.032,
              linestyle=(0, (4.5, 3.2)), linewidth=1.05,
              edgecolor="black", facecolor="none", zorder=1)
)
note(MARG + 0.004, t3 + 0.030, "untrusted", fs=8.4)
arrow(b3 - 0.016, b3 - GAP)

t4, b4 = box(b3 - GAP, H,
             ["TIER B:  deterministic verifier, plain code",
              "(1) $x \\in A_{E,f}$   (2) $|v - V_E| \\leq 0.005$   (3) $a^*$ addressed"],
             [10.6, 9.0])

# the corrective retry, up the right-hand side and back into the narrator
ax.add_patch(
    FancyArrowPatch(
        (R + 0.014, t4 - 0.004), (R + 0.014, b3 - 0.010),
        arrowstyle="-|>", mutation_scale=12, linewidth=1.15,
        color="black", connectionstyle="arc3,rad=0.55", zorder=2,
    )
)
note(MARG + 0.052, (b3 + t4) / 2 - 0.004,
     "violation:\nretry ($\\leq$2),\nnaming the\noffending items", fs=8.0)

arrow(b4, b4 - 0.112)
note(CX + 0.028, b4 - 0.059,
     "after the retry budget:\nstrip unsound, flag omission", fs=8.4)

t5, b5 = box(b4 - 0.112, H,
             ["User",
              "validated citations + verified claims,",
              "then free text marked NOT verified"],
             [10.6, 9.0, 9.0])
note(MARG, (t5 + b5) / 2, "unverified prose\nstill reaches\nthe user",
     fs=8.4, weight="bold", style="normal")

# ---- the scope band: what the contract does and does not promise ------------ #
BY = b5 - 0.040
ax.add_patch(
    Rectangle((L - 0.022, BY - 0.072), (R - L) + 0.044 + 0.30, 0.072,
              linewidth=1.15, edgecolor="black", facecolor="#ececec", zorder=3)
)
ax.text(CX + 0.14, BY - 0.021,
        "GUARANTEED:   response  $\\leftrightarrow$  $E$",
        ha="center", va="center", fontsize=9.4, fontweight="bold", zorder=4)
ax.text(CX + 0.14, BY - 0.051,
        "NOT GUARANTEED:   $E$  $\\leftrightarrow$  ground truth",
        ha="center", va="center", fontsize=9.4, fontweight="bold", zorder=4)

fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
for path in OUT:
    fig.savefig(path, dpi=200)
    print(f"OK: {path}")
