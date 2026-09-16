# Figures

Four figures, all in place. Three came back as images from an image model and
are committed as rasters; one is generated from committed code.

| In the paper | File | Source |
|---|---|---|
| Fig. 1 architecture | `fig1_contract.png` | `../figure1.webp`, image model |
| Fig. 2 seven arms | `fig2_arms.pdf` | `fig_arms.py`, matplotlib, vector |
| Fig. 3 paraphrase sweep | `fig3_sweep.png` | `../figure3.png`, image model |
| Fig. 4 per-rule defects | `fig4_defects.png` | `../figure4.png`, image model |

Regenerate Figure 2 with:

```
python paper/emse_latex/fig_arms.py
```

Its numbers are restated from Table 3 of the manuscript, which is produced by
`scripts/reproduce_hallucination_ablation.py`. When a number in that table
moves, the script has to move with it — that is the price of restating rather
than reading, and it is paid deliberately so the figure is legible without a
data file.

## Known weaknesses in the three raster figures

These are raster at roughly 210--315 DPI. EMSE accepts them, but a reviewer
who asks for vector is entitled to. If any of the three needs to be redrawn,
the data below is what it must carry.

- **Fig. 4** reuses the same wavy hatch for `leak_duplicate_rows` and
  `wrong_metric_for_imbalance`, and prints the `advise+audit` segment labels
  (`0`, `8`) stacked on top of each other. Neither misstates a value; both are
  worth fixing if the figure is ever regenerated.
- **Fig. 3** annotations are small relative to the panel.

### Fig. 2 data (also in `fig_arms.py`)

label, estimate, lo, hi, k/N

```
A-L1             no enforcement               42.0  35.4  48.9  84/200
A-GR-STOCK       Guardrails AI, stock         35.0  28.7  41.8  70/200
A-STRICT-STATIC  provider strict, no enum      4.0   2.0   7.7   8/200
A-GR-OURS        Guardrails' loop, our Tier B  0.0   0.0   1.9   0/200
A-CONTRACT       Tier A + Tier B               0.0   0.0   1.9   0/200
A-STRESS         Tier B alone, no enum         0.0   0.0   1.9   0/200
```

`A-STATIC-ENUM-STALE` is in Table 3 and deliberately not in the figure: it is
degenerate on this task, and a seventh zero next to three earned zeros reads as
four successes when it is three.

### Fig. 3 data

Six variants in ascending blended rate: terse, baseline, expert, cautious,
helpful, mechanical.

```
misfiled   4  36  51  68  71  71
invented   7   0   5   2   5   1
```

### Fig. 4 data

```
                              control  +revise  advise  +audit
leak_duplicate_rows                24       24      14       8
no_validation                       7       10       9       0
target_in_features                  2        0       0       0
no_seed                             0        1       0       0
leak_fit_before_split               0        0       1       0
wrong_metric_for_imbalance          0        0       0       0
total                              33       35      24       8
```

`wrong_metric_for_imbalance` is zero everywhere because its precondition — a
minority class below 20 % — is false of all three datasets. It stays in the
legend labelled "(never fires)": a sixth of the instrument that cannot fire on
this data is a finding, not an omission.
