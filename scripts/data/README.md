# FabBench datasets (Phase 2 corpora)

Fetched and normalized by `scripts/fetch_fabbench_datasets.py` (re-run it to
reproduce; `--verify` checks every frozen task instance against the shipped
detector with zero API calls). Files are committed so the benchmark is
runnable from a bare clone; the SHA256 below pins the exact bytes used by
every run record.

| file | rows x cols | target | source | license/terms |
|---|---|---|---|---|
| `insurance.csv` | 1338 x 7 | `charges` | Lantz, *Machine Learning with R* — [stedy/Machine-Learning-with-R-datasets](https://github.com/stedy/Machine-Learning-with-R-datasets) | freely redistributed teaching dataset (simulated from US census data) |
| `heart_cleveland.csv` | 303 x 14 | `chol` | [UCI Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+disease), processed Cleveland subset; header row added, `?` -> NaN | CC BY 4.0 — credit: Janosi, Steinbrunn, Pfisterer, Detrano (1988) |
| `telco_churn.csv` | 7043 x 21 | `MonthlyCharges` | [IBM Telco Customer Churn sample](https://github.com/IBM/telco-customer-churn-on-icp4d) | IBM sample dataset, widely redistributed for demos/benchmarks |
| `ames_housing.csv` | 2930 x 82 | `SalePrice` | De Cock, *JSE* 19(3) 2011 — [AmesHousing.txt](https://jse.amstat.org/v19n3/decock/AmesHousing.txt); TSV -> CSV | public dataset published for educational use |

## Extended corpus (FabBench artifact breadth — NOT in the preregistered paper core)

These widen the released benchmark's domain coverage; the paper's measured
instances remain the frozen 13 (A1). All 6 injectors verified detectable on
each (`--verify-extended`, 18/18).

| file | rows x cols | target | source | license/terms |
|---|---|---|---|---|
| `abalone.csv` | 4177 x 9 | `rings` | [UCI Abalone](https://archive.ics.uci.edu/ml/datasets/abalone) (marine biology) | CC BY 4.0 |
| `wine_quality_red.csv` | 1599 x 12 | `alcohol` | [UCI Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality), red (chemistry) | CC BY 4.0 — Cortez et al. 2009 |
| `airfoil_self_noise.csv` | 1503 x 6 | `sound_pressure_level` | [UCI Airfoil Self-Noise](https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise) (NASA aeroacoustics) | CC BY 4.0 |

**Design note (verified 2026-07-07):** `monotone_log` and `binned_target`
require a CONTINUOUS target. On a discrete low-cardinality target (wine's
`quality`, ~6 levels) heavy ties push their correlations below the 0.99
detector threshold — which is why wine's benchmark target is the continuous
`alcohol`. Keep this in mind when adding datasets.

SHA256 (2026-07-07 fetch):

```
21bf566ec4bba11a0151d2fabdfd01fccb12556f694f214df48488bd6e0ba77c  insurance.csv
953d45066c221ee00cf8de72cfa4ab2086eb4e231fef6dda8fc3051b17840e61  heart_cleveland.csv
4cd901efab211ccb154051a93a1ab5e9298e37e8fd29bbf2c097da23cee36017  telco_churn.csv
5d5d25c60c165143749013a89a86c604fd55bac07c2a2e6e0d6c9dacf65e6226  ames_housing.csv
e56ba50db5684b117d288f52647987f9fe69d95b040bcda4f55040b2d43bd1da  abalone.csv
d73b81349be9fb18465f19db6b718a9d39322b70695a6cc847c8673da0d4d999  wine_quality_red.csv
60da0d7fcc6b96b021232cfa157eb9eb3cc9e66484dca0ae92299ff4148a2016  airfoil_self_noise.csv
```

## Real-world case studies (natural leaks — fetched on demand, NOT committed)

Fetch with `python scripts/fetch_fabbench_datasets.py --fetch-cases`; check
the pre-stated expectations with `--verify-cases`. Registered as amendment
A2 in `paper/analysis_plan.md` §8.

| case | rows x cols | role | documented leak | source |
|---|---|---|---|---|
| `bodyfat.csv` | 252 x 15 | **positive** — detector fires on a leak nobody injected | `Density` generates the target via Siri's 1956 equation; \|Spearman\| ~ 0.993 | OpenML 560; Johnson, *J. Stat. Educ.* 4(1), 1996 (no explicit redistribution license -> not committed) |
| `sambanis_civil_war.csv` | 7140 x 286 | **negative control** — detector must stay silent; measures false-positive fabrication on innocent evidence | procedural (imputation before split; Kapoor & Narayanan, *Patterns* 2023); max corr ~ 0.65, no duplicates | Harvard Dataverse doi:10.7910/DVN/KRKWK8, CC0 (12 MB -> fetched, not committed) |

The frozen (dataset x injector) instance list lives in
`scripts/fetch_fabbench_datasets.py::FROZEN_INSTANCES` and is registered as
amendment A1 in `paper/analysis_plan.md` §8. The extended corpus is
deliberately excluded from A1 — running it is optional artifact material.
