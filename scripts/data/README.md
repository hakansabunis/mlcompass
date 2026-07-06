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

SHA256 (2026-07-07 fetch):

```
21bf566ec4bba11a0151d2fabdfd01fccb12556f694f214df48488bd6e0ba77c  insurance.csv
953d45066c221ee00cf8de72cfa4ab2086eb4e231fef6dda8fc3051b17840e61  heart_cleveland.csv
4cd901efab211ccb154051a93a1ab5e9298e37e8fd29bbf2c097da23cee36017  telco_churn.csv
5d5d25c60c165143749013a89a86c604fd55bac07c2a2e6e0d6c9dacf65e6226  ames_housing.csv
```

The frozen (dataset x injector) instance list lives in
`scripts/fetch_fabbench_datasets.py::FROZEN_INSTANCES` and is registered as
amendment A1 in `paper/analysis_plan.md` §8.
