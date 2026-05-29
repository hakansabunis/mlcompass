# mlcompass example datasets

Three small synthetic datasets covering the most common tabular ML
scenarios. Each one is designed so `mlcompass advise` produces a
distinctive, useful recommendation.

| File                        | Task                          | What it shows off                              |
| --------------------------- | ----------------------------- | ---------------------------------------------- |
| `titanic_small.csv`         | Binary classification         | Mild imbalance, missing data, mixed types      |
| `house_prices.csv`          | Regression                    | Outliers, skewed numeric targets, neighborhoods|
| `customer_churn.csv`        | Imbalanced binary             | Heavy class imbalance, high-cardinality country|

All three are generated deterministically by `generate.py` (with a
fixed RNG seed) and committed to the repo. Re-run `generate.py` if you
want to regenerate them; the contents are identical.

## Quick start

```bash
mlcompass init demo
mlcompass advise examples/titanic_small.csv
mlcompass advise examples/house_prices.csv --target price
mlcompass advise examples/customer_churn.csv
```

## Expected advisor highlights

### `titanic_small.csv` (binary classification, baseline scenario)

- 200 rows, 7 columns: `age`, `sex`, `pclass`, `fare`, `embarked`,
  `siblings`, `survived` (target).
- ~38% positive class.
- ~10% of `age` is missing (realistic Titanic-style data).
- Advisor should recommend XGBoost / Logistic Regression as baselines,
  note the small dataset (suggest CV), and flag the missing-age imputation.

### `house_prices.csv` (regression)

- 300 rows, 7 columns: `sq_ft`, `bedrooms`, `bathrooms`, `age_years`,
  `neighborhood`, `lot_size`, `price` (target).
- ~5 deliberate price outliers in the top tail (luxury homes).
- Advisor should recommend gradient boosting + linear regression baseline,
  suggest log-transforming `price` or `sq_ft`, and call out the outliers.

### `customer_churn.csv` (heavy imbalance)

- 500 rows, 8 columns: `tenure_months`, `monthly_charges`, `total_charges`,
  `country`, `plan_type`, `support_calls`, `signup_date`, `churn` (target).
- ~5% positive class (real-world telecom-like imbalance).
- `country` has 30 distinct values (mid-cardinality categorical).
- `signup_date` is a datetime — advisor should suggest deriving features.
- Advisor should heavily emphasise the imbalance (focal loss, AUC),
  call out `country` cardinality handling, and recommend date-derived
  features.

## Regenerating

```bash
python examples/generate.py
```

Output paths and row counts are stable across runs (seed = 42).
