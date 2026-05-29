"""Generate the three example datasets deterministically.

Usage:
    python examples/generate.py

Produces:
    examples/titanic_small.csv
    examples/house_prices.csv
    examples/customer_churn.csv

Everything is seeded so reruns produce identical files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
HERE = Path(__file__).parent


# --------------------------------------------------------------------------- #
# 1. Titanic-style binary classification                                      #
# --------------------------------------------------------------------------- #


def make_titanic_small(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    pclass = rng.choice([1, 2, 3], size=n, p=[0.25, 0.20, 0.55])
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    embarked = rng.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09])
    siblings = rng.poisson(lam=0.5, size=n).clip(0, 6)
    age = rng.normal(loc=29.5, scale=14.0, size=n).clip(0.5, 80)

    # Fare correlates with class (mean: 80 / 22 / 14)
    fare_means = {1: 80, 2: 22, 3: 14}
    fare_std = {1: 60, 2: 15, 3: 8}
    fare = np.array(
        [rng.normal(fare_means[c], fare_std[c]) for c in pclass]
    ).clip(0, 500)

    # Survival probability — women + 1st class + young → higher
    logit = (
        -1.0
        + 2.5 * (sex == "female")
        - 0.9 * (pclass == 3)
        + 0.3 * (pclass == 1)
        - 0.012 * age
    )
    p_survive = 1 / (1 + np.exp(-logit))
    survived = (rng.random(n) < p_survive).astype(int)

    # Introduce missingness in age (~10%)
    missing_idx = rng.choice(n, size=int(n * 0.1), replace=False)
    age_with_missing = age.copy()
    age_with_missing[missing_idx] = np.nan

    return pd.DataFrame(
        {
            "pclass": pclass,
            "sex": sex,
            "age": np.round(age_with_missing, 1),
            "siblings": siblings,
            "fare": np.round(fare, 2),
            "embarked": embarked,
            "survived": survived,
        }
    )


# --------------------------------------------------------------------------- #
# 2. House price regression                                                   #
# --------------------------------------------------------------------------- #


def make_house_prices(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)

    neighborhoods = ["downtown", "suburb", "rural", "uptown", "waterfront"]
    nb_weights = [0.20, 0.40, 0.20, 0.15, 0.05]
    neighborhood = rng.choice(neighborhoods, size=n, p=nb_weights)

    sq_ft = rng.normal(loc=1800, scale=600, size=n).clip(500, 6000)
    bedrooms = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.20, 0.45, 0.25, 0.05])
    bathrooms = (bedrooms - rng.integers(0, 2, size=n)).clip(1, None).astype(int)
    age_years = rng.integers(0, 100, size=n)
    lot_size = rng.normal(loc=6500, scale=3000, size=n).clip(1000, 30000)

    # Neighborhood premium
    nb_premium = {
        "downtown": 350,
        "uptown": 420,
        "waterfront": 800,
        "suburb": 250,
        "rural": 150,
    }
    base_psf = np.array([nb_premium[n_] for n_ in neighborhood])

    # Linear-ish price model with noise
    price = (
        sq_ft * base_psf
        + bedrooms * 8_000
        + bathrooms * 5_000
        - age_years * 600
        + lot_size * 8
        + rng.normal(0, 25_000, size=n)
    ).clip(40_000, None)

    # Inject 5 luxury outliers
    outlier_idx = rng.choice(n, size=5, replace=False)
    price[outlier_idx] *= rng.uniform(3.5, 6.0, size=5)

    return pd.DataFrame(
        {
            "sq_ft": np.round(sq_ft).astype(int),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age_years,
            "neighborhood": neighborhood,
            "lot_size": np.round(lot_size).astype(int),
            "price": np.round(price).astype(int),
        }
    )


# --------------------------------------------------------------------------- #
# 3. Imbalanced telecom churn                                                 #
# --------------------------------------------------------------------------- #


def make_customer_churn(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)

    # 30 distinct countries — mid-cardinality categorical
    countries = [f"C{i:02d}" for i in range(30)]
    country = rng.choice(countries, size=n)

    plan_type = rng.choice(
        ["basic", "standard", "premium", "enterprise"],
        size=n,
        p=[0.45, 0.30, 0.20, 0.05],
    )

    tenure_months = rng.integers(1, 72, size=n)
    monthly_charges = rng.uniform(20, 150, size=n)
    total_charges = monthly_charges * tenure_months + rng.normal(0, 50, size=n)
    support_calls = rng.poisson(lam=1.2, size=n)

    # Signup date range — last 6 years
    days_back = rng.integers(0, 6 * 365, size=n)
    signup_dates = pd.Timestamp("2026-05-29") - pd.to_timedelta(days_back, unit="D")

    # Churn is rare (~5%), driven by short tenure + high support calls
    logit = (
        -4.5
        - 0.04 * tenure_months
        + 0.5 * support_calls
        + 0.01 * (monthly_charges - 80)
        + 0.6 * (plan_type == "basic")
    )
    p_churn = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < p_churn).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": np.round(monthly_charges, 2),
            "total_charges": np.round(total_charges, 2),
            "country": country,
            "plan_type": plan_type,
            "support_calls": support_calls,
            "signup_date": signup_dates.strftime("%Y-%m-%d"),
            "churn": churn,
        }
    )


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def main() -> None:
    generators = [
        ("titanic_small.csv", make_titanic_small),
        ("house_prices.csv", make_house_prices),
        ("customer_churn.csv", make_customer_churn),
    ]

    for name, fn in generators:
        df = fn()
        out_path = HERE / name
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path.relative_to(HERE.parent)}  ({len(df)} rows × {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
