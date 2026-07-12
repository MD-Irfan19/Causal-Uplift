import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 8000

BASE_SALES_NOISE_STD = 8.0
OUTCOME_NOISE_STD = 10.0

rng = np.random.default_rng(RANDOM_SEED)


def generate_covariates(n=N_CUSTOMERS):
    """
    Generate synthetic customer covariates with realistic correlations,
    mirroring the structure we confirmed is derivable from the real
    Starbucks dataset.
    """
    customer_id = np.arange(1, n + 1)

    # Age: roughly mirrors real profile.json distribution (18-90, right-skewed)
    age = rng.normal(loc=55, scale=17, size=n).clip(18, 90).round().astype(int)

    # Income: correlated with age (older -> slightly higher income, capped)
    income_base = rng.normal(loc=65000, scale=21000, size=n)
    income = (income_base + (age - 55) * 200).clip(30000, 120000).round(-2)

    # Tenure: days since signup, 0 to ~3 years
    tenure_days = rng.exponential(scale=400, size=n).clip(1, 1100).round().astype(int)

    # Total transactions: correlated with tenure (longer tenure -> more transactions),
    # with noise so it's not a deterministic function
    total_transactions = (
        (tenure_days / 40) + rng.normal(loc=0, scale=3, size=n)
    ).clip(1, 40).round().astype(int)

    # Purchase frequency: transactions per 30 days of tenure
    purchase_frequency = (total_transactions / (tenure_days / 30)).clip(0.05, 5).round(3)

    # Avg spend: correlated with income (higher income -> higher avg spend per transaction)
    avg_spend = (
        10 + (income / 100000) * 15 + rng.normal(loc=0, scale=5, size=n)
    ).clip(2, 60).round(2)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "income": income,
        "tenure_days": tenure_days,
        "avg_spend": avg_spend,
        "purchase_frequency": purchase_frequency,
        "total_transactions": total_transactions,
    })

    return df

def assign_treatment(df):
    """
    Confounded treatment assignment: higher income and longer tenure
    increase the probability of receiving the discount offer. This
    intentionally biases the naive (unadjusted) comparison, since
    treated customers are systematically different from untreated ones
    even before any treatment effect is applied.
    """
    # Standardize inputs so coefficients are interpretable regardless of scale
    income_z = (df["income"] - df["income"].mean()) / df["income"].std()
    tenure_z = (df["tenure_days"] - df["tenure_days"].mean()) / df["tenure_days"].std()

    logit = -0.2 + 0.5 * income_z + 0.4 * tenure_z
    propensity = 1 / (1 + np.exp(-logit))  # sigmoid

    treatment = rng.binomial(n=1, p=propensity)

    df = df.copy()
    df["true_propensity"] = propensity.round(4)
    df["treatment"] = treatment

    return df

def compute_true_uplift(df):
    """
    Ground-truth heterogeneous treatment effect function. This is
    deliberately hand-specified so we know the "right answer" the
    Causal Forest should recover later.

    Story encoded here:
      - Newer customers (tenure < 180 days) respond more strongly
      - Lower average spenders respond more strongly (more price-sensitive)
      - High-income customers (> 100k) respond less (less price-sensitive)
    """
    uplift = (
        30
        + 40 * (df["tenure_days"] < 180).astype(int)
        + 25 * (df["avg_spend"] < 50).astype(int)
        - 20 * (df["income"] > 100000).astype(int)
    )
    return uplift

def generate_outcome(df):
    """
    Observed outcome = baseline spend (function of covariates, no
    treatment) + treatment_effect (only applied if treated) + noise.

    Note: true_uplift is stored for validation purposes ONLY. It must
    NEVER be used as a model input -- only to check, after the fact,
    whether the Causal Forest's predicted effect matches this ground
    truth.
    """
    df = df.copy()

    # Baseline spend: what this customer would spend with NO treatment,
    # as a function of their covariates (correlated with avg_spend and
    # purchase_frequency, plus noise)
    baseline_spend = (
        df["avg_spend"] * df["purchase_frequency"] * 4  # ~monthly baseline spend
        + rng.normal(loc=0, scale=BASE_SALES_NOISE_STD, size=len(df))
    ).clip(lower=0)

    true_uplift = compute_true_uplift(df)

    # Outcome only receives the uplift if actually treated
    outcome = (
        baseline_spend
        + df["treatment"] * true_uplift
        + rng.normal(loc=0, scale=OUTCOME_NOISE_STD, size=len(df))
    ).clip(lower=0).round(2)

    df["true_uplift"] = true_uplift
    df["baseline_spend"] = baseline_spend.round(2)
    df["outcome"] = outcome

    return df

if __name__ == "__main__":
    df = generate_covariates()
    df = assign_treatment(df)
    df = generate_outcome(df)

    print(f"Mean true_uplift (population average, i.e. true ATE): {df['true_uplift'].mean():.2f}")
    print(f"\nMean outcome, treated vs control:")
    print(df.groupby("treatment")["outcome"].mean())

    naive_effect = df.loc[df["treatment"] == 1, "outcome"].mean() - df.loc[df["treatment"] == 0, "outcome"].mean()
    print(f"\nNaive effect estimate (treated mean - control mean): {naive_effect:.2f}")
    print(f"True ATE (mean of true_uplift): {df['true_uplift'].mean():.2f}")
    print(f"Bias in naive estimate: {naive_effect - df['true_uplift'].mean():.2f}")

    df.to_csv("data/synthetic/synthetic_customers.csv", index=False)
    print("\nSaved to data/synthetic/synthetic_customers.csv")