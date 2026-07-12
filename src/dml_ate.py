import pandas as pd
from econml.dml import LinearDML
from sklearn.linear_model import LassoCV, LogisticRegressionCV

COVARIATES = ["age", "income", "tenure_days", "avg_spend", "purchase_frequency", "total_transactions"]

# Naive baseline numbers on real data, computed in baseline_models.py
# (Phase 3). Hardcoded here since dml_ate.py doesn't recompute the
# naive comparison itself -- kept in one place to avoid drift; if
# baseline_models.py changes, update these two values.
NAIVE_REAL_EFFECT = 10.73
NAIVE_REAL_CI = (4.61, 16.84)


# ============================================================
# Core model fitting
# ============================================================

def fit_linear_dml(Y, T, X):
    """
    Fit a LinearDML model with discrete_treatment=True. This flag
    matters: without it, econml assumes T is continuous and fits
    model_t as a regression rather than a classifier, which distorts
    the residualization step for a binary treatment like ours. See
    write-up Pitfalls section for the before/after comparison.
    """
    est = LinearDML(
        model_y=LassoCV(),
        model_t=LogisticRegressionCV(max_iter=1000),
        discrete_treatment=True,
        random_state=42
    )
    est.fit(Y, T, X=X, W=None)
    return est


def get_ate_summary(est, X):
    """ATE point estimate + 95% CI for the fitted model."""
    ate = est.ate(X)
    ci = est.ate_interval(X, alpha=0.05)
    return {"ate": ate, "ci_lower": ci[0], "ci_upper": ci[1]}


# ============================================================
# Synthetic validation
# ============================================================

def run_synthetic_validation(synthetic_path="data/synthetic/synthetic_customers.csv"):
    """
    Fits LinearDML on synthetic data and compares against the known
    true_uplift ground truth and the naive (unadjusted) estimate.
    Returns the fitted estimator, X (for reuse), and a metrics dict.
    """
    df = pd.read_csv(synthetic_path)

    Y = df["outcome"].values
    T = df["treatment"].values
    X = df[COVARIATES].values

    est = fit_linear_dml(Y, T, X)
    ate_summary = get_ate_summary(est, X)

    true_ate = df["true_uplift"].mean()
    naive_effect = (
        df.loc[df["treatment"] == 1, "outcome"].mean()
        - df.loc[df["treatment"] == 0, "outcome"].mean()
    )

    metrics = {
        "true_ate": true_ate,
        "naive_effect": naive_effect,
        "naive_bias": naive_effect - true_ate,
        "dml_ate": ate_summary["ate"],
        "dml_bias": ate_summary["ate"] - true_ate,
        "dml_ci_lower": ate_summary["ci_lower"],
        "dml_ci_upper": ate_summary["ci_upper"],
    }

    return est, X, metrics


# ============================================================
# Real data
# ============================================================

def run_real_data_dml(processed_path="data/processed/processed_starbucks.csv"):
    """
    Fits LinearDML on the real Starbucks data and compares against
    the naive baseline computed separately in baseline_models.py
    (Phase 3).
    """
    df_real = pd.read_csv(processed_path)

    Y = df_real["outcome"].values
    T = df_real["treatment"].values
    X = df_real[COVARIATES].values

    est = fit_linear_dml(Y, T, X)
    ate_summary = get_ate_summary(est, X)

    metrics = {
        "naive_effect": NAIVE_REAL_EFFECT,
        "naive_ci": NAIVE_REAL_CI,
        "dml_ate": ate_summary["ate"],
        "dml_ci_lower": ate_summary["ci_lower"],
        "dml_ci_upper": ate_summary["ci_upper"],
    }

    return est, X, metrics


# ============================================================
# Script entry point 
# ============================================================

if __name__ == "__main__":

    # --- Part 1: Synthetic validation ---
    est_syn, X_syn, syn_metrics = run_synthetic_validation()

    print("=== Synthetic Data Validation (discrete_treatment=True) ===")
    print(f"True ATE (ground truth):      {syn_metrics['true_ate']:.2f}")
    print(f"Naive estimate:                {syn_metrics['naive_effect']:.2f}  (bias: {syn_metrics['naive_bias']:+.2f})")
    print(f"LinearDML estimate:            {syn_metrics['dml_ate']:.2f}  (bias: {syn_metrics['dml_bias']:+.2f})")
    print(f"LinearDML 95% CI:              ({syn_metrics['dml_ci_lower']:.2f}, {syn_metrics['dml_ci_upper']:.2f})")

    # --- Part 2: Real data ---
    est_real, X_real, real_metrics = run_real_data_dml()

    print("\n=== LinearDML on Real Starbucks Data (discrete_treatment=True) ===")
    print(f"Naive estimate:        {real_metrics['naive_effect']:.2f}  "
          f"(95% CI: {real_metrics['naive_ci'][0]}, {real_metrics['naive_ci'][1]})")
    print(f"LinearDML estimate:    {real_metrics['dml_ate']:.2f}")
    print(f"LinearDML 95% CI:      ({real_metrics['dml_ci_lower']:.2f}, {real_metrics['dml_ci_upper']:.2f})")