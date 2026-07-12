import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

COVARIATES = ["age", "income", "tenure_days", "avg_spend", "purchase_frequency", "total_transactions"]


# ============================================================
# Core model fitting
# ============================================================

def fit_causal_forest(Y, T, X, min_samples_leaf=50):
    """
    Fit a CausalForestDML model. min_samples_leaf=50 was chosen after
    an earlier run with the default (10) produced unstable leaf-level
    effect estimates given the real data's treatment/control imbalance
    (13,042 treated vs 1,783 control) -- see the write-up's Pitfalls
    section for details.
    """
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=200, min_samples_leaf=min_samples_leaf, random_state=42),
        model_t=RandomForestClassifier(n_estimators=200, min_samples_leaf=min_samples_leaf, random_state=42),
        discrete_treatment=True,
        n_estimators=500,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    est.fit(Y, T, X=X, W=None)
    return est


# ============================================================
# Synthetic validation
# ============================================================

def run_synthetic_validation(synthetic_path="data/synthetic/synthetic_customers.csv"):
    """
    Fits CausalForestDML on synthetic data and checks recovered CATEs
    against the known true_uplift ground truth. Returns the enriched
    dataframe and a metrics dict.
    """
    df_syn = pd.read_csv(synthetic_path)

    Y = df_syn["outcome"].values
    T = df_syn["treatment"].values
    X = df_syn[COVARIATES].values

    est = fit_causal_forest(Y, T, X)

    df_syn["predicted_cate"] = est.effect(X)

    metrics = {
        "ate": est.ate(X),
        "correlation": df_syn["predicted_cate"].corr(df_syn["true_uplift"]),
        "rmse": np.sqrt(((df_syn["predicted_cate"] - df_syn["true_uplift"]) ** 2).mean()),
    }

    return df_syn, est, metrics


# ============================================================
# Real data: fit + CATE + confidence intervals
# ============================================================

def fit_real_causal_forest(processed_path="data/processed/processed_starbucks.csv"):
    """
    Fits CausalForestDML on the real Starbucks data and attaches
    predicted CATE + per-customer confidence intervals.
    """
    df_real = pd.read_csv(processed_path)

    Y = df_real["outcome"].values
    T = df_real["treatment"].values
    X = df_real[COVARIATES].values

    est = fit_causal_forest(Y, T, X)

    df_real["predicted_cate"] = est.effect(X)
    ci_lower, ci_upper = est.effect_interval(X, alpha=0.05)
    df_real["cate_ci_lower"] = ci_lower
    df_real["cate_ci_upper"] = ci_upper

    return df_real, est, X


def get_ate_summary(est, X):
    """ATE point estimate + 95% CI for the fitted forest."""
    ate = est.ate(X)
    ci = est.ate_interval(X, alpha=0.05)
    return {"ate": ate, "ci_lower": ci[0], "ci_upper": ci[1]}


# ============================================================
# Feature importance
# ============================================================

def compute_feature_importance(est):
    """
    Built-in feature_importances_ from the forest -- used instead of
    SHAP given the 1-day timebox and the fact that 93% of individual
    CATEs aren't statistically distinguishable from zero (SHAP would
    mostly be explaining noise). See write-up for reasoning.
    """
    importances = pd.Series(est.feature_importances_, index=COVARIATES)
    return importances.sort_values(ascending=False)


def plot_feature_importance(importances, save_path="reports/figures/feature_importance.png"):
    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh", color="steelblue")
    plt.xlabel("Importance (share of variance in treatment effect explained)")
    plt.title("What Drives Heterogeneity in Discount Response?")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


# ============================================================
# CATE distribution + reliability check
# ============================================================

def summarize_cate_distribution(df_real):
    n_negative = (df_real["predicted_cate"] < 0).sum()
    n_positive = (df_real["predicted_cate"] > 0).sum()
    return {
        "describe": df_real["predicted_cate"].describe(),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "pct_positive": 100 * n_positive / len(df_real),
        "pct_negative": 100 * n_negative / len(df_real),
    }


def plot_cate_distribution(df_real, save_path="reports/figures/real_cate_distribution.png"):
    plt.figure(figsize=(8, 5))
    plt.hist(df_real["predicted_cate"], bins=50, color="steelblue", edgecolor="white")
    plt.axvline(0, color="red", linestyle="--", label="Zero effect")
    plt.xlabel("Predicted Individual Treatment Effect (CATE)")
    plt.ylabel("Number of Customers")
    plt.title("Distribution of Predicted Uplift — Real Starbucks Customers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def summarize_cate_reliability(df_real):
    """
    Checks what fraction of individual CATE estimates have a 95% CI
    that excludes zero (i.e. are statistically distinguishable from
    no effect). This is the check that revealed only ~7% of real-data
    CATEs are individually reliable.
    """
    ci_excludes_zero = (df_real["cate_ci_lower"] > 0) | (df_real["cate_ci_upper"] < 0)
    ci_width = df_real["cate_ci_upper"] - df_real["cate_ci_lower"]

    return {
        "n_reliable": ci_excludes_zero.sum(),
        "pct_reliable": 100 * ci_excludes_zero.sum() / len(df_real),
        "n_unreliable": (~ci_excludes_zero).sum(),
        "pct_unreliable": 100 * (~ci_excludes_zero).mean(),
        "mean_ci_width": ci_width.mean(),
        "median_ci_width": ci_width.median(),
    }


# ============================================================
# Subgroup analysis
# ============================================================

def add_subgroup_buckets(df_real):
    df_real = df_real.copy()
    df_real["tenure_bucket"] = pd.cut(
        df_real["tenure_days"], bins=[0, 180, 365, 730, 2000],
        labels=["<6mo", "6-12mo", "1-2yr", "2yr+"]
    )
    df_real["income_bucket"] = pd.cut(
        df_real["income"], bins=[0, 50000, 70000, 90000, 200000],
        labels=["<50k", "50-70k", "70-90k", "90k+"]
    )
    return df_real


def summarize_subgroup_cate(df_real, bucket_col):
    return df_real.groupby(bucket_col, observed=True)["predicted_cate"].agg(["mean", "std", "count"])


def test_high_income_effect(df_real):
    """Formal t-test: is the 90k+ income bucket's CATE meaningfully different from the rest?"""
    high_income = df_real.loc[df_real["income_bucket"] == "90k+", "predicted_cate"]
    rest = df_real.loc[df_real["income_bucket"] != "90k+", "predicted_cate"]
    t_stat, p_val = stats.ttest_ind(high_income, rest, equal_var=False)
    return {"t_stat": t_stat, "p_value": p_val}


# ============================================================
# Decile analysis (used for the Qini curve validation)
# ============================================================

def add_cate_deciles(df_real):
    df_real = df_real.copy()
    df_real["cate_decile"] = pd.qcut(df_real["predicted_cate"], 10, labels=False, duplicates="drop")
    return df_real


def summarize_deciles(df_real):
    decile_summary = df_real.groupby("cate_decile").agg(
        mean_predicted_cate=("predicted_cate", "mean"),
        mean_actual_outcome=("outcome", "mean"),
        n_customers=("predicted_cate", "count"),
        treatment_rate=("treatment", "mean")
    )

    def realized_uplift(g):
        return g.loc[g["treatment"] == 1, "outcome"].mean() - g.loc[g["treatment"] == 0, "outcome"].mean()

    decile_realized_uplift = df_real.groupby("cate_decile").apply(realized_uplift, include_groups=False)
    decile_realized_uplift.name = "realized_uplift"

    decile_n_control = df_real.groupby("cate_decile").apply(
        lambda g: (g["treatment"] == 0).sum(), include_groups=False
    )
    decile_n_control.name = "n_control"

    return pd.concat([decile_summary, decile_realized_uplift, decile_n_control], axis=1)


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":

    # --- Part 1: Synthetic re-validation ---
    df_syn, est_syn, syn_metrics = run_synthetic_validation()
    print("=== Synthetic Re-validation (discrete_treatment=True) ===")
    print(f"CausalForestDML ATE: {syn_metrics['ate']:.2f}  (LinearDML was 68.62, true ATE 68.75)")
    print(f"Correlation (predicted CATE vs true uplift): {syn_metrics['correlation']:.3f}")
    print(f"RMSE: {syn_metrics['rmse']:.2f}")
    df_syn.to_csv("data/synthetic/synthetic_customers_with_cate.csv", index=False)

    # --- Part 2: Real data ---
    df_real, est_real, X_real = fit_real_causal_forest()

    # Feature importance
    importances = compute_feature_importance(est_real)
    print("\n=== CausalForestDML Feature Importances (heterogeneity drivers) ===")
    print(importances)
    plot_feature_importance(importances)

    # ATE
    ate_summary = get_ate_summary(est_real, X_real)
    print("\n=== CausalForestDML on Real Starbucks Data ===")
    print(f"CausalForestDML ATE: {ate_summary['ate']:.2f}  (LinearDML was 7.20, naive was 10.73)")
    print(f"CausalForestDML ATE 95% CI: ({ate_summary['ci_lower']:.2f}, {ate_summary['ci_upper']:.2f})")

    # CATE distribution
    cate_dist = summarize_cate_distribution(df_real)
    print(f"\nCATE distribution summary:")
    print(cate_dist["describe"])
    print(f"\nCustomers with positive predicted uplift: {cate_dist['n_positive']} ({cate_dist['pct_positive']:.1f}%)")
    print(f"Customers with negative predicted uplift:  {cate_dist['n_negative']} ({cate_dist['pct_negative']:.1f}%)")
    plot_cate_distribution(df_real)

    df_real.to_csv("data/processed/processed_starbucks_with_cate.csv", index=False)
    print("\nSaved data/processed/processed_starbucks_with_cate.csv")

    # Reliability check
    reliability = summarize_cate_reliability(df_real)
    print(f"\nCustomers with CATE CI excluding zero: {reliability['n_reliable']} ({reliability['pct_reliable']:.1f}%)")
    print(f"Customers with CATE CI straddling zero: {reliability['n_unreliable']} ({reliability['pct_unreliable']:.1f}%)")
    print(f"\nMean CATE CI width: {reliability['mean_ci_width']:.2f}")
    print(f"Median CATE CI width: {reliability['median_ci_width']:.2f}")

    # Subgroup analysis
    df_real = add_subgroup_buckets(df_real)

    print("\n=== Average CATE by tenure bucket ===")
    print(summarize_subgroup_cate(df_real, "tenure_bucket"))

    print("\n=== Average CATE by income bucket ===")
    print(summarize_subgroup_cate(df_real, "income_bucket"))

    income_test = test_high_income_effect(df_real)
    print(f"\n90k+ vs. rest: t={income_test['t_stat']:.2f}, p={income_test['p_value']:.6f}")

    # Decile analysis
    df_real = add_cate_deciles(df_real)
    decile_check = summarize_deciles(df_real)
    print(f"\n=== Decile Check: Predicted CATE vs Realized Uplift ===")
    print(decile_check)

    df_real.to_csv("data/processed/processed_starbucks_with_cate.csv", index=False)

    # Correlation check (explains why avg_spend outranks income in importance)
    print("\n=== Covariate correlations (explains feature importance ranking) ===")
    print(df_real[["avg_spend", "income", "total_transactions"]].corr())