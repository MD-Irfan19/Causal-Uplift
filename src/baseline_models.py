import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

COVARIATES = ["age", "income", "tenure_days", "avg_spend", "purchase_frequency", "total_transactions"]


# ============================================================
# Propensity score model
# ============================================================

def fit_propensity_model(df, covariates=COVARIATES):
    """
    Fits P(treatment=1 | covariates) via logistic regression. Returns
    the fitted model and the dataframe with a propensity_score column
    attached.
    """
    X = df[covariates].values
    T = df["treatment"].values

    model = LogisticRegression(max_iter=1000)
    model.fit(X, T)

    df = df.copy()
    df["propensity_score"] = model.predict_proba(X)[:, 1]
    return model, df


def summarize_propensity_overlap(df):
    """
    Flags customers with poor treatment/control overlap (propensity
    near 0 or 1) -- causal effects for these customers are unreliable
    since there's no comparable counterfactual group.
    """
    poor_overlap = ((df["propensity_score"] < 0.05) | (df["propensity_score"] > 0.95)).sum()

    return {
        "n_poor_overlap": poor_overlap,
        "pct_poor_overlap": 100 * poor_overlap / len(df),
        "by_group": df.groupby("treatment")["propensity_score"].describe(),
    }


def compute_propensity_auc(df):
    """
    AUC of the propensity model. Close to 0.5 means the covariates
    barely predict treatment assignment (offer assignment close to
    random); notably higher suggests real targeting/confounding on
    these covariates.
    """
    return roc_auc_score(df["treatment"], df["propensity_score"])


def plot_propensity_overlap(df, save_path="reports/figures/propensity_overlap.png"):
    plt.figure(figsize=(8, 5))
    plt.hist(df.loc[df["treatment"] == 1, "propensity_score"], bins=40, alpha=0.5, label="Treated", density=True)
    plt.hist(df.loc[df["treatment"] == 0, "propensity_score"], bins=40, alpha=0.5, label="Control", density=True)
    plt.xlabel("Propensity Score (P(treated | covariates))")
    plt.ylabel("Density")
    plt.title("Propensity Score Overlap: Treated vs Control")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


# ============================================================
# Naive baseline
# ============================================================

def compute_naive_baseline(df):
    """
    Unadjusted treated-vs-control comparison: the "wrong answer"
    anchor that LinearDML/CausalForestDML estimates get compared
    against later.
    """
    treated_outcome = df.loc[df["treatment"] == 1, "outcome"]
    control_outcome = df.loc[df["treatment"] == 0, "outcome"]

    naive_effect = treated_outcome.mean() - control_outcome.mean()

    t_stat, p_value = stats.ttest_ind(treated_outcome, control_outcome, equal_var=False)

    se_diff = np.sqrt(
        treated_outcome.var(ddof=1) / len(treated_outcome)
        + control_outcome.var(ddof=1) / len(control_outcome)
    )
    ci_low = naive_effect - 1.96 * se_diff
    ci_high = naive_effect + 1.96 * se_diff

    return {
        "mean_treated": treated_outcome.mean(),
        "mean_control": control_outcome.mean(),
        "naive_effect": naive_effect,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def compute_covariate_balance(df, covariates=COVARIATES):
    """
    Mean covariate values by treatment group, plus the raw difference.
    Small differences here are consistent with a weak propensity AUC
    (little confounding on observed covariates); large differences
    would signal real imbalance to correct for.
    """
    balance = df.groupby("treatment")[covariates].mean().T
    balance["difference"] = balance[1] - balance[0]
    return balance


# ============================================================
# Script entry point 
# ============================================================

if __name__ == "__main__":
    df = pd.read_csv("data/processed/processed_starbucks.csv")

    # Propensity overlap
    prop_model, df = fit_propensity_model(df)
    plot_propensity_overlap(df)

    overlap = summarize_propensity_overlap(df)
    print(f"Customers with poor overlap (propensity < 0.05 or > 0.95): "
          f"{overlap['n_poor_overlap']} ({overlap['pct_poor_overlap']:.1f}%)")
    print(f"\nPropensity score summary by group:")
    print(overlap["by_group"])

    auc = compute_propensity_auc(df)
    print(f"\nPropensity model AUC: {auc:.3f}  (0.5 = no predictive power, 1.0 = perfect separation)")

    # Naive baseline
    naive = compute_naive_baseline(df)
    print(f"\n--- Naive Baseline ---")
    print(f"Mean outcome, treated:   {naive['mean_treated']:.2f}")
    print(f"Mean outcome, control:   {naive['mean_control']:.2f}")
    print(f"Naive effect estimate:   {naive['naive_effect']:.2f}")
    print(f"95% CI:                  ({naive['ci_low']:.2f}, {naive['ci_high']:.2f})")
    print(f"t-statistic:              {naive['t_stat']:.3f}")
    print(f"p-value:                  {naive['p_value']:.4f}")

    # Covariate balance
    balance = compute_covariate_balance(df)
    print(f"\n--- Covariate Balance (treated vs control) ---")
    print(balance)