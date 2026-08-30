import numpy as np
import pandas as pd

from config import (
    CATE_COLUMN,
    MIN_TARGET_FRACTION,
    MAX_TARGET_FRACTION,
)


# ============================================================
# Validation
# ============================================================

def validate_target_fraction(target_fraction):
    """
    Validate the requested targeting fraction.

    Expected range:
        0.0 = 0%
        0.4 = 40%
        1.0 = 100%
    """

    if not (
        MIN_TARGET_FRACTION
        <= target_fraction
        <= MAX_TARGET_FRACTION
    ):
        raise ValueError(
            "target_fraction must be between 0.0 and 1.0."
        )


# ============================================================
# Customer Ranking
# ============================================================

def rank_customers(df):
    """
    Rank customers from highest to lowest predicted CATE.
    """

    if CATE_COLUMN not in df.columns:
        raise ValueError(
            f"Required column '{CATE_COLUMN}' "
            "not found in dataset."
        )

    return (
        df.sort_values(
            CATE_COLUMN,
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# Customer Selection
# ============================================================

def select_top_customers(
    df_sorted,
    target_fraction
):
    """
    Select the top X% of customers according to predicted CATE.

    Example:
        target_fraction = 0.40
        → select top 40% of customers.
    """

    validate_target_fraction(target_fraction)

    total_customers = len(df_sorted)

    n_targeted = int(
        total_customers * target_fraction
    )

    targeted = df_sorted.iloc[:n_targeted].copy()

    return targeted


# ============================================================
# Uplift Ceiling
# ============================================================

def calculate_positive_cate_ceiling(df):
    """
    Calculate the theoretical maximum achievable uplift.

    Only positive predicted CATE values are included.

    This follows the definition used in the original
    policy_sim.py.
    """

    if CATE_COLUMN not in df.columns:
        raise ValueError(
            f"Required column '{CATE_COLUMN}' "
            "not found in dataset."
        )

    positive_cate = df.loc[
        df[CATE_COLUMN] > 0,
        CATE_COLUMN
    ]

    return positive_cate.sum()


# ============================================================
# Policy Metrics
# ============================================================

def calculate_policy_metrics(
    targeted,
    df,
    target_fraction,
    baseline_fraction,
    cost_per_customer
):
    """
    Calculate the performance of a simulated targeting policy.

    Metrics:
        - Number targeted
        - Captured uplift
        - % uplift captured
        - Estimated cost
        - Cost saved vs current policy
        - % cost saved vs current policy
    """

    total_customers = len(df)

    n_targeted = len(targeted)

    # --------------------------------------------------------
    # Estimated incremental uplift
    # --------------------------------------------------------

    captured_uplift = targeted[
        CATE_COLUMN
    ].sum()

    # --------------------------------------------------------
    # Theoretical maximum uplift
    # --------------------------------------------------------

    ceiling = calculate_positive_cate_ceiling(df)

    if ceiling != 0:
        pct_uplift_captured = (
            captured_uplift / ceiling
        )
    else:
        pct_uplift_captured = np.nan

    # --------------------------------------------------------
    # Simulated policy cost
    # --------------------------------------------------------

    estimated_cost = (
        n_targeted * cost_per_customer
    )

    # --------------------------------------------------------
    # Baseline/current policy cost
    # --------------------------------------------------------

    baseline_targeted = int(
        total_customers * baseline_fraction
    )

    baseline_cost = (
        baseline_targeted * cost_per_customer
    )

    # --------------------------------------------------------
    # Cost savings
    # --------------------------------------------------------

    cost_saved = (
        baseline_cost - estimated_cost
    )

    if baseline_cost != 0:
        pct_cost_saved = (
            cost_saved / baseline_cost
        )
    else:
        pct_cost_saved = np.nan

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "target_fraction": target_fraction,
        "n_targeted": n_targeted,
        "captured_uplift": captured_uplift,
        "pct_uplift_captured": pct_uplift_captured,
        "estimated_cost": estimated_cost,
        "cost_saved": cost_saved,
        "pct_cost_saved": pct_cost_saved,
    }


# ============================================================
# Main Policy Simulation Function
# ============================================================

def simulate_policy(
    df,
    target_fraction,
    baseline_fraction,
    cost_per_customer
):
    """
    Simulate a targeting policy based on predicted CATE.

    The function:
        1. Ranks customers by predicted CATE.
        2. Selects the top X%.
        3. Calculates uplift and cost.
        4. Compares the simulated policy against
           the current/baseline targeting level.
    """

    # --------------------------------------------------------
    # Rank customers
    # --------------------------------------------------------

    df_sorted = rank_customers(df)

    # --------------------------------------------------------
    # Select customers
    # --------------------------------------------------------

    targeted = select_top_customers(
        df_sorted,
        target_fraction
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    results = calculate_policy_metrics(
        targeted=targeted,
        df=df_sorted,
        target_fraction=target_fraction,
        baseline_fraction=baseline_fraction,
        cost_per_customer=cost_per_customer
    )

    return results

# ============================================================
# Policy Scenario Generator
# ============================================================

def generate_policy_scenarios(
    df,
    baseline_fraction,
    cost_per_customer,
    step=0.10
):
    """
    Generate policy results for multiple targeting percentages.

    By default, evaluates:
        0%, 10%, 20%, ..., 100%

    Parameters
    ----------
    df : pandas.DataFrame
        Customer dataset containing predicted CATE values.

    baseline_fraction : float
        Actual/current targeting fraction.

    cost_per_customer : float
        Average cost of targeting one customer.

    step : float
        Difference between consecutive targeting percentages.
        Default = 0.10 (10 percentage points).

    Returns
    -------
    pandas.DataFrame
        One row per simulated targeting policy.
    """

    if step <= 0 or step > 1:
        raise ValueError(
            "step must be greater than 0 and no greater than 1."
        )

    scenarios = []

    # Generate targeting fractions.
    # np.arange can sometimes produce floating-point
    # values such as 0.30000000004, so round them.
    fractions = np.arange(
        0.0,
        1.0 + step,
        step
    )

    fractions = [
        round(float(fraction), 10)
        for fraction in fractions
        if fraction <= 1.0
    ]

    for target_fraction in fractions:

        result = simulate_policy(
            df=df,
            target_fraction=target_fraction,
            baseline_fraction=baseline_fraction,
            cost_per_customer=cost_per_customer
        )

        scenarios.append(result)

    return pd.DataFrame(scenarios)