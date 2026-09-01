import pandas as pd


# ============================================================
# Validation
# ============================================================

def validate_scenarios(scenarios):
    """
    Validate the scenario dataframe.
    """

    if scenarios is None or scenarios.empty:
        raise ValueError(
            "Scenario dataframe cannot be empty."
        )

    required_columns = [
        "pct_targeted",
        "n_targeted",
        "captured_uplift",
        "pct_uplift_captured",
        "estimated_cost",
        "cost_saved_vs_current",
        "pct_cost_saved_vs_current"
    ]

    missing = [
        column
        for column in required_columns
        if column not in scenarios.columns
    ]

    if missing:
        raise ValueError(
            f"Missing scenario columns: {missing}"
        )


# ============================================================
# Budget Filtering
# ============================================================

def filter_by_budget(
    scenarios,
    budget
):
    """
    Keep only policies whose estimated cost is within
    the available budget.
    """

    validate_scenarios(scenarios)

    if budget < 0:
        raise ValueError(
            "Budget cannot be negative."
        )

    feasible = scenarios[
        scenarios["estimated_cost"] <= budget
    ].copy()

    return feasible


# ============================================================
# Best Policy Selection
# ============================================================

def select_best_policy(
    scenarios,
    budget=None
):
    """
    Select the policy that captures the highest uplift.

    If a budget is supplied, only policies within that
    budget are considered.
    """

    validate_scenarios(scenarios)

    candidates = scenarios.copy()

    if budget is not None:
        candidates = filter_by_budget(
            candidates,
            budget
        )

    if candidates.empty:
        return None

    best = candidates.loc[
        candidates["captured_uplift"].idxmax()
    ].copy()

    return best


# ============================================================
# Efficiency Calculation
# ============================================================

def calculate_policy_efficiency(row):
    """
    Calculate uplift captured per dollar spent.
    """

    cost = row["estimated_cost"]

    if cost <= 0:
        return 0.0

    return (
        row["captured_uplift"] / cost
    )


# ============================================================
# Add Efficiency Metric
# ============================================================

def add_efficiency_metric(scenarios):
    """
    Add uplift-per-dollar efficiency to every policy.
    """

    validate_scenarios(scenarios)

    result = scenarios.copy()

    result["uplift_per_dollar"] = (
        result.apply(
            calculate_policy_efficiency,
            axis=1
        )
    )

    return result


# ============================================================
# Recommendation Generation
# ============================================================

def generate_recommendation(
    scenarios,
    budget=None
):
    """
    Generate a human-readable recommendation based on
    the best feasible policy.
    """

    validate_scenarios(scenarios)

    enriched = add_efficiency_metric(
        scenarios
    )

    best = select_best_policy(
        enriched,
        budget
    )

    if best is None:

        return {
            "status": "NO_FEASIBLE_POLICY",
            "message": (
                "No available targeting policy "
                "fits within the specified budget."
            )
        }

    targeting_percentage = (
        best["pct_targeted"] * 100
    )

    message = (
        f"Target approximately "
        f"{int(best['n_targeted']):,} customers "
        f"({targeting_percentage:.2f}% of the customer base). "
        f"This policy captures approximately "
        f"{best['pct_uplift_captured']:.2%} of the "
        f"theoretical positive-CATE uplift at an estimated "
        f"cost of ${best['estimated_cost']:,.2f}."
    )

    return {
        "status": "RECOMMENDED",
        "targeting_percentage": targeting_percentage,
        "targeted_customers": int(
            best["n_targeted"]
        ),
        "captured_uplift": (
            best["captured_uplift"]
        ),
        "uplift_captured_percentage": (
            best["pct_uplift_captured"] * 100
        ),
        "estimated_cost": (
            best["estimated_cost"]
        ),
        "cost_saved": (
            best["cost_saved_vs_current"]
        ),
        "cost_saved_percentage": (
            best["pct_cost_saved_vs_current"] * 100
        ),
        "uplift_per_dollar": (
            best["uplift_per_dollar"]
        ),
        "message": message
    }


# ============================================================
# Policy Comparison
# ============================================================

def compare_policies(
    scenarios,
    budget=None
):
    """
    Return all feasible policies ranked by captured uplift.
    """

    validate_scenarios(scenarios)

    enriched = add_efficiency_metric(
        scenarios
    )

    if budget is not None:

        enriched = filter_by_budget(
            enriched,
            budget
        )

    if enriched.empty:
        return enriched

    return enriched.sort_values(
        "captured_uplift",
        ascending=False
    ).reset_index(drop=True)