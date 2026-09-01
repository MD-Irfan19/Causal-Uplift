import pandas as pd


# ============================================================
# Column Detection
# ============================================================

def _find_column(df, candidates):

    for column in candidates:
        if column in df.columns:
            return column

    return None


# ============================================================
# Scenario Normalization
# ============================================================

def _normalize_scenarios(scenarios):

    if scenarios is None or scenarios.empty:
        raise ValueError(
            "Scenario dataframe cannot be empty."
        )

    df = scenarios.copy()

    # --------------------------------------------------------
    # Targeting percentage
    # --------------------------------------------------------

    pct_column = _find_column(
        df,
        [
            "pct_targeted",
            "target_fraction",
            "target_percentage",
            "targeting_fraction",
            "targeting_percentage"
        ]
    )

    if pct_column is None:
        raise ValueError(
            "Could not find targeting percentage column. "
            f"Available columns: {list(df.columns)}"
        )

    df["pct_targeted"] = df[pct_column]

    if df["pct_targeted"].max() > 1:
        df["pct_targeted"] = (
            df["pct_targeted"] / 100
        )

    # --------------------------------------------------------
    # Number of targeted customers
    # --------------------------------------------------------

    n_column = _find_column(
        df,
        [
            "n_targeted",
            "targeted_customers",
            "customers_targeted",
            "n_customers"
        ]
    )

    if n_column is not None:

        df["n_targeted"] = df[n_column]

    else:

        total_column = _find_column(
            df,
            [
                "total_customers",
                "total_customer_count"
            ]
        )

        if total_column is not None:

            df["n_targeted"] = (
                df["pct_targeted"] *
                df[total_column]
            ).astype(int)

        else:
            raise ValueError(
                "Could not determine number of targeted customers."
            )

    # --------------------------------------------------------
    # Captured uplift
    # --------------------------------------------------------

    uplift_column = _find_column(
        df,
        [
            "captured_uplift",
            "uplift",
            "total_uplift",
            "predicted_uplift"
        ]
    )

    if uplift_column is None:
        raise ValueError(
            "Could not find captured uplift column."
        )

    df["captured_uplift"] = df[uplift_column]

    # --------------------------------------------------------
    # Uplift captured percentage
    # --------------------------------------------------------

    captured_pct_column = _find_column(
        df,
        [
            "pct_uplift_captured",
            "uplift_captured",
            "uplift_captured_pct",
            "uplift_percentage"
        ]
    )

    if captured_pct_column is not None:

        df["pct_uplift_captured"] = (
            df[captured_pct_column]
        )

        if df["pct_uplift_captured"].max() > 1:

            df["pct_uplift_captured"] = (
                df["pct_uplift_captured"] / 100
            )

    else:

        max_uplift = df["captured_uplift"].max()

        if max_uplift > 0:

            df["pct_uplift_captured"] = (
                df["captured_uplift"] /
                max_uplift
            )

        else:

            df["pct_uplift_captured"] = 0.0

    # --------------------------------------------------------
    # Estimated cost
    # --------------------------------------------------------

    cost_column = _find_column(
        df,
        [
            "estimated_cost",
            "cost",
            "campaign_cost",
            "actual_cost"
        ]
    )

    if cost_column is None:
        raise ValueError(
            "Could not find estimated cost column."
        )

    df["estimated_cost"] = df[cost_column]

    # --------------------------------------------------------
    # Cost saved
    # --------------------------------------------------------

    saved_column = _find_column(
        df,
        [
            "cost_saved_vs_current",
            "cost_saved",
            "cost_savings"
        ]
    )

    if saved_column is not None:

        df["cost_saved_vs_current"] = (
            df[saved_column]
        )

    else:

        df["cost_saved_vs_current"] = 0.0

    # --------------------------------------------------------
    # Cost saved percentage
    # --------------------------------------------------------

    saved_pct_column = _find_column(
        df,
        [
            "pct_cost_saved_vs_current",
            "pct_cost_saved",
            "cost_saved_percentage"
        ]
    )

    if saved_pct_column is not None:

        df["pct_cost_saved_vs_current"] = (
            df[saved_pct_column]
        )

        if (
            df["pct_cost_saved_vs_current"]
            .abs()
            .max()
            > 1
        ):

            df["pct_cost_saved_vs_current"] = (
                df["pct_cost_saved_vs_current"] / 100
            )

    else:

        df["pct_cost_saved_vs_current"] = 0.0

    return df


# ============================================================
# Validation
# ============================================================

def validate_scenarios(scenarios):

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
            f"Missing normalized scenario columns: {missing}"
        )


# ============================================================
# Budget Filtering
# ============================================================

def filter_by_budget(
    scenarios,
    budget
):

    normalized = _normalize_scenarios(
        scenarios
    )

    validate_scenarios(
        normalized
    )

    if budget < 0:
        raise ValueError(
            "Budget cannot be negative."
        )

    feasible = normalized[
        normalized["estimated_cost"] <= budget
    ].copy()

    return feasible


# ============================================================
# Best Policy
# ============================================================

def select_best_policy(
    scenarios,
    budget=None
):

    normalized = _normalize_scenarios(
        scenarios
    )

    validate_scenarios(
        normalized
    )

    candidates = normalized.copy()

    if budget is not None:

        candidates = candidates[
            candidates["estimated_cost"] <= budget
        ]

    if candidates.empty:
        return None

    best = candidates.loc[
        candidates["captured_uplift"].idxmax()
    ].copy()

    return best


# ============================================================
# Efficiency
# ============================================================

def calculate_policy_efficiency(row):

    cost = row["estimated_cost"]

    if cost <= 0:
        return 0.0

    return (
        row["captured_uplift"] /
        cost
    )


def add_efficiency_metric(
    scenarios
):

    normalized = _normalize_scenarios(
        scenarios
    )

    validate_scenarios(
        normalized
    )

    normalized["uplift_per_dollar"] = (
        normalized.apply(
            calculate_policy_efficiency,
            axis=1
        )
    )

    return normalized


# ============================================================
# Phase 9.10
# Granular Policy Optimization
# ============================================================

def generate_granular_policies(
    df,
    cost_per_customer,
    baseline_fraction,
    step=0.01
):
    """
    Generate policies at fine-grained targeting levels.

    Example with step=0.01:

        0%, 1%, 2%, ..., 100%

    Customers are ranked by predicted CATE and the top
    customers are selected for each targeting percentage.
    """

    if df is None or df.empty:
        raise ValueError(
            "Customer dataframe cannot be empty."
        )

    if cost_per_customer < 0:
        raise ValueError(
            "Cost per customer cannot be negative."
        )

    if not 0 < step <= 1:
        raise ValueError(
            "step must be greater than 0 and <= 1."
        )

    if "predicted_cate" not in df.columns:
        raise ValueError(
            "Dataframe must contain 'predicted_cate'."
        )

    if "treatment" not in df.columns:
        raise ValueError(
            "Dataframe must contain 'treatment'."
        )

    # --------------------------------------------------------
    # Sort customers by predicted CATE
    # --------------------------------------------------------

    sorted_df = (
        df.sort_values(
            "predicted_cate",
            ascending=False
        )
        .reset_index(drop=True)
    )

    total_customers = len(sorted_df)

    # --------------------------------------------------------
    # Current policy information
    # --------------------------------------------------------

    current_targeted = int(
        sorted_df["treatment"].sum()
    )

    current_cost = (
        current_targeted *
        cost_per_customer
    )

    # --------------------------------------------------------
    # Theoretical positive-CATE ceiling
    # --------------------------------------------------------

    positive_cate_ceiling = (
        sorted_df.loc[
            sorted_df["predicted_cate"] > 0,
            "predicted_cate"
        ].sum()
    )

    # --------------------------------------------------------
    # Generate targeting levels
    # --------------------------------------------------------

    targeting_levels = []

    current = 0.0

    while current <= 1.000001:

        targeting_levels.append(
            round(current, 4)
        )

        current += step

    # Make sure 100% is included.
    if targeting_levels[-1] != 1.0:

        targeting_levels.append(1.0)

    # --------------------------------------------------------
    # Evaluate every policy
    # --------------------------------------------------------

    policies = []

    for fraction in targeting_levels:

        n_targeted = int(
            total_customers *
            fraction
        )

        if n_targeted > 0:

            subset = sorted_df.iloc[
                :n_targeted
            ]

            captured_uplift = (
                subset["predicted_cate"].sum()
            )

        else:

            captured_uplift = 0.0

        estimated_cost = (
            n_targeted *
            cost_per_customer
        )

        cost_saved = (
            current_cost -
            estimated_cost
        )

        if positive_cate_ceiling != 0:

            pct_uplift_captured = (
                captured_uplift /
                positive_cate_ceiling
            )

        else:

            pct_uplift_captured = 0.0

        if current_cost != 0:

            pct_cost_saved = (
                cost_saved /
                current_cost
            )

        else:

            pct_cost_saved = 0.0

        if estimated_cost > 0:

            uplift_per_dollar = (
                captured_uplift /
                estimated_cost
            )

        else:

            uplift_per_dollar = 0.0

        policies.append(
            {
                "pct_targeted": fraction,

                "n_targeted": n_targeted,

                "captured_uplift":
                    captured_uplift,

                "pct_uplift_captured":
                    pct_uplift_captured,

                "estimated_cost":
                    estimated_cost,

                "cost_saved_vs_current":
                    cost_saved,

                "pct_cost_saved_vs_current":
                    pct_cost_saved,

                "uplift_per_dollar":
                    uplift_per_dollar
            }
        )

    return pd.DataFrame(
        policies
    )


# ============================================================
# Phase 9.10
# Optimize Under Budget
# ============================================================

def optimize_policy(
    df,
    budget,
    cost_per_customer,
    baseline_fraction,
    step=0.01
):
    """
    Find the highest-uplift targeting policy that satisfies
    the specified campaign budget.

    Returns a dictionary containing the optimal policy and
    the complete granular policy table.
    """

    if budget < 0:
        raise ValueError(
            "Budget cannot be negative."
        )

    policies = generate_granular_policies(
        df=df,
        cost_per_customer=cost_per_customer,
        baseline_fraction=baseline_fraction,
        step=step
    )

    feasible = policies[
        policies["estimated_cost"] <= budget
    ].copy()

    if feasible.empty:

        return {
            "status": "NO_FEASIBLE_POLICY",
            "message": (
                "No targeting policy can be executed "
                "within the selected budget."
            ),
            "optimal_policy": None,
            "policies": policies
        }

    # --------------------------------------------------------
    # Select policy with maximum predicted uplift
    # --------------------------------------------------------

    optimal = feasible.loc[
        feasible["captured_uplift"].idxmax()
    ].copy()

    return {
        "status": "OPTIMAL_POLICY_FOUND",

        "optimal_policy": optimal,

        "policies": policies,

        "message": (
            f"Optimal targeting level: "
            f"{optimal['pct_targeted']:.0%}. "
            f"This policy targets "
            f"{int(optimal['n_targeted']):,} customers "
            f"and captures "
            f"{optimal['pct_uplift_captured']:.2%} "
            f"of the theoretical positive-CATE uplift."
        )
    }


# ============================================================
# Automated Recommendation
# ============================================================

def generate_recommendation(
    scenarios,
    budget=None
):

    enriched = add_efficiency_metric(
        scenarios
    )

    best = select_best_policy(
        enriched,
        budget=budget
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

        "targeting_percentage":
            targeting_percentage,

        "targeted_customers":
            int(best["n_targeted"]),

        "captured_uplift":
            float(best["captured_uplift"]),

        "uplift_captured_percentage":
            float(
                best["pct_uplift_captured"] * 100
            ),

        "estimated_cost":
            float(best["estimated_cost"]),

        "cost_saved":
            float(best["cost_saved_vs_current"]),

        "cost_saved_percentage":
            float(
                best["pct_cost_saved_vs_current"] * 100
            ),

        "uplift_per_dollar":
            float(best["uplift_per_dollar"]),

        "message":
            message
    }


# ============================================================
# Policy Comparison
# ============================================================

def compare_policies(
    scenarios,
    budget=None
):

    enriched = add_efficiency_metric(
        scenarios
    )

    if budget is not None:

        enriched = enriched[
            enriched["estimated_cost"] <= budget
        ].copy()

    if enriched.empty:
        return enriched

    return enriched.sort_values(
        "captured_uplift",
        ascending=False
    ).reset_index(drop=True)