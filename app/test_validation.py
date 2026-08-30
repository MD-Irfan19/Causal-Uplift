import math
import pandas as pd

from data.loader import (
    load_customer_data,
    load_portfolio_data,
    get_avg_discount_reward
)

from engine.policy import (
    simulate_policy,
    calculate_positive_cate_ceiling
)


# ============================================================
# Configuration
# ============================================================

PROCESSED_PATH = (
    "data/processed/processed_starbucks_with_cate.csv"
)

PORTFOLIO_PATH = (
    "data/raw/portfolio.json"
)

TOLERANCE = 1e-6


# ============================================================
# Helper
# ============================================================

def assert_close(name, original, new):
    """
    Compare two numerical values with a small tolerance.
    """

    if math.isclose(
        original,
        new,
        rel_tol=TOLERANCE,
        abs_tol=TOLERANCE
    ):
        print(f"PASS  {name}")
        print(f"      Original : {original:.6f}")
        print(f"      New      : {new:.6f}")
        print()

    else:
        print(f"FAIL  {name}")
        print(f"      Original : {original:.6f}")
        print(f"      New      : {new:.6f}")
        print(
            f"      Difference: "
            f"{abs(original - new):.6f}"
        )
        print()

        raise AssertionError(
            f"{name} does not match."
        )


# ============================================================
# Load data
# ============================================================

df = load_customer_data(PROCESSED_PATH)

portfolio_df = load_portfolio_data(PORTFOLIO_PATH)


# ============================================================
# Basic dataset information
# ============================================================

total_customers = len(df)

current_targeted = df["treatment"].sum()

current_fraction = (
    current_targeted / total_customers
)

avg_reward = get_avg_discount_reward(
    portfolio_df
)


# ============================================================
# Calculate original-policy intermediate values
# ============================================================

df_sorted = (
    df.sort_values(
        "predicted_cate",
        ascending=False
    )
    .reset_index(drop=True)
)

original_ceiling = (
    df_sorted.loc[
        df_sorted["predicted_cate"] > 0,
        "predicted_cate"
    ].sum()
)


# ============================================================
# Calculate same value using new engine
# ============================================================

new_ceiling = calculate_positive_cate_ceiling(
    df_sorted
)


# ============================================================
# Header
# ============================================================

print()
print("=" * 60)
print("PHASE 9.4 — POLICY ENGINE VALIDATION")
print("=" * 60)

print(
    f"Total customers       : {total_customers:,}"
)

print(
    f"Current targeted      : {current_targeted:,}"
)

print(
    f"Current targeting     : {current_fraction:.2%}"
)

print(
    f"Average discount cost : ${avg_reward:.2f}"
)

print()


# ============================================================
# Test 1 — Positive CATE ceiling
# ============================================================

print("-" * 60)
print("TEST 1 — POSITIVE CATE CEILING")
print("-" * 60)

assert_close(
    "Positive CATE ceiling",
    original_ceiling,
    new_ceiling
)


# ============================================================
# Test 2 — 40% policy
# ============================================================

print("-" * 60)
print("TEST 2 — 40% TARGETING POLICY")
print("-" * 60)

original_40 = None

n_40 = int(
    total_customers * 0.40
)

subset_40 = df_sorted.iloc[:n_40]

original_40_uplift = (
    subset_40["predicted_cate"].sum()
)

original_40_uplift_pct = (
    original_40_uplift / original_ceiling
)

original_40_cost = (
    n_40 * avg_reward
)

original_current_cost = (
    current_targeted * avg_reward
)

original_40_cost_saved = (
    original_current_cost - original_40_cost
)

original_40_cost_saved_pct = (
    original_40_cost_saved /
    original_current_cost
)


new_40 = simulate_policy(
    df=df,
    target_fraction=0.40,
    baseline_fraction=current_fraction,
    cost_per_customer=avg_reward
)


assert_close(
    "40% captured uplift",
    original_40_uplift,
    new_40["captured_uplift"]
)

assert_close(
    "40% uplift captured %",
    original_40_uplift_pct,
    new_40["pct_uplift_captured"]
)

assert_close(
    "40% estimated cost",
    original_40_cost,
    new_40["estimated_cost"]
)

assert_close(
    "40% cost saved",
    original_40_cost_saved,
    new_40["cost_saved"]
)

assert_close(
    "40% cost saved %",
    original_40_cost_saved_pct,
    new_40["pct_cost_saved"]
)


# ============================================================
# Test 3 — 20% policy
# ============================================================

print("-" * 60)
print("TEST 3 — 20% TARGETING POLICY")
print("-" * 60)

n_20 = int(
    total_customers * 0.20
)

subset_20 = df_sorted.iloc[:n_20]

original_20_uplift = (
    subset_20["predicted_cate"].sum()
)

original_20_uplift_pct = (
    original_20_uplift / original_ceiling
)

original_20_cost = (
    n_20 * avg_reward
)

original_20_cost_saved = (
    original_current_cost - original_20_cost
)

original_20_cost_saved_pct = (
    original_20_cost_saved /
    original_current_cost
)


new_20 = simulate_policy(
    df=df,
    target_fraction=0.20,
    baseline_fraction=current_fraction,
    cost_per_customer=avg_reward
)


assert_close(
    "20% captured uplift",
    original_20_uplift,
    new_20["captured_uplift"]
)

assert_close(
    "20% uplift captured %",
    original_20_uplift_pct,
    new_20["pct_uplift_captured"]
)

assert_close(
    "20% estimated cost",
    original_20_cost,
    new_20["estimated_cost"]
)

assert_close(
    "20% cost saved",
    original_20_cost_saved,
    new_20["cost_saved"]
)

assert_close(
    "20% cost saved %",
    original_20_cost_saved_pct,
    new_20["pct_cost_saved"]
)


# ============================================================
# Test 4 — 60% policy
# ============================================================

print("-" * 60)
print("TEST 4 — 60% TARGETING POLICY")
print("-" * 60)

n_60 = int(
    total_customers * 0.60
)

subset_60 = df_sorted.iloc[:n_60]

original_60_uplift = (
    subset_60["predicted_cate"].sum()
)

original_60_uplift_pct = (
    original_60_uplift / original_ceiling
)

original_60_cost = (
    n_60 * avg_reward
)

original_60_cost_saved = (
    original_current_cost - original_60_cost
)

original_60_cost_saved_pct = (
    original_60_cost_saved /
    original_current_cost
)


new_60 = simulate_policy(
    df=df,
    target_fraction=0.60,
    baseline_fraction=current_fraction,
    cost_per_customer=avg_reward
)


assert_close(
    "60% captured uplift",
    original_60_uplift,
    new_60["captured_uplift"]
)

assert_close(
    "60% uplift captured %",
    original_60_uplift_pct,
    new_60["pct_uplift_captured"]
)

assert_close(
    "60% estimated cost",
    original_60_cost,
    new_60["estimated_cost"]
)

assert_close(
    "60% cost saved",
    original_60_cost_saved,
    new_60["cost_saved"]
)

assert_close(
    "60% cost saved %",
    original_60_cost_saved_pct,
    new_60["pct_cost_saved"]
)


# ============================================================
# Final result
# ============================================================

print("=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)

print()
print("All validation tests passed successfully.")
print()
print(
    "The new app/engine/policy.py produces the same "
    "results as the original policy logic."
)
print()