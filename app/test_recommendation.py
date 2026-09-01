import pandas as pd

from engine.recommendation import (
    add_efficiency_metric,
    select_best_policy,
    generate_recommendation,
    compare_policies
)


print()
print("=" * 65)
print("RECOMMENDATION ENGINE TEST")
print("=" * 65)


# ============================================================
# Test Data
# ============================================================

scenarios = pd.DataFrame([
    {
        "pct_targeted": 0.10,
        "n_targeted": 1482,
        "captured_uplift": 8119.42,
        "pct_uplift_captured": 0.7830,
        "estimated_cost": 4446.00,
        "cost_saved_vs_current": 34680.00,
        "pct_cost_saved_vs_current": 0.8864
    },
    {
        "pct_targeted": 0.20,
        "n_targeted": 2965,
        "captured_uplift": 9195.51,
        "pct_uplift_captured": 0.8868,
        "estimated_cost": 8895.00,
        "cost_saved_vs_current": 30231.00,
        "pct_cost_saved_vs_current": 0.7727
    },
    {
        "pct_targeted": 0.30,
        "n_targeted": 4447,
        "captured_uplift": 9785.64,
        "pct_uplift_captured": 0.9437,
        "estimated_cost": 13341.00,
        "cost_saved_vs_current": 25785.00,
        "pct_cost_saved_vs_current": 0.6590
    },
    {
        "pct_targeted": 0.40,
        "n_targeted": 5930,
        "captured_uplift": 10113.41,
        "pct_uplift_captured": 0.9753,
        "estimated_cost": 17790.00,
        "cost_saved_vs_current": 21336.00,
        "pct_cost_saved_vs_current": 0.5453
    },
    {
        "pct_targeted": 0.50,
        "n_targeted": 7412,
        "captured_uplift": 10294.33,
        "pct_uplift_captured": 0.9927,
        "estimated_cost": 22236.00,
        "cost_saved_vs_current": 16890.00,
        "pct_cost_saved_vs_current": 0.4317
    },
    {
        "pct_targeted": 0.60,
        "n_targeted": 8895,
        "captured_uplift": 10368.48,
        "pct_uplift_captured": 0.9999,
        "estimated_cost": 26685.00,
        "cost_saved_vs_current": 12441.00,
        "pct_cost_saved_vs_current": 0.3180
    }
])


# ============================================================
# TEST 1 — Efficiency
# ============================================================

print()
print("-" * 65)
print("TEST 1 — POLICY EFFICIENCY")
print("-" * 65)

enriched = add_efficiency_metric(
    scenarios
)

print(
    enriched[
        [
            "pct_targeted",
            "captured_uplift",
            "estimated_cost",
            "uplift_per_dollar"
        ]
    ].to_string(index=False)
)


# ============================================================
# TEST 2 — Best Overall Policy
# ============================================================

print()
print("-" * 65)
print("TEST 2 — BEST OVERALL POLICY")
print("-" * 65)

best = select_best_policy(
    scenarios
)

print(
    f"Targeting percentage : "
    f"{best['pct_targeted']:.0%}"
)

print(
    f"Customers targeted   : "
    f"{int(best['n_targeted']):,}"
)

print(
    f"Captured uplift      : "
    f"{best['captured_uplift']:,.2f}"
)

print(
    f"Estimated cost       : "
    f"${best['estimated_cost']:,.2f}"
)


# ============================================================
# TEST 3 — Budget-Constrained Recommendation
# ============================================================

print()
print("-" * 65)
print("TEST 3 — $15,000 BUDGET")
print("-" * 65)

recommendation = generate_recommendation(
    scenarios,
    budget=15000
)

print(
    recommendation["message"]
)

print(
    f"Uplift per dollar    : "
    f"{recommendation['uplift_per_dollar']:.4f}"
)


# ============================================================
# TEST 4 — Policy Ranking
# ============================================================

print()
print("-" * 65)
print("TEST 4 — POLICY RANKING")
print("-" * 65)

comparison = compare_policies(
    scenarios,
    budget=20000
)

print(
    comparison[
        [
            "pct_targeted",
            "captured_uplift",
            "estimated_cost",
            "uplift_per_dollar"
        ]
    ].to_string(index=False)
)


# ============================================================
# Final
# ============================================================

print()
print("=" * 65)
print("RECOMMENDATION ENGINE TEST COMPLETE")
print("=" * 65)