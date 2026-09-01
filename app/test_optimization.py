from data.loader import (
    load_customer_data,
    load_portfolio_data,
    get_avg_discount_reward
)

from engine.recommendation import (
    generate_granular_policies,
    optimize_policy
)


print()
print("=" * 60)
print("GRANULAR POLICY OPTIMIZATION")
print("=" * 60)


# ============================================================
# Load data
# ============================================================

df = load_customer_data()

portfolio_df = load_portfolio_data()

avg_reward = get_avg_discount_reward(
    portfolio_df
)


# ============================================================
# Current policy
# ============================================================

total_customers = len(df)

current_targeted = int(
    df["treatment"].sum()
)

current_fraction = (
    current_targeted /
    total_customers
)


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


# ============================================================
# Test 1 — Generate granular policies
# ============================================================

print()
print("-" * 60)
print("TEST 1 — GRANULAR POLICY GENERATION")
print("-" * 60)


policies = generate_granular_policies(
    df=df,
    cost_per_customer=avg_reward,
    baseline_fraction=current_fraction,
    step=0.01
)


print(
    f"Number of policies generated : {len(policies)}"
)

print(
    f"Minimum targeting level      : "
    f"{policies['pct_targeted'].min():.0%}"
)

print(
    f"Maximum targeting level      : "
    f"{policies['pct_targeted'].max():.0%}"
)


assert len(policies) == 101
assert policies["pct_targeted"].min() == 0.0
assert policies["pct_targeted"].max() == 1.0

print()
print("PASS — 101 granular policies generated")


# ============================================================
# Test 2 — 40% policy
# ============================================================

print()
print("-" * 60)
print("TEST 2 — 40% POLICY")
print("-" * 60)


policy_40 = policies[
    policies["pct_targeted"] == 0.40
].iloc[0]


print(
    f"Customers targeted : "
    f"{int(policy_40['n_targeted']):,}"
)

print(
    f"Captured uplift     : "
    f"{policy_40['captured_uplift']:,.2f}"
)

print(
    f"Estimated cost      : "
    f"${policy_40['estimated_cost']:,.2f}"
)


assert int(
    policy_40["n_targeted"]
) == int(
    total_customers * 0.40
)

print()
print("PASS — 40% policy")


# ============================================================
# Test 3 — Budget optimization
# ============================================================

print()
print("-" * 60)
print("TEST 3 — BUDGET OPTIMIZATION")
print("-" * 60)


budget = 15000.0


optimization = optimize_policy(
    df=df,
    budget=budget,
    cost_per_customer=avg_reward,
    baseline_fraction=current_fraction,
    step=0.01
)


assert (
    optimization["status"]
    == "OPTIMAL_POLICY_FOUND"
)


optimal = optimization[
    "optimal_policy"
]


print(
    f"Budget              : ${budget:,.2f}"
)

print(
    f"Optimal targeting   : "
    f"{optimal['pct_targeted']:.0%}"
)

print(
    f"Customers targeted  : "
    f"{int(optimal['n_targeted']):,}"
)

print(
    f"Captured uplift     : "
    f"{optimal['captured_uplift']:,.2f}"
)

print(
    f"Uplift captured     : "
    f"{optimal['pct_uplift_captured']:.2%}"
)

print(
    f"Estimated cost      : "
    f"${optimal['estimated_cost']:,.2f}"
)

print(
    f"Cost saved          : "
    f"${optimal['cost_saved_vs_current']:,.2f}"
)


assert (
    optimal["estimated_cost"]
    <= budget
)


print()
print(
    "PASS — Optimal policy is within budget"
)


# ============================================================
# Test 4 — Granularity check
# ============================================================

print()
print("-" * 60)
print("TEST 4 — 1% GRANULARITY CHECK")
print("-" * 60)


unique_steps = (
    policies["pct_targeted"]
    .diff()
    .dropna()
    .round(4)
    .unique()
)


print(
    "Unique targeting step values :",
    unique_steps
)


assert 0.01 in unique_steps

print()
print(
    "PASS — Policies are evaluated at 1% intervals"
)


# ============================================================
# Final
# ============================================================

print()
print(
    "Granular policy optimization is working correctly."
)

print()