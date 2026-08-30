from data.loader import (
    load_customer_data,
    load_portfolio_data,
    get_avg_discount_reward
)

from engine.policy import simulate_policy


# ============================================================
# Load data
# ============================================================

df = load_customer_data()

portfolio_df = load_portfolio_data()


# ============================================================
# Calculate actual baseline targeting
# ============================================================

current_targeted = df["treatment"].sum()

total_customers = len(df)

current_fraction = (
    current_targeted / total_customers
)


# ============================================================
# Calculate cost
# ============================================================

avg_reward = get_avg_discount_reward(
    portfolio_df
)


# ============================================================
# Run 40% policy
# ============================================================

result = simulate_policy(
    df=df,
    target_fraction=0.40,
    baseline_fraction=current_fraction,
    cost_per_customer=avg_reward
)


# ============================================================
# Print results
# ============================================================

print("\n========================================")
print("WHAT-IF POLICY SIMULATOR")
print("========================================")

print(f"Total customers       : {total_customers:,}")

print(
    f"Current targeted      : "
    f"{current_targeted:,}"
)

print(
    f"Current targeting     : "
    f"{current_fraction:.2%}"
)

print(
    f"Average discount cost : "
    f"${avg_reward:.2f}"
)

print("\n--- Simulated Policy ---")

print(
    f"Targeting percentage  : "
    f"{result['target_fraction']:.2%}"
)

print(
    f"Customers targeted    : "
    f"{result['n_targeted']:,}"
)

print(
    f"Captured uplift       : "
    f"{result['captured_uplift']:,.2f}"
)

print(
    f"Uplift captured       : "
    f"{result['pct_uplift_captured']:.2%}"
)

print(
    f"Estimated cost        : "
    f"${result['estimated_cost']:,.2f}"
)

print(
    f"Cost saved            : "
    f"${result['cost_saved']:,.2f}"
)

print(
    f"Cost saved (%)        : "
    f"{result['pct_cost_saved']:.2%}"
)

print("========================================\n")