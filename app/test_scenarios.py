# app/test_scenarios.py

from data.loader import (
    load_customer_data,
    load_portfolio_data,
    get_avg_discount_reward
)

from engine.policy import generate_policy_scenarios


# ============================================================
# Load data
# ============================================================

df = load_customer_data()

portfolio_df = load_portfolio_data()


# ============================================================
# Determine actual/current targeting
# ============================================================

current_targeted = df["treatment"].sum()

total_customers = len(df)

current_fraction = (
    current_targeted / total_customers
)


# ============================================================
# Determine cost per targeted customer
# ============================================================

avg_reward = get_avg_discount_reward(
    portfolio_df
)


# ============================================================
# Generate policy scenarios
# ============================================================

scenarios = generate_policy_scenarios(
    df=df,
    baseline_fraction=current_fraction,
    cost_per_customer=avg_reward,
    step=0.10
)


# ============================================================
# Display results
# ============================================================

print("\n==============================================")
print("WHAT-IF POLICY SCENARIOS")
print("==============================================")

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

print("\n----------------------------------------------")
print("POLICY SCENARIO TABLE")
print("----------------------------------------------")

# Create a copy for prettier display
display_df = scenarios.copy()

display_df["target_fraction"] = (
    display_df["target_fraction"] * 100
)

display_df["pct_uplift_captured"] = (
    display_df["pct_uplift_captured"] * 100
)

display_df["pct_cost_saved"] = (
    display_df["pct_cost_saved"] * 100
)

display_df = display_df.rename(
    columns={
        "target_fraction": "Target %",
        "n_targeted": "Customers",
        "captured_uplift": "Captured Uplift",
        "pct_uplift_captured": "Uplift Captured %",
        "estimated_cost": "Estimated Cost",
        "cost_saved": "Cost Saved",
        "pct_cost_saved": "Cost Saved %",
    }
)

print(
    display_df.to_string(
        index=False,
        formatters={
            "Target %": "{:.0f}".format,
            "Captured Uplift": "{:,.2f}".format,
            "Uplift Captured %": "{:.2f}".format,
            "Estimated Cost": "${:,.2f}".format,
            "Cost Saved": "${:,.2f}".format,
            "Cost Saved %": "{:.2f}".format,
        }
    )
)


# ============================================================
# Save scenario table
# ============================================================

output_path = (
    "data/processed/what_if_policy_scenarios.csv"
)

scenarios.to_csv(
    output_path,
    index=False
)

print("\n----------------------------------------------")
print(
    f"Saved scenario table to:\n{output_path}"
)
print("==============================================\n")