import sys
from pathlib import Path

import streamlit as st


# ============================================================
# Make the app directory importable
# ============================================================

APP_DIR = Path(__file__).resolve().parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


# ============================================================
# Project imports
# ============================================================

from data.loader import (
    load_customer_data,
    load_portfolio_data,
    get_avg_discount_reward
)

from engine.policy import (
    simulate_policy,
    generate_policy_scenarios
)

from engine.budget import calculate_budget_summary

from engine.targeting import (
    prepare_target_list,
    get_targeting_summary
)

from components.tables import (
    display_targeting_table,
    create_download_button
)

from components.charts import (
    create_uplift_chart,
    create_cost_savings_chart,
    create_cost_chart
)

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Causal Uplift — What-If Simulator",
    # page_icon="📊",
    layout="wide"
)


# ============================================================
# Title
# ============================================================

st.title("Causal Uplift — What-If Policy Simulator")

st.markdown(
    """
    ### Explore the impact of different customer-targeting policies

    Adjust the targeting percentage and see how the predicted
    uplift and estimated campaign cost change.
    """
)


# ============================================================
# Load Data
# ============================================================

@st.cache_data
def load_data():

    customer_df = load_customer_data()

    portfolio_df = load_portfolio_data()

    return customer_df, portfolio_df


df, portfolio_df = load_data()


# ============================================================
# Current Policy Information
# ============================================================

total_customers = len(df)

current_targeted = int(
    df["treatment"].sum()
)

current_fraction = (
    current_targeted / total_customers
)

avg_reward = get_avg_discount_reward(
    portfolio_df
)


# ============================================================
# Current Policy Metrics
# ============================================================

st.subheader("Current Policy")

current_col1, current_col2, current_col3 = st.columns(3)

with current_col1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with current_col2:
    st.metric(
        "Currently Targeted",
        f"{current_targeted:,}"
    )

with current_col3:
    st.metric(
        "Current Targeting",
        f"{current_fraction:.2%}"
    )


st.divider()


# ============================================================
# Targeting Control
# ============================================================

st.subheader("What-If Targeting")

target_percentage = st.slider(
    "Select percentage of customers to target",
    min_value=0,
    max_value=100,
    value=40,
    step=1
)

target_fraction = target_percentage / 100


st.write(
    f"**Selected targeting level: "
    f"{target_percentage}%**"
)


# ============================================================
# Run Simulation
# ============================================================

result = simulate_policy(
    df=df,
    target_fraction=target_fraction,
    baseline_fraction=current_fraction,
    cost_per_customer=avg_reward
)


# ============================================================
# Generate scenario data for visualization
# ============================================================

@st.cache_data
def generate_scenarios(
    customer_df,
    baseline_fraction,
    cost_per_customer
):
    return generate_policy_scenarios(
        df=customer_df,
        baseline_fraction=baseline_fraction,
        cost_per_customer=cost_per_customer,
        step=0.10
    )


scenarios = generate_scenarios(
    df,
    current_fraction,
    avg_reward
)


# ============================================================
# Simulation Results
# ============================================================

st.subheader("Simulation Results")


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Customers Targeted",
        f"{result['n_targeted']:,}"
    )


with col2:
    st.metric(
        "Captured Uplift",
        f"{result['captured_uplift']:,.2f}"
    )


with col3:
    st.metric(
        "Uplift Captured",
        f"{result['pct_uplift_captured']:.2%}"
    )


with col4:
    st.metric(
        "Estimated Cost",
        f"${result['estimated_cost']:,.2f}"
    )


# ============================================================
# Cost Savings
# ============================================================

st.subheader("Cost Impact")

cost_col1, cost_col2 = st.columns(2)


with cost_col1:

    st.metric(
        "Cost Saved vs Current Policy",
        f"${result['cost_saved']:,.2f}"
    )


with cost_col2:

    st.metric(
        "Cost Saved %",
        f"{result['pct_cost_saved']:.2%}"
    )

# ============================================================
# Budget-Constrained Simulation
# ============================================================

st.divider()

st.subheader("Budget-Constrained Simulation")

st.markdown(
    """
    Instead of choosing how many customers to target,
    specify the maximum campaign budget. The simulator
    determines how many customers can be targeted within
    that budget.
    """
)


# ============================================================
# Budget Input
# ============================================================

budget = st.number_input(
    "Maximum Campaign Budget ($)",
    min_value=0.0,
    max_value=100000.0,
    value=15000.0,
    step=500.0
)


# ============================================================
# Calculate Budget Allocation
# ============================================================

budget_summary = calculate_budget_summary(
    budget=budget,
    cost_per_customer=avg_reward,
    total_customers=total_customers
)


# ============================================================
# Run Budget-Based Policy
# ============================================================

budget_result = simulate_policy(
    df=df,
    target_fraction=budget_summary["targeting_fraction"],
    baseline_fraction=current_fraction,
    cost_per_customer=avg_reward
)


# ============================================================
# Budget Metrics
# ============================================================

st.markdown("### Budget Allocation")


budget_col1, budget_col2, budget_col3, budget_col4 = st.columns(4)


with budget_col1:

    st.metric(
        "Customers Targeted",
        f"{budget_summary['targeted_customers']:,}"
    )


with budget_col2:

    st.metric(
        "Targeting Percentage",
        f"{budget_summary['targeting_percentage']:.2f}%"
    )


with budget_col3:

    st.metric(
        "Estimated Campaign Cost",
        f"${budget_summary['actual_cost']:,.2f}"
    )


with budget_col4:

    st.metric(
        "Remaining Budget",
        f"${budget_summary['remaining_budget']:,.2f}"
    )


# ============================================================
# Budget Policy Results
# ============================================================

st.markdown("### Expected Impact")


impact_col1, impact_col2, impact_col3 = st.columns(3)


with impact_col1:

    st.metric(
        "Captured Uplift",
        f"{budget_result['captured_uplift']:,.2f}"
    )


with impact_col2:

    st.metric(
        "Uplift Captured",
        f"{budget_result['pct_uplift_captured']:.2%}"
    )


with impact_col3:

    st.metric(
        "Cost Saved vs Current",
        f"${budget_result['cost_saved']:,.2f}"
    )


# ============================================================
# Budget Interpretation
# ============================================================

st.info(
    f"""
    With a maximum budget of **${budget:,.2f}**, the simulator
    can target **{budget_summary['targeted_customers']:,} customers**
    ({budget_summary['targeting_percentage']:.2f}% of the customer base).

    This is expected to capture
    **{budget_result['pct_uplift_captured']:.2%}**
    of the theoretical positive-CATE uplift.

    Estimated campaign expenditure:
    **${budget_summary['actual_cost']:,.2f}**.
    """
)


# ============================================================
# Customer Targeting
# ============================================================

st.divider()

st.subheader("Customer Targeting")

st.markdown(
    """
    The simulator ranks customers by their predicted CATE
    and identifies the highest-priority customers for targeting.
    """
)


# ============================================================
# Determine Target Count
# ============================================================

targeting_mode = st.radio(
    "Choose targeting basis",
    [
        "What-If Targeting %",
        "Budget Constraint"
    ],
    horizontal=True
)


if targeting_mode == "What-If Targeting %":

    selected_target_count = int(
        total_customers * target_fraction
    )

else:

    selected_target_count = (
        budget_summary["targeted_customers"]
    )


# ============================================================
# Targeting Summary
# ============================================================

targeting_summary = get_targeting_summary(
    df,
    selected_target_count
)


summary_col1, summary_col2, summary_col3 = st.columns(3)


with summary_col1:

    st.metric(
        "Customers Selected",
        f"{targeting_summary['targeted_customers']:,}"
    )


with summary_col2:

    st.metric(
        "Average Predicted CATE",
        f"{targeting_summary['average_predicted_cate']:.4f}"
    )


with summary_col3:

    st.metric(
        "Total Predicted CATE",
        f"{targeting_summary['total_predicted_cate']:,.2f}"
    )


# ============================================================
# Target Customer List
# ============================================================

target_list = prepare_target_list(
    df,
    selected_target_count
)


st.markdown(
    "### Highest-Priority Customers"
)

st.caption(
    "Customers are ranked by predicted CATE. "
    "Higher predicted CATE indicates higher estimated "
    "incremental treatment effect."
)


display_targeting_table(
    target_list,
    max_rows=100
)


# ============================================================
# Download
# ============================================================

if not target_list.empty:

    create_download_button(
        target_list,
        filename="causal_uplift_target_customer_list.csv"
    )


# ============================================================
# Interpretation
# ============================================================

st.divider()

st.subheader("Policy Interpretation")

st.write(
    f"""
    By targeting **{target_percentage}%** of customers,
    the simulator selects the customers with the highest
    predicted CATE values.

    This corresponds to **{result['n_targeted']:,} customers**,
    capturing **{result['pct_uplift_captured']:.2%}** of the
    theoretical positive-CATE uplift.

    The estimated targeting cost is
    **${result['estimated_cost']:,.2f}**, representing
    **{result['pct_cost_saved']:.2%}** cost savings compared
    with the current targeting policy.
    """
)


# ============================================================
# Policy Trade-off Analysis
# ============================================================

st.divider()

st.subheader("Policy Trade-off Analysis")

st.markdown(
    """
    These charts show how the predicted uplift and campaign
    cost change as the percentage of customers targeted changes.
    """
)


# ============================================================
# Chart 1 — Uplift
# ============================================================

fig_uplift = create_uplift_chart(
    scenarios,
    selected_percentage=target_percentage
)

st.plotly_chart(
    fig_uplift,
    use_container_width=True
)


# ============================================================
# Chart 2 — Cost Savings
# ============================================================

fig_savings = create_cost_savings_chart(
    scenarios,
    selected_percentage=target_percentage
)

st.plotly_chart(
    fig_savings,
    use_container_width=True
)


# ============================================================
# Chart 3 — Estimated Cost
# ============================================================

fig_cost = create_cost_chart(
    scenarios,
    selected_percentage=target_percentage
)

st.plotly_chart(
    fig_cost,
    use_container_width=True
)

# ============================================================
# Data Information
# ============================================================

with st.expander("Dataset Information"):

    st.write(
        f"Total rows: **{len(df):,}**"
    )

    st.write(
        f"Dataset columns: **{len(df.columns)}**"
    )

    st.write(
        "The simulator ranks customers using "
        "`predicted_cate`."
    )

    st.write(
        f"Average discount reward used as the "
        f"cost proxy: **${avg_reward:.2f}**"
    )