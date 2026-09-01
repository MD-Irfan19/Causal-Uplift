import pandas as pd
import plotly.graph_objects as go


# ============================================================
# Helper
# ============================================================

def prepare_scenario_data(scenarios):
    """
    Prepare scenario data for visualization.

    The policy engine stores target_fraction as a decimal
    between 0 and 1. For visualization, convert it to
    percentage form.
    """

    df = scenarios.copy()

    df["target_percentage"] = (
        df["target_fraction"] * 100
    )

    return df


# ============================================================
# Chart 1 — Uplift Captured
# ============================================================

def create_uplift_chart(
    scenarios,
    selected_percentage=None
):
    """
    Create Targeting % vs Uplift Captured chart.
    """

    df = prepare_scenario_data(scenarios)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["target_percentage"],
            y=df["pct_uplift_captured"] * 100,
            mode="lines+markers",
            name="Uplift Captured"
        )
    )

    # Highlight selected targeting percentage
    if selected_percentage is not None:

        selected = df[
            df["target_percentage"]
            == selected_percentage
        ]

        if not selected.empty:

            fig.add_trace(
                go.Scatter(
                    x=selected["target_percentage"],
                    y=selected["pct_uplift_captured"] * 100,
                    mode="markers",
                    marker=dict(size=12),
                    name="Selected Policy"
                )
            )

    fig.update_layout(
        title="Targeting Percentage vs Uplift Captured",
        xaxis_title="Customers Targeted (%)",
        yaxis_title="Uplift Captured (%)",
        yaxis=dict(range=[0, 105]),
        hovermode="x unified"
    )

    return fig


# ============================================================
# Chart 2 — Cost Savings
# ============================================================

def create_cost_savings_chart(
    scenarios,
    selected_percentage=None
):
    """
    Create Targeting % vs Cost Savings chart.
    """

    df = prepare_scenario_data(scenarios)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["target_percentage"],
            y=df["pct_cost_saved"] * 100,
            mode="lines+markers",
            name="Cost Saved"
        )
    )

    if selected_percentage is not None:

        selected = df[
            df["target_percentage"]
            == selected_percentage
        ]

        if not selected.empty:

            fig.add_trace(
                go.Scatter(
                    x=selected["target_percentage"],
                    y=selected["pct_cost_saved"] * 100,
                    mode="markers",
                    marker=dict(size=12),
                    name="Selected Policy"
                )
            )

    fig.update_layout(
        title="Targeting Percentage vs Cost Savings",
        xaxis_title="Customers Targeted (%)",
        yaxis_title="Cost Saved (%)",
        hovermode="x unified"
    )

    return fig


# ============================================================
# Chart 3 — Estimated Cost
# ============================================================

def create_cost_chart(
    scenarios,
    selected_percentage=None
):
    """
    Create Targeting % vs Estimated Cost chart.
    """

    df = prepare_scenario_data(scenarios)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["target_percentage"],
            y=df["estimated_cost"],
            mode="lines+markers",
            name="Estimated Cost"
        )
    )

    if selected_percentage is not None:

        selected = df[
            df["target_percentage"]
            == selected_percentage
        ]

        if not selected.empty:

            fig.add_trace(
                go.Scatter(
                    x=selected["target_percentage"],
                    y=selected["estimated_cost"],
                    mode="markers",
                    marker=dict(size=12),
                    name="Selected Policy"
                )
            )

    fig.update_layout(
        title="Targeting Percentage vs Estimated Cost",
        xaxis_title="Customers Targeted (%)",
        yaxis_title="Estimated Cost ($)",
        hovermode="x unified"
    )

    return fig