import streamlit as st


# ============================================================
# Recommendation Metrics
# ============================================================

def display_recommendation_metrics(
    recommendation
):
    """
    Display the main recommendation metrics.
    """

    if recommendation["status"] != "RECOMMENDED":

        st.warning(
            recommendation["message"]
        )

        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Customers Targeted",
            f"{recommendation['targeted_customers']:,}"
        )

    with col2:

        st.metric(
            "Targeting Level",
            f"{recommendation['targeting_percentage']:.2f}%"
        )

    with col3:

        st.metric(
            "Uplift Captured",
            f"{recommendation['uplift_captured_percentage']:.2f}%"
        )

    with col4:

        st.metric(
            "Estimated Cost",
            f"${recommendation['estimated_cost']:,.2f}"
        )


# ============================================================
# Cost Metrics
# ============================================================

def display_cost_metrics(
    recommendation
):
    """
    Display cost-related recommendation metrics.
    """

    if recommendation["status"] != "RECOMMENDED":
        return

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Cost Saved vs Current",
            f"${recommendation['cost_saved']:,.2f}"
        )

    with col2:

        st.metric(
            "Cost Saved %",
            f"{recommendation['cost_saved_percentage']:.2f}%"
        )


# ============================================================
# Recommendation Message
# ============================================================

def display_recommendation_message(
    recommendation
):
    """
    Display the recommendation explanation.
    """

    if recommendation["status"] != "RECOMMENDED":

        st.warning(
            recommendation["message"]
        )

        return

    st.success(
        "Recommended Strategy"
    )

    st.info(
        recommendation["message"]
    )