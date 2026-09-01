import pandas as pd


# ============================================================
# Validation
# ============================================================

def validate_customer_data(df):
    """
    Validate that the customer dataframe contains the
    information required for targeting.
    """

    required_columns = [
        "person",
        "predicted_cate"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def validate_target_count(
    target_count,
    total_customers
):
    """
    Validate the requested number of customers.
    """

    if target_count < 0:
        raise ValueError(
            "Target count cannot be negative."
        )

    if target_count > total_customers:
        raise ValueError(
            "Target count cannot exceed total customers."
        )


# ============================================================
# Customer Ranking
# ============================================================

def rank_customers(df):
    """
    Rank customers by predicted CATE in descending order.

    Higher predicted CATE = higher targeting priority.
    """

    validate_customer_data(df)

    ranked = (
        df.sort_values(
            "predicted_cate",
            ascending=False
        )
        .reset_index(drop=True)
        .copy()
    )

    ranked["targeting_rank"] = (
        ranked.index + 1
    )

    return ranked


# ============================================================
# Select Top Customers
# ============================================================

def select_top_customers(
    df,
    target_count
):
    """
    Select the top N customers according to predicted CATE.
    """

    validate_customer_data(df)

    total_customers = len(df)

    validate_target_count(
        target_count,
        total_customers
    )

    ranked = rank_customers(df)

    selected = (
        ranked
        .iloc[:target_count]
        .copy()
    )

    return selected


# ============================================================
# Add Targeting Decision
# ============================================================

def create_targeting_labels(
    df,
    target_count
):
    """
    Add a targeting decision to every customer.

    TARGET = selected for campaign
    NOT TARGETED = excluded from campaign
    """

    ranked = rank_customers(df)

    validate_target_count(
        target_count,
        len(ranked)
    )

    ranked["targeting_decision"] = "NOT TARGETED"

    ranked.loc[
        ranked.index < target_count,
        "targeting_decision"
    ] = "TARGET"

    return ranked


# ============================================================
# Customer Targeting Summary
# ============================================================

def get_targeting_summary(
    df,
    target_count
):
    """
    Generate summary statistics for the selected
    customer population.
    """

    selected = select_top_customers(
        df,
        target_count
    )

    total_customers = len(df)

    selected_count = len(selected)

    targeting_percentage = (
        selected_count / total_customers
        if total_customers > 0
        else 0
    )

    if selected_count > 0:

        average_cate = (
            selected["predicted_cate"].mean()
        )

        total_cate = (
            selected["predicted_cate"].sum()
        )

        minimum_cate = (
            selected["predicted_cate"].min()
        )

        maximum_cate = (
            selected["predicted_cate"].max()
        )

    else:

        average_cate = 0.0
        total_cate = 0.0
        minimum_cate = 0.0
        maximum_cate = 0.0

    return {
        "total_customers": total_customers,
        "targeted_customers": selected_count,
        "targeting_percentage": targeting_percentage,
        "average_predicted_cate": average_cate,
        "total_predicted_cate": total_cate,
        "minimum_predicted_cate": minimum_cate,
        "maximum_predicted_cate": maximum_cate
    }


# ============================================================
# Export Target List
# ============================================================

def prepare_target_list(df, target_count):
    """
    Prepare a clean customer list suitable for display
    or CSV export.
    """

    selected = select_top_customers(
        df,
        target_count
    )

    preferred_columns = [
        "targeting_rank",
        "person",
        "predicted_cate",
        "age",
        "gender",
        "income",
        "tenure_days",
        "total_transactions",
        "avg_spend",
        "purchase_frequency"
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in selected.columns
    ]

    target_list = selected[
        available_columns
    ].copy()

    return target_list