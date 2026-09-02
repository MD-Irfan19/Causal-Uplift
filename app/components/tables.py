import pandas as pd
import streamlit as st


# ============================================================
# Customer Targeting Table
# ============================================================

def display_targeting_table(
    target_list,
    max_rows=100
):
    """
    Display the highest-priority customers in the dashboard.
    """

    if target_list.empty:

        st.info(
            "No customers selected for targeting."
        )

        return

    display_df = target_list.head(max_rows).copy()

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )


# ============================================================
# Download Target List
# ============================================================

def create_download_button(
    target_list,
    filename="target_customer_list.csv"
):
    """
    Create a CSV download button for the selected
    customer targeting list.
    """

    csv_data = target_list.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Target Customer List",
        data=csv_data,
        file_name=filename,
        mime="text/csv"
    )