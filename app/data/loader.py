import pandas as pd


def load_customer_data(
    path="data/processed/processed_starbucks_with_cate.csv"
):
    """
    Load the processed Starbucks customer dataset
    containing predicted CATE values.
    """

    return pd.read_csv(path)


def load_portfolio_data(
    path="data/raw/portfolio.json"
):
    """
    Load Starbucks offer portfolio data.
    """

    return pd.read_json(
        path,
        orient="records",
        lines=True
    )


def get_avg_discount_reward(portfolio_df):
    """
    Calculate the average reward across discount offers.

    This follows the same cost-proxy definition used
    in the original policy_sim.py.
    """

    discount_rewards = portfolio_df.loc[
        portfolio_df["offer_type"] == "discount",
        "reward"
    ]

    if discount_rewards.empty:
        raise ValueError(
            "No discount offers found in portfolio data."
        )

    return discount_rewards.mean()