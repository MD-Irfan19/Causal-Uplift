import pandas as pd
import numpy as np

RAW_DIR = "data/raw"


def load_json_lines(path):
    return pd.read_json(path, orient="records", lines=True)

def count_overlaps(windows_df):
    overlap_flags = {}
    for person, group in windows_df.groupby("person"):
        group = group.sort_values("window_start").reset_index(drop=True)
        has_overlap = False
        for i in range(len(group) - 1):
            if group.loc[i + 1, "window_start"] < group.loc[i, "window_end"]:
                has_overlap = True
                break
        overlap_flags[person] = has_overlap
    return pd.Series(overlap_flags)


def has_discount_overlap(group, discount_ids):
    group = group.sort_values("window_start").reset_index(drop=True)
    for i in range(len(group) - 1):
        if group.loc[i + 1, "window_start"] < group.loc[i, "window_end"]:
            ids_involved = {group.loc[i, "offer_id"], group.loc[i + 1, "offer_id"]}
            if ids_involved & set(discount_ids):
                return True
    return False

def build_offer_windows(transcript_df, portfolio_df):
    """
    For every 'offer received' event, compute the active window:
    [received_time, received_time + duration_hours].
    """
    duration_hours = portfolio_df.set_index("id")["duration"] * 24

    received = transcript_df.loc[transcript_df["event"] == "offer received"].copy()
    received["offer_id"] = received["value"].apply(lambda v: v.get("offer id"))
    received["duration_hours"] = received["offer_id"].map(duration_hours)
    received["window_start"] = received["time"]
    received["window_end"] = received["time"] + received["duration_hours"]

    return received[["person", "offer_id", "window_start", "window_end"]].reset_index(drop=True)


def build_viewed_events(transcript_df):
    """
    Extract 'offer viewed' events: which customer viewed which offer, and when.
    Needed for last-touch attribution.
    """
    viewed = transcript_df.loc[transcript_df["event"] == "offer viewed"].copy()
    viewed["offer_id"] = viewed["value"].apply(lambda v: v.get("offer id"))
    viewed = viewed.rename(columns={"time": "viewed_time"})

    return viewed[["person", "offer_id", "viewed_time"]].reset_index(drop=True)

def get_transactions(transcript_df):
    """Extract transaction events: person, time, amount."""
    trans = transcript_df.loc[transcript_df["event"] == "transaction"].copy()
    trans["amount"] = trans["value"].apply(lambda v: v.get("amount"))
    trans = trans.reset_index(drop=True)
    trans["transaction_id"] = trans.index
    return trans[["transaction_id", "person", "time", "amount"]]

def attribute_transactions(transactions_df, windows_df, viewed_df, portfolio_df):
    """
    Last-touch attribution: for each transaction, find all offer windows
    active at that moment for that customer, then attribute the purchase
    to whichever active offer was most recently viewed before the
    transaction. If no active offer was ever viewed, mark as unattributed.
    """
    # Step A: join transactions to all offer windows for the same person
    # (many-to-many on person, then filter to only windows active at
    # transaction time)
    merged = transactions_df.merge(windows_df, on="person", how="left")
    active = merged[
        (merged["time"] >= merged["window_start"]) &
        (merged["time"] <= merged["window_end"])
    ].copy()

    # Step B: join to viewed events for the same person + offer_id,
    # keep only views that happened before the transaction
    active = active.merge(
        viewed_df, on=["person", "offer_id"], how="left", suffixes=("", "_view")
    )
    active = active[
        active["viewed_time"].notna() & (active["viewed_time"] <= active["time"])
    ]

    # Step C: for each transaction, keep only the most recently viewed
    # active offer (last-touch)
    active = active.sort_values("viewed_time", ascending=False)
    attributed = active.drop_duplicates(subset="transaction_id", keep="first")
    attributed = attributed[["transaction_id", "offer_id"]].rename(
        columns={"offer_id": "attributed_offer_id"}
    )

    # Step D: merge attribution back onto the full transaction set;
    # transactions with no match are unattributed (organic)
    result = transactions_df.merge(attributed, on="transaction_id", how="left")

    offer_type_map = portfolio_df.set_index("id")["offer_type"]
    result["attributed_offer_type"] = result["attributed_offer_id"].map(offer_type_map)
    result["attributed_offer_type"] = result["attributed_offer_type"].fillna("unattributed")

    return result

def build_customer_level_table(profile_df, windows_df, attributed_df):
    """
    Customer-level treatment/outcome table.

    treatment: intent-to-treat = 1 if customer ever RECEIVED a discount
    offer (regardless of viewing/completion), 0 otherwise. This avoids
    conditioning on post-treatment behavior (viewing), which would
    introduce selection bias.

    outcome: total transaction spend over the full observation period,
    for every customer, treated or not.

    discount_attributed_spend: diagnostic only (not used as treatment/
    outcome) -- how much spend was last-touch attributed to a discount,
    for customers who received one.
    """
    discount_ids = {
        "0b1e1539f2cc45b7b9fa7c272da2e1d7",
        "2298d6c36e964ae4a3e7e9706d1fb8c2",
        "fafdcd668e3743c1bb461111dcafc2a4",
        "2906b810c7d4411798c6938adc9daaa5",
    }

    received_discount = (
        windows_df[windows_df["offer_id"].isin(discount_ids)]["person"]
        .unique()
    )

    total_spend = attributed_df.groupby("person")["amount"].sum().rename("outcome")

    discount_attributed_spend = (
        attributed_df[attributed_df["attributed_offer_type"] == "discount"]
        .groupby("person")["amount"].sum()
        .rename("discount_attributed_spend")
    )

    customer_df = profile_df.rename(columns={"id": "person"}).copy()
    customer_df["treatment"] = customer_df["person"].isin(received_discount).astype(int)
    customer_df = customer_df.merge(total_spend, on="person", how="left")
    customer_df = customer_df.merge(discount_attributed_spend, on="person", how="left")

    customer_df["outcome"] = customer_df["outcome"].fillna(0)
    customer_df["discount_attributed_spend"] = customer_df["discount_attributed_spend"].fillna(0)

    return customer_df

def add_covariates(customer_df, transactions_df):
    """
    Add tenure_days (derived from became_member_on), avg_spend,
    purchase_frequency, and total_transactions. Also flags the known
    placeholder rows (age == 118, missing income/gender) for exclusion.
    """
    df = customer_df.copy()

    df["became_member_on"] = pd.to_datetime(df["became_member_on"], format="%Y%m%d")
    reference_date = df["became_member_on"].max()
    df["tenure_days"] = (reference_date - df["became_member_on"]).dt.days

    txn_stats = transactions_df.groupby("person").agg(
        total_transactions=("amount", "count"),
        avg_spend=("amount", "mean"),
    )
    df = df.merge(txn_stats, on="person", how="left")

    df["total_transactions"] = df["total_transactions"].fillna(0)
    df["avg_spend"] = df["avg_spend"].fillna(0)

    # Floor the denominator at 30 days (i.e. minimum "1 month" window) to
    # prevent near-zero tenure from producing exploded frequency values.
    # This is a standard rate-stabilization technique: without it, brand-new
    # customers with even a couple of transactions get absurd "transactions
    # per 30 days" figures purely from dividing by a tiny denominator.
    tenure_floor = df["tenure_days"].clip(lower=30)
    df["purchase_frequency"] = (df["total_transactions"] / (tenure_floor / 30)).round(3)

    df["is_missing_demographics"] = (df["age"] == 118)

    return df

if __name__ == "__main__":
    portfolio_df = load_json_lines(f"{RAW_DIR}/portfolio.json")
    profile_df = load_json_lines(f"{RAW_DIR}/profile.json")
    transcript_df = load_json_lines(f"{RAW_DIR}/transcript.json")

    windows_df = build_offer_windows(transcript_df, portfolio_df)
    viewed_df = build_viewed_events(transcript_df)
    transactions_df = get_transactions(transcript_df)
    attributed_df = attribute_transactions(transactions_df, windows_df, viewed_df, portfolio_df)

    customer_df = build_customer_level_table(profile_df, windows_df, attributed_df)
    customer_df = add_covariates(customer_df, transactions_df)

    print(f"Total customers: {customer_df.shape[0]}")
    print(f"Missing demographics (age==118): {customer_df['is_missing_demographics'].sum()}")

    print(f"\nCovariate summary (excluding missing-demographics rows):")
    clean = customer_df[~customer_df["is_missing_demographics"]]
    print(clean[["age", "income", "tenure_days", "avg_spend", "purchase_frequency", "total_transactions"]].describe())

    print(f"\nTreatment group sizes after exclusion:")
    print(clean["treatment"].value_counts())

    clean.to_csv("data/processed/processed_starbucks.csv", index=False)
    print(f"\nSaved {clean.shape[0]} customers to data/processed/processed_starbucks.csv")