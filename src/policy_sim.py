import pandas as pd
import numpy as np

TOP_PCT_POLICY_B = 0.40  # "reliable zone" cutoff 


# ============================================================
# Data loading helpers
# ============================================================

def load_policy_inputs(
    processed_path="data/processed/processed_starbucks_with_cate.csv",
    portfolio_path="data/raw/portfolio.json",
):
    df = pd.read_csv(processed_path)
    portfolio_df = pd.read_json(portfolio_path, orient="records", lines=True)
    return df, portfolio_df


def get_avg_discount_reward(portfolio_df):
    """
    Average reward across the 4 discount offers -- used as a per-
    targeted-customer cost proxy (what Starbucks pays out per
    completed discount redemption).
    """
    return portfolio_df.loc[portfolio_df["offer_type"] == "discount", "reward"].mean()


def get_positive_cate_ceiling(df_sorted):
    """
    Ceiling = sum of only the POSITIVE predicted CATEs. Represents the
    theoretical best case: perfectly targeting every responder and
    paying nothing for non-responders. Using the full-population sum
    (including negative CATEs) as the denominator would let "captured
    %" exceed 100% once negative-CATE customers are excluded, which is
    misleading -- see write-up Pitfalls section.
    """
    return df_sorted.loc[df_sorted["predicted_cate"] > 0, "predicted_cate"].sum()


# ============================================================
# Policy evaluation
# ============================================================

def evaluate_policy(df_sorted, pct_targeted, ceiling, avg_reward, total_customers, current_targeted):
    """
    Simulates targeting the top `pct_targeted` fraction of customers
    by predicted CATE. Returns captured incremental value (relative to
    the theoretical ceiling) and cost/savings relative to the current
    policy.
    """
    n = int(total_customers * pct_targeted)
    subset = df_sorted.iloc[:n]

    captured_uplift = subset["predicted_cate"].sum()
    pct_uplift_captured = captured_uplift / ceiling if ceiling != 0 else np.nan

    cost = n * avg_reward
    current_cost = current_targeted * avg_reward
    cost_saved = current_cost - cost
    pct_cost_saved = cost_saved / current_cost if current_cost != 0 else np.nan

    return {
        "pct_targeted": pct_targeted,
        "n_targeted": n,
        "captured_uplift": captured_uplift,
        "pct_uplift_captured": pct_uplift_captured,
        "estimated_cost": cost,
        "cost_saved_vs_current": cost_saved,
        "pct_cost_saved_vs_current": pct_cost_saved,
    }


def run_policy_comparison(df, portfolio_df, top_pct_policy_b=TOP_PCT_POLICY_B):
    """
    Compares three policies:
      - Current: whatever was actually targeted in the real data
      - Policy A: target every customer with positive predicted CATE
      - Policy B: target the top N% by predicted CATE rank (the
        "reliable zone" cutoff from Phase 6's decile/Qini analysis)

    Returns the results dataframe plus the intermediate values used to
    compute it (useful for a notebook to reference directly).
    """
    df_sorted = df.sort_values("predicted_cate", ascending=False).reset_index(drop=True)
    total_customers = len(df_sorted)

    avg_discount_reward = get_avg_discount_reward(portfolio_df)
    positive_cate_ceiling = get_positive_cate_ceiling(df_sorted)

    current_targeted = df_sorted["treatment"].sum()
    current_pct_targeted = current_targeted / total_customers

    policy_a_pct = (df_sorted["predicted_cate"] > 0).sum() / total_customers

    policy_current = evaluate_policy(
        df_sorted, current_pct_targeted, positive_cate_ceiling, avg_discount_reward,
        total_customers, current_targeted
    )
    policy_a = evaluate_policy(
        df_sorted, policy_a_pct, positive_cate_ceiling, avg_discount_reward,
        total_customers, current_targeted
    )
    policy_b = evaluate_policy(
        df_sorted, top_pct_policy_b, positive_cate_ceiling, avg_discount_reward,
        total_customers, current_targeted
    )

    results = pd.DataFrame([
        {"policy": f"Current ({current_pct_targeted:.0%} targeted)", **policy_current},
        {"policy": "A: Positive CATE only", **policy_a},
        {"policy": f"B: Top {top_pct_policy_b:.0%} by rank", **policy_b},
    ])

    context = {
        "avg_discount_reward": avg_discount_reward,
        "positive_cate_ceiling": positive_cate_ceiling,
        "current_targeted": current_targeted,
        "current_pct_targeted": current_pct_targeted,
    }

    return results, context


# ============================================================
# Script entry point 
# ============================================================

if __name__ == "__main__":
    df, portfolio_df = load_policy_inputs()

    results, context = run_policy_comparison(df, portfolio_df)

    print(f"Average discount reward (cost proxy per targeted customer): "
          f"${context['avg_discount_reward']:.2f}")
    print(f"\nCurrent policy: {context['current_targeted']} customers targeted "
          f"({context['current_pct_targeted']:.1%} of base)")
    print(f"Maximum achievable uplift (sum of all positive CATEs): "
          f"{context['positive_cate_ceiling']:.2f}")

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print("\n=== Policy Comparison (uplift % now relative to max achievable) ===")
    print(results.to_string(index=False))

    results.to_csv("data/processed/policy_comparison.csv", index=False)
    print("\nSaved data/processed/policy_comparison.csv")