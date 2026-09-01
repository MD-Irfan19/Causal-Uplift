from data.loader import load_customer_data

from engine.targeting import (
    rank_customers,
    select_top_customers,
    get_targeting_summary,
    prepare_target_list
)


print()
print("=" * 60)
print("PHASE 9.8 — CUSTOMER TARGETING TEST")
print("=" * 60)


# ============================================================
# Load Data
# ============================================================

df = load_customer_data()

print()
print(f"Total customers : {len(df):,}")


# ============================================================
# TEST 1 — Ranking
# ============================================================

ranked = rank_customers(df)

print()
print("-" * 60)
print("TEST 1 — CUSTOMER RANKING")
print("-" * 60)

print(
    ranked[
        [
            "targeting_rank",
            "person",
            "predicted_cate"
        ]
    ].head(5).to_string(index=False)
)


# ============================================================
# TEST 2 — Top 100 Customers
# ============================================================

target_count = 100

selected = select_top_customers(
    df,
    target_count
)

print()
print("-" * 60)
print("TEST 2 — TOP 100 CUSTOMERS")
print("-" * 60)

print(
    f"Customers selected : {len(selected):,}"
)

print(
    f"Highest CATE       : "
    f"{selected['predicted_cate'].max():.6f}"
)

print(
    f"Lowest CATE        : "
    f"{selected['predicted_cate'].min():.6f}"
)


# ============================================================
# TEST 3 — Summary
# ============================================================

summary = get_targeting_summary(
    df,
    target_count
)

print()
print("-" * 60)
print("TEST 3 — TARGETING SUMMARY")
print("-" * 60)

print(
    f"Targeted customers : "
    f"{summary['targeted_customers']:,}"
)

print(
    f"Targeting %        : "
    f"{summary['targeting_percentage']:.2%}"
)

print(
    f"Average CATE       : "
    f"{summary['average_predicted_cate']:.6f}"
)

print(
    f"Total CATE         : "
    f"{summary['total_predicted_cate']:.6f}"
)


# ============================================================
# TEST 4 — Export Preparation
# ============================================================

target_list = prepare_target_list(
    df,
    target_count
)

print()
print("-" * 60)
print("TEST 4 — TARGET LIST")
print("-" * 60)

print(
    target_list.head(10).to_string(
        index=False
    )
)


# ============================================================
# Final
# ============================================================

print()
print("=" * 60)
print("CUSTOMER TARGETING TEST COMPLETE")
print("=" * 60)