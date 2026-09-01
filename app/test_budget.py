from engine.budget import (
    calculate_budget_target,
    calculate_budget_target_fraction,
    calculate_budget_summary
)


TOTAL_CUSTOMERS = 14825
COST_PER_CUSTOMER = 3.00


print()
print("=" * 55)
print("PHASE 9.7 — BUDGET ENGINE TEST")
print("=" * 55)


# ============================================================
# Test 1
# ============================================================

budget = 15000

targeted = calculate_budget_target(
    budget,
    COST_PER_CUSTOMER,
    TOTAL_CUSTOMERS
)

print()
print("TEST 1 — $15,000 BUDGET")
print("-" * 55)

print(
    f"Budget               : ${budget:,.2f}"
)

print(
    f"Customers targeted   : {targeted:,}"
)

print(
    f"Targeting percentage : "
    f"{targeted / TOTAL_CUSTOMERS:.2%}"
)


# ============================================================
# Test 2
# ============================================================

budget = 30000

summary = calculate_budget_summary(
    budget,
    COST_PER_CUSTOMER,
    TOTAL_CUSTOMERS
)

print()
print("TEST 2 — $30,000 BUDGET")
print("-" * 55)

print(
    f"Budget               : "
    f"${summary['budget']:,.2f}"
)

print(
    f"Customers targeted   : "
    f"{summary['targeted_customers']:,}"
)

print(
    f"Targeting percentage : "
    f"{summary['targeting_percentage']:.2f}%"
)

print(
    f"Actual cost          : "
    f"${summary['actual_cost']:,.2f}"
)

print(
    f"Remaining budget     : "
    f"${summary['remaining_budget']:,.2f}"
)

print(
    f"Budget utilization   : "
    f"{summary['budget_utilization_percentage']:.2f}%"
)


# ============================================================
# Test 3 — Zero Budget
# ============================================================

budget = 0

targeted = calculate_budget_target(
    budget,
    COST_PER_CUSTOMER,
    TOTAL_CUSTOMERS
)

print()
print("TEST 3 — ZERO BUDGET")
print("-" * 55)

print(
    f"Customers targeted   : {targeted:,}"
)


# ============================================================
# Final
# ============================================================

print()
print("=" * 55)
print("BUDGET ENGINE TEST COMPLETE")
print("=" * 55)