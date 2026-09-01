import math


# ============================================================
# Validation
# ============================================================

def validate_budget(budget):
    """
    Validate the user-provided campaign budget.
    """

    if budget < 0:
        raise ValueError(
            "Budget must be greater than or equal to 0."
        )


def validate_cost(cost_per_customer):
    """
    Validate the estimated cost per targeted customer.
    """

    if cost_per_customer <= 0:
        raise ValueError(
            "Cost per customer must be greater than 0."
        )


# ============================================================
# Budget Calculation
# ============================================================

def calculate_budget_target(
    budget,
    cost_per_customer,
    total_customers
):
    """
    Calculate the maximum number of customers that can
    be targeted within the specified budget.
    """

    validate_budget(budget)
    validate_cost(cost_per_customer)

    if total_customers < 0:
        raise ValueError(
            "Total customers cannot be negative."
        )

    max_customers = math.floor(
        budget / cost_per_customer
    )

    max_customers = min(
        max_customers,
        total_customers
    )

    return max_customers


# ============================================================
# Budget → Targeting Fraction
# ============================================================

def calculate_budget_target_fraction(
    budget,
    cost_per_customer,
    total_customers
):
    """
    Convert the budget into a targeting fraction.
    """

    targeted_customers = calculate_budget_target(
        budget,
        cost_per_customer,
        total_customers
    )

    if total_customers == 0:
        return 0.0

    return targeted_customers / total_customers


# ============================================================
# Budget Utilization
# ============================================================

def calculate_budget_utilization(
    targeted_customers,
    cost_per_customer,
    budget
):
    """
    Calculate how much of the available budget is used.
    """

    if budget <= 0:
        return 0.0

    actual_cost = (
        targeted_customers * cost_per_customer
    )

    return actual_cost / budget


# ============================================================
# Complete Budget Simulation
# ============================================================

def calculate_budget_summary(
    budget,
    cost_per_customer,
    total_customers
):
    """
    Return all important budget-related values.
    """

    targeted_customers = calculate_budget_target(
        budget,
        cost_per_customer,
        total_customers
    )

    targeting_fraction = (
        targeted_customers / total_customers
        if total_customers > 0
        else 0.0
    )

    actual_cost = (
        targeted_customers * cost_per_customer
    )

    remaining_budget = (
        budget - actual_cost
    )

    utilization = calculate_budget_utilization(
        targeted_customers,
        cost_per_customer,
        budget
    )

    return {
        "budget": budget,
        "targeted_customers": targeted_customers,
        "targeting_fraction": targeting_fraction,
        "targeting_percentage": targeting_fraction * 100,
        "actual_cost": actual_cost,
        "remaining_budget": remaining_budget,
        "budget_utilization": utilization,
        "budget_utilization_percentage": utilization * 100,
    }