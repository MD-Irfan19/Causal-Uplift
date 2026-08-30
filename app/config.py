# ============================================================
# What-If Policy Simulator Configuration
# ============================================================

# Column names in the processed customer dataset
CATE_COLUMN = "predicted_cate"
TREATMENT_COLUMN = "treatment"

# Valid targeting range
MIN_TARGET_FRACTION = 0.0
MAX_TARGET_FRACTION = 1.0

# Default cost assumption
# This will be replaced/overridden by the actual average
# discount reward obtained from portfolio.json.
DEFAULT_COST_PER_CUSTOMER = 3.0