"""Leakage-aware feature policy for pre-shipment late-delivery prediction."""

TARGET = "Late_delivery_risk"

# These fields are not valid inputs for a prediction made before shipment.
LEAKAGE_COLUMNS = [
    "Days for shipping (real)",
    "Delivery Status",
    "Order Status",
    TARGET,
    "Benefit per order",
    "Sales per customer",
    "Order Profit Per Order",
]

# Direct identifiers or personal data: excluded by default.
IDENTIFIER_OR_PERSONAL_COLUMNS = [
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Customer Zipcode",
    "Customer Id",
    "Order Customer Id",
]

# Initial candidates: each must be available at the intended prediction time.
# The pipeline only uses columns that actually exist in the CSV.
CANDIDATE_FEATURES = [
    "Type",
    "Days for shipment (scheduled)",
    "Category Id",
    "Category Name",
    "Customer Country",
    "Customer Segment",
    "Customer State",
    "Department Id",
    "Department Name",
    "Latitude",
    "Longitude",
    "Market",
    "Order City",
    "Order Country",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Quantity",
    "Order Region",
    "Order State",
    "Product Name",
    "Product Price",
    "Shipping Mode",
]

def get_candidate_features(columns):
    """Return approved candidate columns present in a dataset, in stable order."""
    available = set(columns)
    return [col for col in CANDIDATE_FEATURES if col in available]

def validate_feature_policy(features):
    """Raise if target, leakage, or personal/identifier columns are selected."""
    forbidden = set(LEAKAGE_COLUMNS) | set(IDENTIFIER_OR_PERSONAL_COLUMNS)
    bad = sorted(set(features) & forbidden)
    if bad:
        raise ValueError(f"Forbidden leakage/identifier columns selected: {bad}")
