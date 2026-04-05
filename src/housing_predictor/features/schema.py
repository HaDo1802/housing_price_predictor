"""Shared feature schema and API field mappings."""

# ---------------------------------------------------------------------------
# Core feature lists — single source of truth used by preprocessor,
# training config, and API serving layer.
# Column names match gold.mart_property_current exactly.
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "living_area",
    "latitude",
    "longitude",
    "normalized_lot_area_value",
    "days_on_zillow",
]

CATEGORICAL_FEATURES = [
    "property_type",
    "vegas_district",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "price"

# ---------------------------------------------------------------------------
# Columns to drop before training — zero variance, leakage, or identifiers
# ---------------------------------------------------------------------------

DROP_COLUMNS = [
    "property_id",
    "snapshot_date",
    "street_address",
    "city",
    "state",
    "zip_code",           # 8 missing, weaker signal than vegas_district + lat/lon
    "zestimate",          # 100% null
    "rentzestimate",      # 100% null
    "listing_status",     # 100% FOR_SALE — zero variance
    "normalized_lot_area_unit",  # 100% sqft — zero variance
    "price_per_sqft",     # derived from price/living_area — data leakage
]

# ---------------------------------------------------------------------------
# Outlier filter bounds — applied to training split only (never to test/prod)
# ---------------------------------------------------------------------------

OUTLIER_FILTERS = {
    "price_per_sqft_min": 80.0,    # below this = data error or land-only
    "price_per_sqft_max": 800.0,   # above this = ultra-luxury, distorts model
    "living_area_min": 400.0,      # below this = unrealistic
    "bedrooms_max": 10,            # above this = mansion/multi-unit, small sample
    "bathrooms_max": 10,
}

# Property types to exclude from training (too few rows or fundamentally different market)
EXCLUDED_PROPERTY_TYPES = [
    "MOBILE",       # priced off lot-lease, not sqft — median $124k vs $490k overall
]

# ---------------------------------------------------------------------------
# API field mappings (kept for serving layer compatibility)
# ---------------------------------------------------------------------------

API_TO_MODEL_FIELDS = {
    "bedrooms": "bedrooms",
    "bathrooms": "bathrooms",
    "living_area": "living_area",
    "livingarea": "living_area",      # backward compat for old API clients
    "latitude": "latitude",
    "longitude": "longitude",
    "normalized_lot_area_value": "normalized_lot_area_value",
    "days_on_zillow": "days_on_zillow",
    "property_type": "property_type",
    "propertytype": "property_type",  # backward compat
    "vegas_district": "vegas_district",
}

FEATURE_DISPLAY_LABELS = {
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "living_area": "Living Area (sqft)",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "normalized_lot_area_value": "Lot Size (sqft)",
    "days_on_zillow": "Days on Market",
    "property_type": "Property Type",
    "vegas_district": "District",
}

CATEGORICAL_OPTIONS = {
    "property_type": [
        "SINGLE_FAMILY",
        "TOWNHOUSE",
        "CONDO",
        "MULTI_FAMILY",
    ],
    "vegas_district": [
        "Summerlin",
        "Centennial",
        "Winchester",
        "Spring Valley",
        "Enterprise",
        "Mountains Edge",
        "Downtown Las Vegas",
        "Paradise",
        "Green Valley",
        "North Las Vegas",
        "The Strip",
        "Anthem",
    ],
}

VEGAS_DISTRICT_CENTROIDS = {
    "Summerlin": {"latitude": 36.1699, "longitude": -115.3378},
    "Green Valley": {"latitude": 36.0417, "longitude": -115.0833},
    "Henderson": {"latitude": 36.0395, "longitude": -114.9817},
    "Downtown Las Vegas": {"latitude": 36.1699, "longitude": -115.1398},
    "North Las Vegas": {"latitude": 36.1989, "longitude": -115.1175},
    "Spring Valley": {"latitude": 36.1021, "longitude": -115.2450},
    "Paradise": {"latitude": 36.0972, "longitude": -115.1467},
    "Enterprise": {"latitude": 36.0253, "longitude": -115.2419},
    "Centennial": {"latitude": 36.2839, "longitude": -115.2710},
    "Mountains Edge": {"latitude": 36.0089, "longitude": -115.2747},
    "The Strip": {"latitude": 36.1147, "longitude": -115.1728},
    "Winchester": {"latitude": 36.1420, "longitude": -115.0987},
    "Anthem": {"latitude": 35.9853, "longitude": -115.1017},
}