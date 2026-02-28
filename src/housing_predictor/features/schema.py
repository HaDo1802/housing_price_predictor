"""Shared feature schema and API field mappings."""

MODEL_FEATURES = [
    "bedrooms",
    "bathrooms",
    "livingarea",
    "propertytype",
    "vegas_district",
]

API_TO_MODEL_FIELDS = {
    "bedrooms": "bedrooms",
    "bathrooms": "bathrooms",
    "livingarea": "livingarea",
    "living_area": "livingarea",
    "propertytype": "propertytype",
    "property_type": "propertytype",
    "vegas_district": "vegas_district",
}

NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "livingarea",
]

CATEGORICAL_FEATURES = [
    "propertytype",
    "vegas_district",
]

FEATURE_DISPLAY_LABELS = {
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "livingarea": "Living Area (sqft)",
    "propertytype": "Property Type",
    "vegas_district": "Vegas District",
}

CATEGORICAL_OPTIONS = {
    "propertytype": [
        "SINGLE_FAMILY",
        "TOWNHOUSE",
        "CONDO",
        "MULTI_FAMILY",
        "MOBILE",
    ],
    "vegas_district": [
        "Summerlin",
        "Green Valley",
        "Henderson",
        "Downtown Las Vegas",
        "North Las Vegas",
        "Spring Valley",
        "Paradise",
        "Enterprise",
        "Centennial",
        "Mountains Edge",
        "The Strip",
        "Winchester",
        "Anthem",
    ],
}

TARGET_COLUMN = "price"
