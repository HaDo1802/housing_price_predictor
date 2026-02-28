"""Shared feature schema and API field mappings."""

MODEL_FEATURES = [
    "bedrooms",
    "bathrooms",
    "livingarea",
    "latitude",
    "longitude",
    "propertytype",
]

API_TO_MODEL_FIELDS = {
    "bedrooms": "bedrooms",
    "bathrooms": "bathrooms",
    "livingarea": "livingarea",
    "living_area": "livingarea",
    "latitude": "latitude",
    "longitude": "longitude",
    "propertytype": "propertytype",
    "property_type": "propertytype",
}

NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "livingarea",
    "latitude",
    "longitude",
]

CATEGORICAL_FEATURES = [
    "propertytype",
]

FEATURE_DISPLAY_LABELS = {
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "livingarea": "Living Area (sqft)",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "propertytype": "Property Type",
}

CATEGORICAL_OPTIONS = {
    "propertytype": [
        "SINGLE_FAMILY",
        "TOWNHOUSE",
        "CONDO",
        "MULTI_FAMILY",
        "MOBILE",
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

TARGET_COLUMN = "price"
