import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@pytest.fixture
def sample_housing_df():
    """Canonical small fixture using the current training feature contract."""
    return pd.DataFrame(
        {
            "bedrooms": [2, 3, 3, 4, 5, 3],
            "bathrooms": [1.5, 2.0, 2.5, 3.0, 4.0, 2.0],
            "living_area": [950, 1400, 1650, 2200, 3100, 1500],
            "latitude": [36.1021, 36.1699, 36.1147, 36.2839, 36.1699, 36.1420],
            "longitude": [-115.2450, -115.3378, -115.1728, -115.2710, -115.3378, -115.0987],
            "normalized_lot_area_value": [1800, 2500, 2200, 4200, 6100, 2600],
            "property_type": [
                "CONDO",
                "SINGLE_FAMILY",
                "TOWNHOUSE",
                "SINGLE_FAMILY",
                "SINGLE_FAMILY",
                "CONDO",
            ],
            "vegas_district": [
                "Spring Valley",
                "Summerlin",
                "The Strip",
                "Centennial",
                "Summerlin",
                "Winchester",
            ],
            "price": [225000, 420000, 355000, 610000, 980000, 295000],
        }
    )


@pytest.fixture
def raw_training_df():
    """Larger raw fixture that mirrors the warehouse-style training input."""
    districts = [
        ("Summerlin", 36.1699, -115.3378),
        ("Centennial", 36.2839, -115.2710),
        ("Spring Valley", 36.1021, -115.2450),
        ("Paradise", 36.0972, -115.1467),
    ]
    property_types = ["SINGLE_FAMILY", "TOWNHOUSE", "CONDO"]

    rows = []
    for idx in range(48):
        district, latitude, longitude = districts[idx % len(districts)]
        bedrooms = 2 + (idx % 4)
        bathrooms = 1.5 + (idx % 4) * 0.5
        living_area = 1100 + idx * 35
        lot_area = 2200 + idx * 60
        property_type = "MOBILE" if idx % 17 == 0 else property_types[idx % len(property_types)]
        price = float(living_area * 235 + bedrooms * 12000 + lot_area * 4)
        rows.append(
            {
                "property_id": 10_000 + idx,
                "snapshot_date": "2026-04-01",
                "street_address": f"{idx} Test Ave",
                "city": "Las Vegas",
                "state": "NV",
                "zip_code": "89101",
                "vegas_district": district,
                "latitude": latitude,
                "longitude": longitude,
                "property_type": property_type,
                "price": price,
                "zestimate": None,
                "rentzestimate": None,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "living_area": float(living_area),
                "normalized_lot_area_value": float(lot_area),
                "normalized_lot_area_unit": "sqft",
                "days_on_zillow": float(5 + (idx % 30)),
                "listing_status": "FOR_SALE",
                "price_per_sqft": price / living_area,
            }
        )

    return pd.DataFrame(rows)
