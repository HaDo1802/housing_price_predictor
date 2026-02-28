import pandas as pd
import pytest


@pytest.fixture
def sample_housing_df():
    return pd.DataFrame(
        {
            "bedrooms": [2, 3, 3, 4, 5, 3],
            "bathrooms": [1.5, 2.0, 2.5, 3.0, 4.0, 2.0],
            "livingarea": [950, 1400, 1650, 2200, 3100, 1500],
            "propertytype": [
                "CONDO",
                "SINGLE_FAMILY",
                "TOWNHOUSE",
                "SINGLE_FAMILY",
                "SINGLE_FAMILY",
                "CONDO",
            ],
            "latitude": [
                36.1021,
                36.1699,
                36.1147,
                36.2839,
                36.1699,
                36.1420,
            ],
            "longitude": [
                -115.2450,
                -115.3378,
                -115.1728,
                -115.2710,
                -115.3378,
                -115.0987,
            ],
            "price": [225000, 420000, 355000, 610000, 980000, 295000],
        }
    )
