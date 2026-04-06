import pytest
from pydantic import ValidationError

from serving.api.schemas import HouseFeatures


def test_house_features_accepts_legacy_payload_without_new_optional_fields():
    """Older clients should remain valid when new optional inputs are introduced."""
    payload = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "livingarea": 1600,
        "latitude": 36.12,
        "longitude": -115.17,
        "propertytype": "SINGLE_FAMILY",
    }

    parsed = HouseFeatures(**payload)

    assert parsed.normalized_lot_area_value is None
    assert parsed.days_on_zillow is None


def test_house_features_still_rejects_invalid_core_fields():
    """Backward compatibility should not weaken validation of required inputs."""
    with pytest.raises(ValidationError):
        HouseFeatures(
            bedrooms=-1,
            bathrooms=2.0,
            livingarea=1600,
            latitude=36.12,
            longitude=-115.17,
            propertytype="SINGLE_FAMILY",
        )
