from pathlib import Path
import json
import pytest

from src.pipelines.inference_pipeline import InferencePipeline


def test_model_artifacts_exist():
    model_dir = Path("models/production")
    assert model_dir.exists(), "models/production directory is missing"

    for name in ("model.pkl", "preprocessor.pkl", "metadata.json"):
        assert (model_dir / name).exists(), f"Missing required artifact: {name}"


def test_metadata_has_feature_names():
    metadata_path = Path("models/production/metadata.json")
    with metadata_path.open() as f:
        metadata = json.load(f)

    feature_names = metadata.get("feature_names", [])
    assert isinstance(feature_names, list), "feature_names must be a list"
    assert len(feature_names) > 0, "feature_names should not be empty"
    assert all(isinstance(name, str) for name in feature_names), "feature names must be strings"


def test_validate_model_dir_raises_when_missing(tmp_path):
    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline.model_dir = tmp_path

    with pytest.raises(FileNotFoundError):
        pipeline._validate_model_dir()
