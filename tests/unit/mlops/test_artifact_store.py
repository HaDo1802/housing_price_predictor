import json
from types import SimpleNamespace

from predictor import artifact_store


def test_sync_production_to_local_materializes_production_snapshot(tmp_path, monkeypatch):
    preprocessor_src = tmp_path / "preprocessor.pkl"
    preprocessor_src.write_bytes(b"preprocessor")

    metadata_src = tmp_path / "metadata.json"
    metadata_src.write_text(json.dumps({"model_type": "HistGradientBoostingRegressor"}))

    config_src = tmp_path / "config.yaml"
    config_src.write_text("training:\n  registry_model_name: housing_price_predictor\n")

    class FakeClient:
        def get_model_version(self, model_name, version):
            return SimpleNamespace(
                run_id="run_123",
                current_stage="Production",
            )

    monkeypatch.setattr(artifact_store, "resolve_version", lambda model_name, stage=None: "7")
    monkeypatch.setattr(artifact_store, "MlflowClient", lambda: FakeClient())
    monkeypatch.setattr(artifact_store.mlflow.sklearn, "load_model", lambda uri: {"uri": uri})

    def fake_download(run_id, artifact_path):
        mapping = {
            "preprocessor.pkl": str(preprocessor_src),
            "metadata.json": str(metadata_src),
            "config/config.yaml": str(config_src),
            "config.yaml": str(config_src),
        }
        return mapping.get(artifact_path)

    monkeypatch.setattr(artifact_store, "_download_optional_run_artifact", fake_download)

    result = artifact_store.sync_production_to_local(
        model_name="housing_price_predictor",
        output_dir=tmp_path / "production",
    )

    metadata_path = tmp_path / "production" / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    assert result["version"] == "7"
    assert result["run_id"] == "run_123"
    assert (tmp_path / "production" / "model.pkl").exists()
    assert (tmp_path / "production" / "preprocessor.pkl").exists()
    assert metadata["registry_version"] == "7"
    assert metadata["promotion_stage"] == "Production"
