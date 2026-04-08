from types import SimpleNamespace

from predictor import registry


def test_resolve_version_prefers_latest_matching_stage(monkeypatch):
    versions = [
        SimpleNamespace(version="1", current_stage="Staging"),
        SimpleNamespace(version="2", current_stage="Production"),
        SimpleNamespace(version="3", current_stage="Production"),
    ]
    monkeypatch.setattr(registry, "list_versions", lambda model_name: versions)

    resolved = registry.resolve_version("housing_price_predictor", stage="Production")

    assert resolved == "3"


def test_evaluate_and_promote_skips_registration_when_candidate_is_worse(monkeypatch):
    monkeypatch.setattr(
        registry, "_get_production_metric", lambda *args, **kwargs: 0.91
    )
    monkeypatch.setattr(registry, "register_model", lambda *args, **kwargs: "99")
    monkeypatch.setattr(registry, "promote_version", lambda *args, **kwargs: "99")
    monkeypatch.setattr(registry, "MlflowClient", lambda: object())

    result = registry.evaluate_and_promote(
        model_name="housing_price_predictor",
        run_id="run_123",
        current_metric=0.90,
    )

    assert result["passed"] is False
    assert result["registered"] is False
    assert result["promoted"] is False
    assert result["version"] is None
