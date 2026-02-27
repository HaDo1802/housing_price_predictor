"""Model information endpoints."""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/model", tags=["Model"])


def _raw_feature_names(pipeline) -> list:
    """Best-effort raw input feature list (before one-hot expansion)."""
    preprocessor = getattr(pipeline, "preprocessor", None)
    if preprocessor is None:
        return []

    numeric = getattr(preprocessor, "numeric_features", None) or []
    categorical = getattr(preprocessor, "categorical_features", None) or []
    return list(numeric) + list(categorical)


@router.get("/info")
async def model_info(request: Request):
    pipeline = request.app.state.inference_pipeline
    if pipeline is None:
        load_error = getattr(request.app.state, "model_load_error", None)
        detail = "Model not loaded"
        if load_error:
            detail = f"{detail}: {load_error}"
        raise HTTPException(status_code=503, detail=detail)

    transformed_feature_names = pipeline.metadata.get("feature_names", [])
    raw_feature_names = _raw_feature_names(pipeline)
    if not raw_feature_names:
        # Fallback when preprocessor does not expose raw feature fields.
        raw_feature_names = transformed_feature_names

    return {
        "model_type": pipeline.metadata.get("model_type"),
        "hyperparameters": pipeline.metadata.get("hyperparameters"),
        "metrics": {
            "test": pipeline.metadata.get("test_metrics"),
            "validation": pipeline.metadata.get("val_metrics"),
        },
        "features": {
            "count": len(raw_feature_names),
            "names": raw_feature_names,
        },
        # "transformed_features": {
        #     "count": len(transformed_feature_names),
        #     "names": transformed_feature_names,
        # },
        "training_info": {
            "train_size": pipeline.metadata.get("train_size"),
            "val_size": pipeline.metadata.get("val_size"),
            "test_size": pipeline.metadata.get("test_size"),
        },
    }
