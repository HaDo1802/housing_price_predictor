"""Model factory helpers."""

import inspect

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge

MODEL_REGISTRY = {
    "gradient_boosting": GradientBoostingRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    "random_forest": RandomForestRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "ridge": Ridge,
    "Ridge": Ridge,
}


class TrainerFactory:
    """Build the configured estimator stack."""

    @staticmethod
    def _resolve_model_class(model_type: str):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model_type '{model_type}'.")
        return MODEL_REGISTRY[model_type]

    @staticmethod
    def _wrap_target_transform(model, target_transform: str):
        if target_transform == "log1p":
            return TransformedTargetRegressor(
                regressor=model,
                func=np.log1p,
                inverse_func=np.expm1,
            )
        return model

    @classmethod
    def get_model(cls, config):
        """Return the configured estimator, filtering unsupported hyperparameters."""
        model_class = cls._resolve_model_class(config.model.model_type)
        valid_params = set(inspect.signature(model_class.__init__).parameters)
        valid_params.discard("self")

        model_params = {
            key: value
            for key, value in dict(config.model.hyperparameters or {}).items()
            if key in valid_params
        }
        if "random_state" in valid_params:
            model_params["random_state"] = config.model.random_state

        model = model_class(**model_params)
        return cls._wrap_target_transform(model, config.preprocessing.target_transform)

    @staticmethod
    def get_inner_model(model):
        """Return the underlying regressor, even when wrapped."""
        return getattr(model, "regressor_", getattr(model, "regressor", model))
