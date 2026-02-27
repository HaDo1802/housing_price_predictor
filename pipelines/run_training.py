"""Training pipeline entrypoint."""

import json
import logging

from housing_predictor.pipelines.training import TrainingPipeline


def _to_builtin(value):
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("mlflow.store.db.utils").setLevel(logging.ERROR)
    logging.getLogger("alembic").setLevel(logging.ERROR)
    logging.getLogger("sqlalchemy").setLevel(logging.ERROR)

    metrics = TrainingPipeline("conf/config.yaml").run()
    print(json.dumps(_to_builtin(metrics), indent=2))
