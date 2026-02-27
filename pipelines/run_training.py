"""Training pipeline entrypoint."""

import logging
import sys

from housing_predictor.pipelines.training import TrainingPipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    register_model = False
    metrics = TrainingPipeline("conf/config.yaml").run(register_model=register_model)
    print(metrics)
