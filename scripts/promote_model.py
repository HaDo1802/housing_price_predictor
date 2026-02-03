"""CLI for MLflow model stage transitions."""

import argparse
from typing import List

from mlflow.tracking import MlflowClient


def list_models(client: MlflowClient, model_name: str = None) -> None:
    """List registered models, versions, stages, and key metrics."""
    if model_name:
        registered_models = [client.get_registered_model(model_name)]
    else:
        registered_models = client.search_registered_models()

    if not registered_models:
        print("No registered models found.")
        return

    for rm in registered_models:
        print(f"\nModel: {rm.name}")
        versions = sorted(rm.latest_versions, key=lambda v: int(v.version))
        if not versions:
            print("  (no versions)")
            continue

        for mv in versions:
            run = client.get_run(mv.run_id)
            test_r2 = run.data.metrics.get("test_r2")
            val_r2 = run.data.metrics.get("val_r2")
            print(
                f"  - v{mv.version} | stage={mv.current_stage or 'None'} | "
                f"test_r2={test_r2} | val_r2={val_r2} | run_id={mv.run_id}"
            )


def transition_stage(
    client: MlflowClient,
    model_name: str,
    version: str,
    stage: str,
) -> None:
    """Transition a specific model version to a target stage."""
    before = client.get_model_version(model_name, version)
    previous_stage = before.current_stage or "None"

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )

    after = client.get_model_version(model_name, version)
    print(
        f"Updated {model_name} v{version}: {previous_stage} -> {after.current_stage}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote/demote MLflow model versions")
    parser.add_argument(
        "--model-name",
        default="housing_price_predictor",
        help="Registered model name (default: housing_price_predictor)",
    )
    parser.add_argument(
        "--version",
        help="Model version to transition",
    )
    parser.add_argument(
        "--stage",
        choices=["Staging", "Production", "Archived"],
        help="Target stage for selected version",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list models/versions and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = MlflowClient()

    list_models(client, model_name=None)

    if args.list_only:
        return

    if args.version and args.stage:
        transition_stage(
            client=client,
            model_name=args.model_name,
            version=str(args.version),
            stage=args.stage,
        )
        print("\nCurrent registry state:")
        list_models(client, model_name=args.model_name)
        return

    print("\nNo transition executed. Provide both --version and --stage to change a model stage.")


if __name__ == "__main__":
    main()
