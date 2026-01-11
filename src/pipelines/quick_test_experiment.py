"""
Quick Test: Run 3 Different Experiments

This shows you how to run experiments with DIFFERENT hyperparameters
so you can see meaningful comparisons in MLflow UI.

Usage:
    python scripts/quick_test.py
    
Then:
    mlflow ui
"""

import logging
import sys
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).parent.parent.parent))

import mlflow
from src.pipelines.production_training_pipeline import TrainingPipeline

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(name: str, learning_rate: float, max_depth: int):
    """Run a single experiment with specific hyperparameters."""
    
    logger.info("=" * 80)
    logger.info(f"Running: {name}")
    logger.info(f"  learning_rate: {learning_rate}")
    logger.info(f"  max_depth: {max_depth}")
    logger.info("=" * 80)
    
    # Load base config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Update hyperparameters
    config['model']['hyperparameters']['learning_rate'] = learning_rate
    config['model']['hyperparameters']['max_depth'] = max_depth
    config['training']['run_name'] = name
    
    # Save temporary config
    temp_config = "config/temp_config.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Run pipeline
    pipeline = TrainingPipeline(temp_config)
    metrics = pipeline.run()
    
    logger.info(f"✓ {name} complete: Test R² = {metrics['test']['r2']:.4f}\n")
    
    # Clean up
    Path(temp_config).unlink(missing_ok=True)
    
    return metrics


def main():
    """Run 3 experiments with different configurations."""
    
    # Set experiment name
    mlflow.set_experiment("house_price_quick_test")
    
    logger.info("\n🚀 Running 3 experiments with DIFFERENT hyperparameters...\n")
    
    # Experiment 1: Baseline (slow learning)
    metrics1 = run_experiment(
        name="baseline_slow",
        learning_rate=0.05,
        max_depth=3
    )
    
    # Experiment 2: Medium (your current config)
    metrics2 = run_experiment(
        name="medium_balanced",
        learning_rate=0.1,
        max_depth=5
    )
    
    # Experiment 3: Aggressive (fast learning)
    metrics3 = run_experiment(
        name="aggressive_deep",
        learning_rate=0.2,
        max_depth=7
    )
    
    # Summary
    logger.info("=" * 80)
    logger.info("ALL EXPERIMENTS COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nResults:")
    logger.info(f"1. baseline_slow:    R² = {metrics1['test']['r2']:.4f}")
    logger.info(f"2. medium_balanced:  R² = {metrics2['test']['r2']:.4f}")
    logger.info(f"3. aggressive_deep:  R² = {metrics3['test']['r2']:.4f}")
    
    # Find best
    results = [
        ("baseline_slow", metrics1['test']['r2']),
        ("medium_balanced", metrics2['test']['r2']),
        ("aggressive_deep", metrics3['test']['r2'])
    ]
    best_name, best_r2 = max(results, key=lambda x: x[1])
    
    logger.info(f"\n✓ Best: {best_name} (R² = {best_r2:.4f})")
    
    logger.info("\n" + "=" * 80)
    logger.info("NOW VIEW IN MLFLOW UI:")
    logger.info("=" * 80)
    logger.info("1. Run: mlflow ui")
    logger.info("2. Open: http://localhost:5000")
    logger.info("3. Look for experiment: 'house_price_quick_test'")
    logger.info("4. You'll see 3 runs with DIFFERENT metrics")
    logger.info("5. Select all 3 runs (checkboxes)")
    logger.info("6. Click 'Compare' button")
    logger.info("7. See side-by-side comparison!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()