"""
Hyperparameter Tuning Script

Automatically runs multiple experiments with different hyperparameters
and logs everything to MLflow for comparison.

Usage:
    python scripts/hyperparameter_search.py
    
Then view results:
    mlflow ui
"""

import logging
import sys
from pathlib import Path
from itertools import product
import yaml

# Add project to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import mlflow
from src.pipelines.training_pipeline import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_hyperparameter_search():
    """
    Run grid search over hyperparameters and log to MLflow.
    """
    
    # Set experiment
    mlflow.set_experiment("house_price_hyperparameter_search")
    
    # Define hyperparameter grid
    learning_rates = [0.05, 0.1, 0.15]
    max_depths = [3, 5, 7]
    n_estimators_list = [50, 100]
    
    # Load base config
    config_path = "config/config.yaml"
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Total combinations
    total_runs = len(learning_rates) * len(max_depths) * len(n_estimators_list)
    logger.info(f"Starting hyperparameter search: {total_runs} total runs")
    logger.info(f"Learning rates: {learning_rates}")
    logger.info(f"Max depths: {max_depths}")
    logger.info(f"N estimators: {n_estimators_list}")
    
    # Track results
    results = []
    
    # Grid search
    run_num = 0
    for lr, depth, n_est in product(learning_rates, max_depths, n_estimators_list):
        run_num += 1
        
        logger.info("=" * 80)
        logger.info(f"Run {run_num}/{total_runs}")
        logger.info(f"  learning_rate: {lr}")
        logger.info(f"  max_depth: {depth}")
        logger.info(f"  n_estimators: {n_est}")
        logger.info("=" * 80)
        
        # Update config with new hyperparameters
        config = base_config.copy()
        config['model']['hyperparameters']['learning_rate'] = lr
        config['model']['hyperparameters']['max_depth'] = depth
        config['model']['hyperparameters']['n_estimators'] = n_est
        config['training']['run_name'] = f"lr{lr}_depth{depth}_nest{n_est}"
        
        # Save temporary config
        temp_config_path = "config/temp_config.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        try:
            # Run pipeline
            pipeline = TrainingPipeline(temp_config_path)
            metrics = pipeline.run()
            
            # Store results
            results.append({
                'learning_rate': lr,
                'max_depth': depth,
                'n_estimators': n_est,
                'test_r2': metrics['test']['r2'],
                'test_rmse': metrics['test']['rmse'],
                'val_r2': metrics['validation']['r2']
            })
            
            logger.info(f"✓ Run {run_num} complete: R² = {metrics['test']['r2']:.4f}")
            
        except Exception as e:
            logger.error(f"✗ Run {run_num} failed: {e}")
            continue
    
    # Summary
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER SEARCH COMPLETE!")
    logger.info("=" * 80)
    
    # Sort by test R²
    results_sorted = sorted(results, key=lambda x: x['test_r2'], reverse=True)
    
    logger.info("\nTop 5 Configurations:")
    for i, result in enumerate(results_sorted[:5], 1):
        logger.info(
            f"{i}. lr={result['learning_rate']}, "
            f"depth={result['max_depth']}, "
            f"n_est={result['n_estimators']} → "
            f"R²={result['test_r2']:.4f}"
        )
    
    # Best configuration
    best = results_sorted[0]
    logger.info("\n" + "=" * 80)
    logger.info("BEST CONFIGURATION:")
    logger.info(f"  learning_rate: {best['learning_rate']}")
    logger.info(f"  max_depth: {best['max_depth']}")
    logger.info(f"  n_estimators: {best['n_estimators']}")
    logger.info(f"  Test R²: {best['test_r2']:.4f}")
    logger.info(f"  Test RMSE: {best['test_rmse']:.4f}")
    logger.info("=" * 80)
    
   
    # Clean up temp config
    Path(temp_config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    run_hyperparameter_search()