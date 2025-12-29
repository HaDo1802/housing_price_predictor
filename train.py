"""
Main entry point for the ML pipeline

This script runs the complete machine learning workflow and trains multiple models.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.ml_pipeline import main

if __name__ == "__main__":
    main()
