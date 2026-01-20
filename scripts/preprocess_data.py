#!/usr/bin/env python3
"""
Data preprocessing script for training data preparation

Usage:
    python scripts/preprocess_data.py --input data/raw/solar.csv --output data/processed/solar_processed.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipeline import DataPreprocessingPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main preprocessing function"""
    
    parser = argparse.ArgumentParser(description="Preprocess solar energy data")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file path")
    parser.add_argument("--type", "-t", default="solar", choices=["solar", "consumption"],
                       help="Data type: solar or consumption")
    parser.add_argument("--system-capacity", "-c", type=float, default=5.0,
                       help="System capacity in kW (for solar data)")
    parser.add_argument("--validate", "-v", action="store_true", help="Validate data before preprocessing")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Read data
        logger.info(f"Reading data from {args.input}")
        data = pd.read_csv(args.input)
        logger.info(f"Loaded {len(data)} rows")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessingPipeline()
        
        # Preprocess based on type
        if args.type == "solar":
            logger.info("Preprocessing as solar generation data")
            processed_data = preprocessor.preprocess_solar_data(
                data,
                system_capacity_kw=args.system_capacity,
                validate=args.validate
            )
        else:
            logger.info("Preprocessing as consumption data")
            processed_data = preprocessor.preprocess_consumption_data(data)
        
        # Save processed data
        logger.info(f"Saving processed data to {args.output}")
        processed_data.to_csv(args.output, index=False)
        
        # Print summary statistics
        logger.info("=" * 50)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Input rows: {len(data)}")
        logger.info(f"Output rows: {len(processed_data)}")
        logger.info(f"Rows removed: {len(data) - len(processed_data)} ({(len(data)-len(processed_data))/len(data)*100:.1f}%)")
        logger.info(f"\nFeatures created: {len(processed_data.columns)}")
        logger.info(f"\nFeature list:")
        for i, col in enumerate(processed_data.columns, 1):
            logger.info(f"  {i:2d}. {col}")
        
        logger.info(f"\nData types:")
        logger.info(f"{processed_data.dtypes}")
        
        logger.info(f"\nBasic statistics:")
        logger.info(f"{processed_data.describe()}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
