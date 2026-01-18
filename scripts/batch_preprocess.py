#!/usr/bin/env python3
"""
Batch preprocessing script.

Processes all raw CSV files in data/raw/ and outputs to data/processed/.
Automatically detects data type, handles NSRDB metadata, and applies full pipeline.
"""
import sys
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Setup path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.utils.logger import get_logger
from src.preprocessing.pipeline import DataPreprocessingPipeline
from src.config.schemas import DataContractValidator, NSRDB_SCHEMA, DataSource

logger = get_logger(__name__)


def detect_data_source(df: pd.DataFrame) -> str:
    """Auto-detect data source type from DataFrame columns"""
    cols_lower = {c.lower() for c in df.columns}
    
    # Check for NSRDB solar columns
    if any(c in cols_lower for c in ['ghi', 'dni', 'dhi', 'clearsky_ghi']):
        return DataSource.NSRDB
    
    # Check for consumption columns
    if any(c in cols_lower for c in ['active_power_kw', 'active_energy_kwh', 'power_kw']):
        return DataSource.METER
    
    # Check for weather columns
    if all(c in cols_lower for c in ['temperature', 'humidity', 'wind_speed']):
        return DataSource.WEATHER
    
    return "unknown"


def preprocess_file(
    input_path: Path,
    output_dir: Path,
    system_capacity_kw: float = 5.0,
    validate: bool = True,
    max_workers: int = 1
) -> dict:
    """
    Process a single CSV file.
    
    Returns:
        {
            'file': filename,
            'status': 'success' | 'error',
            'rows_input': int,
            'rows_output': int,
            'rows_removed': int,
            'output_path': str,
            'error': str (if failed)
        }
    """
    result = {
        'file': input_path.name,
        'status': 'pending',
        'rows_input': 0,
        'rows_output': 0,
        'rows_removed': 0,
        'output_path': None,
        'error': None
    }
    
    try:
        logger.info(f"Processing {input_path.name}...")
        
        # Read CSV with encoding detection
        df = None
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for enc in encodings:
            try:
                df = pd.read_csv(
                    input_path,
                    encoding=enc,
                    on_bad_lines='skip',
                    engine='python'
                )
                logger.info(f"  ✓ Read with encoding: {enc}")
                break
            except Exception as e:
                logger.debug(f"  Failed to read with {enc}: {str(e)[:60]}")
                continue
        
        if df is None or len(df) == 0:
            raise ValueError("Could not read CSV with any encoding")
        
        result['rows_input'] = len(df)
        
        # Detect data source
        data_source = detect_data_source(df)
        logger.info(f"  Detected source: {data_source}")
        
        # Handle NSRDB metadata rows
        if data_source == DataSource.NSRDB:
            header_idx = None
            for idx, line in enumerate(open(input_path, 'r', encoding='utf-8', errors='ignore')):
                normalized = line.strip().lower().replace(' ', '')
                if 'year,month,day,hour,minute' in normalized:
                    header_idx = idx
                    break
            
            if header_idx is not None and header_idx >= 1:
                logger.info(f"  NSRDB: skipping {header_idx} metadata rows")
                df = pd.read_csv(
                    input_path,
                    skiprows=header_idx,
                    on_bad_lines='skip',
                    engine='python'
                )
                result['rows_input'] = len(df)
        
        # Validate schema if enabled
        if validate:
            validation = DataContractValidator.validate_columns(df, data_source, skip_missing=True)
            if not validation['valid']:
                logger.warning(f"  Schema validation: {validation['errors']}")
            
            range_check = DataContractValidator.validate_ranges(df, data_source)
            if not range_check['valid']:
                logger.warning(f"  Range violations: {len(range_check['violations'])} columns")
        
        # Preprocess
        pipeline = DataPreprocessingPipeline()
        
        if data_source == DataSource.NSRDB:
            # Rename columns to canonical names if needed
            canonicalized = DataContractValidator.canonicalize_columns(df, data_source)
            if canonicalized:
                df = df.rename(columns=canonicalized)
                logger.info(f"  Canonicalized {len(canonicalized)} columns")
            
            processed = pipeline.preprocess_solar_data(
                df,
                system_capacity_kw=system_capacity_kw,
                validate=validate
            )
        elif data_source == DataSource.METER:
            processed = pipeline.preprocess_consumption_data(df)
        else:
            logger.warning(f"  Unknown data source, skipping preprocessing")
            processed = df
        
        # Save output
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_name = f"{data_source}_processed_{timestamp}_{input_path.stem}.csv"
        output_path = output_dir / output_name
        
        processed.to_csv(output_path, index=False)
        
        result['status'] = 'success'
        result['rows_output'] = len(processed)
        result['rows_removed'] = result['rows_input'] - result['rows_output']
        result['output_path'] = str(output_path)
        
        pct_removed = (result['rows_removed'] / result['rows_input'] * 100) if result['rows_input'] else 0
        logger.info(
            f"  ✓ Success: {result['rows_input']} → {result['rows_output']} rows "
            f"({pct_removed:.1f}% removed)"
        )
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        logger.error(f"  ✗ Error: {e}")
    
    return result


def batch_preprocess(
    input_dir: str = "ml-service/data/raw",
    output_dir: str = "ml-service/data/processed",
    system_capacity_kw: float = 5.0,
    validate: bool = True,
    max_workers: int = 4,
    pattern: str = "*.csv"
) -> dict:
    """
    Process all CSV files in input_dir.
    
    Args:
        input_dir: Directory containing raw CSV files
        output_dir: Output directory for processed files
        system_capacity_kw: System capacity for solar validation
        validate: Enable schema/range validation
        max_workers: Number of parallel workers
        pattern: File pattern to process
    
    Returns:
        Summary dict
    """
    input_path = Path(input_dir).expanduser()
    output_path = Path(output_dir).expanduser()
    
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return {'files_processed': 0, 'success': 0, 'errors': 0, 'results': []}
    
    # Find CSV files
    csv_files = list(input_path.glob(pattern))
    logger.info(f"Found {len(csv_files)} files to process")
    
    if not csv_files:
        logger.warning(f"No CSV files found matching {pattern}")
        return {'files_processed': 0, 'success': 0, 'errors': 0, 'results': []}
    
    results = []
    success_count = 0
    error_count = 0
    
    # Process files (optionally in parallel)
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    preprocess_file,
                    csv_file,
                    output_path,
                    system_capacity_kw,
                    validate
                ): csv_file
                for csv_file in csv_files
            }
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result['status'] == 'success':
                    success_count += 1
                else:
                    error_count += 1
    else:
        # Sequential processing
        for csv_file in csv_files:
            result = preprocess_file(csv_file, output_path, system_capacity_kw, validate)
            results.append(result)
            if result['status'] == 'success':
                success_count += 1
            else:
                error_count += 1
    
    summary = {
        'files_processed': len(csv_files),
        'success': success_count,
        'errors': error_count,
        'results': results,
        'output_dir': str(output_path)
    }
    
    # Log summary
    logger.info("=" * 60)
    logger.info("BATCH PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Files processed: {len(csv_files)}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Output directory: {output_path}")
    
    for result in results:
        status_icon = "✓" if result['status'] == 'success' else "✗"
        if result['status'] == 'success':
            logger.info(
                f"  {status_icon} {result['file']}: "
                f"{result['rows_input']} → {result['rows_output']} rows"
            )
        else:
            logger.info(f"  {status_icon} {result['file']}: {result['error']}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Batch preprocess CSV files for ML training"
    )
    parser.add_argument(
        "--input-dir",
        default="ml-service/data/raw",
        help="Input directory with raw CSV files"
    )
    parser.add_argument(
        "--output-dir",
        default="ml-service/data/processed",
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--capacity",
        type=float,
        default=5.0,
        help="System capacity in kW (for solar validation)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip schema/range validation"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="File pattern to match"
    )
    
    args = parser.parse_args()
    
    summary = batch_preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        system_capacity_kw=args.capacity,
        validate=not args.no_validate,
        max_workers=args.workers,
        pattern=args.pattern
    )
    
    return 0 if summary['errors'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
