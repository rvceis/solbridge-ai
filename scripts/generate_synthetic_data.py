#!/usr/bin/env python3
"""
Generate synthetic training data for ML models

Usage:
    python scripts/generate_synthetic_data.py --samples 10000 --output data/raw/synthetic.csv
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_solar_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic solar generation data"""
    logger.info(f"Generating {n_samples} synthetic solar records")
    
    data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(n_samples):
        timestamp = start_date + timedelta(hours=i)
        hour = timestamp.hour
        day_of_year = timestamp.timetuple().tm_yday
        
        # Solar generation follows bell curve pattern during day
        if 6 <= hour <= 18:
            # Peak at solar noon (12)
            peak_factor = np.sin((hour - 6) * np.pi / 12)
            base_generation = 5.0 * peak_factor
            
            # Random cloud cover effect
            cloud_cover = np.random.uniform(0, 100)
            cloud_factor = 1 - (cloud_cover / 100) * 0.7
            
            # Seasonal variation
            seasonal_factor = 0.8 + 0.4 * np.cos((day_of_year - 172) * 2 * np.pi / 365)
            
            generation = base_generation * cloud_factor * seasonal_factor
            generation = max(0, generation + np.random.normal(0, 0.2))  # Add noise
        else:
            generation = np.random.uniform(0, 0.1)  # Night: minimal generation
        
        # System parameters
        temperature = 25 + 10 * np.sin((hour - 6) * np.pi / 12) + np.random.normal(0, 2)
        
        data.append({
            'timestamp': timestamp.isoformat(),
            'device_id': f'SM_H{np.random.randint(100, 999)}_001',
            'power_kw': max(0, generation),
            'energy_kwh': generation * 0.25,  # 15-min reading
            'voltage': 230 + np.random.normal(0, 5),
            'current': generation / 0.23 + np.random.normal(0, 1),
            'frequency': 50 + np.random.normal(0, 0.1),
            'power_factor': 0.98 + np.random.normal(0, 0.01),
            'temperature': temperature,
            'cloud_cover': cloud_cover if 6 <= hour <= 18 else 0,
            'system_capacity_kw': 5.0
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated solar data shape: {df.shape}")
    return df


def generate_consumption_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic consumption data"""
    logger.info(f"Generating {n_samples} synthetic consumption records")
    
    data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(n_samples):
        timestamp = start_date + timedelta(hours=i)
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = day_of_week >= 5
        
        # Base consumption pattern
        if is_weekend:
            # Higher consumption on weekends
            if 6 <= hour <= 10:
                base = 1.5  # Morning peak
            elif 10 <= hour <= 17:
                base = 0.8  # Midday low
            elif 17 <= hour <= 23:
                base = 2.5  # Evening peak
            else:
                base = 0.4  # Night
        else:
            # Weekday pattern
            if 6 <= hour <= 9:
                base = 1.0  # Morning
            elif 9 <= hour <= 18:
                base = 0.5  # Work hours (people away)
            elif 18 <= hour <= 23:
                base = 2.0  # Evening peak
            else:
                base = 0.3  # Night
        
        # Add random appliance usage
        if np.random.random() < 0.15:  # 15% chance of appliance spike
            base += np.random.uniform(1, 3)
        
        consumption = base + np.random.normal(0, 0.3)
        consumption = max(0.1, consumption)
        
        # Temperature effect (AC usage in hot hours)
        temp = 25 + 8 * np.sin((hour - 6) * np.pi / 12)
        if temp > 30:
            consumption *= 1.2
        
        data.append({
            'timestamp': timestamp.isoformat(),
            'user_id': f'buyer_{np.random.randint(1000, 9999)}',
            'power_kw': consumption,
            'energy_kwh': consumption * 0.25,
            'voltage': 230 + np.random.normal(0, 3),
            'current': consumption / 0.23 + np.random.normal(0, 0.5),
            'frequency': 50 + np.random.normal(0, 0.1),
            'power_factor': 0.95 + np.random.normal(0, 0.02),
            'temperature': temp,
            'humidity': 50 + 20 * np.sin((hour - 6) * np.pi / 12) + np.random.normal(0, 5),
            'household_size': np.random.randint(2, 6),
            'has_ac': True
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated consumption data shape: {df.shape}")
    return df


def generate_weather_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic weather data"""
    logger.info(f"Generating {n_samples} synthetic weather records")
    
    data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(n_samples):
        timestamp = start_date + timedelta(hours=i)
        hour = timestamp.hour
        day_of_year = timestamp.timetuple().tm_yday
        
        # Temperature cycle
        base_temp = 25 + 8 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
        hourly_temp = base_temp + 8 * np.sin((hour - 6) * np.pi / 12)
        temperature = hourly_temp + np.random.normal(0, 2)
        
        # Humidity pattern
        humidity = 60 + 20 * np.sin((hour - 6) * np.pi / 12) + np.random.normal(0, 5)
        humidity = np.clip(humidity, 20, 95)
        
        # Wind speed
        wind_speed = 3 + np.random.exponential(2)
        
        # Cloud cover (mostly clear, occasional clouds)
        if np.random.random() < 0.8:  # 80% clear days
            cloud_cover = np.random.uniform(0, 30)
        else:  # 20% cloudy/rainy
            cloud_cover = np.random.uniform(30, 100)
        
        # Irradiance (estimated)
        if 6 <= hour <= 18:
            clear_irradiance = 900 * np.sin((hour - 6) * np.pi / 12)
            irradiance = clear_irradiance * (1 - cloud_cover / 100 * 0.8)
        else:
            irradiance = 0
        
        data.append({
            'timestamp': timestamp.isoformat(),
            'latitude': 12.9716 + np.random.normal(0, 0.1),
            'longitude': 77.5946 + np.random.normal(0, 0.1),
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'cloud_cover': cloud_cover,
            'irradiance': max(0, irradiance + np.random.normal(0, 50)),
            'description': np.random.choice(['clear', 'partly cloudy', 'cloudy', 'rainy'])
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated weather data shape: {df.shape}")
    return df


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--samples", "-s", type=int, default=10000,
                       help="Number of samples to generate")
    parser.add_argument("--output-dir", "-o", default="data/raw",
                       help="Output directory")
    parser.add_argument("--type", "-t", default="all",
                       choices=["all", "solar", "consumption", "weather"],
                       help="Data type to generate")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.type in ["all", "solar"]:
            solar_data = generate_solar_data(args.samples)
            output_file = output_dir / "solar_synthetic.csv"
            solar_data.to_csv(output_file, index=False)
            logger.info(f"Saved to {output_file}")
        
        if args.type in ["all", "consumption"]:
            consumption_data = generate_consumption_data(args.samples)
            output_file = output_dir / "consumption_synthetic.csv"
            consumption_data.to_csv(output_file, index=False)
            logger.info(f"Saved to {output_file}")
        
        if args.type in ["all", "weather"]:
            weather_data = generate_weather_data(args.samples)
            output_file = output_dir / "weather_synthetic.csv"
            weather_data.to_csv(output_file, index=False)
            logger.info(f"Saved to {output_file}")
        
        logger.info("Synthetic data generation complete")
        return 0
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
