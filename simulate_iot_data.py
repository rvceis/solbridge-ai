"""
IoT Data Simulator for Solar Sharing Platform
Simulates real solar panel readings for testing without physical devices

This script generates realistic IoT data that mimics:
1. Solar panel power output (based on time of day, weather)
2. Voltage and current readings
3. Temperature sensors
4. Battery state of charge (if applicable)

Usage:
  python simulate_iot_data.py --duration 24 --interval 5  # 24 hours, 5-min intervals
  python simulate_iot_data.py --realtime --device_id DEV001  # Real-time simulation
"""

import argparse
import json
import time
import random
import math
from datetime import datetime, timedelta
import requests
import sys
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:3000/api/v1"
DEFAULT_DEVICE_ID = "SOLAR_SIM_001"


class SolarPanelSimulator:
    """Simulates a solar panel with realistic power curves"""
    
    def __init__(
        self,
        capacity_kw: float = 5.0,
        efficiency: float = 0.18,
        latitude: float = 12.97,  # Bangalore
        longitude: float = 77.59
    ):
        self.capacity_kw = capacity_kw
        self.efficiency = efficiency
        self.latitude = latitude
        self.longitude = longitude
        
        # Panel specs
        self.voc = 40.0  # Open circuit voltage
        self.isc = 10.0  # Short circuit current
        self.vmp = 32.0  # Max power voltage
        self.imp = 8.0   # Max power current
        
    def get_solar_position(self, dt: datetime) -> dict:
        """Calculate approximate solar position"""
        day_of_year = dt.timetuple().tm_yday
        hour = dt.hour + dt.minute / 60
        
        # Solar declination
        declination = 23.45 * math.sin(math.radians(360/365 * (284 + day_of_year)))
        
        # Hour angle
        hour_angle = 15 * (hour - 12)
        
        # Solar elevation
        lat_rad = math.radians(self.latitude)
        dec_rad = math.radians(declination)
        hour_rad = math.radians(hour_angle)
        
        elevation = math.degrees(math.asin(
            math.sin(lat_rad) * math.sin(dec_rad) +
            math.cos(lat_rad) * math.cos(dec_rad) * math.cos(hour_rad)
        ))
        
        # Solar azimuth (simplified)
        azimuth = 180 + hour_angle
        
        return {
            "elevation": max(0, elevation),
            "azimuth": azimuth,
            "day_of_year": day_of_year
        }
    
    def get_weather_factor(self, dt: datetime) -> float:
        """Simulate weather effects (cloud cover, etc.)"""
        # Random weather patterns with some persistence
        hour = dt.hour
        
        # Morning fog (6-9 AM)
        if 6 <= hour < 9:
            return random.uniform(0.6, 0.9)
        
        # Clear midday (10 AM - 3 PM)
        if 10 <= hour < 15:
            return random.uniform(0.85, 1.0)
        
        # Afternoon clouds (3-6 PM)
        if 15 <= hour < 18:
            return random.uniform(0.7, 0.95)
        
        # Night
        return 0.0
    
    def generate_reading(self, dt: datetime = None) -> dict:
        """Generate a single realistic reading"""
        if dt is None:
            dt = datetime.now()
        
        solar_pos = self.get_solar_position(dt)
        elevation = solar_pos["elevation"]
        
        # Base irradiance from solar position
        if elevation <= 0:
            irradiance = 0
        else:
            # Peak irradiance ~1000 W/m² at solar noon
            irradiance = 1000 * math.sin(math.radians(elevation))
        
        # Apply weather factor
        weather_factor = self.get_weather_factor(dt)
        irradiance *= weather_factor
        
        # Calculate power output
        if irradiance > 0:
            power_kw = (irradiance / 1000) * self.capacity_kw * self.efficiency
            power_kw *= random.uniform(0.95, 1.02)  # Small random variation
            power_kw = max(0, min(power_kw, self.capacity_kw))
        else:
            power_kw = 0
        
        # Calculate voltage and current
        if power_kw > 0:
            # Voltage varies with temperature and irradiance
            temp_factor = 1 - 0.004 * (25 - 25)  # Temperature coefficient
            voltage = self.vmp * temp_factor * (0.9 + 0.1 * (irradiance / 1000))
            current = (power_kw * 1000) / voltage if voltage > 0 else 0
        else:
            voltage = 0
            current = 0
        
        # Temperature (panel temperature)
        ambient_temp = 25 + 10 * math.sin(math.radians((dt.hour - 6) * 15))
        ambient_temp = max(15, min(40, ambient_temp))
        panel_temp = ambient_temp + (irradiance / 1000) * 20  # Panels heat up
        
        # Energy produced (cumulative for the day)
        # This would normally be tracked over time
        
        return {
            "timestamp": dt.isoformat(),
            "power_kw": round(power_kw, 3),
            "power_w": round(power_kw * 1000, 1),
            "voltage": round(voltage, 2),
            "current": round(current, 2),
            "irradiance": round(irradiance, 1),
            "panel_temperature": round(panel_temp, 1),
            "ambient_temperature": round(ambient_temp, 1),
            "solar_elevation": round(elevation, 1),
            "weather_factor": round(weather_factor, 2),
            "efficiency": round(self.efficiency * 100, 1),
            "status": "producing" if power_kw > 0.01 else "idle"
        }


class IoTDataGenerator:
    """Generate and optionally send IoT data to backend"""
    
    def __init__(self, device_id: str, api_url: str = None, auth_token: str = None):
        self.device_id = device_id
        self.api_url = api_url or API_BASE_URL
        self.auth_token = auth_token
        self.simulator = SolarPanelSimulator()
        
    def generate_batch(self, start_time: datetime, duration_hours: int, interval_minutes: int) -> list:
        """Generate a batch of readings"""
        readings = []
        current_time = start_time
        end_time = start_time + timedelta(hours=duration_hours)
        
        while current_time < end_time:
            reading = self.simulator.generate_reading(current_time)
            reading["device_id"] = self.device_id
            readings.append(reading)
            current_time += timedelta(minutes=interval_minutes)
        
        return readings
    
    def send_reading(self, reading: dict) -> bool:
        """Send a single reading to the backend API"""
        if not self.auth_token:
            print("Warning: No auth token, skipping API call")
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "device_id": self.device_id,
                "readings": [{
                    "measurement_type": "solar_generation",
                    "value": reading["power_kw"],
                    "unit": "kW",
                    "metadata": {
                        "voltage": reading["voltage"],
                        "current": reading["current"],
                        "temperature": reading["panel_temperature"],
                        "irradiance": reading["irradiance"]
                    }
                }]
            }
            
            response = requests.post(
                f"{self.api_url}/iot/readings",
                headers=headers,
                json=payload,
                timeout=5
            )
            
            return response.status_code == 200 or response.status_code == 201
            
        except Exception as e:
            print(f"Error sending reading: {e}")
            return False
    
    def run_realtime(self, interval_seconds: int = 60):
        """Run real-time simulation"""
        print(f"Starting real-time IoT simulation for device: {self.device_id}")
        print(f"Interval: {interval_seconds} seconds")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                reading = self.simulator.generate_reading()
                reading["device_id"] = self.device_id
                
                # Print reading
                print(f"[{reading['timestamp']}]")
                print(f"  Power: {reading['power_kw']:.3f} kW ({reading['power_w']:.1f} W)")
                print(f"  Voltage: {reading['voltage']:.2f} V, Current: {reading['current']:.2f} A")
                print(f"  Panel Temp: {reading['panel_temperature']:.1f}°C")
                print(f"  Irradiance: {reading['irradiance']:.1f} W/m²")
                print(f"  Status: {reading['status']}")
                
                # Send to API if configured
                if self.auth_token:
                    success = self.send_reading(reading)
                    print(f"  API: {'✓ Sent' if success else '✗ Failed'}")
                
                print()
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\nSimulation stopped.")


def save_readings_to_file(readings: list, filename: str):
    """Save readings to JSON file"""
    output_dir = Path(__file__).parent / "data" / "simulated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(readings, f, indent=2)
    
    print(f"Saved {len(readings)} readings to: {filepath}")
    return filepath


def save_readings_to_csv(readings: list, filename: str):
    """Save readings to CSV file for easy analysis"""
    import csv
    
    output_dir = Path(__file__).parent / "data" / "simulated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    
    if readings:
        keys = readings[0].keys()
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(readings)
    
    print(f"Saved {len(readings)} readings to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="IoT Data Simulator for Solar Sharing Platform")
    parser.add_argument("--device_id", type=str, default=DEFAULT_DEVICE_ID,
                       help="Device ID for the simulated panel")
    parser.add_argument("--duration", type=int, default=24,
                       help="Duration in hours for batch generation")
    parser.add_argument("--interval", type=int, default=5,
                       help="Interval in minutes between readings")
    parser.add_argument("--realtime", action="store_true",
                       help="Run real-time simulation")
    parser.add_argument("--api_url", type=str, default=API_BASE_URL,
                       help="Backend API URL")
    parser.add_argument("--token", type=str, default=None,
                       help="JWT auth token for API calls")
    parser.add_argument("--output", type=str, default=None,
                       help="Output filename (without extension)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Solar Sharing Platform - IoT Data Simulator")
    print("=" * 60)
    print(f"Device ID: {args.device_id}")
    
    generator = IoTDataGenerator(
        device_id=args.device_id,
        api_url=args.api_url,
        auth_token=args.token
    )
    
    if args.realtime:
        # Real-time mode
        generator.run_realtime(interval_seconds=args.interval)
    else:
        # Batch generation mode
        print(f"Generating {args.duration} hours of data at {args.interval}-minute intervals...")
        
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        readings = generator.generate_batch(
            start_time=start_time,
            duration_hours=args.duration,
            interval_minutes=args.interval
        )
        
        # Calculate statistics
        power_values = [r["power_kw"] for r in readings]
        total_energy = sum(power_values) * (args.interval / 60)  # kWh
        peak_power = max(power_values)
        avg_power = sum(power_values) / len(power_values)
        producing_hours = sum(1 for p in power_values if p > 0.01) * (args.interval / 60)
        
        print(f"\nGenerated {len(readings)} readings")
        print(f"  Total Energy: {total_energy:.2f} kWh")
        print(f"  Peak Power: {peak_power:.3f} kW")
        print(f"  Avg Power: {avg_power:.3f} kW")
        print(f"  Producing Hours: {producing_hours:.1f} hours")
        
        # Save to files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = args.output or f"solar_sim_{args.device_id}_{timestamp}"
        
        save_readings_to_json(readings, f"{base_filename}.json")
        save_readings_to_csv(readings, f"{base_filename}.csv")
        
        print("\nSimulation complete!")


def save_readings_to_json(readings: list, filename: str):
    """Save readings to JSON file"""
    output_dir = Path(__file__).parent / "data" / "simulated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(readings, f, indent=2)
    
    print(f"Saved {len(readings)} readings to: {filepath}")
    return filepath


if __name__ == "__main__":
    main()
