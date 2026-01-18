"""
Data contracts and schema definitions for solar/consumption/weather datasets.

This module defines the expected column names, types, units, and ranges
for NSRDB (solar), meter (consumption), and weather data.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class DataSource(str, Enum):
    """Data source types"""
    NSRDB = "nsrdb"  # Solar irradiance
    METER = "meter"  # Consumption data
    WEATHER = "weather"  # Weather data
    SOLAR_INVERTER = "solar_inverter"  # Real-time solar generation


@dataclass
class ColumnSchema:
    """Schema for a single column"""
    name: str
    type: str  # 'float', 'int', 'string', 'datetime'
    unit: str  # e.g., 'W/m2', 'kW', 'degC', '%'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""
    required: bool = True


class NSRDB_SCHEMA:
    """NSRDB solar irradiance dataset schema (PSM v3.2.2)"""
    
    METADATA_ROWS = 2  # First 2 rows are metadata
    
    COLUMNS: List[ColumnSchema] = [
        # Temporal
        ColumnSchema("Year", "int", "year", 1998, 2023, "Calendar year"),
        ColumnSchema("Month", "int", "month", 1, 12, "Calendar month"),
        ColumnSchema("Day", "int", "day", 1, 31, "Calendar day"),
        ColumnSchema("Hour", "int", "hour", 0, 23, "Hour of day (0-23 UTC)"),
        ColumnSchema("Minute", "int", "minute", 0, 59, "Minute of hour"),
        
        # Location
        ColumnSchema("Latitude", "float", "degrees", -90, 90, "Site latitude"),
        ColumnSchema("Longitude", "float", "degrees", -180, 180, "Site longitude"),
        ColumnSchema("Elevation", "float", "meters", -300, 8850, "Site elevation"),
        
        # Solar Irradiance (W/m2) - main targets
        ColumnSchema("GHI", "float", "W/m2", 0, 1500, "Global horizontal irradiance"),
        ColumnSchema("DNI", "float", "W/m2", 0, 1500, "Direct normal irradiance"),
        ColumnSchema("DHI", "float", "W/m2", 0, 1500, "Diffuse horizontal irradiance"),
        ColumnSchema("GHIUncertainty", "float", "percent", 0, 100, "GHI uncertainty"),
        ColumnSchema("DNIUncertainty", "float", "percent", 0, 100, "DNI uncertainty"),
        ColumnSchema("DHIUncertainty", "float", "percent", 0, 100, "DHI uncertainty"),
        
        # Irradiance (plane-of-array)
        ColumnSchema("POA_Global", "float", "W/m2", 0, 1500, "Plane-of-array global"),
        ColumnSchema("POA_Direct", "float", "W/m2", 0, 1500, "Plane-of-array direct"),
        ColumnSchema("POA_Diffuse", "float", "W/m2", 0, 1500, "Plane-of-array diffuse"),
        
        # Weather
        ColumnSchema("Temperature", "float", "degC", -40, 60, "Air temperature"),
        ColumnSchema("Pressure", "float", "mb", 800, 1100, "Atmospheric pressure"),
        ColumnSchema("RelativeHumidity", "float", "percent", 0, 100, "Relative humidity"),
        ColumnSchema("Precipitable Water", "float", "cm", 0, 10, "Precipitable water"),
        ColumnSchema("WindSpeed", "float", "m/s", 0, 30, "Wind speed at 10m"),
        ColumnSchema("WindDirection", "float", "degrees", 0, 360, "Wind direction"),
        
        # Sky/Cloud (fill flags)
        ColumnSchema("ClearSky GHI", "float", "W/m2", 0, 1500, "Modeled clear-sky GHI"),
        ColumnSchema("ClearSky DNI", "float", "W/m2", 0, 1500, "Modeled clear-sky DNI"),
        ColumnSchema("ClearSky DHI", "float", "W/m2", 0, 1500, "Modeled clear-sky DHI"),
        ColumnSchema("Dew Point", "float", "degC", -40, 40, "Dew point temperature"),
        ColumnSchema("Solar Zenith Angle", "float", "degrees", 0, 180, "Solar zenith angle"),
    ]
    
    # Map possible NSRDB header variations to canonical names
    COLUMN_ALIASES: Dict[str, List[str]] = {
        "GHI": ["Global Horizontal Irradiance", "ghi"],
        "DNI": ["Direct Normal Irradiance", "dni"],
        "DHI": ["Diffuse Horizontal Irradiance", "dhi"],
        "Temperature": ["Air Temperature", "temp", "temperature_c"],
        "Pressure": ["Atmospheric Pressure", "press"],
        "RelativeHumidity": ["Relative Humidity", "humidity", "rh"],
        "WindSpeed": ["Wind Speed", "wind_speed", "ws"],
        "WindDirection": ["Wind Direction", "wind_dir"],
    }


class METER_SCHEMA:
    """Smart meter consumption dataset schema"""
    
    COLUMNS: List[ColumnSchema] = [
        # Temporal
        ColumnSchema("timestamp", "datetime", "", description="UTC timestamp"),
        
        # Consumption (kWh or kW)
        ColumnSchema("active_energy_kwh", "float", "kWh", 0, 1000, "Active energy (cumulative)"),
        ColumnSchema("active_power_kw", "float", "kW", 0, 100, "Active power demand"),
        ColumnSchema("reactive_power_kvar", "float", "kVAR", 0, 100, "Reactive power"),
        ColumnSchema("apparent_power_kva", "float", "kVA", 0, 100, "Apparent power"),
        ColumnSchema("power_factor", "float", "", 0, 1, "Power factor"),
        
        # Voltage/Current
        ColumnSchema("voltage_v", "float", "V", 190, 260, "Line voltage"),
        ColumnSchema("current_a", "float", "A", 0, 100, "Line current"),
        ColumnSchema("frequency_hz", "float", "Hz", 49.5, 50.5, "Grid frequency"),
        
        # Quality flags
        ColumnSchema("data_quality_flag", "int", "", 0, 1, "0=valid, 1=questionable"),
        ColumnSchema("estimated_flag", "int", "", 0, 1, "0=measured, 1=estimated"),
    ]


class WEATHER_SCHEMA:
    """Weather dataset schema (subset of NSRDB)"""
    
    COLUMNS: List[ColumnSchema] = [
        ColumnSchema("timestamp", "datetime", "", description="UTC timestamp"),
        ColumnSchema("temperature_c", "float", "degC", -40, 60),
        ColumnSchema("humidity_percent", "float", "percent", 0, 100),
        ColumnSchema("wind_speed_ms", "float", "m/s", 0, 50),
        ColumnSchema("pressure_mb", "float", "mb", 800, 1100),
        ColumnSchema("cloud_cover_percent", "float", "percent", 0, 100),
    ]


class DataContractValidator:
    """Validates data against defined schemas"""
    
    SCHEMAS: Dict[str, List[ColumnSchema]] = {
        DataSource.NSRDB: NSRDB_SCHEMA.COLUMNS,
        DataSource.METER: METER_SCHEMA.COLUMNS,
        DataSource.WEATHER: WEATHER_SCHEMA.COLUMNS,
    }
    
    @classmethod
    def get_schema(cls, source: str) -> Optional[List[ColumnSchema]]:
        """Get schema for a data source"""
        try:
            ds = DataSource(source.lower())
            return cls.SCHEMAS[ds]
        except (ValueError, KeyError):
            return None
    
    @classmethod
    def validate_columns(cls, df, source: str, skip_missing: bool = False) -> Dict[str, Any]:
        """
        Validate DataFrame columns against schema.
        
        Args:
            df: Input DataFrame
            source: Data source type (e.g., 'nsrdb', 'meter')
            skip_missing: If True, allow missing columns; if False, fail on missing
        
        Returns:
            {
                'valid': bool,
                'missing_columns': list,
                'extra_columns': list,
                'errors': list
            }
        """
        schema = cls.get_schema(source)
        if not schema:
            return {
                'valid': False,
                'errors': [f"Unknown data source: {source}"],
                'missing_columns': [],
                'extra_columns': []
            }
        
        required_cols = {s.name for s in schema if s.required}
        df_cols = set(df.columns)
        
        missing = required_cols - df_cols
        extra = df_cols - required_cols
        errors = []
        
        if missing and not skip_missing:
            errors.append(f"Missing required columns: {missing}")
        
        return {
            'valid': len(errors) == 0,
            'missing_columns': list(missing),
            'extra_columns': list(extra),
            'errors': errors
        }
    
    @classmethod
    def validate_ranges(cls, df, source: str) -> Dict[str, Any]:
        """
        Validate column ranges against schema.
        
        Returns:
            {
                'valid': bool,
                'violations': {col_name: violation_details}
            }
        """
        schema = cls.get_schema(source)
        if not schema:
            return {'valid': False, 'violations': {}}
        
        violations = {}
        for col_schema in schema:
            col = col_schema.name
            if col not in df.columns:
                continue
            
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            col_violations = []
            
            if col_schema.min_value is not None:
                below_min = (col_data < col_schema.min_value).sum()
                if below_min > 0:
                    col_violations.append(f"{below_min} values below min {col_schema.min_value}")
            
            if col_schema.max_value is not None:
                above_max = (col_data > col_schema.max_value).sum()
                if above_max > 0:
                    col_violations.append(f"{above_max} values above max {col_schema.max_value}")
            
            if col_violations:
                violations[col] = col_violations
        
        return {
            'valid': len(violations) == 0,
            'violations': violations
        }
    
    @classmethod
    def canonicalize_columns(cls, df, source: str) -> Dict[str, str]:
        """
        Map common column name variations to canonical names.
        
        Returns:
            Mapping of original → canonical names
        """
        if source.lower() != DataSource.NSRDB:
            return {}
        
        mapping = {}
        df_cols_lower = {c.lower(): c for c in df.columns}
        
        for canonical, aliases in NSRDB_SCHEMA.COLUMN_ALIASES.items():
            for alias in [canonical.lower()] + [a.lower() for a in aliases]:
                if alias in df_cols_lower:
                    mapping[df_cols_lower[alias]] = canonical
                    break
        
        return mapping
