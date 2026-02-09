"""
Athlete configuration loader for multi-athlete pipeline v60.

Loads athlete-specific settings from athlete.yml and provides backward
compatibility with v51 hardcoded config.py values.

Usage:
    from athlete_config import AthleteConfig
    
    config = AthleteConfig.load("athletes/paul/athlete.yml")
    print(config.mass_kg)
    print(config.power_mode)  # "stryd" or "gap"
    print(config.peak_cp_watts)  # Only for Stryd mode
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # Graceful degradation for v51 compatibility


@dataclass
class StrydMassCorrection:
    """Single mass correction period for Stryd."""
    start_date: str
    end_date: str
    configured_kg: float
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> StrydMassCorrection:
        return cls(
            start_date=str(d["start_date"]),
            end_date=str(d["end_date"]),
            configured_kg=float(d["configured_kg"])
        )


@dataclass
class StrydConfig:
    """Stryd-specific configuration."""
    peak_cp_watts: float
    eras: Dict[str, str]
    mass_corrections: List[StrydMassCorrection]
    re_reference_era: str
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> StrydConfig:
        return cls(
            peak_cp_watts=float(d.get("peak_cp_watts", 372)),
            eras=d.get("eras", {}),
            mass_corrections=[StrydMassCorrection.from_dict(mc) for mc in d.get("mass_corrections", [])],
            re_reference_era=str(d.get("re_reference_era", "s4"))
        )


@dataclass
class GAPConfig:
    """GAP mode configuration."""
    re_constant: float
    peak_cp_watts: Optional[float] = None  # Bootstrapped from first race
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GAPConfig:
        return cls(
            re_constant=float(d.get("re_constant", 0.92)),
            peak_cp_watts=d.get("peak_cp_watts")
        )


@dataclass
class PowerConfig:
    """Power/fitness mode configuration."""
    mode: str  # "stryd" or "gap"
    stryd: Optional[StrydConfig] = None
    gap: Optional[GAPConfig] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PowerConfig:
        mode = str(d.get("mode", "stryd"))
        return cls(
            mode=mode,
            stryd=StrydConfig.from_dict(d.get("stryd", {})) if mode == "stryd" else None,
            gap=GAPConfig.from_dict(d.get("gap", {})) if mode == "gap" else None
        )


@dataclass
class DataSourceConfig:
    """Data source configuration."""
    source: str  # "intervals", "fit_folder", "strava_export"
    intervals_athlete_id: Optional[str] = None
    intervals_api_key: Optional[str] = None
    fit_folder_path: Optional[str] = None
    strava_export_zip_path: Optional[str] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DataSourceConfig:
        source = str(d.get("source", "intervals"))
        
        # Try environment variables first, then YAML
        intervals_id = os.getenv("INTERVALS_ATHLETE_ID")
        intervals_key = os.getenv("INTERVALS_API_KEY")
        
        intervals_dict = d.get("intervals", {})
        if not intervals_id:
            intervals_id = intervals_dict.get("athlete_id")
        if not intervals_key:
            intervals_key = intervals_dict.get("api_key")
        
        return cls(
            source=source,
            intervals_athlete_id=intervals_id,
            intervals_api_key=intervals_key,
            fit_folder_path=d.get("fit_folder", {}).get("path"),
            strava_export_zip_path=d.get("strava_export", {}).get("zip_path")
        )


@dataclass
class PipelineConfig:
    """Optional pipeline parameter overrides."""
    rf_window_duration_s: float = 2400.0
    rf_trend_window_days: int = 42
    rf_trend_min_periods: int = 5
    ctl_time_constant: int = 42
    atl_time_constant: int = 7
    temp_baseline_c: float = 10.0
    easy_rf_hr_min: int = 120
    easy_rf_np_cp_max: float = 0.85
    easy_rf_vi_max: float = 1.10
    easy_rf_dist_min_km: float = 4.0
    easy_rf_ema_span: int = 15
    easy_rf_z_window: int = 30
    alert1_rfl_drop: float = 0.02
    alert1_ctl_rise: float = 3.0
    alert1_window_days: int = 28
    alert1b_rfl_gap: float = 0.02
    alert1b_peak_window_days: int = 90
    alert1b_race_window_days: int = 7
    alert2_tsb_threshold: float = -15.0
    alert2_count: int = 3
    alert2_window: int = 5
    alert3b_z_threshold: float = -2.0
    alert5_gap_threshold: float = -0.03
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PipelineConfig:
        easy = d.get("easy_rf", {})
        alerts = d.get("alerts", {})
        
        return cls(
            rf_window_duration_s=float(d.get("rf_window_duration_s", 2400.0)),
            rf_trend_window_days=int(d.get("rf_trend_window_days", 42)),
            rf_trend_min_periods=int(d.get("rf_trend_min_periods", 5)),
            ctl_time_constant=int(d.get("ctl_time_constant", 42)),
            atl_time_constant=int(d.get("atl_time_constant", 7)),
            temp_baseline_c=float(d.get("temp_baseline_c", 10.0)),
            easy_rf_hr_min=int(easy.get("hr_min", 120)),
            easy_rf_np_cp_max=float(easy.get("np_cp_max", 0.85)),
            easy_rf_vi_max=float(easy.get("vi_max", 1.10)),
            easy_rf_dist_min_km=float(easy.get("dist_min_km", 4.0)),
            easy_rf_ema_span=int(easy.get("ema_span", 15)),
            easy_rf_z_window=int(easy.get("z_window", 30)),
            alert1_rfl_drop=float(alerts.get("alert1_rfl_drop", 0.02)),
            alert1_ctl_rise=float(alerts.get("alert1_ctl_rise", 3.0)),
            alert1_window_days=int(alerts.get("alert1_window_days", 28)),
            alert1b_rfl_gap=float(alerts.get("alert1b_rfl_gap", 0.02)),
            alert1b_peak_window_days=int(alerts.get("alert1b_peak_window_days", 90)),
            alert1b_race_window_days=int(alerts.get("alert1b_race_window_days", 7)),
            alert2_tsb_threshold=float(alerts.get("alert2_tsb_threshold", -15.0)),
            alert2_count=int(alerts.get("alert2_count", 3)),
            alert2_window=int(alerts.get("alert2_window", 5)),
            alert3b_z_threshold=float(alerts.get("alert3b_z_threshold", -2.0)),
            alert5_gap_threshold=float(alerts.get("alert5_gap_threshold", -0.03))
        )


@dataclass
class AthleteConfig:
    """Complete athlete configuration."""
    name: str
    mass_kg: float
    date_of_birth: str
    gender: str
    power: PowerConfig
    data: DataSourceConfig
    pipeline: PipelineConfig
    
    # Convenience properties
    @property
    def power_mode(self) -> str:
        """Get power mode: 'stryd' or 'gap'."""
        return self.power.mode
    
    @property
    def peak_cp_watts(self) -> Optional[float]:
        """Get peak CP (from Stryd config or GAP bootstrap)."""
        if self.power_mode == "stryd" and self.power.stryd:
            return self.power.stryd.peak_cp_watts
        elif self.power_mode == "gap" and self.power.gap:
            return self.power.gap.peak_cp_watts
        return None
    
    @property
    def stryd_eras(self) -> Dict[str, str]:
        """Get Stryd era dates (empty dict if GAP mode)."""
        if self.power_mode == "stryd" and self.power.stryd:
            return self.power.stryd.eras
        return {}
    
    @property
    def stryd_mass_corrections(self) -> List[StrydMassCorrection]:
        """Get Stryd mass corrections (empty list if GAP mode)."""
        if self.power_mode == "stryd" and self.power.stryd:
            return self.power.stryd.mass_corrections
        return []
    
    @property
    def re_reference_era(self) -> str:
        """Get RE reference era (default 's4' if GAP mode)."""
        if self.power_mode == "stryd" and self.power.stryd:
            return self.power.stryd.re_reference_era
        return "s4"  # Not used in GAP mode, but provide default
    
    @property
    def re_constant(self) -> float:
        """Get RE constant for GAP mode (0.92 default)."""
        if self.power_mode == "gap" and self.power.gap:
            return self.power.gap.re_constant
        return 0.92
    
    @classmethod
    def load(cls, yaml_path: str) -> AthleteConfig:
        """
        Load athlete configuration from YAML file.
        
        Args:
            yaml_path: Path to athlete.yml file
            
        Returns:
            AthleteConfig instance
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is invalid or required fields missing
        """
        if yaml is None:
            raise ImportError("PyYAML is required for athlete config. Install with: pip install pyyaml")
        
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Athlete config not found: {yaml_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML structure in {yaml_path}")
        
        athlete_data = data.get("athlete", {})
        
        return cls(
            name=str(athlete_data.get("name", "Unknown")),
            mass_kg=float(athlete_data.get("mass_kg", 76.0)),
            date_of_birth=str(athlete_data.get("date_of_birth", "1970-01-01")),
            gender=str(athlete_data.get("gender", "male")),
            power=PowerConfig.from_dict(data.get("power", {})),
            data=DataSourceConfig.from_dict(data.get("data", {})),
            pipeline=PipelineConfig.from_dict(data.get("pipeline", {}))
        )
    
    @classmethod
    def load_v51_defaults(cls) -> AthleteConfig:
        """
        Load v51 hardcoded defaults for backward compatibility.
        This returns Paul's configuration as it exists in v51 config.py.
        """
        return cls(
            name="Paul",
            mass_kg=76.0,
            date_of_birth="1969-05-27",
            gender="male",
            power=PowerConfig(
                mode="stryd",
                stryd=StrydConfig(
                    peak_cp_watts=372,
                    eras={
                        "pre_stryd": "1900-01-01",
                        "v1": "2017-05-05",
                        "repl": "2017-09-12",
                        "air": "2019-09-07",
                        "s4": "2023-01-03",
                        "s5": "2025-12-17",
                    },
                    mass_corrections=[
                        StrydMassCorrection("2017-07-08", "2017-09-10", 77.0),
                        StrydMassCorrection("2017-09-11", "2019-09-06", 79.0),
                    ],
                    re_reference_era="s4"
                ),
                gap=None
            ),
            data=DataSourceConfig(
                source="intervals",
                intervals_athlete_id=os.getenv("INTERVALS_ATHLETE_ID"),
                intervals_api_key=os.getenv("INTERVALS_API_KEY")
            ),
            pipeline=PipelineConfig()
        )


# Convenience function for backward compatibility
def load_athlete_config(yaml_path: Optional[str] = None) -> AthleteConfig:
    """
    Load athlete configuration.
    
    Args:
        yaml_path: Path to athlete.yml. If None, loads v51 defaults.
        
    Returns:
        AthleteConfig instance
    """
    if yaml_path is None:
        return AthleteConfig.load_v51_defaults()
    return AthleteConfig.load(yaml_path)


if __name__ == "__main__":
    # Test loading
    import sys
    
    if len(sys.argv) > 1:
        config = AthleteConfig.load(sys.argv[1])
    else:
        config = AthleteConfig.load_v51_defaults()
    
    print(f"Athlete: {config.name}")
    print(f"Power mode: {config.power_mode}")
    print(f"Peak CP: {config.peak_cp_watts}")
    print(f"Mass: {config.mass_kg} kg")
    print(f"DOB: {config.date_of_birth}")
    print(f"Data source: {config.data.source}")
