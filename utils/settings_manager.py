"""
Centralized settings management for the ML pipeline.
Parses a unified JSON configuration into strictly typed dataclasses.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .pipeline_enums import OptimizationGoal
from .pipeline_errors import ConfigurationFault

logger = logging.getLogger(__name__)

# =========================================================================
# SETTINGS DATACLASSES
# =========================================================================


@dataclass
class SchemaSettings:
    mandatory_features: List[str]
    categorical_features: List[str]
    target_variable: str


@dataclass
class TemporalSettings:
    train_start_date: str
    train_end_date: str
    prediction_window_days: int


@dataclass
class SegmentSettings:
    min_activity_threshold: int
    max_activity_threshold: int = None


@dataclass
class PipelineSettings:
    """Master typed object representing the entire application state."""
    project_name: str
    schema: SchemaSettings
    temporal: TemporalSettings
    segments: Dict[str, SegmentSettings]


# =========================================================================
# SETTINGS MANAGER
# =========================================================================

class SettingsManager:
    """Handles the ingestion and materialization of pipeline settings."""

    def __init__(
        self,
        settings_path: str = "configs/config.json",
        goal: OptimizationGoal = OptimizationGoal.BALANCED,
    ) -> None:
        self.settings_path = Path(settings_path)
        self.goal = goal
        logger.info(f"SettingsManager initialized | Goal: {self.goal.value}")

    def load(self) -> PipelineSettings:
        """Loads and parses the JSON configuration."""
        try:
            raw_data = self._read_json(self.settings_path)
            return self._build_settings_object(raw_data)
        except Exception as e:
            raise ConfigurationFault(f"Failed to load settings: {str(e)}")

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise ConfigurationFault(
                f"Settings file missing at {path}", code="SYS-201")
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _build_settings_object(self, data: Dict[str, Any]) -> PipelineSettings:
        schema = SchemaSettings(**data.get("schema", {}))
        temporal = TemporalSettings(**data.get("temporal", {}))

        segments = {
            k: SegmentSettings(**v)
            for k, v in data.get("segments", {}).items()
        }

        return PipelineSettings(
            project_name=data.get("project_name", "Unknown"),
            schema=schema,
            temporal=temporal,
            segments=segments
        )
