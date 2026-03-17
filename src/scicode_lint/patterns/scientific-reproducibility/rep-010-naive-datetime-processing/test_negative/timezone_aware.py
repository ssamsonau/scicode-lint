"""Sensor data pipeline with timezone-aware timestamps throughout."""

from datetime import UTC, datetime, timedelta

import numpy as np


class SensorRecorder:
    """Record sensor measurements with proper timezone handling."""

    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id
        self.readings = []

    def record(self, value: float) -> dict:
        """Store reading with UTC timestamp."""
        entry = {
            "sensor": self.sensor_id,
            "value": value,
            "timestamp": datetime.now(UTC),
        }
        self.readings.append(entry)
        return entry

    def get_readings_since(self, hours: int) -> list[dict]:
        """Filter readings from last N hours."""
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        return [r for r in self.readings if r["timestamp"] >= cutoff]


def aggregate_daily(readings: list[dict]) -> dict[str, float]:
    """Aggregate readings by UTC date."""
    daily = {}
    for r in readings:
        day = r["timestamp"].strftime("%Y-%m-%d")
        daily.setdefault(day, []).append(r["value"])
    return {day: np.mean(vals) for day, vals in daily.items()}
