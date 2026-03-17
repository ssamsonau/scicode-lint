"""Recording timestamps in scientific data."""

from datetime import datetime

import pandas as pd


def record_measurement(value, sensor_id):
    """Record a sensor measurement with timestamp."""
    timestamp = datetime.now()
    return {
        "timestamp": timestamp,
        "sensor_id": sensor_id,
        "value": value,
    }


def create_experiment_dataframe(measurements):
    """Create dataframe with measurement data."""
    df = pd.DataFrame(measurements)
    df["recorded_at"] = datetime.now()
    return df
