import time
from datetime import datetime


def log_experiment(experiment_name, results):
    timestamp = datetime.now()
    return {"name": experiment_name, "timestamp": timestamp.isoformat(), "results": results}


def get_current_time():
    return datetime.utcnow()


def record_event(event_type):
    event = {
        "type": event_type,
        "time": datetime.now(),
        "unix_ts": datetime.fromtimestamp(time.time()),
    }
    return event


def save_measurement(sensor_data):
    """Save sensor measurement with timestamp in scientific record."""
    return {
        "data": sensor_data,
        "recorded_at": datetime.now(),
    }
