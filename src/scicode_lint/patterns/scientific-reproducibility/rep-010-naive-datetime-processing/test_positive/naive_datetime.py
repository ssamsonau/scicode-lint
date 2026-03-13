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


def create_run_id():
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
