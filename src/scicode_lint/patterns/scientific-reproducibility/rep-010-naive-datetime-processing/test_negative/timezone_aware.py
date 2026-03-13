from datetime import datetime, timezone


def log_experiment(experiment_name, results):
    timestamp = datetime.now(timezone.utc)
    return {"name": experiment_name, "timestamp": timestamp.isoformat(), "results": results}


def get_current_time():
    return datetime.now(tz=timezone.utc)


def record_event(event_type):
    event = {
        "type": event_type,
        "time": datetime.now(timezone.utc),
    }
    return event


def create_run_id():
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


class ExperimentTracker:
    def __init__(self, name):
        self.name = name
        self.start_time = datetime.now(timezone.utc)
        self.events = []

    def log_event(self, event_name):
        self.events.append(
            {
                "event": event_name,
                "timestamp": datetime.now(timezone.utc),
            }
        )

    def get_duration(self):
        return datetime.now(timezone.utc) - self.start_time
