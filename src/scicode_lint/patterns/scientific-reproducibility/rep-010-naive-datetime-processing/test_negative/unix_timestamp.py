import time


def log_experiment_start():
    timestamp = time.time()
    return {"experiment_started": timestamp}


def record_metrics(loss, accuracy):
    timestamp = time.time()
    return {"timestamp": timestamp, "loss": loss, "accuracy": accuracy}


def get_elapsed_time(start_timestamp):
    return time.time() - start_timestamp


class TimestampLogger:
    def __init__(self):
        self.start = time.time()
        self.events = []

    def log(self, event_name):
        self.events.append(
            {
                "event": event_name,
                "timestamp": time.time(),
                "elapsed": time.time() - self.start,
            }
        )

    def get_log(self):
        return self.events
