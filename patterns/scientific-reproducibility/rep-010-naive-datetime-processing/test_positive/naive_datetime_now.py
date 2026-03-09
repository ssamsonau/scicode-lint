from datetime import datetime


def log_experiment_start():
    start_time = datetime.now()
    return {"experiment_started": start_time.isoformat()}


def record_training_metrics(loss, accuracy):
    timestamp = datetime.utcnow()
    return {"timestamp": timestamp, "loss": loss, "accuracy": accuracy}


if __name__ == "__main__":
    result = log_experiment_start()
    print(f"Started: {result}")
