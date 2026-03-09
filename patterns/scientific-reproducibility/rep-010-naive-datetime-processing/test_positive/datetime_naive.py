from datetime import datetime


def log_event(event_type):
    return {"type": event_type, "timestamp": datetime.now()}


def get_current_time():
    return datetime.now()


def parse_timestamp(ts_string):
    return datetime.strptime(ts_string, "%Y-%m-%d %H:%M:%S")
