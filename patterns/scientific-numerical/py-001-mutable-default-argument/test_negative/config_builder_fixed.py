def build_config(name, options=None):
    if options is None:
        options = {}
    options["name"] = name
    return options


def create_metrics(values, history=None):
    if history is None:
        history = []
    history.extend(values)
    return {"current": values, "history": history}


class Accumulator:
    def __init__(self, initial_values=None):
        self.values = initial_values if initial_values is not None else []

    def add(self, value):
        self.values.append(value)
