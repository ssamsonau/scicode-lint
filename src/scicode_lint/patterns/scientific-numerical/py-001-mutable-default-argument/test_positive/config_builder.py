def build_config(name, options={}):
    options["name"] = name
    return options


def create_metrics(values, history=[]):
    history.extend(values)
    return {"current": values, "history": history}


class Accumulator:
    def __init__(self, initial_values=[]):
        self.values = initial_values

    def add(self, value):
        self.values.append(value)
