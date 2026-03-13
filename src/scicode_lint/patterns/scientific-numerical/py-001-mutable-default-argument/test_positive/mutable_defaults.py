import numpy as np


def add_measurement(value, history=[]):
    history.append(value)
    return history


def configure_model(params, settings={}):
    settings["configured"] = True
    return settings


def collect_results(result, accumulator=[]):
    accumulator.append(result)
    mean = np.mean(accumulator)
    return mean


def track_gradients(grad, gradient_history=[]):
    gradient_history.append(grad)
    if len(gradient_history) > 10:
        gradient_history.pop(0)
    return gradient_history


measurements1 = add_measurement(1.0)
measurements2 = add_measurement(2.0)

config1 = configure_model({"lr": 0.01})
config2 = configure_model({"lr": 0.001})
