import numpy as np


def collect_scalars(data_source):
    values = []
    for item in data_source:
        values.append(item.compute_value())
    return np.array(values)


def gather_measurements(sensors):
    readings = []
    for sensor in sensors:
        readings.append(sensor.read())
    return np.array(readings)


def vectorized_transform(data):
    return np.array([x * 2 + 1 for x in data])


def stack_at_end(matrices):
    result_list = []
    for m in matrices:
        result_list.append(m.flatten())
    return np.vstack(result_list)
