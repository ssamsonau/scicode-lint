import numpy as np


def memoized_computation(x, cache={}):
    if x in cache:
        return cache[x]
    result = np.exp(x)
    cache[x] = result
    return result


def build_features(data, feature_list=[]):
    for i in range(len(data)):
        feature_list.append(data[i] * 2)
    return feature_list


def register_callback(func, callbacks=[]):
    callbacks.append(func)
    return len(callbacks)


def store_activations(layer_output, activation_store={}):
    layer_id = len(activation_store)
    activation_store[layer_id] = layer_output
    return activation_store


val1 = memoized_computation(1.0)
val2 = memoized_computation(2.0)

features = build_features(np.array([1, 2, 3]))
