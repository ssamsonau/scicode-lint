def pass_to_external_library(arr, external_processor):
    safe_copy = arr.copy()
    result = external_processor(safe_copy)
    return result


def prepare_for_unknown_function(data, func):
    prepared = data.copy()
    return func(prepared)
