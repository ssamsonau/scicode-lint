class FeatureNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, tensor):
        self.mean = tensor.mean(dim=0)
        self.std = tensor.std(dim=0) + 1e-8

    def transform(self, tensor):
        return (tensor - self.mean) / self.std


def normalize_sensor_readings(train_readings, test_readings):
    normalizer = FeatureNormalizer()
    normalizer.fit(train_readings)
    train_normed = normalizer.transform(train_readings)
    test_normed = normalizer.transform(test_readings)
    return train_normed, test_normed
