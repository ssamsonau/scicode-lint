import numpy as np


def compute_features(samples, feature_extractor):
    features = []
    for sample in samples:
        row_features = feature_extractor(sample)
        features.append(np.array(row_features))
    return np.vstack(features)


def transform_dataset(data_list):
    transformed = []
    for data in data_list:
        arr = np.array(data)
        normalized = (arr - arr.mean()) / arr.std()
        transformed.append(normalized)
    return transformed


def generate_embeddings(texts, model):
    embeddings = []
    for text in texts:
        embedding = np.empty(512)
        model.encode(text, out=embedding)
        embeddings.append(embedding)
    return np.stack(embeddings)
