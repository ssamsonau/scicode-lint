import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def preprocess_features(X, y):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.25, random_state=42)


def scale_all_data(features, labels):
    combined = np.vstack([features[:800], features[800:]])
    scaler = RobustScaler()
    scaled = scaler.fit_transform(combined)
    return scaled[:800], scaled[800:]
