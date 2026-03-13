from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def train_rare_event_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model


def split_multiclass_data(X, y, test_fraction=0.25):
    return train_test_split(X, y, test_size=test_fraction, shuffle=True, stratify=y)
