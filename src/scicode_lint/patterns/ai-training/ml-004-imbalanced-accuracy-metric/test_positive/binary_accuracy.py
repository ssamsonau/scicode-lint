from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def evaluate_model(model, X_test, y_test):
    # Class distribution: ~3% positive (rare disease detection)
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)


def train_and_score(X_train, y_train, X_test, y_test):
    # Imbalanced data: less than 5% positive cases
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc
