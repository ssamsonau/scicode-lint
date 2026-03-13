from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return f1_score(y_test, predictions)


def train_and_score(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    bal_acc = balanced_accuracy_score(y_test, model.predict(X_test))
    return model, bal_acc
