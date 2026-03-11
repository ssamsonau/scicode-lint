from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_fraud_detector(X, y):
    # Note: fraud is rare (typically < 1% of transactions)
    print(f"Class distribution: {y.sum()}/{len(y)} fraud cases ({100*y.sum()/len(y):.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)
    return model, score


def evaluate_classifier(model, X_test, y_test):
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)
