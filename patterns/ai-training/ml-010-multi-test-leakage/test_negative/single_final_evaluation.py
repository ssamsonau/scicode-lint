from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_and_evaluate(X, y):
    """Simple train/test split with no hyperparameter tuning.

    This is CORRECT because:
    - No tuning/selection happens (fixed hyperparameters)
    - X_test is truly held out and only used once for final evaluation
    - Without tuning, there's no risk of overfitting to "validation" data
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    final_accuracy = model.score(X_test, y_test)
    print(f"Final test accuracy: {final_accuracy}")

    return model, final_accuracy
