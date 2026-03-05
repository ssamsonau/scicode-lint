from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split


def split_imbalanced_without_stratify(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train positive rate: {y_train.mean():.2%}")
    print(f"Test positive rate: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


def kfold_without_stratification(X, y):
    model = LogisticRegression()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"Cross-validation scores: {scores}")
    print(f"Mean AUC: {scores.mean():.3f}")
    print(f"Std AUC: {scores.std():.3f}")
    return scores


def no_stratify_rare_classes(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if y_test.sum() == 0:
        print("WARNING: No positive examples in test set!")
    return X_train, X_test, y_train, y_test
