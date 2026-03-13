from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


def split_with_stratification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Full data positive rate: {y.mean():.2%}")
    print(f"Train positive rate: {y_train.mean():.2%}")
    print(f"Test positive rate: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


def stratified_kfold_correct(X, y):
    model = LogisticRegression()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"Cross-validation scores: {scores}")
    print(f"Mean AUC: {scores.mean():.3f}")
    print(f"Std AUC: {scores.std():.3f}")
    return scores


def stratified_split_rare_classes(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Test set positives: {y_test.sum()}")
    print(f"Test positive rate: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


def stratified_with_multiple_classes(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test
