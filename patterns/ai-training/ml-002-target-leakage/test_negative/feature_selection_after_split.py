from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split


def select_features_after_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    selector = SelectKBest(f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected, y_train, y_test


def feature_selection_in_pipeline(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [("selector", SelectKBest(f_classif, k=10)), ("classifier", LogisticRegression())]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    return predictions
