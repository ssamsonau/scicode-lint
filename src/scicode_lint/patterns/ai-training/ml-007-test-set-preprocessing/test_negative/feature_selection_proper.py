from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split


def train_with_feature_selection(X, y, k=20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    model = RandomForestClassifier()
    model.fit(X_train_selected, y_train)
    return model, selector, X_test_selected, y_test
