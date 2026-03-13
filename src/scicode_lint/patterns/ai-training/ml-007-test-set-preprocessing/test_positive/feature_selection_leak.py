from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split


def train_with_feature_selection(X, y, k=20):
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, X_test, y_test
