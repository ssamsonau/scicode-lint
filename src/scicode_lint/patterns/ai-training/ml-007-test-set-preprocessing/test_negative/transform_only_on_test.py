from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_correctly(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def preprocess_with_pipeline(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    return predictions


def multiple_preprocessing_steps(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca, y_train, y_test
