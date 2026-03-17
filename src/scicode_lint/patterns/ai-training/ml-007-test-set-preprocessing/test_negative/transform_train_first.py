from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:
    """Preprocessor that correctly fits only on training data."""

    def __init__(self, numeric_cols: list[str], categorical_cols: list[str]):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.transformer = ColumnTransformer(
            [
                ("num", StandardScaler(), numeric_cols),
                (
                    "cat",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    categorical_cols,
                ),
            ]
        )
        self._fitted = False

    def fit(self, train_df):
        """Fit transformers on training data only."""
        self.transformer.fit(train_df)
        self._fitted = True
        return self

    def transform(self, df):
        """Transform any dataset using fitted parameters."""
        if not self._fitted:
            raise ValueError("Must fit on training data first")
        return self.transformer.transform(df)


def create_preprocessing_pipeline(model):
    """Create sklearn Pipeline ensuring correct fit order."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def cross_validation_with_pipeline(X, y, model, cv=5):
    """Cross-validation with preprocessing inside fold - no leakage."""
    from sklearn.model_selection import cross_val_score

    pipeline = create_preprocessing_pipeline(model)
    return cross_val_score(pipeline, X, y, cv=cv)
