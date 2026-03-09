import numpy as np
from sklearn.model_selection import KFold, train_test_split


def encode_categories(df, target_col, cat_cols, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for col in cat_cols:
        df[f"{col}_encoded"] = np.nan
        for train_idx, val_idx in kf.split(df):
            means = df.iloc[train_idx].groupby(col)[target_col].mean()
            df.loc[df.index[val_idx], f"{col}_encoded"] = df.iloc[val_idx][col].map(means)
    return df


def prepare_data(df, target_col, categorical_cols):
    X = df.drop(columns=[target_col] + categorical_cols)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for col in categorical_cols:
        encoding = df.loc[X_train.index].groupby(col)[target_col].mean().to_dict()
        X_train[f"{col}_enc"] = df.loc[X_train.index, col].map(encoding)
        X_test[f"{col}_enc"] = df.loc[X_test.index, col].map(encoding)
    return X_train, X_test, y_train, y_test
