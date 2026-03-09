import numpy as np


def rank_candidates(scores, names):
    order = np.argsort(-scores)
    return [names[i] for i in order]


def get_feature_ranking(model, feature_names):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    return [(feature_names[i], importances[i]) for i in idx]


def sort_dataframe_by_column(df, column):
    return df.iloc[df[column].values.argsort()]
