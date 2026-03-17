import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def patient_level_split(imaging_df, patient_col="patient_id", test_size=0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    groups = imaging_df[patient_col]
    train_idx, test_idx = next(gss.split(imaging_df, groups=groups))
    return imaging_df.iloc[train_idx], imaging_df.iloc[test_idx]


def temporal_holdout(df, date_col="acquisition_date", cutoff="2025-01-01"):
    df[date_col] = pd.to_datetime(df[date_col])
    train = df[df[date_col] < cutoff].copy()
    test = df[df[date_col] >= cutoff].copy()
    return train, test
