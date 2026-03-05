from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score


def cross_validate_user_sessions(df):
    X = df[["session_duration", "pages_viewed", "user_age"]]
    y = df["converted"]
    model = LogisticRegression()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"Cross-validation AUC scores: {scores}")
    return scores


def cross_validate_transactions(df):
    X = df[["amount", "merchant_category", "user_history"]]
    y = df["is_fraud"]
    model = LogisticRegression()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return scores
