import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=2000,
    n_features=20,
    weights=[0.85, 0.15],
    flip_y=0.02,
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kf = KFold(n_splits=5, shuffle=True)

model = LogisticRegression(max_iter=500)

scores = []
for train_idx, val_idx in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    scores.append(accuracy_score(y_val, preds))

print(f"Mean CV accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
