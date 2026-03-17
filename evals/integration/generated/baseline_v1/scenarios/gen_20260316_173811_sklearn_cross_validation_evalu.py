import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

X_raw, y = make_classification(
    n_samples=3000,
    n_features=30,
    n_informative=12,
    n_redundant=5,
    weights=[0.93, 0.07],
    random_state=42,
)

df = pd.DataFrame(X_raw, columns=[f"feature_{i}" for i in range(X_raw.shape[1])])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.values)

cv = StratifiedKFold(n_splits=5, shuffle=True)

model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced")
scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")

print("Cross-validation results:")
print(f"  Mean accuracy: {scores.mean():.4f}")
print(f"  Std: {scores.std():.4f}")
print(f"  Per-fold: {np.round(scores, 4)}")
