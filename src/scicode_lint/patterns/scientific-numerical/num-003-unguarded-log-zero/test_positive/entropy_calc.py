import numpy as np


class BayesianClassifier:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.class_log_priors = None
        self.feature_log_likelihoods = None

    def fit(self, X, y):
        class_counts = np.bincount(y, minlength=self.n_classes)
        self.class_log_priors = np.log(class_counts / len(y))

        self.feature_log_likelihoods = {}
        for c in range(self.n_classes):
            mask = y == c
            feature_means = X[mask].mean(axis=0)
            self.feature_log_likelihoods[c] = np.log(feature_means)

    def predict_log_probability(self, X):
        log_probs = np.zeros((len(X), self.n_classes))
        for c in range(self.n_classes):
            log_probs[:, c] = self.class_log_priors[c] + X @ self.feature_log_likelihoods[c]
        return log_probs


def mutual_information(joint_prob_matrix):
    marginal_x = joint_prob_matrix.sum(axis=1)
    marginal_y = joint_prob_matrix.sum(axis=0)
    outer = np.outer(marginal_x, marginal_y)
    mi = np.sum(joint_prob_matrix * np.log(joint_prob_matrix / outer))
    return mi
