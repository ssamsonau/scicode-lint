"""Survey data analysis with unstable sort on Likert-scale responses."""

import numpy as np


class SurveyAnalyzer:
    """Analyze survey responses with discrete integer ratings."""

    def __init__(self, responses: np.ndarray, respondent_ids: np.ndarray):
        self.responses = responses
        self.respondent_ids = respondent_ids

    def get_ranking(self) -> np.ndarray:
        """Rank respondents by total score - many ties in integer sums."""
        totals = self.responses.sum(axis=1)
        return self.respondent_ids[np.argsort(totals)[::-1]]

    def top_questions_by_variance(self, k: int = 5) -> np.ndarray:
        """Find questions with highest variance - binned values have ties."""
        variances = np.round(self.responses.var(axis=0), decimals=1)
        return np.argsort(variances)[::-1][:k]

    def cluster_by_pattern(self) -> np.ndarray:
        """Sort respondents by response pattern hash - integer hashes have ties."""
        pattern_hash = np.array([hash(tuple(row)) % 100 for row in self.responses])
        return np.argsort(pattern_hash)
