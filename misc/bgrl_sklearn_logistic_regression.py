from sklearn.linear_model import LogisticRegression
from misc.eval import BaseSKLearnEvaluator


class sklearnLREvaluator(BaseSKLearnEvaluator):
    def __init__(self, params=None):
        if params is None:
            params = {
                "C": [
                    1 / (2e-10),
                    1 / (2e-8),
                    1 / (2e-6),
                    1 / (2e-4),
                    1 / (2e-2),
                    1,
                    1 / (2e2),
                    1 / (2e4),
                    1 / (2e6),
                    1 / (2e8),
                    1 / (2e10),
                ]
            }
        super(sklearnLREvaluator, self).__init__(
            LogisticRegression(solver="liblinear"), params
        )
