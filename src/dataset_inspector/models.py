from sklearn.linear_model import LinearRegression # Regression
from sklearn.linear_model import LogisticRegression # Classification


class Model:
    def __init__(self, task: str):
        self.task = task

    def make_baseline(self):
        match self.task:
            case "regression":
                model = LinearRegression()
            case "classification":
                model = LogisticRegression()
            case _:
                return -1
        return model