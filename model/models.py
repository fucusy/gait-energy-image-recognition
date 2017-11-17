#! encoding=utf8

from sklearn.ensemble import RandomForestClassifier
from model.base_model import Model


class RandomForestClassification(Model):

    def __init__(self):
        Model.__init__(self)
        """
            Best parameters found by grid search:
            {'n_estimators': 500, 'max_depth': 100, 'max_features': 100}
        """
        self.model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=2016, verbose=1, max_depth=100, max_features=100)

    def fit(self, x_train, y_train, x_test=None):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)

