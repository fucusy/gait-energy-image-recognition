from sklearn import grid_search
from sklearn.metrics import make_scorer, f1_score

class Model(object):

    def __init__(self):
        self.model = None

    def fit(self, x_train, y_train, x_test):
        pass

    def predict(self, x_test):
        pass

    def grid_search_fit_(self, clf, param_grid, x_train, y_train):
        model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, cv=2, verbose=20
                                         , scoring=make_scorer(f1_score))
        model.fit(x_train, y_train)
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        # f = "%s/%s/model.dump.pickle" % (config.project.project_path, sys.argv[1])
        # pickle.dump(model, f)
        return model        
