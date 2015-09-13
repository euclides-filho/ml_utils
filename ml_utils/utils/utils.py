__author__ = 'euclides'
import re
import pandas as pd
from time import time as t
from sklearn.grid_search import GridSearchCV
td = lambda t0: t() - t0

def stringify2(model, feature_set):
    return "%s:%s" % (re.sub(r"[a-z]", '', model.__class__.__name__), feature_set)


def stringify(model):
    return "%s" % (re.sub(r"[a-z]", '', model.__class__.__name__))


def plot_fi(model, train):
    dfp = pd.DataFrame([dict(zip(train.columns, model.feature_importances_))]).T
    dfp.sort(0, ascending=0, inplace=1)
    dfp["cumsum"] = dfp[0].cumsum()
    dfp.sort(0, ascending=1, inplace=1)
    hsize = train.shape[1]/3.5
    hsize = 8 if hsize < 8 else hsize
    dfp.plot(kind="barh", figsize=(20, hsize))
    dfp.sort(0, ascending=0, inplace=1)
    return dfp

def get_best_model(model, param_grid, x, y, scoring, cv=4, refit=True, n_jobs=-1, verbose=1):
    t0 = t()
    grid = GridSearchCV(model, param_grid=param_grid, scoring=scoring, cv=cv,
                        refit=refit, n_jobs=n_jobs, verbose=verbose)
    grid.fit(x, y)
    best_params = grid.best_params_
    best_model = grid.best_estimator_ if refit else model
    best_score = grid.best_score_

    if verbose:
        for item in grid.grid_scores_:
            print("%s %s %s" % ('\tGRIDSCORES\t', stringify(best_model), item))
        print('BEST SCORE\t%s\t%2.6f in %2.2f seconds' % (stringify2(best_model, best_params), abs(best_score), td(t0)))

    return best_model, dict(best_params=best_params,
                best_score=grid.best_score_,
                grid_scores_=grid.grid_scores_)
