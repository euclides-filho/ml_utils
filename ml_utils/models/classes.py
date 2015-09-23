from __future__ import division, print_function
__author__ = 'Euclides Fernandes Filho <euclides5414@gmail.com>'
"""
ml_utils
Copyright (C) 2015  Euclides Fernandes Filho <euclides5414@gmail.com>
http://www.gnu.org/licenses/gpl-2.0.html#SEC4
"""

import xgboost as xgbb
from sklearn.base import BaseEstimator

class XGBoostBase(BaseEstimator):
    def __init__(self, num_rounds=100, objective="reg:linear", eta=0.05, min_child_weight=5, subsample=0.8,
                 colsample_bytree=0.8, scale_pos_weight=1.0, silent=True, max_depth=7, max_delta_step=2,
                 evals=(), obj=None, feval=None, early_stopping_rounds=None, evals_result=None,
                 verbose_eval=True, **kwargs):
        """
        num_boost_round: int
            Number of boosting iterations.
        watchlist (evals): list of pairs (DMatrix, string)
            List of items to be evaluated during training, this allows user to watch
            performance on the validation set.
        obj : function
            Customized objective function.
        feval : function
            Customized evaluation function.
        early_stopping_rounds: int
            Activates early stopping. Validation error needs to decrease at least
            every <early_stopping_rounds> round(s) to continue training.
            Requires at least one item in evals.
            If there's more than one, will use the last.
            Returns the model from the last iteration (not the best one).
            If early stopping occurs, the model will have two additional fields:
            bst.best_score and bst.best_iteration.
        evals_result: dict
            This dictionary stores the evaluation results of all the items in watchlist
            verbose_eval : bool
            If `verbose_eval` then the evaluation metric on the validation set, if
            given, is printed at each boosting stage.
        """

        self.objective = objective
        self.eta = eta
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.silent = silent
        self.max_depth = max_depth
        self.num_rounds = num_rounds
        self.max_delta_step = max_delta_step
        self.kwargs = kwargs

        self.model = None

        self.evals = evals
        self.obj = obj
        self.feval = feval
        self.early_stopping_rounds = early_stopping_rounds
        self.evals_result = evals_result
        self.verbose_eval = verbose_eval

        pass

    def get_params(self, deep=True):
        params = dict()
        params["objective"] = self.objective
        params["eta"] = self.eta
        params["min_child_weight"] = self.min_child_weight
        params["subsample"] = self.subsample
        params["colsample_bytree"] = self.colsample_bytree
        params["scale_pos_weight"] = self.scale_pos_weight
        params["silent"] = self.silent
        params["max_depth"] = self.max_depth
        params["num_rounds"] = self.num_rounds
        params["max_delta_step"] = self.max_delta_step

        for k, v in self.kwargs:
            params[k] = v

        return params

    def set_params(self, **params):
        for k, v in params.iteritems():
            setattr(self, k, v)

    def fit(self, X, y):
        xg_train = xgbb.DMatrix(X, label=y)
        params = self.get_params()
        param_list = list(params.items())

        self.model = xgbb.train(param_list, xg_train, self.num_rounds, evals=self.evals, obj=self.obj,
                                feval=self.feval, early_stopping_rounds=self.early_stopping_rounds,
                                evals_result=self.evals_result, verbose_eval=self.verbose_eval)
        return self

    def predict(self, X):
        xg_test = xgbb.DMatrix(X)
        pred = self.model.predict(xg_test)
        return pred
