from __future__ import division, print_function
__author__ = 'euclides'

from ml_utils.utils.logger import get_loggers
logi, logd, logw, loge = get_loggers(__name__)

import pandas as pd
import numpy as np
import sklearn
import sklearn.base
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
import ml_utils.utils as u

from ml_utils.cv import KFoldPred, KStratifiedPred
from ml_utils.utils import stringify2
try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle


SEED = 17
np.random.seed(SEED)


class KFoldEnsemble:
    def __init__(self, x, y, models, ensemble_model, scoring=None, n_folds=3, random_state=SEED,
                 shuffle=False, n_jobs=-1, stratified=False, preprocessor=None, verbose=0,
                 ensemble_grid_params=None, score_greater_is_better=False):

        assert isinstance(models, (list, tuple, set)), type(models)
        assert isinstance(ensemble_model, sklearn.base.BaseEstimator), \
            "%s != %s" % (type(ensemble_model), type(sklearn.base.BaseEstimator))
        self.X = x
        self.y = y
        self.ensemble_model = ensemble_model
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.models = models
        self.stratified = stratified
        self.random_state = random_state
        self.predictors_n_jobs = n_jobs
        self.scoring = scoring
        self.preprocessor = preprocessor
        self.verbose = verbose
        self.ensemble_scaler = None
        self.score_greater_is_better = score_greater_is_better
        self.base_predictor = KStratifiedPred if self.stratified else KFoldPred
        # TODO n_jobs split ensemble and CV

        self.true_col = "TRUE"
        self.cols = map(lambda i: "%s" % stringify2(i[1], i[0]), enumerate(models))
        self.predictions_cv = None
        self.predictions = None
        self.ensemble_grid_params = ensemble_grid_params
        self.predictors = {}

    def fit(self, external_cols=None):
        self.predictors = {}
        self.predictions_cv = pd.DataFrame()
        for i_model, model in enumerate(self.models):
            model_name = stringify2(model, i_model)
            if self.verbose:
                logd(model_name)
            t0 = u.t()
            i_predictor = self.base_predictor(self.X, self.y, model, scoring=self.scoring,
                                              n_folds=self.n_folds, random_state=self.random_state,
                                              shuffle=self.shuffle, n_jobs=self.predictors_n_jobs,
                                              preprocessor=self.preprocessor, verbose=self.verbose)
            col = model_name
            i_predictor.fit()
            i_prediction_cv = i_predictor.predict()
            if not len(self.predictions_cv):
                self.predictions_cv = i_prediction_cv.rename(columns={i_predictor.cv_col: col})  # [i_predictor.cv_col]
            else:
                df = i_prediction_cv[[i_predictor.cv_col]].rename(columns={i_predictor.cv_col: col})
                # TODO assert index is not duplicate
                self.predictions_cv = self.predictions_cv.merge(df, left_index=True, right_index=True)

            i_predictor.fit_test()
            self.predictors[model_name] = i_predictor
            if self.verbose:
                logd("Fit %s in %2.2f seconds" % (model_name, u.td(t0)))
        self.fit_ensemble(external_cols=external_cols)

    def fit_ensemble(self, external_cols=None):
        t0 = u.t()
        _x = self.predictions_cv[self.cols] if self.predictions_cv is not None else pd.DataFrame()
        if external_cols is not None:
            if not isinstance(external_cols, pd.DataFrame):
                external_cols = pd.DataFrame(external_cols)
            for col in external_cols.columns:
                _x["ADD_%s" % col] = external_cols[col]

        _y = self.predictions_cv[self.true_col]
        self.ensemble_scaler = StandardScaler()
        x = self.ensemble_scaler.fit_transform(_x)
        if self.ensemble_grid_params:
            scorer = make_scorer(self.scoring, greater_is_better=self.score_greater_is_better)
            self.ensemble_model, _ = \
                u.get_best_model(self.ensemble_model, self.ensemble_grid_params, x, _y,
                                 scoring=scorer, cv=self.n_folds, refit=True)
        else:
            self.ensemble_model.fit(x, _y)

        if self.verbose:
            logd("Fit Ensemble in %2.2f seconds" % u.td(t0))
        self.predictions_cv["ENS"] = self.ensemble_model.predict(x)
        self.predictions_cv = self.predictions_cv[self.cols + ["ENS", self.true_col]]

    def predict(self, x, external_cols=None):
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        self.predictions = pd.DataFrame(columns=self.cols, index=x.index) \
            if self.predictions_cv is not None else pd.DataFrame()
        for model_name, i_predictor in self.predictors.iteritems():
            i_prediction = i_predictor.predict_test(x)
            self.predictions[model_name] = i_prediction

        if external_cols is not None:
            if not isinstance(external_cols, pd.DataFrame):
                external_cols = pd.DataFrame(external_cols)
            for col in external_cols.columns:
                self.predictions["ADD_%s" % col] = external_cols[col]

        return self.ensemble_model.predict(self.ensemble_scaler.transform(self.predictions))

    def pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def score_ensemble(self):
        if self.scoring:
            df = self.predictions_cv
            for col in df.columns:
                if col == "TRUE":
                    continue
                logd(col, self.scoring(df[col], df["TRUE"]))
