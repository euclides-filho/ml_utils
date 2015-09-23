from __future__ import division, print_function
__author__ = 'Euclides Fernandes Filho <euclides5414@gmail.com>'
"""
ml_utils
Copyright (C) 2015  Euclides Fernandes Filho <euclides5414@gmail.com>
http://www.gnu.org/licenses/gpl-2.0.html#SEC4
"""
from ml_utils.utils.logger import get_loggers

logi, logd, logw, loge = get_loggers(__name__)

import pandas as pd
import numpy as np
import sklearn
import sklearn.base
import multiprocessing
import ml_utils.utils as u
SEED = 17


def _fit(args):
    return KFoldPredBase.fit_static(args)


class KFoldPredBase:
    def __init__(self, model, folder, scoring=None, random_state=SEED,
                 predict_probas=False, n_jobs=-1, preprocessor=None, verbose=0):
        assert n_jobs, n_jobs
        assert isinstance(n_jobs, int), type(n_jobs)

        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        if predict_probas:
            assert hasattr(model, "predict_probas")
        if hasattr(model, "random_state"):
            model.random_state = random_state
        assert isinstance(model, sklearn.base.BaseEstimator), \
            "%s != %s" % (type(model), type(sklearn.base.BaseEstimator))
        if preprocessor:
            assert hasattr(preprocessor, "fit")
            assert hasattr(preprocessor, "transform")
            assert hasattr(preprocessor, "fit_transform")
            assert isinstance(preprocessor, sklearn.base.BaseEstimator), \
                "%s != %s" % (type(model), type(sklearn.base.BaseEstimator))

        self.model = model
        self.n_jobs = n_jobs
        self.random_state = random_state
        cores = multiprocessing.cpu_count()
        self.n_jobs_ = min(self.n_jobs, cores) if self.n_jobs > 0 else cores + self.n_jobs + 1
        self.score_mean = None
        self.score_std = None
        self.fitted = False
        self.fitted_test = False
        self.verbose = verbose
        if not isinstance(self.X, pd.DataFrame):
            self.X = pd.DataFrame(self.X)
        if not isinstance(self.y, (pd.DataFrame, pd.Series)):
            self.y = pd.Series(self.y)

        self.preprocessor = preprocessor
        self.folder = folder
        self.models = {}
        self.preprocessors = {}
        self.scoring = scoring
        if self.scoring:
            self.scores = {}
            self.score = 0
        self.predict_probas = predict_probas
        self.true_col = "TRUE"
        self.cv_col = "CV"
        if self.predict_probas:
            u = self.y.unique()
            u.sort()
            assert isinstance(u[0], int,), type(u[0])
            cols = list(map(lambda i: self.cv_col + "_L%i" % i, u))
            self.predictions = pd.DataFrame(columns=cols + [self.true_col])
        else:
            self.predictions = pd.DataFrame(columns=[self.cv_col, self.true_col])

    def yield_cv(self, include_model=True):
        for iCV, (train_idx, cv_idx) in enumerate(self.folder):
            i_x, i_y = self.X.iloc[train_idx], self.y.iloc[train_idx]
            icv_x, icv_y = self.X.iloc[cv_idx], self.y.iloc[cv_idx]
            i_model = self.model.__class__(**self.model.get_params()) if include_model else None
            i_preprocessor = self.preprocessor.__class__(**self.preprocessor.get_params()) \
                if self.preprocessor else None

            yield iCV, i_x, i_y, icv_x, icv_y, i_model, i_preprocessor

    @staticmethod
    def fit_static(args):
        icv, i_x, i_y, _, _, i_model, i_preprocessor = args
        if i_preprocessor:
            i_x = i_preprocessor.fit_transform(i_x)
        i_model = i_model.fit(i_x, i_y)
        return icv, i_model, i_preprocessor

    def fit_test(self):
        t0 = u.t()
        self.fitted_test = False
        x = self.preprocessor.fit_transform(self.X) if self.preprocessor else self.X
        self.model.fit(x, self.y)
        self.fitted_test = True
        if self.verbose:
            logd("Fit Test in %2.2f seconds | %s" % (u.td(t0), x.shape))

    def predict_test(self, x):
        assert self.fitted_test
        return self.model.predict(x) if not self.predict_probas else self.model.predict_proba(x)

    def fit(self):
        t0 = u.t()
        self.fitted = False
        np.random.seed(self.random_state)
        if self.n_jobs_ != 1:
            pool = multiprocessing.Pool(self.n_jobs_)
            try:
                iter_params = []
                for it in self.yield_cv():
                    iter_params.append(it)
                res = pool.map(_fit, iter_params)
                for iCV, i_model, i_preprocessor in res:
                    self.models[iCV] = i_model
                    self.preprocessors[iCV] = i_preprocessor

            finally:
                pool.close()
        else:
            for args in self.yield_cv():
                icv, i_model, i_preprocessor = KFoldPredBase.fit_static(args)
                self.models[icv] = i_model
                self.preprocessors[icv] = i_preprocessor
        self.fitted = True
        if self.verbose:
            logd("Fit ALL CVs in %2.2f seconds" % u.td(t0))
        return self

    @staticmethod
    def stack_y(y1, y2):
        if len(y1.shape) != len(y2.shape) or len(y1.shape) == 1:
            if len(y1.shape) == 1:
                y1 = y1.reshape(len(y1), 1)
            if len(y2.shape) == 1:
                y2 = y2.reshape(len(y2), 1)
        return np.hstack((y1, y2))

    def predict(self):
        assert self.fitted
        np.random.seed(self.random_state)
        self.predictions = pd.DataFrame(columns=self.predictions.columns)
        if self.scoring:
            self.scores = {}
            self.score = 0

        for iCV, i_X, i_y, icv_x, iCV_y, _, _ in self.yield_cv():
            if self.preprocessor:
                old_max = icv_x.max().max()
                icv_x = self.preprocessors[iCV].transform(icv_x)
                if self.verbose and False:
                    logd("%2.2f %2.2f" % (old_max, icv_x.max().max()))

            if self.predict_probas:
                i_prediction = self.models[iCV].predict_proba(icv_x)
            else:
                i_prediction = self.models[iCV].predict(icv_x)

            if self.scoring and not self.predict_probas:
                self.scores[iCV] = self.scoring(i_prediction, iCV_y)
                if self.verbose:
                    logi("%i - Score: %f" % (iCV, self.scores[iCV]))

            i_pred_true = pd.DataFrame(KFoldPredBase.stack_y(i_prediction, iCV_y),
                                       columns=self.predictions.columns, index=iCV_y.index)

            self.predictions = pd.concat((self.predictions, i_pred_true))
        if self.scoring and not self.predict_probas:
            self.score_mean = pd.Series(self.scores.values()).mean()
            self.score_std = pd.Series(self.scores.values()).std()
        return self.predictions


class KFoldPred(KFoldPredBase):
    def __init__(self, x, y, model, scoring=None, n_folds=3, random_state=SEED,
                 shuffle=False, n_jobs=-1, preprocessor=None, verbose=0):
        self.X = x
        self.y = y
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.folder = sklearn.cross_validation.KFold(n=self.X.shape[0], n_folds=self.n_folds, indices=None,
                                                     shuffle=self.shuffle, random_state=random_state)

        KFoldPredBase.__init__(self, model=model, scoring=scoring, random_state=random_state,
                               folder=self.folder, predict_probas=False, n_jobs=n_jobs, preprocessor=preprocessor,
                               verbose=verbose)


class KStratifiedPred(KFoldPredBase):
    def __init__(self, x, y, model, scoring=None, n_folds=3, random_state=SEED,
                 shuffle=False, predict_probas=False, n_jobs=-1, preprocessor=None, verbose=0):
        self.X = x
        self.y = y
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

        self.folder = sklearn.cross_validation.StratifiedKFold(y=self.y, n_folds=self.n_folds, indices=None,
                                                               shuffle=self.shuffle, random_state=random_state)

        KFoldPredBase.__init__(self, model=model, scoring=scoring, random_state=random_state,
                               folder=self.folder, predict_probas=predict_probas,
                               n_jobs=n_jobs, preprocessor=preprocessor, verbose=verbose)


