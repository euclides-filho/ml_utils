from __future__ import division, print_function
__author__ = 'Euclides Fernandes Filho <euclides5414@gmail.com>'
"""
ml_utils
Copyright (C) 2015  Euclides Fernandes Filho <euclides5414@gmail.com>
http://www.gnu.org/licenses/gpl-2.0.html#SEC4
"""
from ml_utils.utils.logger import get_loggers

logi, logd, logw, loge = get_loggers(__name__)

import numpy as np
import pandas as pd
import traceback
from sklearn.cross_validation import KFold, _BaseKFold
from sklearn.metrics.scorer import _BaseScorer

try:
    from multiprocessing import Pool, cpu_count
    CORES = cpu_count()
except:
    CORES = 1
    logw(traceback.format_exc())

import ml_utils.utils as u


def mp_aux(p):
    model, X, y, f, scoring, good_features, cv, use_std, debug = p
    assert isinstance(X, pd.DataFrame)

    if f not in good_features:
        tt0 = u.t()
        feats = good_features + [f]

        Xt = X[feats]

        if isinstance(cv, (int, float)):
            cv_splitter = y
            skf = KFold(len(cv_splitter), n_folds=cv, shuffle=False)
            n_folds = cv
        else:
            assert isinstance(cv, _BaseKFold), type(cv)
            skf = cv
            n_folds = skf.n_folds
        cv_scores = np.zeros((n_folds,), dtype=np.float64)

        for iCV, (train_index, test_index) in enumerate(skf):
            Xt_train, Xt_cv = Xt.iloc[train_index], Xt.iloc[test_index]
            yt_train, yt_cv = y[train_index], y[test_index]

            #model = model.__class__(**model.get_params())

            model.fit(Xt_train, yt_train)
            cv_scores[iCV] = scoring(model, Xt_cv, yt_cv)

        score, std = cv_scores.mean(), cv_scores.std()
        final_score = (score + std) if use_std else score
        score_name = scoring._score_func.__name__
        if debug:
            logd('Feature: "%s" %s: %f +/- %f in %2.1f s' % (f, score_name, score, std, (u.t() - tt0)))
        return final_score, f
    else:
        return None


def combine(model, X, y, cols, scoring, good_features=list(), cv=4, use_std=False, debug=False):
    for f in cols:
        if f not in set(good_features):
            yield (model, X, y, f, scoring, good_features, cv, use_std, debug)


def greedy_forward(model, X, y, scoring, good_features=None, cv=4, use_std=False, n_jobs=-1, debug=False):
    if good_features is None:
        good_features = list()
    else:
        assert isinstance(good_features, (set, list))

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    assert isinstance(scoring, _BaseScorer), type(scoring)
    ###########################################################
    score_hist = []

    cols = X.columns.tolist()

    # Greedy feature forward selection loop
    score_name = scoring._score_func.__name__
    iter = 0
    all_scores = {}
    while (len(score_hist) < len(cols)) and (len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]):
        t0 = u.t()
        scores = []

        params_comb = combine(model, X, y, cols, scoring, good_features, cv, use_std, debug)

        if n_jobs == 1:
            for p in params_comb:
                fs_score, f = mp_aux(p)
                scores.append((fs_score, f))
        else:
            nCORES = min(n_jobs, CORES) if n_jobs > 0 else CORES + n_jobs + 1
            pool = Pool(processes=nCORES)
            res = pool.map(mp_aux, params_comb)

            for fs_score, f in res:
                scores.append((fs_score, f))
                if debug:
                    logd('Feature: "%s" %s: %f ' % (f, score_name, fs_score))
            pool.close()

        best_score, best_f = sorted(scores)[-1]
        if debug:
            all_scores[iter] = scores
        good_features.append(best_f)
        score_hist.append((best_score, best_f))
        logi('Feature added: "%s" and %s: %f' % (best_f, score_name, best_score))
        logi('Current features: %s len: %i in %2.1f s' % (good_features, len(good_features), (u.t() - t0)))
        iter += 1

    # Remove last added feature from good_features
    best_score = score_hist[-1][0]
    if score_hist[-1][0] < score_hist[-2][0]:
        good_features.remove(score_hist[-1][1])
        best_score = score_hist[-2][0]
    logi('Selected features %s and SCORE: %f' % (good_features, best_score))
    ###########################################################

    return good_features, score_hist, all_scores
