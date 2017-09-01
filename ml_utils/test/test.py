from __future__ import division, print_function
__author__ = 'Euclides Fernandes Filho <euclides5414@gmail.com>'
"""
ml_utils
Copyright (C) 2015  Euclides Fernandes Filho <euclides5414@gmail.com>
http://www.gnu.org/licenses/gpl-2.0.html#SEC4
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression as LR
from sklearn.ensemble import RandomForestRegressor as RFR, ExtraTreesRegressor as ETR
from sklearn.preprocessing import StandardScaler

from ml_utils.ensemble import KFoldEnsemble


SEED = 17
np.random.seed(SEED)


def test():
    _regression()
    _feature_selection()


def _regression():
    X = np.random.rand(1000, 50)
    y = np.random.rand(1000)
    preprocessor = StandardScaler()
    models = [RFR(), ETR(), LR()]
    param_grid = {"alpha": [0.001, 0.01, 0.1, 0, 1, 10, 100, 1000],
                  "fit_intercept": [False, True]}
    ens = KFoldEnsemble(X, y, models=models, ensemble_model=Ridge(),
                        scoring=mean_squared_error, preprocessor=preprocessor,
                        verbose=1, ensemble_grid_params=param_grid)

    external_cols = np.random.rand(1000, 2)
    ens.fit(external_cols=external_cols)

    X_test = np.random.rand(5000, 50)
    external_cols = np.random.rand(5000, 2)
    _ = ens.predict(X_test, external_cols=external_cols)

    df = ens.predictions_cv
    for col in df.columns:
        print(col, mean_squared_error(df[col], df["TRUE"]))


def _feature_selection():
    from ml_utils.feature_selection import greedy_forward
    from sklearn.linear_model import Ridge
    from sklearn.metrics import make_scorer, mean_squared_error
    MSE = make_scorer(mean_squared_error, greater_is_better=False)
    import pandas as pd
    import numpy as np
    np.random.seed(17)

    df = pd.DataFrame(np.random.rand(1000, 20))
    y = pd.Series(np.random.rand(1000), index=df.index)

    model = Ridge(alpha=10)

    good_features, score_hist, all_scores = greedy_forward(model, df, y, scoring=MSE, cv=4, n_jobs=-1)
    print(len(good_features))
    print(good_features)


if __name__ == "__main__":
    test()
