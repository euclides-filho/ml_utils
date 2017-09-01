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
import traceback

class LabelEncoderThreshold:
    def __init__(self, min_threshold=2, null_threshold=0.95, min_thresholds=None):
        # TODO drop beyond null_threshold, support different min_threshold for each column
        self.min_thresholds = min_thresholds if min_thresholds is not None else {}
        self.null_threshold = null_threshold
        self.df = None
        self.df_transformed = None
        self.columns_data = None
        self.min_threshold = min_threshold
        assert isinstance(self.min_thresholds, dict)

    def fit_transform(self, df):
        self.df = df
        self.df_transformed, self.columns_data = \
            LabelEncoderThreshold.categorize_df(self.df, self.min_threshold,
                                                self.null_threshold, self.min_thresholds)
        return self.df_transformed

    def transform(self, dt_to):
        return LabelEncoderThreshold.transform_cats(dt_to, self.columns_data)

    @staticmethod
    def categorize_col(df, col_name, min_threshold=2):
        dfg = df[[col_name]].copy()
        dfg["count"] = 1
        df_counts = dfg.groupby(col_name).count()
        df_counts = df_counts.sort_values(by=["count"], ascending=[0])
        categories = {}

        cur_idx = 0
        for val in list(df_counts.index):
            count = df_counts.loc[val]["count"]
            if count >= min_threshold:
                categories[val] = cur_idx
                cur_idx += 1
            else:
                categories[val] = cur_idx
        categories[None] = -1
        categories[np.nan] = -1

        df[col_name] = df[col_name].apply(lambda x: categories[x] if x in categories else x)
        return df, categories, df_counts

    @staticmethod
    def categorize_df(df, min_threshold, null_threshold=0.95, min_thresholds=None):
        columns_data = {}
        col, col_name = "", ""
        # noinspection PyBroadException
        try:
            df = df.copy()
            for col in df:
                if df[col].isnull().mean() > null_threshold:
                    logi("Column: '%s' with more than %2.2f of nulls" % (col, null_threshold))

            for col_name in df.columns:
                if df[col_name].dtype not in (object, str):
                    continue
                min_threshold = min_thresholds[col_name] \
                    if min_thresholds and col_name in min_thresholds else min_threshold
                df, categories, df_counts = LabelEncoderThreshold.categorize_col(df, col_name,
                                                                                 min_threshold=min_threshold)
                columns_data[col_name] = {"categories": categories, "counts": df_counts}
        except:
            loge("%s %s %s" % (col, col_name, traceback.format_exc()))

        return df, columns_data

    @staticmethod
    def transform_cats(df, columns_categories):
        for col in columns_categories:
            if col in df.columns:
                d = columns_categories[col]["categories"]
                max_val = max(d.values())
                f = lambda x: d.get(x, max_val)
                df[col] = df[col].apply(f)
        return df
