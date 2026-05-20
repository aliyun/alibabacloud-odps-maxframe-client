# Copyright 1999-2026 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from maxframe.dataframe.reduction.aggregation import DataFrameAggregate
from maxframe.dataframe.reduction.all import DataFrameAll
from maxframe.dataframe.reduction.any import DataFrameAny
from maxframe.dataframe.reduction.argmax import DataFrameArgMax
from maxframe.dataframe.reduction.argmin import DataFrameArgMin
from maxframe.dataframe.reduction.core import CustomReduction, NamedAgg
from maxframe.dataframe.reduction.count import DataFrameCount
from maxframe.dataframe.reduction.cummax import DataFrameCummax
from maxframe.dataframe.reduction.cummin import DataFrameCummin
from maxframe.dataframe.reduction.cumprod import DataFrameCumprod
from maxframe.dataframe.reduction.cumsum import DataFrameCumsum
from maxframe.dataframe.reduction.custom_reduction import DataFrameCustomReduction
from maxframe.dataframe.reduction.idxmax import DataFrameIdxMax
from maxframe.dataframe.reduction.idxmin import DataFrameIdxMin
from maxframe.dataframe.reduction.kurtosis import DataFrameKurtosis
from maxframe.dataframe.reduction.max import DataFrameMax
from maxframe.dataframe.reduction.mean import DataFrameMean
from maxframe.dataframe.reduction.median import DataFrameMedian
from maxframe.dataframe.reduction.min import DataFrameMin
from maxframe.dataframe.reduction.mode import DataFrameMode
from maxframe.dataframe.reduction.nunique import DataFrameNunique
from maxframe.dataframe.reduction.prod import DataFrameProd
from maxframe.dataframe.reduction.reduction_size import DataFrameSize
from maxframe.dataframe.reduction.sem import DataFrameSem
from maxframe.dataframe.reduction.skew import DataFrameSkew
from maxframe.dataframe.reduction.str_concat import (
    DataFrameStrConcat,
    build_str_concat_object,
)
from maxframe.dataframe.reduction.sum import DataFrameSum
from maxframe.dataframe.reduction.unique import DataFrameUnique, unique
from maxframe.dataframe.reduction.var import DataFrameVar


def _install():
    from maxframe.dataframe.core import DATAFRAME_TYPE, INDEX_TYPE, SERIES_TYPE
    from maxframe.dataframe.reduction.aggregation import aggregate
    from maxframe.dataframe.reduction.all import all_dataframe, all_index, all_series
    from maxframe.dataframe.reduction.any import any_dataframe, any_index, any_series
    from maxframe.dataframe.reduction.argmax import argmax_series_index
    from maxframe.dataframe.reduction.argmin import argmin_series_index
    from maxframe.dataframe.reduction.count import count_dataframe, count_series
    from maxframe.dataframe.reduction.cov import cov_dataframe, cov_series
    from maxframe.dataframe.reduction.cummax import cummax
    from maxframe.dataframe.reduction.cummin import cummin
    from maxframe.dataframe.reduction.cumprod import cumprod
    from maxframe.dataframe.reduction.cumsum import cumsum
    from maxframe.dataframe.reduction.idxmax import idxmax_dataframe, idxmax_series
    from maxframe.dataframe.reduction.idxmin import idxmin_dataframe, idxmin_series
    from maxframe.dataframe.reduction.kurtosis import kurt_dataframe, kurt_series
    from maxframe.dataframe.reduction.max import max_dataframe, max_index, max_series
    from maxframe.dataframe.reduction.mean import mean_dataframe, mean_series
    from maxframe.dataframe.reduction.median import median_dataframe, median_series
    from maxframe.dataframe.reduction.min import min_dataframe, min_index, min_series
    from maxframe.dataframe.reduction.mode import mode_dataframe, mode_series
    from maxframe.dataframe.reduction.nunique import nunique_dataframe, nunique_series
    from maxframe.dataframe.reduction.prod import prod_dataframe, prod_series
    from maxframe.dataframe.reduction.reduction_size import size_dataframe, size_series
    from maxframe.dataframe.reduction.sem import sem_dataframe, sem_series
    from maxframe.dataframe.reduction.skew import skew_dataframe, skew_series
    from maxframe.dataframe.reduction.std import std_dataframe, std_series
    from maxframe.dataframe.reduction.sum import sum_dataframe, sum_series
    from maxframe.dataframe.reduction.var import var_dataframe, var_series

    funcs = [
        ("agg", aggregate, aggregate),
        ("aggregate", aggregate, aggregate),
        ("all", all_series, all_dataframe),
        ("any", any_series, any_dataframe),
        ("argmax", argmax_series_index, None),
        ("argmin", argmin_series_index, None),
        ("count", count_series, count_dataframe),
        ("cov", cov_series, cov_dataframe),
        ("cummax", cummax, cummax),
        ("cummin", cummin, cummin),
        ("cumprod", cumprod, cumprod),
        ("cumsum", cumsum, cumsum),
        ("idxmax", idxmax_series, idxmax_dataframe),
        ("idxmin", idxmin_series, idxmin_dataframe),
        ("kurt", kurt_series, kurt_dataframe),
        ("kurtosis", kurt_series, kurt_dataframe),
        ("max", max_series, max_dataframe),
        ("mean", mean_series, mean_dataframe),
        ("median", median_series, median_dataframe),
        ("min", min_series, min_dataframe),
        ("mode", mode_series, mode_dataframe),
        ("nunique", nunique_series, nunique_dataframe),
        ("prod", prod_series, prod_dataframe),
        ("product", prod_series, prod_dataframe),
        ("sem", sem_series, sem_dataframe),
        ("skew", skew_series, skew_dataframe),
        ("std", std_series, std_dataframe),
        ("sum", sum_series, sum_dataframe),
        ("unique", unique, None),
        ("var", var_series, var_dataframe),
        ("_reduction_size", size_series, size_dataframe),
    ]
    for func_name, series_func, df_func in funcs:
        if df_func is not None:  # pragma: no branch
            for t in DATAFRAME_TYPE:
                setattr(t, func_name, df_func)
        if series_func is not None:  # pragma: no branch
            for t in SERIES_TYPE:
                setattr(t, func_name, series_func)

    for t in INDEX_TYPE:
        setattr(t, "agg", aggregate)
        setattr(t, "aggregate", aggregate)
        setattr(t, "all", all_index)
        setattr(t, "any", any_index)
        setattr(t, "min", min_index)
        setattr(t, "max", max_index)
        setattr(t, "argmin", argmin_series_index)
        setattr(t, "argmax", argmax_series_index)


_install()
del _install
