# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

import functools
import operator
from functools import reduce
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from .... import dataframe as md
from ....lib.dtypes_extension import ArrowDtype
from ....tensor import Tensor
from ....tests.utils import assert_mf_index_dtype
from ....udf import ODPSFunction
from ...core import DataFrame, IndexValue, OutputType, Series
from ...datasource.dataframe import from_pandas as from_pandas_df
from ...datasource.series import from_pandas as from_pandas_series
from .. import (
    CustomReduction,
    DataFrameAggregate,
    DataFrameAll,
    DataFrameAny,
    DataFrameCount,
    DataFrameCummax,
    DataFrameCummin,
    DataFrameCumprod,
    DataFrameCumsum,
    DataFrameKurtosis,
    DataFrameMax,
    DataFrameMean,
    DataFrameMedian,
    DataFrameMin,
    DataFrameNunique,
    DataFrameProd,
    DataFrameSem,
    DataFrameSkew,
    DataFrameSum,
    DataFrameVar,
)
from ..aggregation import where_function
from ..core import ReductionCompiler

pytestmark = pytest.mark.pd_compat


class FunctionOptions(NamedTuple):
    has_skipna: bool = True
    has_numeric_only: bool = True
    has_bool_only: bool = False


reduction_functions = [
    ("sum", DataFrameSum, FunctionOptions()),
    ("prod", DataFrameProd, FunctionOptions()),
    ("min", DataFrameMin, FunctionOptions()),
    ("max", DataFrameMax, FunctionOptions()),
    ("count", DataFrameCount, FunctionOptions(has_skipna=False)),
    ("mean", DataFrameMean, FunctionOptions()),
    ("var", DataFrameVar, FunctionOptions()),
    ("skew", DataFrameSkew, FunctionOptions()),
    ("kurt", DataFrameKurtosis, FunctionOptions()),
    ("sem", DataFrameSem, FunctionOptions()),
    ("all", DataFrameAll, FunctionOptions(has_numeric_only=False, has_bool_only=True)),
    ("any", DataFrameAny, FunctionOptions(has_numeric_only=False, has_bool_only=True)),
    ("median", DataFrameMedian, FunctionOptions()),
]


@pytest.mark.parametrize("func_name,op,func_opts", reduction_functions)
def test_series_reduction(func_name, op, func_opts: FunctionOptions):
    data = pd.Series(range(20), index=[str(i) for i in range(20)])
    series = getattr(from_pandas_series(data, chunk_size=3), func_name)()

    assert isinstance(series, Tensor)
    assert isinstance(series.op, op)
    assert series.shape == ()

    data = pd.Series(np.random.rand(25), name="a")
    if func_opts.has_skipna:
        kwargs = dict(axis="index", skipna=False)
    else:
        kwargs = dict()
    series = getattr(from_pandas_series(data, chunk_size=7), func_name)(**kwargs)

    assert isinstance(series, Tensor)
    assert series.shape == ()


@pytest.mark.parametrize("func_name,op,func_opts", reduction_functions)
def test_dataframe_reduction(func_name, op, func_opts: FunctionOptions):
    data = pd.DataFrame(
        {"a": list(range(20)), "b": list(range(20, 0, -1))},
        index=[str(i) for i in range(20)],
    )
    reduction_df = getattr(from_pandas_df(data, chunk_size=3), func_name)()

    assert isinstance(reduction_df, Series)
    assert isinstance(reduction_df.op, op)
    assert isinstance(reduction_df.index_value._index_value, IndexValue.Index)
    assert reduction_df.shape == (2,)

    data = pd.DataFrame(np.random.rand(20, 10))
    reduction_df = getattr(from_pandas_df(data, chunk_size=3), func_name)()

    assert isinstance(reduction_df, Series)
    assert_mf_index_dtype(reduction_df.index_value._index_value, np.int64)
    assert reduction_df.shape == (10,)

    data = pd.DataFrame(np.random.rand(20, 20), index=[str(i) for i in range(20)])
    reduction_df = getattr(from_pandas_df(data, chunk_size=4), func_name)(
        axis="columns"
    )

    assert reduction_df.shape == (20,)

    with pytest.raises(NotImplementedError):
        getattr(from_pandas_df(data, chunk_size=3), func_name)(level=0, axis=1)


cum_reduction_functions = [
    ("cummin", DataFrameCummin, FunctionOptions()),
    ("cummax", DataFrameCummax, FunctionOptions()),
    ("cumprod", DataFrameCumprod, FunctionOptions()),
    ("cumsum", DataFrameCumsum, FunctionOptions()),
]


@pytest.mark.parametrize("func_name,op,func_opts", cum_reduction_functions)
def test_cum_series_reduction(func_name, op, func_opts: FunctionOptions):
    data = pd.Series({"a": list(range(20))}, index=[str(i) for i in range(20)])
    series = getattr(from_pandas_series(data, chunk_size=3), func_name)()

    assert isinstance(series, Series)
    assert series.shape == (20,)

    data = pd.Series(np.random.rand(25), name="a")
    if func_opts.has_skipna:
        kwargs = dict(axis="index", skipna=False)
    else:
        kwargs = dict()
    series = getattr(from_pandas_series(data, chunk_size=7), func_name)(**kwargs)

    assert isinstance(series, Series)
    assert series.shape == (25,)


@pytest.mark.parametrize("func_name,op,func_opts", cum_reduction_functions)
def test_cum_dataframe_reduction(func_name, op, func_opts: FunctionOptions):
    data = pd.DataFrame(
        {"a": list(range(20)), "b": list(range(20, 0, -1))},
        index=[str(i) for i in range(20)],
    )
    reduction_df = getattr(from_pandas_df(data, chunk_size=3), func_name)()

    assert isinstance(reduction_df, DataFrame)
    assert isinstance(reduction_df.index_value._index_value, IndexValue.Index)
    assert reduction_df.shape == (20, 2)

    data = pd.DataFrame(np.random.rand(20, 10))
    reduction_df = getattr(from_pandas_df(data, chunk_size=3), func_name)()

    assert isinstance(reduction_df, DataFrame)
    assert isinstance(reduction_df.index_value._index_value, IndexValue.RangeIndex)
    assert reduction_df.shape == (20, 10)


def test_nunique():
    data = pd.DataFrame(
        np.random.randint(0, 6, size=(20, 10)),
        columns=["c" + str(i) for i in range(10)],
    )
    df = from_pandas_df(data, chunk_size=3)
    result = df.nunique()

    assert result.shape == (10,)
    assert result.op.output_types[0] == OutputType.series
    assert isinstance(result.op, DataFrameNunique)

    data2 = data.copy()
    df2 = from_pandas_df(data2, chunk_size=3)
    result2 = df2.nunique(axis=1)

    assert result2.shape == (20,)
    assert result2.op.output_types[0] == OutputType.series
    assert isinstance(result2.op, DataFrameNunique)


def test_unique():
    pd_df = pd.DataFrame(
        {
            "col1": pd.Series(np.random.choice(["a", "b", "c", "d"], 100)),
            "col2": pd.Series(np.random.choice([0, 1, 2, 3], 100)),
        }
    )
    df = from_pandas_df(pd_df, chunk_size=3)
    result = df.agg(["unique"])

    assert result.shape == (1, 2)
    assert result.op.output_types[0] == OutputType.dataframe
    assert isinstance(result.op, DataFrameAggregate)
    pd.testing.assert_series_equal(
        result.dtypes,
        pd.Series(
            [ArrowDtype(pa.list_(pa.string())), ArrowDtype(pa.list_(pa.int64()))],
            index=pd_df.columns,
        ),
    )

    pd_s = pd.Series(np.random.choice(["a", "b", "c", "d"], 100))
    ms = from_pandas_series(pd_s, chunk_size=3)
    result = ms.agg(["unique"])
    assert result.shape == (1,)
    assert result.op.output_types[0] == OutputType.series
    assert isinstance(result.op, DataFrameAggregate)
    assert result.dtype == ArrowDtype(pa.list_(pa.string()))


def test_dataframe_aggregate():
    data = pd.DataFrame(np.random.rand(20, 19))
    agg_funcs = [
        "sum",
        "min",
        "max",
        "mean",
        "var",
        "std",
        "all",
        "any",
        "skew",
        "kurt",
        "sem",
        "median",
    ]

    df = from_pandas_df(data)
    result = df.agg(agg_funcs)
    assert result.shape == (len(agg_funcs), data.shape[1])
    assert list(result.columns_value.to_pandas()) == list(range(19))
    assert list(result.index_value.to_pandas()) == agg_funcs
    assert result.op.output_types[0] == OutputType.dataframe
    assert result.op.func == agg_funcs

    df = from_pandas_df(data, chunk_size=(3, 4))

    result = df.agg("sum")
    assert result.shape == (data.shape[1],)
    assert list(result.index_value.to_pandas()) == list(range(data.shape[1]))
    assert result.op.output_types[0] == OutputType.series
    assert result.op.func == ["sum"]

    result = df.agg("sum", axis=1)
    assert result.shape == (data.shape[0],)
    assert list(result.index_value.to_pandas()) == list(range(data.shape[0]))
    assert result.op.output_types[0] == OutputType.series

    result = df.agg("var", axis=1)
    assert result.shape == (data.shape[0],)
    assert list(result.index_value.to_pandas()) == list(range(data.shape[0]))
    assert result.op.output_types[0] == OutputType.series
    assert result.op.func == ["var"]

    result = df.agg(agg_funcs)
    assert result.shape == (len(agg_funcs), data.shape[1])
    assert list(result.columns_value.to_pandas()) == list(range(data.shape[1]))
    assert list(result.index_value.to_pandas()) == agg_funcs
    assert result.op.output_types[0] == OutputType.dataframe
    assert result.op.func == agg_funcs

    result = df.agg(agg_funcs, axis=1)
    assert result.shape == (data.shape[0], len(agg_funcs))
    assert list(result.columns_value.to_pandas()) == agg_funcs
    assert list(result.index_value.to_pandas()) == list(range(data.shape[0]))
    assert result.op.output_types[0] == OutputType.dataframe
    assert result.op.func == agg_funcs

    dict_fun = {0: "sum", 2: ["var", "max"], 9: ["mean", "var", "std", "median"]}
    all_cols = set(
        reduce(
            operator.add, [[v] if isinstance(v, str) else v for v in dict_fun.values()]
        )
    )
    result = df.agg(dict_fun)
    assert result.shape == (len(all_cols), len(dict_fun))
    assert set(result.columns_value.to_pandas()) == set(dict_fun.keys())
    assert set(result.index_value.to_pandas()) == all_cols
    assert result.op.output_types[0] == OutputType.dataframe
    assert result.op.func[0] == [dict_fun[0]]
    assert result.op.func[2] == dict_fun[2]

    with pytest.raises(TypeError):
        df.agg(sum_0="sum", mean_0="mean", median_0="median")
    with pytest.raises(NotImplementedError):
        df.agg({0: ["sum", "min", "var"], 9: ["mean", "var", "std", "median"]}, axis=1)


def test_series_aggregate():
    data = pd.Series(np.random.rand(20), index=[str(i) for i in range(20)], name="a")
    agg_funcs = [
        "sum",
        "min",
        "max",
        "mean",
        "var",
        "std",
        "all",
        "any",
        "skew",
        "kurt",
        "sem",
        "median",
    ]

    series = from_pandas_series(data)

    result = series.agg(agg_funcs)
    assert result.shape == (len(agg_funcs),)
    assert list(result.index_value.to_pandas()) == agg_funcs
    assert result.op.output_types[0] == OutputType.series
    assert result.op.func == agg_funcs

    series = from_pandas_series(data, chunk_size=3)

    result = series.agg("sum")
    assert result.shape == ()
    assert result.op.output_types[0] == OutputType.scalar

    result = series.agg("median")
    assert result.shape == ()
    assert result.op.output_types[0] == OutputType.scalar

    result = series.median(level=0)
    assert result.shape == (np.nan,)
    assert result.op.output_types[0] == OutputType.series

    result = series.agg(agg_funcs)
    assert result.shape == (len(agg_funcs),)
    assert list(result.index_value.to_pandas()) == agg_funcs
    assert result.op.output_types[0] == OutputType.series
    assert result.op.func == agg_funcs

    with pytest.raises(TypeError):
        series.agg(sum_0=(0, "sum"), mean_0=(0, "mean"))


def test_compile_function():
    compiler = ReductionCompiler()
    ms = md.Series([1, 2, 3])
    # no MaxFrame objects inside closures
    with pytest.raises(ValueError):
        compiler.add_function(functools.partial(lambda x: (x + ms).sum()), ndim=2)
    # function should return a MaxFrame object
    with pytest.raises(ValueError):
        compiler.add_function(lambda x: x is not None, ndim=2)
    # function should perform some sort of reduction in dimensionality
    with pytest.raises(ValueError):
        compiler.add_function(lambda x: x, ndim=2)
    # function should only contain acceptable operators
    with pytest.raises(ValueError):
        compiler.add_function(lambda x: x.sort_values().max(), ndim=1)
    with pytest.raises(ValueError):
        compiler.add_function(lambda x: x.max().shift(1), ndim=2)

    # test agg for all data
    for ndim in [1, 2]:
        compiler = ReductionCompiler()
        compiler.add_function(lambda x: (x**2).count() + 1, ndim=ndim)
        result = compiler.compile()
        # check pre_funcs
        assert len(result.pre_funcs) == 1
        assert b"pow" in result.pre_funcs[0].func_idl
        # check agg_funcs
        assert len(result.agg_funcs) == 1
        assert result.agg_funcs[0].map_func_name == "count"
        assert result.agg_funcs[0].agg_func_name == "sum"
        # check post_funcs
        assert len(result.post_funcs) == 1
        assert result.post_funcs[0].func_name == "<lambda>"
        assert b"add" in result.post_funcs[0].func_idl

        compiler.add_function(
            lambda x: -x.prod() ** 2 + (1 + (x**2).count()), ndim=ndim
        )
        result = compiler.compile()
        # check pre_funcs
        assert len(result.pre_funcs) == 2
        assert (
            b"pow" in result.pre_funcs[0].func_idl
            or b"pow" in result.pre_funcs[1].func_idl
        )
        assert (
            b"pow" not in result.pre_funcs[0].func_idl
            or b"pow" not in result.pre_funcs[1].func_idl
        )
        # check agg_funcs
        assert len(result.agg_funcs) == 2
        assert set(result.agg_funcs[i].map_func_name for i in range(2)) == {
            "count",
            "prod",
        }
        assert set(result.agg_funcs[i].agg_func_name for i in range(2)) == {
            "sum",
            "prod",
        }
        # check post_funcs
        assert len(result.post_funcs) == 2
        assert result.post_funcs[0].func_name == "<lambda_0>"
        assert b"add" in result.post_funcs[0].func_idl
        assert b"add" in result.post_funcs[1].func_idl

        compiler = ReductionCompiler()
        compiler.add_function(
            lambda x: where_function(x.all(), x.count(), 0), ndim=ndim
        )
        result = compiler.compile()
        # check pre_funcs
        assert len(result.pre_funcs) == 1
        assert result.pre_funcs[0].input_key == result.pre_funcs[0].output_key
        # check agg_funcs
        assert len(result.agg_funcs) == 2
        assert set(result.agg_funcs[i].map_func_name for i in range(2)) == {
            "all",
            "count",
        }
        assert set(result.agg_funcs[i].agg_func_name for i in range(2)) == {
            "sum",
            "all",
        }
        # check post_funcs
        assert len(result.post_funcs) == 1
        assert b"where" in result.post_funcs[0].func_idl

        # check boolean expressions
        compiler = ReductionCompiler()
        compiler.add_function(lambda x: (x == "1").sum(), ndim=ndim)
        result = compiler.compile()
        # check pre_funcs
        assert len(result.pre_funcs) == 1
        assert b"eq" in result.pre_funcs[0].func_idl
        # check agg_funcs
        assert len(result.agg_funcs) == 1
        assert result.agg_funcs[0].map_func_name == "sum"
        assert result.agg_funcs[0].agg_func_name == "sum"

    # test agg for specific columns
    compiler = ReductionCompiler()
    compiler.add_function(lambda x: 1 + x.sum(), ndim=2, cols=["a", "b"])
    compiler.add_function(lambda x: -1 + x.sum(), ndim=2, cols=["b", "c"])
    result = compiler.compile()
    # check pre_funcs
    assert len(result.pre_funcs) == 1
    assert set(result.pre_funcs[0].columns) == set("abc")
    # check agg_funcs
    assert len(result.agg_funcs) == 1
    assert result.agg_funcs[0].map_func_name == "sum"
    assert result.agg_funcs[0].agg_func_name == "sum"
    # check post_funcs
    assert len(result.post_funcs) == 2
    assert set("".join(sorted(result.post_funcs[i].columns)) for i in range(2)) == {
        "ab",
        "bc",
    }

    # test agg for multiple columns
    compiler = ReductionCompiler()
    compiler.add_function(lambda x: x.sum(), ndim=2, cols=["a"])
    compiler.add_function(lambda x: x.sum(), ndim=2, cols=["b"])
    compiler.add_function(lambda x: x.min(), ndim=2, cols=["c"])
    result = compiler.compile()
    # check pre_funcs
    assert len(result.pre_funcs) == 1
    assert set(result.pre_funcs[0].columns) == set("abc")
    # check agg_funcs
    assert len(result.agg_funcs) == 2
    assert result.agg_funcs[0].map_func_name == "sum"
    assert result.agg_funcs[0].agg_func_name == "sum"
    # check post_funcs
    assert len(result.post_funcs) == 2
    assert set(result.post_funcs[0].columns) == set("ab")


def test_custom_aggregation():
    class MockReduction1(CustomReduction):
        def agg(self, v1):
            return v1.sum()

    class MockReduction2(CustomReduction):
        def pre(self, value):
            return value + 1, value**2

        def agg(self, v1, v2):
            return v1.sum(), v2.prod()

        def post(self, v1, v2):
            return v1 + v2

    for ndim in [1, 2]:
        compiler = ReductionCompiler()
        compiler.add_function(MockReduction1(), ndim=ndim)
        result = compiler.compile()
        # check agg_funcs
        assert len(result.agg_funcs) == 1
        assert result.agg_funcs[0].map_func_name == "custom_reduction"
        assert result.agg_funcs[0].agg_func_name == "custom_reduction"
        assert isinstance(result.agg_funcs[0].custom_reduction, MockReduction1)
        assert result.agg_funcs[0].output_limit == 1

        compiler = ReductionCompiler()
        compiler.add_function(MockReduction2(), ndim=ndim)
        result = compiler.compile()
        # check agg_funcs
        assert len(result.agg_funcs) == 1
        assert result.agg_funcs[0].map_func_name == "custom_reduction"
        assert result.agg_funcs[0].agg_func_name == "custom_reduction"
        assert isinstance(result.agg_funcs[0].custom_reduction, MockReduction2)
        assert result.agg_funcs[0].output_limit == 2


def test_aggregation_with_odps_function():
    odps_func = ODPSFunction("test_odps_udaf", dtype=np.float64)
    for ndim in [1, 2]:
        compiler = ReductionCompiler()
        compiler.add_function(odps_func, ndim=ndim)
        result = compiler.compile()
        assert result.agg_funcs[0].map_func_name == "custom_reduction"
        assert result.agg_funcs[0].agg_func_name == "custom_reduction"
        assert isinstance(result.agg_funcs[0].custom_reduction, ODPSFunction)
