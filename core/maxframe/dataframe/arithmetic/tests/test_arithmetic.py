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

import datetime
import operator
from dataclasses import dataclass
from math import isinf
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from ....core import OperatorType
from ....core.operator import estimate_size
from ....tests.utils import assert_mf_index_dtype
from ....utils import dataslots
from ...core import IndexValue
from ...datasource.dataframe import from_pandas
from ...datasource.series import from_pandas as from_pandas_series
from ...utils import split_monotonic_index_min_max
from .. import (
    DataFrameAdd,
    DataFrameAnd,
    DataFrameEqual,
    DataFrameFloorDiv,
    DataFrameGreater,
    DataFrameGreaterEqual,
    DataFrameLess,
    DataFrameLessEqual,
    DataFrameMod,
    DataFrameMul,
    DataFrameNotEqual,
    DataFrameOr,
    DataFramePower,
    DataFrameSubtract,
    DataFrameTrueDiv,
    DataFrameXor,
)


def comp_func(name, reverse_name):
    def inner(lhs, rhs):
        try:
            return getattr(lhs, name)(rhs)
        except AttributeError:
            return getattr(rhs, reverse_name)(lhs)

    return inner


@dataslots
@dataclass
class FunctionOptions:
    func: Callable
    op: OperatorType
    func_name: str
    rfunc_name: str


binary_functions = dict(
    add=FunctionOptions(
        func=operator.add, op=DataFrameAdd, func_name="add", rfunc_name="radd"
    ),
    subtract=FunctionOptions(
        func=operator.sub, op=DataFrameSubtract, func_name="sub", rfunc_name="rsub"
    ),
    multiply=FunctionOptions(
        func=operator.mul, op=DataFrameMul, func_name="mul", rfunc_name="rmul"
    ),
    floordiv=FunctionOptions(
        func=operator.floordiv,
        op=DataFrameFloorDiv,
        func_name="floordiv",
        rfunc_name="rfloordiv",
    ),
    truediv=FunctionOptions(
        func=operator.truediv,
        op=DataFrameTrueDiv,
        func_name="truediv",
        rfunc_name="rtruediv",
    ),
    mod=FunctionOptions(
        func=operator.mod, op=DataFrameMod, func_name="mod", rfunc_name="rmod"
    ),
    power=FunctionOptions(
        func=operator.pow, op=DataFramePower, func_name="pow", rfunc_name="rpow"
    ),
    equal=FunctionOptions(
        func=comp_func("eq", "eq"), op=DataFrameEqual, func_name="eq", rfunc_name="eq"
    ),
    not_equal=FunctionOptions(
        func=comp_func("ne", "ne"),
        op=DataFrameNotEqual,
        func_name="ne",
        rfunc_name="ne",
    ),
    greater=FunctionOptions(
        func=comp_func("gt", "lt"), op=DataFrameGreater, func_name="gt", rfunc_name="lt"
    ),
    less=FunctionOptions(
        func=comp_func("lt", "gt"), op=DataFrameLess, func_name="lt", rfunc_name="gt"
    ),
    greater_equal=FunctionOptions(
        func=comp_func("ge", "le"),
        op=DataFrameGreaterEqual,
        func_name="ge",
        rfunc_name="le",
    ),
    less_equal=FunctionOptions(
        func=comp_func("le", "ge"),
        op=DataFrameLessEqual,
        func_name="le",
        rfunc_name="ge",
    ),
    logical_and=FunctionOptions(
        func=operator.and_, op=DataFrameAnd, func_name="__and__", rfunc_name="and"
    ),
    logical_or=FunctionOptions(
        func=operator.or_, op=DataFrameOr, func_name="__or__", rfunc_name="__ror__"
    ),
    logical_xor=FunctionOptions(
        func=operator.xor, op=DataFrameXor, func_name="__xor__", rfunc_name="__rxor__"
    ),
)


def to_boolean_if_needed(func_name, value, split_value=0.5):
    if func_name in ["__and__", "__or__", "__xor__"]:
        return value > split_value
    else:
        return value


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_without_shuffle(func_name, func_opts):
    # all the axes are monotonic
    # data1 with index split into [0...4], [5...9],
    # columns [3...7], [8...12]
    data1 = pd.DataFrame(
        np.random.rand(10, 10), index=np.arange(10), columns=np.arange(3, 13)
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)
    # data2 with index split into [6...11], [2, 5],
    # columns [4...9], [10, 13]
    data2 = pd.DataFrame(
        np.random.rand(10, 10), index=np.arange(11, 1, -1), columns=np.arange(4, 14)
    )
    data2 = to_boolean_if_needed(func_opts.func_name, data2)
    df2 = from_pandas(data2, chunk_size=6)

    df3 = func_opts.func(df1, df2)

    # test df3's index and columns
    pd.testing.assert_index_equal(
        df3.columns_value.to_pandas(), func_opts.func(data1, data2).columns
    )
    assert_mf_index_dtype(df3.index_value.value, np.int64)
    pd.testing.assert_index_equal(
        df3.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df3.index_value.key != df1.index_value.key
    assert df3.index_value.key != df2.index_value.key
    assert df3.shape[1] == 11  # columns is recorded, so we can get it

    # test df3's index and columns after tiling
    pd.testing.assert_index_equal(
        df3.columns_value.to_pandas(), func_opts.func(data1, data2).columns
    )
    assert_mf_index_dtype(df3.index_value.value, np.int64)
    pd.testing.assert_index_equal(
        df3.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df3.index_value.key != df1.index_value.key
    assert df3.index_value.key != df2.index_value.key
    assert df3.shape[1] == 11  # columns is recorded, so we can get it

    result_ctx = {inp.key: 10 for inp in df3.op.inputs}
    estimate_size(result_ctx, df3.op)
    assert result_ctx[df3.key] >= 0 and not isinf(result_ctx[df3.key])


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_dataframe_and_series_with_align_map(func_name, func_opts):
    data1 = pd.DataFrame(
        np.random.rand(10, 10), index=np.arange(10), columns=np.arange(3, 13)
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)
    s1 = df1[3]

    df2 = func_opts.func(df1, s1)

    assert df2.shape == (df1.shape[0], np.nan)
    assert df2.index_value.key == df1.index_value.key


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_dataframe_and_series_identical(func_name, func_opts):
    data1 = pd.DataFrame(
        np.random.rand(10, 10), index=np.arange(10), columns=np.arange(10)
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)
    s1 = from_pandas_series(data1[3], chunk_size=5)

    df2 = func_opts.func(df1, s1)

    assert df2.shape == (10, 10)
    assert df2.index_value.key == df1.index_value.key
    assert df2.columns_value.key == df1.columns_value.key
    assert df2.columns_value.key == s1.index_value.key


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_dataframe_and_series_with_shuffle(func_name, func_opts):
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[4, 9, 3, 2, 1, 5, 8, 6, 7, 10],
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)
    s1 = from_pandas_series(data1[10], chunk_size=6)

    df2 = func_opts.func(df1, s1)

    # test df2's index and columns
    assert df2.shape == (df1.shape[0], np.nan)
    assert df2.index_value.key == df1.index_value.key
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df2.columns_value.key != df1.columns_value.key


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_dataframe_and_series_with_multiindex(func_name, func_opts):
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=pd.MultiIndex.from_arrays(
            [list("AAAAABBBBB"), [4, 9, 3, 2, 1, 5, 8, 6, 7, 10]]
        ),
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)
    s1 = from_pandas_series(data1[10].reset_index(level=0, drop=True), chunk_size=6)

    df2 = getattr(df1, func_opts.func_name)(s1, level=1, axis=0)

    # test df2's index and columns
    assert df2.shape == (np.nan, df1.shape[1])
    assert df2.index_value.key != df1.index_value.key
    assert df2.index_value.names == df1.index_value.names
    assert df2.columns_value.key == df1.columns_value.key


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_series_and_series_with_align_map(func_name, func_opts):
    data1 = pd.DataFrame(
        np.random.rand(10, 10), index=np.arange(10), columns=np.arange(3, 13)
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)

    s1 = df1.iloc[4]
    s2 = df1[3]

    s3 = func_opts.func(s1, s2)

    assert s3.shape == (np.nan,)

    s1_index_min_max = [[3, True, 7, True], [8, True, 12, True]]
    s2_index_min_max = [(0, True, 4, True), (5, True, 9, True)]

    left_index_splits, right_index_splits = split_monotonic_index_min_max(
        s1_index_min_max, True, s2_index_min_max, True
    )


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_series_and_series_identical(func_name, func_opts):
    data1 = pd.DataFrame(
        np.random.rand(10, 10), index=np.arange(10), columns=np.arange(10)
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    s1 = from_pandas_series(data1[1], chunk_size=5)
    s2 = from_pandas_series(data1[3], chunk_size=5)

    s3 = func_opts.func(s1, s2)

    assert s3.shape == (10,)
    assert s3.index_value.key == s1.index_value.key
    assert s3.index_value.key == s2.index_value.key


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_series_and_series_with_shuffle(func_name, func_opts):
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[4, 9, 3, 2, 1, 5, 8, 6, 7, 10],
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    s1 = from_pandas_series(data1.iloc[4], chunk_size=5)
    s2 = from_pandas_series(data1[10], chunk_size=6)

    s3 = func_opts.func(s1, s2)

    # test s3's index
    assert s3.shape == (np.nan,)
    assert s3.index_value.key != s1.index_value.key
    assert s3.index_value.key != s2.index_value.key
    pd.testing.assert_index_equal(
        s3.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_identical_index_and_columns(func_name, func_opts):
    data1 = pd.DataFrame(np.random.rand(10, 10), columns=np.arange(3, 13))
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)
    data2 = pd.DataFrame(np.random.rand(10, 10), columns=np.arange(3, 13))
    data2 = to_boolean_if_needed(func_opts.func_name, data2)
    df2 = from_pandas(data2, chunk_size=5)

    df3 = func_opts.func(df1, df2)

    # test df3's index and columns
    pd.testing.assert_index_equal(
        df3.columns_value.to_pandas(), func_opts.func(data1, data2).columns
    )
    assert isinstance(df3.index_value.value, IndexValue.RangeIndex)
    pd.testing.assert_index_equal(df3.index_value.to_pandas(), pd.RangeIndex(0, 10))
    assert df3.index_value.key == df1.index_value.key
    assert df3.index_value.key == df2.index_value.key
    assert df3.shape == (10, 10)  # columns is recorded, so we can get it


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_with_one_shuffle(func_name, func_opts):
    # only 1 axis is monotonic
    # data1 with index split into [0...4], [5...9],
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.arange(10),
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)
    # data2 with index split into [6...11], [2, 5],
    data2 = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.arange(11, 1, -1),
        columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2],
    )
    data2 = to_boolean_if_needed(func_opts.func_name, data2)
    df2 = from_pandas(data2, chunk_size=6)

    df3 = func_opts.func(df1, df2)

    # test df3's index and columns
    pd.testing.assert_index_equal(
        df3.columns_value.to_pandas(), func_opts.func(data1, data2).columns
    )
    assert_mf_index_dtype(df3.index_value.value, np.int64)
    pd.testing.assert_index_equal(
        df3.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df3.index_value.key != df1.index_value.key
    assert df3.index_value.key != df2.index_value.key
    assert df3.shape[1] == 12  # columns is recorded, so we can get it


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_with_all_shuffle(func_name, func_opts):
    # no axis is monotonic
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=5)
    data2 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3],
        columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2],
    )
    data2 = to_boolean_if_needed(func_opts.func_name, data2)
    df2 = from_pandas(data2, chunk_size=6)

    df3 = func_opts.func(df1, df2)

    # test df3's index and columns
    pd.testing.assert_index_equal(
        df3.columns_value.to_pandas(), func_opts.func(data1, data2).columns
    )
    assert_mf_index_dtype(df3.index_value.value, np.int64)
    pd.testing.assert_index_equal(
        df3.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df3.index_value.key != df1.index_value.key
    assert df3.index_value.key != df2.index_value.key
    assert df3.shape[1] == 12  # columns is recorded, so we can get it

    data4 = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.random.randint(-100, 100, size=(10,)),
        columns=[np.random.bytes(10) for _ in range(10)],
    )
    data4 = to_boolean_if_needed(func_opts.func_name, data4)
    df4 = from_pandas(data4, chunk_size=3)

    data5 = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.random.randint(-100, 100, size=(10,)),
        columns=[np.random.bytes(10) for _ in range(10)],
    )
    data5 = to_boolean_if_needed(func_opts.func_name, data5)
    df5 = from_pandas(data5, chunk_size=3)

    df6 = func_opts.func(df4, df5)

    # test df6's index and columns
    pd.testing.assert_index_equal(
        df6.columns_value.to_pandas(), func_opts.func(data4, data5).columns
    )
    assert_mf_index_dtype(df6.index_value.value, np.int64)
    pd.testing.assert_index_equal(
        df6.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df6.index_value.key != df4.index_value.key
    assert df6.index_value.key != df5.index_value.key
    assert df6.shape[1] == 20  # columns is recorded, so we can get it


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_without_shuffle_and_with_one_chunk(func_name, func_opts):
    # only 1 axis is monotonic
    # data1 with index split into [0...4], [5...9],
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.arange(10),
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=(5, 10))
    # data2 with index split into [6...11], [2, 5],
    data2 = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.arange(11, 1, -1),
        columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2],
    )
    data2 = to_boolean_if_needed(func_opts.func_name, data2)
    df2 = from_pandas(data2, chunk_size=(6, 10))

    df3 = func_opts.func(df1, df2)

    # test df3's index and columns
    pd.testing.assert_index_equal(
        df3.columns_value.to_pandas(), func_opts.func(data1, data2).columns
    )
    assert_mf_index_dtype(df3.index_value.value, np.int64)
    pd.testing.assert_index_equal(
        df3.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df3.index_value.key != df1.index_value.key
    assert df3.index_value.key != df2.index_value.key
    assert df3.shape[1] == 12  # columns is recorded, so we can get it


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_both_one_chunk(func_name, func_opts):
    # no axis is monotonic, but 1 chunk for all axes
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=10)
    data2 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3],
        columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2],
    )
    data2 = to_boolean_if_needed(func_opts.func_name, data2)
    df2 = from_pandas(data2, chunk_size=10)

    df3 = func_opts.func(df1, df2)

    # test df3's index and columns
    pd.testing.assert_index_equal(
        df3.columns_value.to_pandas(), func_opts.func(data1, data2).columns
    )
    assert_mf_index_dtype(df3.index_value.value, np.int64)
    pd.testing.assert_index_equal(
        df3.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df3.index_value.key != df1.index_value.key
    assert df3.index_value.key != df2.index_value.key
    assert df3.shape[1] == 12  # columns is recorded, so we can get it


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_with_shuffle_and_one_chunk(func_name, func_opts):
    # no axis is monotonic
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    data1 = to_boolean_if_needed(func_opts.func_name, data1)
    df1 = from_pandas(data1, chunk_size=(5, 10))
    data2 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3],
        columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2],
    )
    data2 = to_boolean_if_needed(func_opts.func_name, data2)
    df2 = from_pandas(data2, chunk_size=(6, 10))

    df3 = func_opts.func(df1, df2)

    # test df3's index and columns
    pd.testing.assert_index_equal(
        df3.columns_value.to_pandas(), func_opts.func(data1, data2).columns
    )
    assert_mf_index_dtype(df3.index_value.value, np.int64)
    pd.testing.assert_index_equal(
        df3.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df3.index_value.key != df1.index_value.key
    assert df3.index_value.key != df2.index_value.key
    assert df3.shape[1] == 12  # columns is recorded, so we can get it


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_on_same_dataframe(func_name, func_opts):
    data = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.random.randint(-100, 100, size=(10,)),
        columns=[np.random.bytes(10) for _ in range(10)],
    )
    data = to_boolean_if_needed(func_opts.func_name, data)
    df = from_pandas(data, chunk_size=3)
    df2 = func_opts.func(df, df)

    # test df2's index and columns
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), func_opts.func(data, data).columns
    )
    assert_mf_index_dtype(df2.index_value.value, np.int64)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )
    assert df2.index_value.key == df.index_value.key
    assert df2.columns_value.key == df.columns_value.key
    assert df2.shape[1] == 10


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_dataframe_and_scalar(func_name, func_opts):
    if func_opts.func_name in ["__and__", "__or__", "__xor__"]:
        # bitwise logical operators doesn\'t support floating point scalars
        return

    data = pd.DataFrame(
        np.random.rand(10, 10), index=np.arange(10), columns=np.arange(3, 13)
    )
    df = from_pandas(data, chunk_size=5)
    # test operator with scalar
    result = func_opts.func(df, 1)
    result2 = getattr(df, func_opts.func_name)(1)

    # test reverse operator with scalar
    result3 = getattr(df, func_opts.rfunc_name)(1)
    result4 = func_opts.func(df, 1)
    result5 = func_opts.func(1, df)

    expected = func_opts.func(data, 2)
    pd.testing.assert_series_equal(result.dtypes, expected.dtypes)

    pd.testing.assert_index_equal(result.columns_value.to_pandas(), data.columns)
    assert_mf_index_dtype(result.index_value.value, np.int64)

    pd.testing.assert_index_equal(result2.columns_value.to_pandas(), data.columns)
    assert_mf_index_dtype(result2.index_value.value, np.int64)

    pd.testing.assert_index_equal(result3.columns_value.to_pandas(), data.columns)
    assert_mf_index_dtype(result3.index_value.value, np.int64)

    pd.testing.assert_index_equal(result4.columns_value.to_pandas(), data.columns)
    assert_mf_index_dtype(result4.index_value.value, np.int64)

    pd.testing.assert_index_equal(result5.columns_value.to_pandas(), data.columns)
    assert_mf_index_dtype(result5.index_value.value, np.int64)

    if "builtin_function_or_method" not in str(type(func_opts.func)):
        # skip NotImplemented test for comparison function
        return

    # test NotImplemented, use other's rfunc instead
    class TestRFunc:
        pass

    setattr(TestRFunc, f"__{func_opts.rfunc_name}__", lambda *_: 1)
    other = TestRFunc()
    ret = func_opts.func(df, other)
    assert ret == 1


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_series_and_scalar(func_name, func_opts):
    if func_opts.func_name in ["__and__", "__or__", "__xor__"]:
        # bitwise logical operators doesn\'t support floating point scalars
        return

    data = pd.Series(range(10), index=[1, 3, 4, 2, 9, 10, 33, 23, 999, 123])
    s1 = from_pandas_series(data, chunk_size=3)
    r = getattr(s1, func_opts.func_name)(456)

    assert r.index_value.key == s1.index_value.key
    assert r.dtype == getattr(data, func_opts.func_name)(456).dtype

    if "builtin_function_or_method" not in str(type(func_opts.func)):
        # skip rfunc test for comparison function
        return

    s1 = from_pandas_series(data, chunk_size=3)
    r = getattr(s1, func_opts.rfunc_name)(789)
    assert r.index_value.key == s1.index_value.key


@pytest.mark.parametrize("func_name, func_opts", binary_functions.items())
def test_check_inputs(func_name, func_opts):
    data = pd.DataFrame(np.random.rand(10, 3))
    data = to_boolean_if_needed(func_opts.func_name, data)
    df = from_pandas(data)

    with pytest.raises(ValueError):
        _ = df + np.random.rand(5, 3)

    with pytest.raises(ValueError):
        _ = df + np.random.rand(10)

    with pytest.raises(ValueError):
        _ = df + np.random.rand(10, 3, 2)

    data = pd.Series(np.random.rand(10))
    series = from_pandas_series(data)

    with pytest.raises(ValueError):
        _ = series + np.random.rand(5, 3)

    with pytest.raises(ValueError):
        _ = series + np.random.rand(5)


def test_abs():
    data1 = pd.DataFrame(
        np.random.rand(10, 10),
        index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    df1 = from_pandas(data1, chunk_size=(5, 10))

    df2 = df1.abs()

    # test df2's index and columns
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), df1.columns_value.to_pandas()
    )
    assert_mf_index_dtype(df2.index_value.value, np.int64)
    assert df2.shape == (10, 10)


def test_not():
    data1 = pd.DataFrame(
        np.random.rand(10, 10) > 0.5,
        index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
        columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
    )
    df1 = from_pandas(data1, chunk_size=(5, 10))

    df2 = ~df1

    # test df2's index and columns
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), df1.columns_value.to_pandas()
    )
    assert_mf_index_dtype(df2.index_value.value, np.int64)
    assert df2.shape == (10, 10)


def test_datetime_arithmetic():
    data1 = (
        pd.Series([pd.Timedelta(days=d) for d in range(10)]) + datetime.datetime.now()
    )
    s1 = from_pandas_series(data1)

    assert (s1 + pd.Timedelta(days=10)).dtype == (data1 + pd.Timedelta(days=10)).dtype
    assert (s1 + datetime.timedelta(days=10)).dtype == (
        data1 + datetime.timedelta(days=10)
    ).dtype
    assert (s1 - pd.Timestamp.now()).dtype == (data1 - pd.Timestamp.now()).dtype
    assert (s1 - datetime.datetime.now()).dtype == (
        data1 - datetime.datetime.now()
    ).dtype
