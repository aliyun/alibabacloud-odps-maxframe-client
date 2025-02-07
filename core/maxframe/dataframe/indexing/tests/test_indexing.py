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

import numpy as np
import pandas as pd
import pytest

from .... import dataframe as md
from .... import tensor as mt
from ....tensor.core import TENSOR_TYPE, Tensor
from ...core import DATAFRAME_TYPE, SERIES_TYPE, DataFrame, Series
from ..iloc import DataFrameIlocGetItem, DataFrameIlocSetItem, IndexingError
from ..loc import DataFrameLocGetItem


def test_iloc_getitem():
    df1 = pd.DataFrame(
        [[1, 3, 3], [4, 2, 6], [7, 8, 9]],
        index=["a1", "a2", "a3"],
        columns=["x", "y", "z"],
    )
    df2 = md.DataFrame(df1, chunk_size=2)

    with pytest.raises(IndexingError):
        _ = df2.iloc[1, 1, 1]

    # index cannot be tuple
    with pytest.raises(IndexingError):
        _ = df2.iloc[((1,),)]

    # index wrong type
    with pytest.raises(TypeError):
        _ = df2.iloc["a1":]

    with pytest.raises(NotImplementedError):
        _ = df2.iloc[0, md.Series(["a2", "a3"])]

    # fancy index should be 1-d
    with pytest.raises(ValueError):
        _ = df2.iloc[[[0, 1], [1, 2]]]

    with pytest.raises(ValueError):
        _ = df2.iloc[1, ...]

    with pytest.raises(IndexError):
        _ = df2.iloc[-4]

    with pytest.raises(IndexError):
        _ = df2.iloc[3]

    # plain index
    df3 = df2.iloc[1]
    assert isinstance(df3, SERIES_TYPE)
    assert isinstance(df3.op, DataFrameIlocGetItem)
    assert df3.shape == (3,)

    # slice index
    df4 = df2.iloc[:, 2:4]
    assert isinstance(df4, DATAFRAME_TYPE)
    assert isinstance(df4.op, DataFrameIlocGetItem)
    assert df4.index_value.key == df2.index_value.key
    assert df4.shape == (3, 1)

    # plain fancy index
    df5 = df2.iloc[[0], [0, 1, 2]]
    assert isinstance(df5, DATAFRAME_TYPE)
    assert isinstance(df5.op, DataFrameIlocGetItem)
    assert df5.shape == (1, 3)

    # fancy index
    df6 = df2.iloc[[1, 2], [0, 1, 2]]
    assert isinstance(df6, DATAFRAME_TYPE)
    assert isinstance(df6.op, DataFrameIlocGetItem)
    assert df6.shape == (2, 3)

    # plain index
    df7 = df2.iloc[1, 2]
    assert isinstance(df7, TENSOR_TYPE)  # scalar
    assert isinstance(df7.op, DataFrameIlocGetItem)
    assert df7.shape == ()

    # test Series iloc getitem

    # slice
    series = md.Series(pd.Series(np.arange(10)), chunk_size=3).iloc[4:8]

    assert series.shape == (4,)

    # fancy index
    series = md.Series(pd.Series(np.arange(10)), chunk_size=3).iloc[[2, 4, 8]]

    assert series.shape == (3,)


def test_iloc_setitem():
    df1 = pd.DataFrame(
        [[1, 3, 3], [4, 2, 6], [7, 8, 9]],
        index=["a1", "a2", "a3"],
        columns=["x", "y", "z"],
    )
    df2 = md.DataFrame(df1, chunk_size=2)

    # plain index
    df3 = md.DataFrame(df1, chunk_size=2)
    df3.iloc[1] = 100
    assert isinstance(df3.op, DataFrameIlocSetItem)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df3.index_value.to_pandas()
    )
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), df3.columns_value.to_pandas()
    )

    # # slice index
    df4 = md.DataFrame(df1, chunk_size=2)
    df4.iloc[:, 2:4] = 1111
    assert isinstance(df4.op, DataFrameIlocSetItem)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df4.index_value.to_pandas()
    )
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), df4.columns_value.to_pandas()
    )

    # plain fancy index
    df5 = md.DataFrame(df1, chunk_size=2)
    df5.iloc[[0], [0, 1, 2]] = 2222
    assert isinstance(df5.op, DataFrameIlocSetItem)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df5.index_value.to_pandas()
    )
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), df5.columns_value.to_pandas()
    )

    # fancy index
    df6 = md.DataFrame(df1, chunk_size=2)
    df6.iloc[[1, 2], [0, 1, 2]] = 3333
    assert isinstance(df6.op, DataFrameIlocSetItem)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df6.index_value.to_pandas()
    )
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), df6.columns_value.to_pandas()
    )

    # plain index
    df7 = md.DataFrame(df1, chunk_size=2)
    df7.iloc[1, 2] = 4444
    assert isinstance(df7.op, DataFrameIlocSetItem)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df7.index_value.to_pandas()
    )
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), df7.columns_value.to_pandas()
    )

    # test Series

    # slice
    series = md.Series(pd.Series(np.arange(10)), chunk_size=3)
    series.iloc[:4] = 2

    assert series.shape == (10,)

    # fancy index
    series = md.Series(pd.Series(np.arange(10)), chunk_size=3)
    series.iloc[[2, 4, 9]] = 3

    assert series.shape == (10,)


def test_dataframe_loc():
    raw = pd.DataFrame(
        [[1, 3, 3], [4, 2, 6], [7, 8, 9]],
        index=["a1", "a2", "a3"],
        columns=["x", "y", "z"],
    )
    df = md.DataFrame(raw, chunk_size=2)
    raw2 = raw.copy()
    raw2.reset_index(inplace=True, drop=True)
    df3 = md.DataFrame(raw2, chunk_size=2)
    s = pd.Series([1, 3, 5], index=["a1", "a2", "a3"])
    series = md.Series(s, chunk_size=2)

    # test return scalar
    df2 = df.loc["a1", "z"]
    assert isinstance(df2, Tensor)
    assert df2.shape == ()
    assert df2.dtype == raw["z"].dtype

    # test return series for index axis
    df2 = df.loc[:, "y"]
    assert isinstance(df2, Series)
    assert df2.shape == (3,)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df.index_value.to_pandas()
    )
    assert df2.name == "y"
    assert df2.index_value.key == df.index_value.key

    # test return series for column axis
    df2 = df.loc["a2", :]
    assert isinstance(df2, Series)
    assert df2.shape == (3,)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df.columns_value.to_pandas()
    )
    assert df2.name == "a2"

    # test slice
    df2 = df.loc["a2":"a3", "y":"z"]
    assert isinstance(df2, DataFrame)
    assert df2.shape == (np.nan, 2)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df.index_value.to_pandas()
    )
    assert df2.index_value.key != df.index_value.key
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), raw.loc[:, "y":"z"].columns
    )
    pd.testing.assert_series_equal(df2.dtypes, raw.loc[:, "y":"z"].dtypes)

    # test fancy index on index axis
    df2 = df.loc[["a3", "a2"], [True, False, True]]
    assert isinstance(df2, DataFrame)
    assert df2.shape == (2, 2)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df.index_value.to_pandas()
    )
    assert df2.index_value.key != df.index_value.key
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), raw.loc[:, [True, False, True]].columns
    )
    pd.testing.assert_series_equal(df2.dtypes, raw.loc[:, [True, False, True]].dtypes)

    # test fancy index which is md.Series on index axis
    df2 = df.loc[md.Series(["a3", "a2"]), [True, False, True]]
    assert isinstance(df2, DataFrame)
    assert df2.shape == (2, 2)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df.index_value.to_pandas()
    )
    assert df2.index_value.key != df.index_value.key
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), raw.loc[:, [True, False, True]].columns
    )
    pd.testing.assert_series_equal(df2.dtypes, raw.loc[:, [True, False, True]].dtypes)

    # test fancy index on columns axis
    df2 = df.loc[[True, False, True], ["z", "x", "y"]]
    assert isinstance(df2, DataFrame)
    assert df2.shape == (2, 3)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df.index_value.to_pandas()
    )
    assert df2.index_value.key != df.index_value.key
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), raw.loc[:, ["z", "x", "y"]].columns
    )
    pd.testing.assert_series_equal(df2.dtypes, raw.loc[:, ["z", "x", "y"]].dtypes)

    df2 = df.loc[md.Series([True, False, True])]
    assert isinstance(df2, DataFrame)
    assert df2.shape == (np.nan, 3)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df.index_value.to_pandas()
    )
    assert df2.index_value.key != df.index_value.key
    pd.testing.assert_index_equal(df2.columns_value.to_pandas(), raw.columns)
    pd.testing.assert_series_equal(df2.dtypes, raw.dtypes)

    df2 = df3.loc[md.Series([True, False, True])]
    assert isinstance(df2, DataFrame)
    assert df2.shape == (np.nan, 3)
    assert isinstance(
        df2.index_value.to_pandas(), type(raw.loc[[True, False, True]].index)
    )
    assert df2.index_value.key != df3.index_value.key
    pd.testing.assert_index_equal(df2.columns_value.to_pandas(), raw.columns)
    pd.testing.assert_series_equal(df2.dtypes, raw.dtypes)

    df2 = df3.loc[md.Series([2, 1])]
    assert isinstance(df2, DataFrame)
    assert df2.shape == (2, 3)
    assert isinstance(df2.index_value.to_pandas(), type(raw2.loc[[2, 1]].index))
    assert df2.index_value.key != df3.index_value.key
    pd.testing.assert_index_equal(df2.columns_value.to_pandas(), raw.columns)
    pd.testing.assert_series_equal(df2.dtypes, raw.dtypes)

    series2 = series.loc["a2"]
    assert isinstance(series2, Tensor)
    assert series2.shape == ()
    assert series2.dtype == s.dtype

    series2 = series.loc[["a2", "a3"]]
    assert isinstance(series2, Series)
    assert series2.shape == (2,)
    assert series2.dtype == s.dtype
    assert series2.name == s.name

    with pytest.raises(IndexingError):
        _ = df.loc["a1", "z", ...]

    with pytest.raises(NotImplementedError):
        _ = df.loc[:, md.Series([True, False, True])]

    with pytest.raises(KeyError):
        _ = df.loc[:, ["non_exist"]]


def test_loc_use_iloc():
    raw = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]], columns=["x", "y", "z"])
    df = md.DataFrame(raw, chunk_size=2)

    assert isinstance(df.loc[:3].op, DataFrameIlocGetItem)
    assert isinstance(df.loc[1:3].op, DataFrameIlocGetItem)
    assert isinstance(df.loc[1].op, DataFrameIlocGetItem)
    # negative
    assert isinstance(df.loc[:-3].op, DataFrameLocGetItem)
    with pytest.raises(KeyError):
        _ = df.loc[-3]
    # index 1 not None
    assert isinstance(df.loc[:3, :"y"].op, DataFrameLocGetItem)
    # index 1 not slice
    assert isinstance(df.loc[:3, [True, False, True]].op, DataFrameLocGetItem)
    assert isinstance(df.loc[[True, False, True]].op, DataFrameLocGetItem)

    raw2 = raw.copy()
    raw2.index = pd.RangeIndex(1, 4)
    df2 = md.DataFrame(raw2, chunk_size=2)

    assert isinstance(df2.loc[:3].op, DataFrameLocGetItem)
    assert isinstance(df2.loc["a3":].op, DataFrameLocGetItem)

    raw2 = raw.copy()
    raw2.index = [f"a{i}" for i in range(3)]
    df2 = md.DataFrame(raw2, chunk_size=2)

    assert isinstance(df2.loc[:3].op, DataFrameLocGetItem)


def test_dataframe_getitem():
    data = pd.DataFrame(np.random.rand(10, 5), columns=["c1", "c2", "c3", "c4", "c5"])
    df = md.DataFrame(data, chunk_size=2)

    series = df["c3"]
    assert isinstance(series, Series)
    assert series.shape == (10,)
    assert series.name == "c3"
    assert series.dtype == data["c3"].dtype
    assert series.index_value == df.index_value

    assert isinstance(series, SERIES_TYPE)

    df1 = df[["c1", "c2", "c3"]]
    assert isinstance(df1, DataFrame)
    assert df1.shape == (10, 3)
    assert df1.index_value == df.index_value
    pd.testing.assert_index_equal(
        df1.columns_value.to_pandas(), data[["c1", "c2", "c3"]].columns
    )
    pd.testing.assert_series_equal(df1.dtypes, data[["c1", "c2", "c3"]].dtypes)


def test_dataframe_getitem_bool():
    data = pd.DataFrame(
        np.random.rand(10, 5),
        columns=["c1", "c2", "c3", "c4", "c5"],
        index=pd.RangeIndex(10, name="i"),
    )
    df = md.DataFrame(data, chunk_size=2)

    mask_data1 = data.c1 > 0.5
    mask_data2 = data.c1 < 0.5
    mask1 = md.Series(mask_data1, chunk_size=2)
    mask2 = md.Series(mask_data2, chunk_size=2)

    r1 = df[mask1]
    r2 = df[mask2]
    r3 = df[mask1]

    assert r1.index_value.key != df.index_value.key
    assert r1.index_value.key != mask1.index_value.key
    assert r1.columns_value.key == df.columns_value.key
    assert r1.columns_value is df.columns_value
    assert r1.index_value.name == "i"

    assert r1.index_value.key != r2.index_value.key
    assert r1.columns_value.key == r2.columns_value.key
    assert r1.columns_value is r2.columns_value

    assert r1.index_value.key == r3.index_value.key
    assert r1.columns_value.key == r3.columns_value.key
    assert r1.columns_value is r3.columns_value


def test_series_getitem():
    data = pd.Series(np.random.rand(10), name="a")
    series = md.Series(data, chunk_size=3)

    result1 = series[2]
    assert result1.shape == ()

    result2 = series[[4, 5, 1, 2, 3]]
    assert result2.shape == (5,)

    data = pd.Series(np.random.rand(10), index=["i" + str(i) for i in range(10)])
    series = md.Series(data, chunk_size=3)

    result1 = series["i2"]
    assert result1.shape == ()

    result2 = series[["i2", "i4"]]
    assert result2.shape == (2,)


def test_series_getitem_with_multi_index():
    arrays = [["m1", "m2", "m3"], ["n1", "n2", "n3"]]
    idx = pd.MultiIndex.from_arrays(arrays, names=("idx_1", "idx_2"))
    s = md.Series([1, 2, 3], index=idx)
    s1 = s[("m1", "n1")]
    assert s1.shape == ()


def test_setitem():
    data = pd.DataFrame(np.random.rand(10, 2), columns=["c1", "c2"])
    df = md.DataFrame(data, chunk_size=4)

    df["new"] = 1
    assert df.shape == (10, 3)
    pd.testing.assert_series_equal(df.inputs[0].dtypes, data.dtypes)


def test_reset_index():
    data = pd.DataFrame(
        [("bird", 389.0), ("bird", 24.0), ("mammal", 80.5), ("mammal", np.nan)],
        index=["falcon", "parrot", "lion", "monkey"],
        columns=("class", "max_speed"),
    )
    df = md.DataFrame(data, chunk_size=2).reset_index()
    r = data.reset_index()

    assert df.shape == (4, 3)
    pd.testing.assert_series_equal(df.dtypes, r.dtypes)
    pd.testing.assert_index_equal(df.columns_value.to_pandas(), r.columns)

    df = md.DataFrame(data, chunk_size=1).reset_index(drop=True)
    r = data.reset_index(drop=True)

    assert df.shape == (4, 2)
    pd.testing.assert_series_equal(df.dtypes, r.dtypes)

    # test Series
    series_data = pd.Series(
        [1, 2, 3, 4], name="foo", index=pd.Index(["a", "b", "c", "d"], name="idx")
    )
    s = md.Series(series_data, chunk_size=2).reset_index()
    r = series_data.reset_index()

    assert s.shape == (4, 2)
    pd.testing.assert_series_equal(s.dtypes, r.dtypes)

    with pytest.raises(TypeError):
        md.Series(series_data, chunk_size=2).reset_index(inplace=True)


def test_reindex():
    raw = pd.DataFrame(np.random.rand(4, 3))

    df = md.DataFrame(raw, chunk_size=2)

    with pytest.raises(TypeError):
        df.reindex(unknown_arg=1)

    with pytest.raises(ValueError):
        df.reindex([1, 2], fill_value=mt.tensor([1, 2]))
