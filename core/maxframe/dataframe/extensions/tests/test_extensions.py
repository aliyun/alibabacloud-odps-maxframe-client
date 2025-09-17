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
from ....tests.utils import assert_mf_index_dtype
from ... import DataFrame
from ...core import DATAFRAME_TYPE, SERIES_TYPE, IndexValue
from ..reshuffle import DataFrameReshuffle


def test_reshuffle():
    df = pd.DataFrame(
        {"a": [3, 4, 5, 3, 5, 4, 1, 2, 3], "b": [1, 3, 4, 5, 6, 5, 4, 4, 4]}
    )
    mdf = md.DataFrame(df, chunk_size=2)

    with pytest.raises(ValueError):
        mdf.mf.reshuffle(sort_by="b")

    r = mdf.mf.reshuffle()
    assert isinstance(r.op, DataFrameReshuffle)
    assert_mf_index_dtype(r.index_value.value, np.int64)

    r = mdf.mf.reshuffle(ignore_index=True)
    assert isinstance(r.op, DataFrameReshuffle)
    assert isinstance(r.index_value.value, IndexValue.RangeIndex)


@pytest.fixture
def df1():
    return DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})


@pytest.fixture
def df2():
    return DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])


@pytest.fixture
def df3():
    return DataFrame(
        [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        columns=["a", "b", "c"],
        index=pd.MultiIndex.from_arrays([[1, 2, 3], [1, 2, 3]], names=["A", "B"]),
    )


@pytest.fixture
def df4():
    return DataFrame(
        {
            "name1": ["a", "b", "c", "d"],
            "name2": ["a", "b", "c", "d"],
            "num": [1, 2, 3, 4],
            "kv": [
                "k1=1.1,k2=3.1,k3=1.0",
                "k1=7.1,k4=8.2",
                "k5=1.2,k7=1.5",
                "k3=1.1,k9=1",
            ],
            "vk": ["v1=1.1,v2=1.2", "v3=1.1,v4=1.2", "v5=1.1,v6=1.2", "v7=1.1,v8=1.2"],
        }
    )


@pytest.fixture
def df5():
    return DataFrame(
        {
            "name1": ["name1", "name2", "name3", "name4", "name5"],
            "name2": ["name1", "name2", "name3", "name4", "name5"],
            "k1": [1.0, None, 7.1, None, None],
            "k2": [3.0, 3.0, None, 1.2, 1.0],
            "k3": [None, 5.1, None, 1.5, None],
            "k5": [10.0, None, None, None, None],
            "k7": [None, None, 8.2, None, None],
            "k9": [None, None, None, None, 1.1],
        }
    )


def test_flatmap(df1, df2, df3):
    def f(x, keys):
        if x["a"] in keys:
            yield [1, 0]
            yield [0, 1]

    apply_df = df1[["a"]].mf.flatmap(
        f,
        dtypes={"a": "int64", "b": "int64"},
    )
    assert apply_df.shape == (np.nan, 2)
    assert df1.index_value.key != apply_df.index_value.key
    assert isinstance(df1.index_value.to_pandas(), pd.RangeIndex)
    assert not isinstance(apply_df.index_value.to_pandas(), pd.RangeIndex)
    apply_df = df2[["a"]].mf.flatmap(
        f,
        dtypes=pd.Series(["int64", "int64"]),
    )
    assert apply_df.shape == (np.nan, 2)
    assert df2.index_value.key != apply_df.index_value.key
    with pytest.raises(TypeError):
        apply_s = df3["a"].mf.flatmap(
            f,
        )
    apply_s = df3["a"].mf.flatmap(
        f,
        dtype="int64",
    )
    assert apply_s.shape == (np.nan,)
    assert df3.index_value.key != apply_s.index_value.key
    assert df3.key != apply_s.index_value.key
    apply_s = df3["a"].mf.flatmap(
        f,
        output_type="dataframe",
        dtypes=["int64", "int64"],
    )
    assert apply_s.shape == (np.nan, 2)
    assert df3.index_value.key != apply_s.index_value.key
    assert df3.key != apply_s.index_value.key


def test_flatjson():
    s1 = md.Series(["{{'a': 1, 'b': false}}"], index=[1])
    df1 = s1.mf.flatjson(
        ["$.a", "$.b"], dtypes=pd.Series(["int32", "bool"], index=["a", "b"])
    )
    assert df1.shape == (1, 2)
    assert df1.index_value.key == s1.index_value.key
    assert isinstance(df1, DATAFRAME_TYPE)
    assert list(df1.dtypes) == [np.dtype("int32"), np.dtype("bool")]
    assert list(df1.dtypes.index) == ["a", "b"]

    df2 = s1.mf.flatjson(["$.a"], dtypes=pd.Series(["int32"], index=["a"]))
    assert df2.shape == (1, 1)
    assert df2.index_value.key == s1.index_value.key
    assert isinstance(df2, DATAFRAME_TYPE)
    assert list(df2.dtypes) == [np.dtype("int32")]
    assert list(df2.dtypes.index) == ["a"]

    s2 = s1.mf.flatjson("$.a", dtype="int32", name="a")
    assert s2.shape == (1,)
    assert s2.index_value.key == s1.index_value.key
    assert isinstance(s2, SERIES_TYPE)
    assert s2.dtype == np.dtype("int32")
    assert s2.name == "a"

    with pytest.raises(ValueError):
        s1.mf.flatjson([], dtypes=pd.Series(["int32", "bool"], index=["a", "b"]))
    with pytest.raises(ValueError):
        s1.mf.flatjson(["$.a"], dtypes=pd.Series(["int32", "bool"], index=["a", "b"]))
    with pytest.raises(ValueError):
        s1.mf.flatjson(["$.a"], dtypes=pd.Series(["int32", "bool"], index=["a", "b"]))
    with pytest.raises(ValueError):
        s1.mf.flatjson(["$.a", "$.b"], dtypes=pd.Series(["bool"], index=["b"]))
    with pytest.raises(ValueError):
        s1.mf.flatjson(
            ["$.a"],
            dtype="int32",
            dtypes=pd.Series(["int32"], index=["a"]),
        )
    with pytest.raises(ValueError):
        s1.mf.flatjson(["$.a"])


def test_extract_kv(df4):
    extract_kv_df = df4.mf.extract_kv(
        columns=["kv", "vk"], kv_delim=",", item_delim="="
    )
    assert extract_kv_df.shape == (4, np.nan)
    assert extract_kv_df.index_value.key == df4.index_value.key
    with pytest.raises(ValueError):
        df4.mf.extract_kv(columns=["name"])
    with pytest.raises(ValueError):
        df4.mf.extract_kv(columns=["num"])


def test_collect_kv(df5):
    collect_kv_df = df5.mf.collect_kv(columns=["k1", "k2", "k3", "k5", "k7", "k9"])
    assert collect_kv_df.shape == (5, 3)
    assert collect_kv_df.index_value.key == df5.index_value.key
    with pytest.raises(ValueError):
        df5.mf.collect_kv(columns=["num"])
