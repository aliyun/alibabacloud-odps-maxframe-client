# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

from ...core import IndexValue
from ...datasource.dataframe import from_pandas
from .. import DataFrameMerge, concat


def test_merge():
    df1 = pd.DataFrame(
        np.arange(20).reshape((4, 5)) + 1, columns=["a", "b", "c", "d", "e"]
    )
    df2 = pd.DataFrame(np.arange(20).reshape((5, 4)) + 1, columns=["a", "b", "x", "y"])

    mdf1 = from_pandas(df1, chunk_size=2)
    mdf2 = from_pandas(df2, chunk_size=3)

    parameters = [
        {},
        {"how": "left", "right_on": "x", "left_index": True},
        {"how": "right", "left_on": "a", "right_index": True},
        {"how": "left", "left_on": "a", "right_on": "x"},
        {"how": "right", "left_on": "a", "right_index": True},
        {"how": "right", "on": "a"},
        {"how": "inner", "on": ["a", "b"]},
    ]

    for kw in parameters:
        df = mdf1.merge(mdf2, **kw)

        assert isinstance(df.op, DataFrameMerge)
        assert df.op.how == kw.get("how", "inner")
        pd.testing.assert_index_equal(
            df.columns_value.to_pandas(), df.columns_value.to_pandas()
        )


def test_merge_invalid_parameters():
    pdf1 = pd.DataFrame(
        np.arange(20).reshape((4, 5)) + 1, columns=["a", "b", "c", "d", "e"]
    )
    pdf2 = pd.DataFrame(np.arange(20).reshape((5, 4)) + 1, columns=["a", "b", "x", "y"])

    df1 = from_pandas(pdf1, chunk_size=2)
    df2 = from_pandas(pdf2, chunk_size=3)

    with pytest.raises(ValueError):
        df1.merge(df2, bloom_filter="wrong")

    with pytest.raises(TypeError):
        df1.merge(df2, bloom_filter_options="wrong")

    with pytest.raises(ValueError):
        df1.merge(df2, bloom_filter_options={"wrong": 1})

    with pytest.raises(ValueError):
        df1.merge(df2, bloom_filter_options={"filter": "wrong"})


def test_join():
    df1 = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]], index=["a1", "a2", "a3"])
    df2 = pd.DataFrame([[1, 2, 3], [1, 5, 6], [7, 8, 9]], index=["a1", "b2", "b3"]) + 1
    df2 = pd.concat([df2, df2 + 1])

    mdf1 = from_pandas(df1, chunk_size=2)
    mdf2 = from_pandas(df2, chunk_size=2)

    parameters = [
        {"lsuffix": "l_", "rsuffix": "r_"},
        {"lsuffix": "l_", "rsuffix": "r_", "how": "left"},
        {"lsuffix": "l_", "rsuffix": "r_", "how": "right"},
        {"lsuffix": "l_", "rsuffix": "r_", "how": "inner"},
        {"lsuffix": "l_", "rsuffix": "r_", "how": "left"},
    ]

    for kw in parameters:
        df = mdf1.join(mdf2, auto_merge="none", bloom_filter=False, **kw)

        assert isinstance(df.op, DataFrameMerge)
        assert df.op.how == kw.get("how", "left")
        pd.testing.assert_index_equal(
            df.columns_value.to_pandas(), df.columns_value.to_pandas()
        )


def test_join_on():
    df1 = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]], columns=["a1", "a2", "a3"])
    df2 = (
        pd.DataFrame([[1, 2, 3], [1, 5, 6], [7, 8, 9]], columns=["a1", "b2", "b3"]) + 1
    )
    df2 = pd.concat([df2, df2 + 1])

    mdf1 = from_pandas(df1, chunk_size=2)
    mdf2 = from_pandas(df2, chunk_size=2)

    parameters = [
        {"lsuffix": "l_", "rsuffix": "r_"},
        {"lsuffix": "l_", "rsuffix": "r_", "how": "left", "on": "a1"},
        {"lsuffix": "l_", "rsuffix": "r_", "how": "right", "on": "a2"},
        {"lsuffix": "l_", "rsuffix": "r_", "how": "inner", "on": "a2"},
        {"lsuffix": "l_", "rsuffix": "r_", "how": "outer", "on": "a2"},
    ]

    for kw in parameters:
        df = mdf1.join(mdf2, auto_merge="none", bloom_filter=False, **kw)

        assert isinstance(df.op, DataFrameMerge)
        assert df.op.how == kw.get("how", "left")
        pd.testing.assert_index_equal(
            df.columns_value.to_pandas(), df.columns_value.to_pandas()
        )


def test_append():
    df1 = pd.DataFrame(np.random.rand(10, 4), columns=list("ABCD"))
    df2 = pd.DataFrame(np.random.rand(10, 4), columns=list("ABCD"))

    mdf1 = from_pandas(df1, chunk_size=3)
    mdf2 = from_pandas(df2, chunk_size=3)
    adf = mdf1.append(mdf2)

    assert adf.shape == (20, 4)
    assert isinstance(adf.index_value.value, IndexValue.Int64Index)

    mdf1 = from_pandas(df1, chunk_size=3)
    mdf2 = from_pandas(df2, chunk_size=3)
    adf = mdf1.append(mdf2, ignore_index=True)

    assert adf.shape == (20, 4)
    assert isinstance(adf.index_value.value, IndexValue.RangeIndex)
    pd.testing.assert_index_equal(adf.index_value.to_pandas(), pd.RangeIndex(20))


def test_concat():
    df1 = pd.DataFrame(np.random.rand(10, 4), columns=list("ABCD"))
    df2 = pd.DataFrame(np.random.rand(10, 4), columns=list("ABCD"))

    mdf1 = from_pandas(df1, chunk_size=4)
    mdf2 = from_pandas(df2, chunk_size=4)
    r = concat([mdf1, mdf2], axis="index")

    assert r.shape == (20, 4)
    pd.testing.assert_series_equal(r.dtypes, df1.dtypes)

    df3 = pd.DataFrame(
        np.random.rand(10, 4), columns=list("ABCD"), index=pd.RangeIndex(10, 20)
    )

    mdf3 = from_pandas(df3, chunk_size=4)
    r = concat([mdf1, mdf3], axis="index")

    assert r.shape == (20, 4)
    pd.testing.assert_series_equal(r.dtypes, df1.dtypes)
    pd.testing.assert_index_equal(r.index_value.to_pandas(), pd.RangeIndex(20))

    df4 = pd.DataFrame(
        np.random.rand(10, 4),
        columns=list("ABCD"),
        index=np.random.permutation(np.arange(10)),
    )

    mdf4 = from_pandas(df4, chunk_size=4)
    r = concat([mdf1, mdf4], axis="index")

    assert r.shape == (20, 4)
    pd.testing.assert_series_equal(r.dtypes, df1.dtypes)
    pd.testing.assert_index_equal(
        r.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )

    r = concat([mdf4, mdf1], axis="index")

    assert r.shape == (20, 4)
    pd.testing.assert_series_equal(r.dtypes, df1.dtypes)
    pd.testing.assert_index_equal(
        r.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )

    r = concat([mdf4, mdf4], axis="index")

    assert r.shape == (20, 4)
    pd.testing.assert_series_equal(r.dtypes, df1.dtypes)
    pd.testing.assert_index_equal(
        r.index_value.to_pandas(), pd.Index([], dtype=np.int64)
    )

    mdf1 = from_pandas(df1, chunk_size=3)
    mdf2 = from_pandas(df2, chunk_size=4)
    r = concat([mdf1, mdf2], axis="columns")

    assert r.shape == (10, 8)
    expected_dtypes = pd.concat([df1, df2], axis="columns").dtypes
    pd.testing.assert_series_equal(r.dtypes, expected_dtypes)

    df1 = pd.DataFrame(np.random.rand(10, 4), columns=list("ABCD"))
    df2 = pd.DataFrame(np.random.rand(10, 3), columns=list("ABC"))
    mdf1 = from_pandas(df1, chunk_size=3)
    mdf2 = from_pandas(df2, chunk_size=3)
    r = concat([mdf1, mdf2], join="inner")
    assert r.shape == (20, 3)
