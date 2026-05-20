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

import pandas as pd
import pyarrow as pa
import pytest

from maxframe.dataframe.core import DATAFRAME_TYPE, INDEX_TYPE, SERIES_TYPE
from maxframe.dataframe.initializer import DataFrame, Index, Series, read_pandas
from maxframe.lib.dtypes_extension import ArrowDtype
from maxframe.tests.utils import require_arrow_dtype


def test_read_pandas():
    df_data = pd.DataFrame([["a", 1], ["b", 2]], columns=["a", "b"])
    assert isinstance(read_pandas(df_data), DATAFRAME_TYPE)

    s_data = pd.Series(["a", "b"])
    assert isinstance(read_pandas(s_data), SERIES_TYPE)

    idx_data = pd.Index(["a", "b"])
    assert isinstance(read_pandas(idx_data), INDEX_TYPE)


def test_init_dataframe_from_maxframe_series():
    s = Series([1, 2, 3, 4], index=[1, 2, 3, 4])

    df = DataFrame(s, index=s.index, columns=["col1"])

    assert isinstance(df, DATAFRAME_TYPE)
    assert df.dtypes.index == ["col1"]

    with pytest.raises(ValueError):
        DataFrame(s, index=s.index, columns=[])

    with pytest.raises(ValueError):
        DataFrame(s, index=s.index, columns="col1")

    with pytest.raises(ValueError):
        DataFrame(s, index=s.index, columns="col2")


def test_init_dataframe_from_maxframe_dataframe():
    df1 = DataFrame({"A": [1, 2, 3, 4], "B": [1, 2, 3, 4]}, index=[1, 2, 3, 4])

    df2 = DataFrame(df1, index=df1.index, columns=["col1", "col2"])

    assert isinstance(df2, DATAFRAME_TYPE)
    assert list(df2.dtypes.index) == ["col1", "col2"]

    with pytest.raises(ValueError):
        DataFrame(df1, index=df1.index, columns=["col1", "col2", "col3"])


@require_arrow_dtype
def test_dataframe_with_pyarrow_dtype_backend():
    """Test DataFrame initialization with dtype_backend='pyarrow'"""
    # Test with dict data
    df = DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}, dtype_backend="pyarrow")
    assert isinstance(df, DATAFRAME_TYPE)
    result_dtypes = df.dtypes
    assert isinstance(result_dtypes["a"], ArrowDtype)
    assert isinstance(result_dtypes["b"], ArrowDtype)

    # Test with list data
    df2 = DataFrame(
        [[1, "a"], [2, "b"]], columns=["col1", "col2"], dtype_backend="pyarrow"
    )
    assert isinstance(df2, DATAFRAME_TYPE)
    result_dtypes2 = df2.dtypes
    assert isinstance(result_dtypes2["col1"], ArrowDtype)
    assert isinstance(result_dtypes2["col2"], ArrowDtype)

    # Test with list objects
    df3 = DataFrame(
        {"list_col": [[1, 2], [3, 4], [5, 6]], "str_col": ["a", "b", "c"]},
        dtype_backend="pyarrow",
    )
    assert isinstance(df3, DATAFRAME_TYPE)
    result_dtypes3 = df3.dtypes
    assert isinstance(result_dtypes3["list_col"], ArrowDtype)
    assert isinstance(result_dtypes3["str_col"], ArrowDtype)
    # Verify list column has list type
    assert isinstance(result_dtypes3["list_col"].pyarrow_dtype, pa.ListType)


@require_arrow_dtype
def test_series_with_pyarrow_dtype_backend():
    """Test Series initialization with dtype_backend='pyarrow'"""
    # Test with list data
    s = Series([1, 2, 3], dtype_backend="pyarrow")
    assert isinstance(s, SERIES_TYPE)
    assert isinstance(s.dtype, ArrowDtype)

    # Test with string data
    s2 = Series(["a", "b", "c"], dtype_backend="pyarrow", name="test_series")
    assert isinstance(s2, SERIES_TYPE)
    assert isinstance(s2.dtype, ArrowDtype)
    assert s2.name == "test_series"

    # Test with list objects
    s3 = Series([[1, 2], [3, 4], [5, 6]], dtype_backend="pyarrow")
    assert isinstance(s3, SERIES_TYPE)
    assert isinstance(s3.dtype, ArrowDtype)
    # Verify it has list type
    assert isinstance(s3.dtype.pyarrow_dtype, pa.ListType)


@require_arrow_dtype
def test_index_with_pyarrow_dtype_backend():
    """Test Index initialization with dtype_backend='pyarrow'"""
    # Test with list data
    idx = Index([1, 2, 3], dtype_backend="pyarrow")
    assert isinstance(idx, INDEX_TYPE)
    assert isinstance(idx.dtype, ArrowDtype)

    # Test with string data
    idx2 = Index(["a", "b", "c"], dtype_backend="pyarrow", name="test_index")
    assert isinstance(idx2, INDEX_TYPE)
    assert isinstance(idx2.dtype, ArrowDtype)
    assert idx2.name == "test_index"

    # Test with MultiIndex
    pd_midx = pd.MultiIndex.from_tuples(
        [(1, "a"), (2, "b"), (3, "c")], names=["num", "char"]
    )
    idx3 = Index(pd_midx, dtype_backend="pyarrow")
    assert isinstance(idx3, INDEX_TYPE)
    # For MultiIndex, check that index_value.value.dtypes is a Series and each level has ArrowDtype
    dtypes = idx3.index_value.value.dtypes
    assert isinstance(dtypes, pd.Series)
    assert len(dtypes) == 2
    assert isinstance(dtypes.iloc[0], ArrowDtype)
    assert isinstance(dtypes.iloc[1], ArrowDtype)
