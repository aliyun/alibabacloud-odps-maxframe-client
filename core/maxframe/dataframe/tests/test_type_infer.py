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

from maxframe.core import OutputType
from maxframe.dataframe.type_infer import (
    MockDataFrame,
    MockSeries,
    _wrap_mock,
    infer_dataframe_return_value,
)
from maxframe.dataframe.typing_ import dtype, infer_dtype
from maxframe.lib.dtypes_extension import ArrowDtype
from maxframe.lib.dtypes_extension.blob import ExternalBlobDtype, SolidBlob


def test_mock_and_wrap_mock():
    import maxframe.dataframe as md

    # Test MockDataFrame
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    mock_df = MockDataFrame(df)

    repr_str = repr(mock_df)
    assert "<MockDataFrame for type inference>" in repr_str
    assert "a" in repr_str and "b" in repr_str

    sliced = mock_df[["a"]]
    assert isinstance(sliced, MockDataFrame)
    assert mock_df.shape == (2, 2)
    assert list(mock_df.columns) == ["a", "b"]
    assert mock_df["a"].sum() == 3

    # Test MockSeries
    s = pd.Series([1, 2, 3], name="test")
    mock_s = MockSeries(s)

    repr_str = repr(mock_s)
    assert "<MockSeries for type inference>" in repr_str

    result = mock_s + 1
    assert isinstance(result, MockSeries)
    assert len(mock_s) == 3
    assert mock_s.sum() == 6

    # Test _wrap_mock wraps pandas objects
    wrapped_df = _wrap_mock(pd.DataFrame({"a": [1]}))
    wrapped_s = _wrap_mock(pd.Series([1, 2]))
    assert isinstance(wrapped_df, MockDataFrame)
    assert isinstance(wrapped_s, MockSeries)

    # Test _wrap_mock no double wrap
    mock_df2 = MockDataFrame({"a": [1]})
    assert _wrap_mock(mock_df2) is mock_df2

    # Test _wrap_mock passthrough non-pandas
    assert _wrap_mock(123) == 123
    assert _wrap_mock("string") == "string"

    # Test infer_dataframe_return_value wraps mock
    sample_df = md.DataFrame(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    _types = []

    def _func(df1):
        _types.append(type(df1).__name__)
        return df1.sum()

    infer_dataframe_return_value(sample_df.data, _func)
    assert "MockDataFrame" in _types

    # Test infer returns user specified type on exception
    def _error_func(df2):
        raise ValueError("Intentional error")

    result = infer_dataframe_return_value(
        sample_df.data, _error_func, output_type=OutputType.series
    )
    assert result.output_type == OutputType.series


def test_infer_dtype_basic():
    """Test infer_dtype with basic scalar types."""
    # Integer
    result = infer_dtype(42)
    assert result == dtype(pa.int64())

    # Float
    result = infer_dtype(3.14)
    assert result == dtype(pa.float64())

    # Boolean
    result = infer_dtype(True)
    assert result == dtype(pa.bool_())

    # String
    result = infer_dtype("hello")
    assert result == dtype(pa.string())

    # Bytes
    result = infer_dtype(b"bytes")
    assert result == dtype(pa.binary())

    # SolidBlob object
    blob = SolidBlob(b"test_data")
    result = infer_dtype(blob)
    assert isinstance(result, ExternalBlobDtype)

    # None should raise TypeError
    with pytest.raises(TypeError, match="Cannot infer dtype"):
        infer_dtype(None)

    # Unsupported custom object
    with pytest.raises(TypeError, match="Cannot infer dtype"):
        infer_dtype(type("CustomClass", (object,), {})())


def test_infer_dtype_collections():
    """Test infer_dtype with collection types and verify sub-types."""
    # List with integer elements
    result = infer_dtype([1, 2, 3])
    assert isinstance(result, ArrowDtype)
    assert pa.types.is_list(result.pyarrow_dtype)
    assert result.pyarrow_dtype.value_type == pa.int64()

    # List with string elements
    result = infer_dtype(["a", "b", "c"])
    assert isinstance(result, ArrowDtype)
    assert pa.types.is_list(result.pyarrow_dtype)
    assert result.pyarrow_dtype.value_type == pa.string()

    # Dict/struct type
    result = infer_dtype({"a": 1, "b": "text"})
    assert isinstance(result, ArrowDtype)
    assert pa.types.is_struct(result.pyarrow_dtype)
    # Verify struct has 2 fields
    struct_type = result.pyarrow_dtype
    assert len(struct_type) == 2

    # List of lists
    result = infer_dtype([[1, 2], [3, 4]])
    assert isinstance(result, ArrowDtype)
    assert pa.types.is_list(result.pyarrow_dtype)

    # Dict with list values
    result = infer_dtype({"a": [1, 2], "b": [3, 4]})
    assert isinstance(result, ArrowDtype)
    assert pa.types.is_struct(result.pyarrow_dtype)
