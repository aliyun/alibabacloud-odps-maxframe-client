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

import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from maxframe.errors import OutputColumnMismatchError, OutputDtypeMismatchError
from maxframe.lib.dtypes_extension import ArrowBlobType, ArrowDtype, ExternalBlobDtype
from maxframe.utils.datatypes import (
    arrow_type_from_str,
    check_dtype_compatibility,
    is_arrow_dtype_supported,
    is_bool_dtype,
    is_datetime64_dtype,
    is_string_dtype,
    make_dtype,
    make_dtypes,
    validate_and_align_output,
    wrap_arrow_dtype,
)


def test_arrow_type_from_string():
    def _assert_arrow_type_convert(tp: pa.DataType) -> None:
        assert arrow_type_from_str(str(tp)) == tp

    _assert_arrow_type_convert(pa.int64())
    _assert_arrow_type_convert(pa.binary(20))
    _assert_arrow_type_convert(pa.decimal128(10, 2))
    _assert_arrow_type_convert(pa.list_(pa.int64()))
    _assert_arrow_type_convert(pa.map_(pa.string(), pa.int64()))
    _assert_arrow_type_convert(pa.date32())
    _assert_arrow_type_convert(pa.date64())
    _assert_arrow_type_convert(
        pa.struct([("key", pa.string()), ("value", pa.list_(pa.int64()))])
    )
    _assert_arrow_type_convert(
        pa.struct([("key", pa.string(), False), ("value", pa.list_(pa.int64()))])
    )
    _assert_arrow_type_convert(
        pa.struct([("key", pa.string()), ("value", pa.list_(pa.int64()), False)])
    )


@pytest.mark.parametrize(
    "val, expected, make_series",
    [
        (int, np.dtype(int), True),
        ([int], [np.dtype(int)], False),
        ([int], pd.Series([np.dtype(int)]), True),
        (np.dtype("float64"), np.dtype("float64"), True),
        ("category", "category", True),
        ("string", "string", True),
        (pd.Timestamp, np.dtype("datetime64[ns]"), True),
        (pd.Timedelta, np.dtype("timedelta64[ns]"), True),
        ({"col": float}, {"col": np.dtype(float)}, False),
        ({"col": float}, pd.Series({"col": np.dtype(float)}), True),
        (
            pd.Series([float], index=["col"]),
            pd.Series([np.dtype(float)], index=["col"]),
            True,
        ),
    ],
)
def test_make_dtypes(val, expected, make_series):
    result = make_dtypes(val, make_series=make_series)
    if isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "input_dtype, expected",
    [
        (wrap_arrow_dtype(pa.int64()), False),
        (wrap_arrow_dtype(pa.string()), True),
        (np.dtype("O"), True),
        (np.dtype("float32"), False),
        (pd.Int64Dtype(), False),
        (pd.StringDtype(), True),
    ],
)
def test_is_string_dtype(input_dtype, expected):
    assert is_string_dtype(input_dtype) is expected


@pytest.mark.parametrize(
    "input_dtype, expected",
    [
        (wrap_arrow_dtype(pa.int64()), False),
        (wrap_arrow_dtype(pa.bool_()), True),
        (np.dtype("bool"), True),
        (np.dtype("float32"), False),
        (pd.Int64Dtype(), False),
        (pd.BooleanDtype(), True),
    ],
)
def test_is_bool_dtype(input_dtype, expected):
    assert is_bool_dtype(input_dtype) is expected


@pytest.mark.parametrize(
    "input_dtype, expected",
    [
        (wrap_arrow_dtype(pa.int64()), False),
        (wrap_arrow_dtype(pa.timestamp("ms")), True),
        (np.dtype("bool"), False),
        (np.dtype("datetime64[ms]"), True),
        (pd.Int64Dtype(), False),
    ],
)
def test_is_datetime64_dtype(input_dtype, expected):
    assert is_datetime64_dtype(input_dtype) is expected


# New tests for previously untested functions
@pytest.mark.parametrize(
    "input_dtype, expected",
    [
        (int, np.dtype(int)),
        ("float64", np.dtype("float64")),
        (np.dtype("bool"), np.dtype("bool")),
        (pd.StringDtype(), pd.StringDtype()),
        (pd.BooleanDtype(), pd.BooleanDtype()),
        ("category", "category"),
        ("string", "string"),
        (pd.Timestamp, np.dtype("datetime64[ns]")),
        (pd.Timedelta, np.dtype("timedelta64[ns]")),
    ],
)
def test_make_dtype(input_dtype, expected):
    result = make_dtype(input_dtype)
    assert result == expected


def test_wrap_arrow_dtype():
    if not is_arrow_dtype_supported():
        pytest.skip("ArrowDtype is not supported")

    # Test wrapping various Arrow types
    wrapped_int = wrap_arrow_dtype(pa.int64())
    # wrap_arrow_dtype returns either an ArrowDtype or the raw Arrow type
    # The important thing is that it preserves the dtype information
    assert wrapped_int is not None

    wrapped_string = wrap_arrow_dtype(pa.string())
    assert isinstance(wrapped_string, pd.StringDtype)

    wrapped_timestamp = wrap_arrow_dtype(pa.timestamp("ms"))
    assert wrapped_timestamp is not None

    # The exact behavior depends on whether ArrowDtype is available
    if is_arrow_dtype_supported():
        # When ArrowDtype is supported, wrap_arrow_dtype may return an ArrowDtype
        # or a special wrapper that has a pyarrow_dtype attribute
        if hasattr(wrapped_int, "pyarrow_dtype"):
            assert wrapped_int.pyarrow_dtype == pa.int64()
        # Otherwise it might return the raw Arrow type
    else:
        # When ArrowDtype is not supported, should return the raw Arrow type
        assert wrapped_int == pa.int64()


def test_validate_and_align_output():
    """Test validate_and_align_output with various check_output_dtypes modes."""
    # Test 1: DataFrame with check_output_dtypes=False skips validation
    df = pd.DataFrame({"a": [1, 2, 3]})  # int64
    expected = pd.Series({"a": np.dtype("float64")})
    result = validate_and_align_output(df, expected, check_output_dtypes=False)
    assert result["a"].dtype == np.dtype("int64")

    # Test 2: DataFrame with check_output_dtypes='ignore' skips validation
    result = validate_and_align_output(df, expected, check_output_dtypes="ignore")
    assert result["a"].dtype == np.dtype("int64")

    # Test 3: DataFrame column check skipped when check_output_dtypes=False
    df_cols = pd.DataFrame({"b": [4, 5, 6], "a": [1, 2, 3]})
    expected_cols = pd.Series({"a": np.dtype("int64")})
    result = validate_and_align_output(
        df_cols, expected_cols, check_output_dtypes=False
    )
    assert list(result.columns) == ["b", "a"]

    # Test 4: DataFrame column check skipped when check_output_dtypes='ignore'
    result = validate_and_align_output(
        df_cols, expected_cols, check_output_dtypes="ignore"
    )
    assert list(result.columns) == ["b", "a"]

    # Test 5: Series with check_output_dtypes=False skips validation
    s = pd.Series([1, 2, 3])  # int64
    expected_s = np.dtype("float64")
    result = validate_and_align_output(s, expected_s, check_output_dtypes=False)
    assert result.dtype == np.dtype("int64")

    # Test 6: Series with check_output_dtypes='ignore' skips validation
    result = validate_and_align_output(s, expected_s, check_output_dtypes="ignore")
    assert result.dtype == np.dtype("int64")

    # Test 7: DataFrame with check_output_dtypes='warns' performs validation and casting
    result = validate_and_align_output(df, expected, check_output_dtypes="warns")
    assert result["a"].dtype == np.dtype("float64")

    # Test 8: DataFrame with check_output_dtypes='raises' raises on column mismatch
    df_missing = pd.DataFrame({"a": [1, 2, 3]})
    expected_missing = pd.Series({"a": np.dtype("int64"), "b": np.dtype("int64")})
    with pytest.raises(OutputColumnMismatchError):
        validate_and_align_output(
            df_missing, expected_missing, check_output_dtypes="raises"
        )

    # Test 9: Series with check_output_dtypes='warns' performs validation and casting
    result = validate_and_align_output(s, expected_s, check_output_dtypes="warns")
    assert result.dtype == np.dtype("float64")

    # Test 10: None expected_dtypes skips validation
    df_test = pd.DataFrame({"a": [1, 2, 3]})
    result = validate_and_align_output(df_test, None, check_output_dtypes="raises")
    pd.testing.assert_frame_equal(result, df_test)

    # Test 11: check_dtype_compatibility rejects float to int (causes truncation)
    with pytest.raises(OutputDtypeMismatchError) as exc_info:
        check_dtype_compatibility(np.dtype("float64"), np.dtype("int64"), "test_col")
    assert "float dtype" in str(exc_info.value).lower()
    assert "integer dtype" in str(exc_info.value).lower()
    assert "truncation" in str(exc_info.value).lower()

    # Test 12: check_dtype_compatibility allows int to float conversion
    result = check_dtype_compatibility(
        np.dtype("int64"), np.dtype("float64"), "test_col"
    )
    assert result is None

    # Test 13: validate_and_align_output warns on float to int with warns mode
    s_incompat = pd.Series([1.5, 2.5, 3.5], dtype=np.dtype("float64"))
    expected_incompat = np.dtype("int64")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = validate_and_align_output(
            s_incompat, expected_incompat, check_output_dtypes="warns"
        )
        assert len(w) > 0
        assert "truncation" in str(w[0].message).lower()

    # Test 14: validate_and_align_output rejects float to int with raises mode
    with pytest.raises(OutputDtypeMismatchError) as exc_info:
        validate_and_align_output(
            s_incompat, expected_incompat, check_output_dtypes="raises"
        )
    assert "truncation" in str(exc_info.value).lower()

    # Test 15: validate_and_align_output casts compatible types
    s_compat = pd.Series([1, 2, 3], dtype=np.dtype("int64"))
    expected_compat = np.dtype("float64")
    result = validate_and_align_output(
        s_compat, expected_compat, check_output_dtypes="warns"
    )
    assert result.dtype == np.dtype("float64")

    # Test 16: check_dtype_compatibility with blob types - matching types
    blob_dtype = ExternalBlobDtype()
    result = check_dtype_compatibility(blob_dtype, blob_dtype, "blob_col")
    assert result is None  # Same blob type should pass

    # Test 17: check_dtype_compatibility with Arrow blob type
    arrow_blob_dtype = ArrowDtype(ArrowBlobType())
    result = check_dtype_compatibility(arrow_blob_dtype, blob_dtype, "arrow_blob_col")
    assert result is None  # Arrow blob to blob should pass

    # Test 18: check_dtype_compatibility with blob to non-blob should fail
    with pytest.raises(OutputDtypeMismatchError):
        check_dtype_compatibility(blob_dtype, np.dtype("int64"), "blob_col")

    # Test 19: check_dtype_compatibility with non-blob to blob should fail
    with pytest.raises(OutputDtypeMismatchError):
        check_dtype_compatibility(np.dtype("int64"), blob_dtype, "int_col")
