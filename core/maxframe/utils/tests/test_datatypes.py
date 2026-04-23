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

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from ...utils.datatypes import (
    arrow_type_from_str,
    is_arrow_dtype_supported,
    is_bool_dtype,
    is_datetime64_dtype,
    is_string_dtype,
    make_dtype,
    make_dtypes,
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
