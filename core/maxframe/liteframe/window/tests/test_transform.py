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

from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from maxframe.config import option_context, options
from maxframe.liteframe.datatypes import infer_decimal_agg_dtype
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.window import (
    LiteFrameWindowTransform,
    _LegacyLiteFrameWindowTransform_2_9,
)
from maxframe.serialization import clear_type_cache, deserialize, serialize
from maxframe.utils import is_arrow_dtype_supported, wrap_arrow_dtype

# ---------------------------------------------------------------------------
# shift / diff / pct_change -- dtype inference and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_arrow_type, fill_value, expected",
    [
        (None, None, pa.int64()),
        (None, 0, pa.int64()),
        (None, 0.5, pa.float64()),
        (pa.float64(), 0.5, pa.float64()),
        (pa.int8(), 0.5, pa.float64()),
        (pa.int8(), 1, pa.int8()),
    ],
    ids=["int-none", "int-int", "int-float", "float-float", "int8-float", "int8-int"],
)
def test_shift_fill_value_dtype(input_arrow_type, fill_value, expected):
    data = (
        [1, 2, 3]
        if input_arrow_type is None
        else pa.array([1, 2, 3], type=input_arrow_type)
    )
    lf = LF({"a": data})
    result = lf.shift(1, fill_value=fill_value)
    assert result.dtypes["a"] == wrap_arrow_dtype(expected)


@pytest.mark.skipif(
    not is_arrow_dtype_supported(), reason="ArrowDtype is not supported"
)
@pytest.mark.parametrize(
    "fill_value, expected",
    [
        (None, pa.decimal128(10, 2)),
        (0, pa.decimal128(22, 2)),
        (Decimal("0"), pa.decimal128(11, 2)),
        (Decimal("9.99"), pa.decimal128(11, 2)),
        (0.5, pa.float64()),
    ],
    ids=["none", "int", "decimal-zero", "decimal-value", "float"],
)
def test_shift_decimal_fill_value_dtype(fill_value, expected):
    lf = LF(
        {
            "a": pd.Series(
                [Decimal("1.00"), Decimal("2.00")],
                dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
            )
        }
    )
    result = lf.shift(1, fill_value=fill_value)
    assert result.dtypes["a"] == wrap_arrow_dtype(expected)


@pytest.mark.parametrize(
    "fill_value, expected",
    [
        (None, pa.bool_()),
        (0, pa.bool_()),
        (1, pa.bool_()),
        (True, pa.bool_()),
        (False, pa.bool_()),
        (0.5, pa.bool_()),
        (1.0, pa.bool_()),
    ],
    ids=[
        "none",
        "int-zero",
        "int-one",
        "bool-true",
        "bool-false",
        "float",
        "float-one",
    ],
)
def test_shift_boolean_fill_value_dtype(fill_value, expected):
    lf = LF({"a": pa.array([True, False], type=pa.bool_())})
    result = lf.shift(1, fill_value=fill_value)
    assert result.dtypes["a"] == wrap_arrow_dtype(expected)


def test_shift_boolean_string_fill_rejected():
    lf = LF({"a": pa.array([True, False], type=pa.bool_())})
    with pytest.raises(TypeError, match="cannot be coerced to boolean"):
        lf.shift(1, fill_value="x")


@pytest.mark.parametrize(
    "input_arrow_type, expected",
    [
        (None, pa.int64()),
        (pa.int8(), pa.int8()),
        (pa.uint8(), pa.int16()),
        (pa.uint16(), pa.int32()),
        (pa.uint32(), pa.int64()),
        (pa.uint64(), pa.int64()),
        (pa.float32(), pa.float32()),
    ],
    ids=["int64", "int8", "uint8", "uint16", "uint32", "uint64", "float32"],
)
def test_diff_dtype(input_arrow_type, expected):
    data = (
        [1, 2, 3]
        if input_arrow_type is None
        else pa.array([1, 2, 3], type=input_arrow_type)
    )
    lf = LF({"a": data})
    result = lf.diff(1)
    assert result.dtypes["a"] == wrap_arrow_dtype(expected)


def test_pct_change_output_dtypes_float64():
    lf = LF({"a": [1, 2, 3]})
    result = lf.pct_change(1)
    assert result.dtypes["a"] == wrap_arrow_dtype(pa.float64())


@pytest.mark.parametrize(
    "kind,call",
    [
        ("shift", lambda lf: lf.shift(1, columns=["a"])),
        ("diff", lambda lf: lf.diff(1, columns=["a"])),
        ("pct_change", lambda lf: lf.pct_change(1, columns=["a"])),
    ],
    ids=["shift", "diff", "pct_change"],
)
def test_partial_columns_skips_non_numeric(kind, call):
    """columns= restricts numeric checks to selected columns only."""
    lf = LF({"a": [1, 2, 3], "name": ["x", "y", "z"]})
    assert isinstance(call(lf).data.op, LiteFrameWindowTransform)


@pytest.mark.parametrize(
    "kind,call",
    [
        ("diff", lambda lf: lf.diff(1)),
        ("pct_change", lambda lf: lf.pct_change(1)),
    ],
    ids=["diff", "pct_change"],
)
def test_non_numeric_raises_type_error(kind, call):
    lf = LF({"a": [1, 2, 3], "name": ["x", "y", "z"]})
    with pytest.raises(TypeError, match="non-numeric"):
        call(lf)


@pytest.mark.parametrize(
    "kind,call",
    [
        ("shift", lambda lf: lf.shift(1, columns="a")),
        ("rolling", lambda lf: lf.rolling(2, columns="a").sum()),
    ],
    ids=["shift", "rolling"],
)
def test_str_columns_param(kind, call):
    """A single string for columns is accepted and normalized to a list."""
    lf = LF({"a": [1, 2, 3], "b": [10, 20, 30]})
    assert call(lf).data.op.columns == ["a"]


def test_groupby_transform_rejects_group_key_in_columns():
    lf = LF({"key": ["a", "b", "a"], "val": [1, 2, 3]})
    with pytest.raises(ValueError, match="Group key column"):
        lf.groupby("key").shift(1, columns=["key"])
    with pytest.raises(ValueError, match="Group key column"):
        lf.groupby("key").diff(1, columns=["key", "val"])
    with pytest.raises(ValueError, match="Group key column"):
        lf.groupby("key").rolling(2, columns=["key"]).sum()


# ---------------------------------------------------------------------------
# rolling -- dtype inference and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_arrow_type, expected",
    [
        (pa.int8(), pa.int64()),
        (pa.int16(), pa.int64()),
        (pa.uint8(), pa.int64()),
        (pa.uint16(), pa.int64()),
        (pa.int32(), pa.int32()),
        (pa.uint32(), pa.uint32()),
        (pa.int64(), pa.int64()),
        (pa.uint64(), pa.uint64()),
        (pa.float32(), pa.float32()),
        (pa.float64(), pa.float64()),
    ],
    ids=[
        "int8",
        "int16",
        "uint8",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
    ],
)
def test_rolling_sum_dtype(input_arrow_type, expected):
    lf = LF({"a": pa.array([1, 2, 3, 4], type=input_arrow_type)})
    result = lf.rolling(2).sum()
    assert result.dtypes["a"] == wrap_arrow_dtype(expected)


@pytest.mark.parametrize(
    "input_arrow_type", [pa.int8(), pa.int64()], ids=["int8", "int64"]
)
def test_rolling_min_max_preserve_dtype(input_arrow_type):
    lf = LF({"a": pa.array([1, 2, 3, 4], type=input_arrow_type)})
    assert lf.rolling(2).min().dtypes["a"] == wrap_arrow_dtype(input_arrow_type)
    assert lf.rolling(2).max().dtypes["a"] == wrap_arrow_dtype(input_arrow_type)


@pytest.mark.parametrize("func", ["mean", "std", "var", "median"])
def test_rolling_float_funcs_produce_float64(func):
    """mean/std/var/median always produce Float64, even for Float32 input."""
    lf = LF({"a": pa.array([1.0, 2.0, 3.0], type=pa.float32())})
    result = getattr(lf.rolling(2), func)()
    assert result.dtypes["a"] == wrap_arrow_dtype(pa.float64())


def test_rolling_count_produces_int64():
    lf = LF({"a": [1, 2, 3, 4]})
    result = lf.rolling(2).count()
    assert result.dtypes["a"] == wrap_arrow_dtype(pa.int64())


def test_rolling_count_allows_non_numeric():
    lf = LF({"a": [1, 2, 3], "name": ["x", "y", "z"]})
    result = lf.rolling(2).count()
    assert result.data.op._get_transform_param("agg_func") == "count"


def test_rolling_sum_non_numeric_raises_type_error():
    lf = LF({"a": [1, 2, 3], "name": ["x", "y", "z"]})
    with pytest.raises(TypeError, match="non-numeric"):
        lf.rolling(2).sum()


@pytest.mark.skipif(
    not is_arrow_dtype_supported(), reason="ArrowDtype is not supported"
)
@pytest.mark.parametrize(
    "func, expected",
    [
        ("sum", pa.decimal128(20, 2)),
        ("min", pa.decimal128(10, 2)),
        ("max", pa.decimal128(10, 2)),
        ("mean", pa.float64()),
        ("count", pa.int64()),
    ],
    ids=["sum", "min", "max", "mean", "count"],
)
def test_rolling_decimal_dtype(func, expected):
    lf = LF(
        {
            "a": pd.Series(
                [Decimal("1.00"), Decimal("2.00")],
                dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
            )
        }
    )
    result = getattr(lf.rolling(2), func)()
    assert result.dtypes["a"] == wrap_arrow_dtype(expected)


def test_rolling_partial_columns_skips_non_numeric():
    lf = LF({"a": [1, 2, 3, 4], "name": ["x", "y", "z", "w"]})
    result = lf.rolling(2, columns=["a"]).sum()
    assert result.dtypes["a"] == wrap_arrow_dtype(pa.int64())
    assert result.dtypes["name"] == lf.dtypes["name"]


@pytest.mark.parametrize(
    "call,exc,match",
    [
        (lambda lf: lf.rolling("2D"), NotImplementedError, "integer"),
        (lambda lf: lf.rolling(0), ValueError, "positive"),
        (lambda lf: lf.rolling(2, min_periods=3), ValueError, "must not exceed"),
        (lambda lf: lf.rolling(2).agg("skew"), ValueError, "Unsupported rolling"),
    ],
    ids=["offset-window", "zero-window", "min-periods-too-large", "unsupported-agg"],
)
def test_rolling_validation_errors(call, exc, match):
    lf = LF({"a": [1, 2, 3, 4, 5]})
    with pytest.raises(exc, match=match):
        call(lf)


# ---------------------------------------------------------------------------
# cum* and expanding -- dtype inference and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func_name,expected_dtype",
    [
        ("cumsum", pa.int64()),
        ("cummax", pa.int64()),
        ("cummin", pa.int64()),
        ("cumprod", pa.int64()),
    ],
)
def test_cum_dtype_inference(func_name, expected_dtype):
    lf = LF({"a": [1, 2, 3]})
    result = getattr(lf, func_name)()
    assert result.dtypes["a"] == wrap_arrow_dtype(expected_dtype)


# cumprod: Polars cum_prod/product widen every integer except UInt64 to Int64.
# The planned dtype must agree with the fetched dtype (review: cumprod dtype).
@pytest.mark.skipif(
    not is_arrow_dtype_supported(), reason="ArrowDtype is not supported"
)
@pytest.mark.parametrize(
    "input_arrow_type,expected_dtype",
    [
        (pa.int8(), pa.int64()),
        (pa.int16(), pa.int64()),
        (pa.int32(), pa.int64()),
        (pa.int64(), pa.int64()),
        (pa.uint8(), pa.int64()),
        (pa.uint16(), pa.int64()),
        (pa.uint32(), pa.int64()),
        (pa.uint64(), pa.uint64()),
    ],
    ids=["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"],
)
def test_cumprod_integer_dtype_widening(input_arrow_type, expected_dtype):
    raw = pd.DataFrame(
        {"a": pd.array([1, 2, 3], dtype=pd.ArrowDtype(input_arrow_type))}
    )
    lf = LF(raw)
    assert lf.cumprod().dtypes["a"] == wrap_arrow_dtype(expected_dtype)


@pytest.mark.parametrize(
    "func,expected_dtype",
    [
        ("sum", pa.int64()),
        ("min", pa.int64()),
        ("max", pa.int64()),
        ("count", pa.int64()),
        ("mean", pa.float64()),
        ("std", pa.float64()),
        ("var", pa.float64()),
    ],
)
def test_expanding_dtype_inference(func, expected_dtype):
    lf = LF({"a": [1, 2, 3, 4]})
    assert lf.expanding(1).agg(func).dtypes["a"] == wrap_arrow_dtype(expected_dtype)


@pytest.mark.skipif(
    not is_arrow_dtype_supported(), reason="ArrowDtype is not supported"
)
def test_cumsum_decimal_dtype():
    lf = LF(
        {
            "a": pd.Series(
                [Decimal("1.00"), Decimal("2.00")],
                dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
            )
        }
    )
    result = lf.cumsum()
    expected = infer_decimal_agg_dtype(wrap_arrow_dtype(pa.decimal128(10, 2)), "sum")
    assert result.dtypes["a"] == expected


@pytest.mark.skipif(
    not is_arrow_dtype_supported(), reason="ArrowDtype is not supported"
)
def test_cumprod_decimal_raises_type_error_plan():
    lf = LF(
        {
            "a": pd.Series(
                [Decimal("1.00"), Decimal("2.00")],
                dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
            )
        }
    )
    with pytest.raises(TypeError, match="decimal"):
        lf.cumprod()


@pytest.mark.skipif(
    not is_arrow_dtype_supported(), reason="ArrowDtype is not supported"
)
def test_expanding_decimal_mean_produces_float64():
    lf = LF(
        {
            "a": pd.Series(
                [Decimal("1.00"), Decimal("2.00")],
                dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
            )
        }
    )
    assert lf.expanding(1).mean().dtypes["a"] == wrap_arrow_dtype(pa.float64())


@pytest.mark.parametrize(
    "call",
    [
        lambda lf: lf.cumsum(),
        lambda lf: lf.cumprod(),
        lambda lf: lf.expanding(1).sum(),
        lambda lf: lf.expanding(1).mean(),
    ],
)
def test_cum_expanding_non_numeric_raises(call):
    lf = LF({"a": [1, 2, 3], "name": ["x", "y", "z"]})
    with pytest.raises(TypeError, match="non-numeric"):
        call(lf)


@pytest.mark.parametrize(
    "call",
    [
        lambda lf: lf.cumsum(columns=["a"]),
        lambda lf: lf.expanding(1, columns=["a"]).sum(),
    ],
)
def test_cum_expanding_partial_columns_skips_non_numeric(call):
    lf = LF({"a": [1, 2, 3], "name": ["x", "y", "z"]})
    assert isinstance(call(lf).data.op, LiteFrameWindowTransform)


@pytest.mark.parametrize(
    "call,exc,match",
    [
        (lambda lf: lf.expanding(1).agg("prod"), ValueError, "Unsupported"),
        (lambda lf: lf.expanding(0), ValueError, "min_periods"),
        (
            lambda lf: lf.groupby("key").cumsum(columns=["key"]),
            ValueError,
            "cannot be transform targets",
        ),
    ],
    ids=["unsupported-agg", "min-periods-invalid", "group-key-in-columns"],
)
def test_expanding_validation_errors(call, exc, match):
    lf = LF({"key": ["a", "b", "a"], "val": [1, 2, 3]})
    with pytest.raises(exc, match=match):
        call(lf)


# ---------------------------------------------------------------------------
# MF-specific window fields (order_by, window_shift, reverse) -- validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call,match",
    [
        (
            lambda lf: lf.expanding(
                1, order_by=["a", "b"], order_by_ascending=[True]
            ).sum(),
            "same length",
        ),
        (lambda lf: lf.expanding(1, order_by="zzz").sum(), "not found"),
    ],
    ids=["ascending-length-mismatch", "missing-column"],
)
def test_order_by_validation_errors(call, match):
    lf = LF({"a": [1, 2, 3], "b": [3, 2, 1]})
    with pytest.raises((ValueError, KeyError), match=match):
        call(lf)


def test_reverse_on_rolling_raises():
    lf = LF({"a": [1, 2, 3]})
    with pytest.raises(TypeError):
        lf.rolling(2, reverse=True)


def test_reverse_on_non_expanding_operator_raises():
    op = LiteFrameWindowTransform(
        kind="rolling_agg",
        window_size=2,
        transform_params={"agg_func": "sum"},
        reverse=True,
    )
    lf = LF({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="expanding"):
        op(lf.data)


def test_rank_without_groupby_raises():
    """Rank is a grouped-only window kind; whole-frame ranking goes through
    LiteFrame.rank (PSRS), not this operator."""
    with pytest.raises(TypeError, match="groupby_params"):
        LiteFrameWindowTransform.for_rank(method="average")


# ---------------------------------------------------------------------------
# Column-name normalization (end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call,expected_col",
    [
        (lambda lf: lf.cumsum(columns="VALUE"), "value"),
        (lambda lf: lf.shift(1, columns="VALUE"), "value"),
    ],
    ids=["cumsum", "shift"],
)
def test_uppercase_columns_normalized(call, expected_col):
    lf = LF({"Value": [1, 2, 3]})
    assert call(lf).data.op.columns == [expected_col]


@pytest.mark.parametrize(
    "call,expected_col,expected_order_by",
    [
        (
            lambda lf: lf.expanding(1, columns="VALUE", order_by="ORD").sum(),
            "value",
            "ord",
        ),
        (
            lambda lf: lf.rolling(2, columns="VALUE", order_by="ORD").sum(),
            "value",
            "ord",
        ),
    ],
    ids=["expanding", "rolling"],
)
def test_uppercase_columns_and_order_by_normalized(
    call, expected_col, expected_order_by
):
    lf = LF({"Value": [1, 2, 3], "Ord": [3, 2, 1]})
    op = call(lf).data.op
    assert op.columns == [expected_col]
    assert op.order_by == [expected_order_by]


# ---------------------------------------------------------------------------
# ffill / bfill -- dtype inference, limit validation, operator fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_fill_preserves_input_dtype(method):
    lf = LF({"a": [1, 2, 3]})
    result = getattr(lf, method)()
    assert result.dtypes["a"] == wrap_arrow_dtype(pa.int64())
    op = result.data.op
    assert op.kind == method
    assert op._get_transform_param("limit") is None


@pytest.mark.parametrize("limit", [2, np.int64(1), np.int32(1)])
@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_fill_limit_set_on_operator(method, limit):
    """Accepted integer-like values (int, numpy int scalars) normalize via
    operator.index to their Python int value (pandas 3.x compat)."""
    op = getattr(LF({"a": [1, 2, 3]}), method)(limit=limit).data.op
    expected = 1 if isinstance(limit, np.integer) else 2
    assert op._get_transform_param("limit") == expected


@pytest.mark.parametrize("limit", [0, -1, -(2**31) - 1])
@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_fill_limit_nonpositive_rejected(method, limit):
    """``<= 0`` (incl. negative Int32 overflow) raises ValueError, not
    OverflowError — the positivity check runs before the range bound."""
    with pytest.raises(ValueError, match="Limit must be greater than 0"):
        getattr(LF({"a": [1, 2, 3]}), method)(limit=limit)


@pytest.mark.parametrize("method", ["ffill", "bfill"])
@pytest.mark.parametrize("limit", [True, False, 1.5, "1", 1.0])
def test_fill_limit_non_integer_rejected(method, limit):
    """bool/float/str are rejected at the API boundary (pandas: 'Limit must
    be an integer'). bool is an int subclass, so it must be excluded
    explicitly — otherwise True silently executes as 1."""
    with pytest.raises(ValueError, match="Limit must be an integer"):
        getattr(LF({"a": [1, 2, 3]}), method)(limit=limit)


@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_fill_limit_out_of_range_rejected(method):
    """Positive out-of-Int32-range integers raise OverflowError."""
    with pytest.raises(OverflowError, match="out of Int32 range"):
        getattr(LF({"a": [1, 2, 3]}), method)(limit=2**31)


@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_fill_partial_columns_passes_through(method):
    lf = LF({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = getattr(lf, method)(columns=["a"])
    # b is a passthrough column -- dtype unchanged.
    assert result.dtypes["b"] == lf.dtypes["b"]
    op = result.data.op
    assert op.columns == ["a"]


@pytest.mark.skipif(
    not is_arrow_dtype_supported(), reason="ArrowDtype is not supported"
)
@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_fill_preserves_decimal_dtype(method):
    lf = LF(
        {
            "a": pd.array(
                [Decimal("1.50"), None, Decimal("3.00")],
                dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
            )
        }
    )
    result = getattr(lf, method)()
    assert result.dtypes["a"] == wrap_arrow_dtype(pa.decimal128(4, 2))


@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_fill_uppercase_columns_normalized(method):
    lf = LF({"Value": [1, 2, 3]})
    result = getattr(lf, method)(columns="VALUE")
    op = result.data.op
    assert op.columns == ["value"]


@pytest.mark.parametrize(
    "client_version,legacy_kwargs,expected_tp,expected_fields,framedriver_version",
    [
        pytest.param(
            "2.9.0",
            dict(
                kind="expanding_agg",
                agg_func="sum",
                remask_input_nan=True,
                skipna=False,
                min_periods=1,
                columns=["a"],
            ),
            {"agg_func": "sum", "remask_input_nan": True},
            {"kind": "expanding_agg", "skipna": False, "min_periods": 1},
            None,
            id="cumsum",
        ),
        pytest.param(
            "2.9.12",
            dict(kind="shift", periods=2, fill_value=0, columns=["a"]),
            {"fill_value": 0, "remask_input_nan": False},
            {"kind": "shift", "periods": 2},
            None,
            id="shift",
        ),
        pytest.param(
            "2.9.0",
            dict(
                kind="rolling_agg",
                window_size=3,
                min_periods=2,
                agg_func="mean",
                center=True,
                columns=["a", "b"],
            ),
            {"agg_func": "mean", "remask_input_nan": False},
            {"kind": "rolling_agg", "window_size": 3, "center": True},
            None,
            id="rolling_agg",
        ),
        # Non-2.9 serializers must not be redirected to the legacy layout.
        pytest.param("2.10.0", None, None, None, None, id="2.10-no-redirect"),
        pytest.param("3.0.0", None, None, None, None, id="3.0-no-redirect"),
        pytest.param(
            "2.9.0.dev1",
            dict(kind="shift", periods=2, fill_value=0, columns=["a"]),
            {"fill_value": 0, "remask_input_nan": False},
            {"kind": "shift", "periods": 2},
            None,
            id="2.9-dev-redirect",
        ),
        pytest.param(None, None, None, None, None, id="unset-no-redirect"),
        # The DPE's serializer is the FrameDriver, so its version wins over
        # the client version (including dev releases).
        pytest.param("2.9.0", None, None, None, "2.10.0", id="framedriver-2.10-wins"),
        pytest.param(None, None, None, None, "2.10.0", id="framedriver-2.10-only"),
        pytest.param(
            "2.10.0",
            dict(kind="shift", periods=2, fill_value=0, columns=["a"]),
            {"fill_value": 0, "remask_input_nan": False},
            {"kind": "shift", "periods": 2},
            "2.9.0",
            id="framedriver-2.9-wins",
        ),
        pytest.param(
            None,
            dict(kind="shift", periods=2, fill_value=0, columns=["a"]),
            {"fill_value": 0, "remask_input_nan": False},
            {"kind": "shift", "periods": 2},
            "2.9.1.dev4",
            id="framedriver-2.9-dev",
        ),
        pytest.param(
            "2.9.0",
            dict(kind="shift", periods=2, fill_value=0, columns=["a"]),
            {"fill_value": 0, "remask_input_nan": False},
            {"kind": "shift", "periods": 2},
            "not-a-version",
            id="invalid-framedriver-falls-back",
        ),
    ],
)
def test_legacy_2_9_serialization(
    client_version, legacy_kwargs, expected_tp, expected_fields, framedriver_version
):
    """2.9 layouts use the legacy adapter; newer layouts do not.

    The serializer version wins: ``framedriver_version`` when set, otherwise
    ``client_version``.
    """

    class _V29Payload(_LegacyLiteFrameWindowTransform_2_9):
        __module__ = "maxframe.liteframe.operators.window"
        __qualname__ = "LiteFrameWindowTransform"

    old29 = _V29Payload(**(legacy_kwargs or {}))
    header, buffers = serialize(old29)

    try:
        with option_context(
            {
                "session.client_version": client_version,
                "session.framedriver_version": framedriver_version,
            }
        ):
            if legacy_kwargs is None:
                assert LiteFrameWindowTransform.__serializable__() is None
                return
            assert (
                LiteFrameWindowTransform.__serializable__()
                is _LegacyLiteFrameWindowTransform_2_9
            )
            clear_type_cache()
            result = deserialize(header, buffers)
    finally:
        clear_type_cache()

    assert type(result) is LiteFrameWindowTransform
    assert result.transform_params == expected_tp
    for k, v in expected_fields.items():
        assert getattr(result, k) == v, f"field {k}"
    assert not hasattr(result, "remask_input_nan")
    assert not hasattr(result, "agg_func")
    assert not hasattr(result, "fill_value")


def test_v2_9_payload_mis_maps_without_client_version():
    """A 2.9 payload without a serializer version is not redirected.

    Documents why the ingress wraps deserialization in ``option_context``.
    """

    class _V29Payload(_LegacyLiteFrameWindowTransform_2_9):
        __module__ = "maxframe.liteframe.operators.window"
        __qualname__ = "LiteFrameWindowTransform"

    old29 = _V29Payload(kind="shift", periods=1, fill_value=7, columns=["a"])
    header, buffers = serialize(old29)

    old_cv = options.session.client_version
    old_fv = options.session.framedriver_version
    try:
        options.session.client_version = None
        options.session.framedriver_version = None
        assert LiteFrameWindowTransform.__serializable__() is None
        clear_type_cache()
        # Without the redirect, the current layout can't deserialize the 2.9
        # payload — it raises rather than producing a silently-corrupted op.
        with pytest.raises(Exception):
            deserialize(header, buffers)
    finally:
        options.session.client_version = old_cv
        options.session.framedriver_version = old_fv
        clear_type_cache()
