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

import datetime
import itertools
import operator

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from maxframe.config import option_context
from maxframe.core.operator import Operator
from maxframe.dataframe.utils import (
    MAX_DECIMAL128_PRECISION,
    _calc_add_precision_scale,
    _calc_div_precision_scale_hive,
    _calc_mean_precision_scale,
    _calc_mul_precision_scale,
    _calc_sum_precision_scale,
    _estimate_integer_precision,
    _generate_value,
    _infer_decimal_agg_dtype,
    _pd_time_has_unit,
    copy_func_scheduling_hints,
    extract_scalar_dtype,
    infer_dtype,
    infer_dtypes,
    pack_func_args,
)
from maxframe.tests.utils import require_arrow_dtype
from maxframe.udf import (
    MarkedFunction,
    with_image_options,
    with_network_options,
    with_python_requirements,
    with_resources,
    with_running_options,
)
from maxframe.utils import wrap_arrow_dtype

try:
    from pandas import ArrowDtype
except:
    ArrowDtype = None


@pytest.fixture
def df1():
    return pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])


def test_pack_function(df1):
    # pack normal function
    @with_resources("a.zip")
    def keep(df):
        return df

    f = pack_func_args(df1, keep)
    assert f(df1).equals(df1)
    assert isinstance(f, MarkedFunction)
    assert f.file_resources == ["a.zip"]
    assert f.vpc_network_link is None
    assert isinstance(f.internal_network_whitelist, list)
    assert isinstance(f.public_network_whitelist, list)

    # pack with args
    @with_python_requirements("numpy")
    def add(a, b):
        return a + b

    f = pack_func_args(df1, add, 1)
    assert f(df1).equals(df1 + 1)
    assert isinstance(f, MarkedFunction)
    assert f.pythonpacks[0].requirements == ("numpy",)
    assert f.vpc_network_link is None
    assert isinstance(f.internal_network_whitelist, list)
    assert isinstance(f.public_network_whitelist, list)

    f = pack_func_args(df1, np.sum)
    # In pandas 2.0+, np.sum(df1) returns a Series, not a scalar
    df1_sum = np.sum(df1)
    f_result = f(df1)
    if isinstance(df1_sum, pd.Series):
        # When df1_sum is a Series, f_result is also a Series
        # Compare them using .equals()
        assert f_result.equals(df1_sum)
    else:
        assert f_result == df1_sum

    @with_resources("a.txt")
    @with_python_requirements("pandas")
    def times_add(df, param, times):
        return df * times + param

    f = pack_func_args(df1, times_add, 5, 6)
    assert f(df1).equals(df1 * 6 + 5)
    assert isinstance(f, MarkedFunction)
    assert f.file_resources == ["a.txt"]
    assert f.pythonpacks[0].requirements == ("pandas",)
    assert f.vpc_network_link is None
    assert isinstance(f.internal_network_whitelist, list)
    assert isinstance(f.public_network_whitelist, list)


@pytest.mark.parametrize(
    "dtype, fill_value, expected",
    [
        (
            wrap_arrow_dtype(pa.list_(pa.string())) if ArrowDtype else None,
            1,
            [pa.scalar("1")],
        ),
        (pa.list_(pa.string()), 1, [pa.scalar("1")]),
        (
            wrap_arrow_dtype(pa.map_(pa.int32(), pa.string())) if ArrowDtype else None,
            1,
            [(pa.scalar(1, pa.int32()), pa.scalar("1"))],
        ),
        (
            pa.map_(pa.int32(), pa.string()),
            1,
            [(pa.scalar(1, pa.int32()), pa.scalar("1"))],
        ),
        (
            wrap_arrow_dtype(
                pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string())])
            )
            if ArrowDtype
            else None,
            1,
            {"a": pa.scalar(1, pa.int32()), "b": pa.scalar("1")},
        ),
        (
            pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string())]),
            1,
            {"a": pa.scalar(1, pa.int32()), "b": pa.scalar("1")},
        ),
        (pa.int32(), 1, pa.scalar(1, pa.int32())),
        (np.datetime64, "2023-01-01", pd.Timestamp("2023-01-01")),
        (np.timedelta64, "1D", pd.Timedelta("1D")),
        (
            pd.CategoricalDtype(categories=["a", "b", "c"]),
            "a",
            pd.CategoricalDtype(["a"]),
        ),
        (np.object_, 1, "1"),
        (np.int32, 1, 1),
    ],
)
def test_generate_value(dtype, fill_value, expected):
    if dtype is None:
        pytest.skip("Arrow Dtype is not supported")
    result = _generate_value(dtype, fill_value)
    assert result == expected


def test_copy_func_scheduling_hints():
    # Test with a regular function (no scheduling hints)
    with option_context() as options:
        options.function.default_running_options = {}  # No default options

        def regular_func(x):
            return x + 1

        op1 = Operator()
        copy_func_scheduling_hints(regular_func, op1)
        # Should not set any attributes since regular function has no hints
        assert not hasattr(op1, "expect_engine") or op1.expect_engine is None
        assert not hasattr(op1, "expect_resources") or op1.expect_resources is None
        assert not hasattr(op1, "gpu") or op1.gpu is None

    # Test with MarkedFunction with scheduling hints

    @with_running_options(engine="DPE", cpu=4, memory="8GiB")
    def marked_func(x):
        return x + 1

    op2 = Operator()
    copy_func_scheduling_hints(marked_func, op2)
    assert op2.expect_engine == "DPE"
    # The expect_resources will include default values for gpu (gu_quota not set when None)
    expected_resources = {"cpu": 4, "memory": "8GiB", "gpu": 0}
    assert op2.expect_resources == expected_resources

    # Test with MarkedFunction with GPU
    @with_running_options(gu=2)
    def gpu_func(x):
        return x + 1

    op3 = Operator()
    copy_func_scheduling_hints(gpu_func, op3)
    assert op3.gpu is True
    # The expect_resources will include the gu value and default values
    # System has default options: {'cpu': 1, 'memory': '4GiB', 'gpu': 0}
    # When GPU is set, copy_func_scheduling_hints auto-sets gu_quota to
    # [options.session.gu_quota_name] which defaults to [None]
    expected_resources = {"gpu": 2, "gu_quota": [None], "cpu": 1, "memory": "4GiB"}
    assert op3.expect_resources == expected_resources

    # Test with MarkedFunction with image_options
    @with_image_options(image_name="python")
    def image_func(x):
        return x + 1

    op4 = Operator()
    copy_func_scheduling_hints(image_func, op4)
    assert op4.image_options == {"name": "python"}


@require_arrow_dtype
def test_decimal_type_inference():
    dtype1 = pd.ArrowDtype(pa.decimal128(MAX_DECIMAL128_PRECISION, 10))
    dtype2 = pd.ArrowDtype(pa.decimal128(MAX_DECIMAL128_PRECISION, 7))
    inferred = infer_dtype(dtype1, dtype2, operator.truediv)
    assert inferred.pyarrow_dtype.precision == MAX_DECIMAL128_PRECISION
    assert inferred.pyarrow_dtype.scale == 18

    dtypes1 = pd.Series(
        [np.dtype("int64"), pd.ArrowDtype(pa.decimal128(MAX_DECIMAL128_PRECISION, 10))]
    )
    dtypes2 = pd.Series(
        [np.dtype("int64"), pd.ArrowDtype(pa.decimal128(MAX_DECIMAL128_PRECISION, 7))]
    )
    inferred = infer_dtypes(dtypes1, dtypes2, operator.truediv)
    assert inferred.iloc[1].pyarrow_dtype.precision == MAX_DECIMAL128_PRECISION
    assert inferred.iloc[1].pyarrow_dtype.scale == 18


@pytest.mark.parametrize(
    "with_network_link, with_public_whitelist, with_internal_whitelist",
    [
        (use_link, use_public, use_internal)
        for use_link, use_public, use_internal in itertools.product(
            [True, False], repeat=3
        )
    ],
)
def test_with_network_options(
    with_network_link, with_public_whitelist, with_internal_whitelist
):
    kwargs = dict()
    if with_network_link:
        kwargs["vpc_network_link"] = "abc"
    if with_public_whitelist:
        kwargs["public_whitelist"] = ["a", "b"]
    if with_internal_whitelist:
        kwargs["internal_whitelist"] = ["e", "f"]

    @with_network_options(**kwargs)
    def test_func(x):
        return x + 1

    if with_network_link:
        assert test_func.vpc_network_link == "abc"
    else:
        assert test_func.vpc_network_link is None

    if with_public_whitelist:
        assert test_func.public_network_whitelist == ["a", "b"]
    else:
        assert not test_func.public_network_whitelist

    if with_internal_whitelist:
        assert test_func.internal_network_whitelist == ["e", "f"]
    else:
        assert not test_func.internal_network_whitelist


@pytest.mark.skipif(not _pd_time_has_unit, reason="pandas time objects has no unit")
@pytest.mark.parametrize(
    "scalar_value,expected_dtype",
    [
        (np.int32(5), np.dtype("int32")),
        (
            lambda: pd.Timestamp("2023-01-01 12:33:41").as_unit("ms"),
            np.dtype("datetime64[ms]"),
        ),
        (
            lambda: pd.Timestamp("2023-01-01 12:33:41").as_unit("ns"),
            np.dtype("datetime64[ns]"),
        ),
        (lambda: pd.Timedelta(days=1).as_unit("ms"), np.dtype("timedelta64[ms]")),
        (lambda: pd.Timedelta(days=1).as_unit("ns"), np.dtype("timedelta64[ns]")),
        (datetime.datetime(2023, 1, 1), np.dtype("datetime64[ms]")),
        (datetime.timedelta(days=1), np.dtype("timedelta64[ms]")),
        ("hello", str),
    ],
)
def test_extract_scalar_dtype(scalar_value, expected_dtype):
    if callable(scalar_value):
        scalar_value = scalar_value()
    result = extract_scalar_dtype(scalar_value)
    if isinstance(expected_dtype, type):
        assert result == expected_dtype
    else:
        assert result == expected_dtype


# Tests for decimal precision/scale calculation


@require_arrow_dtype
@pytest.mark.parametrize(
    "op,expected_precision,expected_scale",
    [
        # Addition/Subtraction
        (operator.add, 12, 3),
        (operator.sub, 12, 3),
        # Multiplication
        (operator.mul, 19, 5),
        # Division (using Hive formula)
        (operator.truediv, 22, 11),
    ],
)
def test_arithmetic_precision(op, expected_precision, expected_scale):
    """Test arithmetic precision/scale calculation using Hive rules"""
    dtype1 = pd.ArrowDtype(pa.decimal128(10, 2))
    dtype2 = pd.ArrowDtype(pa.decimal128(8, 3))

    result = infer_dtype(dtype1, dtype2, op)
    assert result.pyarrow_dtype.precision == expected_precision
    assert result.pyarrow_dtype.scale == expected_scale


@require_arrow_dtype
def test_infer_dtypes_dataframe():
    """Test infer_dtypes for DataFrame operations"""
    dtypes1 = pd.Series(
        {
            "a": pd.ArrowDtype(pa.decimal128(10, 2)),
            "b": pd.ArrowDtype(pa.decimal128(8, 1)),
        }
    )
    dtypes2 = pd.Series(
        {
            "a": pd.ArrowDtype(pa.decimal128(5, 1)),
            "b": pd.ArrowDtype(pa.decimal128(6, 0)),
        }
    )

    result = infer_dtypes(dtypes1, dtypes2, operator.truediv)
    # Using Hive formula
    assert result["a"].pyarrow_dtype.precision == 17
    assert result["a"].pyarrow_dtype.scale == 8


@require_arrow_dtype
@pytest.mark.parametrize(
    "func_name,expected_precision,expected_scale",
    [
        # Sum: Hive uses min(p + 10, 38)
        ("sum", 20, 2),  # min(10 + 10, 38) = 20
        # Mean: Hive uses min(p + 4, 38) and min(s + 4, 18)
        ("mean", 14, 6),  # min(10 + 4, 38) = 14, min(2 + 4, 18) = 6
        # Min/Max preserve input type
        ("min", 10, 2),
        ("max", 10, 2),
    ],
)
def test_aggregation_precision(func_name, expected_precision, expected_scale):
    """Test aggregation precision/scale calculation using Hive rules"""
    input_dtype = pd.ArrowDtype(pa.decimal128(10, 2))

    result = _infer_decimal_agg_dtype(input_dtype, func_name)
    assert result.pyarrow_dtype.precision == expected_precision
    assert result.pyarrow_dtype.scale == expected_scale


@require_arrow_dtype
@pytest.mark.parametrize("agg_func", ["var", "std"])
def test_variance_std_precision(agg_func):
    """Test that var/std use decimal arithmetic with max precision in Hive mode"""
    input_dtype = pd.ArrowDtype(pa.decimal128(10, 2))

    result = _infer_decimal_agg_dtype(input_dtype, agg_func)
    # Hive mode uses decimal arithmetic with max precision
    assert result.pyarrow_dtype.precision == 38


@require_arrow_dtype
@pytest.mark.parametrize(
    "int_type,expected_precision",
    [
        (np.int8, 3),  # max 127
        (np.int16, 5),  # max 32767
        (np.int32, 10),  # max 2147483647
        (np.int64, 19),  # max 9223372036854775807
    ],
)
def test_integer_precision_estimation(int_type, expected_precision):
    """Test integer precision estimation for mixed decimal/integer operations"""
    assert _estimate_integer_precision(int_type) == expected_precision


@require_arrow_dtype
@pytest.mark.parametrize(
    "calc_func,p1,s1,p2,s2,expected_precision,expected_scale",
    [
        # Addition: max(p1-s1, p2-s2) + max(s1, s2) + 1
        (_calc_add_precision_scale, 10, 2, 8, 3, 12, 3),
        # Multiplication: p1 + p2 + 1, s1 + s2
        (_calc_mul_precision_scale, 10, 2, 8, 3, 19, 5),
        # Division (Hive): precision = p1 - s1 + s2 + max(6, s1 + p2 + 1)
        (_calc_div_precision_scale_hive, 10, 2, 5, 1, 17, 8),
    ],
)
def test_calc_precision_scale_functions(
    calc_func, p1, s1, p2, s2, expected_precision, expected_scale
):
    """Test individual decimal calculation functions"""
    precision, scale = calc_func(p1, s1, p2, s2)
    assert precision == expected_precision
    assert scale == expected_scale


@require_arrow_dtype
@pytest.mark.parametrize(
    "calc_func,expected_precision,expected_scale",
    [
        # Sum: Hive uses min(p + 10, 38)
        (_calc_sum_precision_scale, 20, 2),
        # Mean: Hive uses min(p + 4, 38) and min(s + 4, 18)
        (_calc_mean_precision_scale, 14, 6),
    ],
)
def test_calc_aggregation_functions(calc_func, expected_precision, expected_scale):
    """Test aggregation precision/scale calculation functions"""
    precision, scale = calc_func(10, 2)
    assert precision == expected_precision
    assert scale == expected_scale


@require_arrow_dtype
def test_precision_overflow_protection():
    """Test that precision is capped at maximum"""
    dtype1 = pd.ArrowDtype(pa.decimal128(38, 18))
    dtype2 = pd.ArrowDtype(pa.decimal128(38, 18))

    # Test multiplication which would exceed max precision
    result = infer_dtype(dtype1, dtype2, operator.mul)
    # Precision should be capped at 38
    assert result.pyarrow_dtype.precision <= 38
    assert result.pyarrow_dtype.scale <= 18


@require_arrow_dtype
def test_division_by_integer():
    """Test division by integer-scale decimal"""
    dtype1 = pd.ArrowDtype(pa.decimal128(10, 4))
    dtype2 = pd.ArrowDtype(pa.decimal128(10, 0))  # integer-scale decimal

    # Using Hive formula
    result = infer_dtype(dtype1, dtype2, operator.truediv)
    # Should produce scale = max(6, s1 + p2 + 1) = max(6, 4 + 10 + 1) = 15
    assert result.pyarrow_dtype.scale == 15
