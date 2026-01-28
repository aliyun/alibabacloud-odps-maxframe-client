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

import itertools
import operator

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from ...config import option_context
from ...core.operator import Operator
from ...udf import (
    MarkedFunction,
    with_network_options,
    with_python_requirements,
    with_resources,
    with_running_options,
)
from ...utils import wrap_arrow_dtype
from ..utils import (
    MAX_DECIMAL128_PRECISION,
    _generate_value,
    copy_func_scheduling_hints,
    infer_dtype,
    infer_dtypes,
    pack_func_args,
)

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
    assert f.resources == ["a.zip"]
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
    assert f(df1).equals(np.sum(df1))

    @with_resources("a.txt")
    @with_python_requirements("pandas")
    def times_add(df, param, times):
        return df * times + param

    f = pack_func_args(df1, times_add, 5, 6)
    assert f(df1).equals(df1 * 6 + 5)
    assert isinstance(f, MarkedFunction)
    assert f.resources == ["a.txt"]
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
    # The expect_resources will include default values for gpu and gu_quota
    expected_resources = {"cpu": 4, "memory": "8GiB", "gpu": 0, "gu_quota": None}
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
    # The with_running_options decorator will override the gpu value with the gu value
    expected_resources = {"gpu": 2, "gu_quota": None, "cpu": 1, "memory": "4GiB"}
    assert op3.expect_resources == expected_resources


@pytest.mark.skipif(not hasattr(pd, "ArrowDtype"), reason="ArrowDtype not available")
def test_decimal_type_inference():
    dtype1 = pd.ArrowDtype(pa.decimal128(MAX_DECIMAL128_PRECISION, 10))
    dtype2 = pd.ArrowDtype(pa.decimal128(MAX_DECIMAL128_PRECISION, 7))
    inferred = infer_dtype(dtype1, dtype2, operator.truediv)
    assert inferred.pyarrow_dtype.precision == MAX_DECIMAL128_PRECISION
    assert inferred.pyarrow_dtype.scale == 12

    dtypes1 = pd.Series(
        [np.dtype("int64"), pd.ArrowDtype(pa.decimal128(MAX_DECIMAL128_PRECISION, 10))]
    )
    dtypes2 = pd.Series(
        [np.dtype("int64"), pd.ArrowDtype(pa.decimal128(MAX_DECIMAL128_PRECISION, 7))]
    )
    inferred = infer_dtypes(dtypes1, dtypes2, operator.truediv)
    assert inferred.iloc[1].pyarrow_dtype.precision == MAX_DECIMAL128_PRECISION
    assert inferred.iloc[1].pyarrow_dtype.scale == 12


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
