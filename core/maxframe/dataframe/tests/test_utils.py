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
import pyarrow as pa
import pytest

from ...config import option_context
from ...core.operator import Operator
from ...udf import (
    MarkedFunction,
    with_python_requirements,
    with_resources,
    with_running_options,
)
from ...utils import ARROW_DTYPE_NOT_SUPPORTED
from ..utils import _generate_value, copy_func_scheduling_hints, pack_func_args

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

    # pack with args
    @with_python_requirements("numpy")
    def add(a, b):
        return a + b

    f = pack_func_args(df1, add, 1)
    assert f(df1).equals(df1 + 1)
    assert isinstance(f, MarkedFunction)
    assert f.pythonpacks[0].requirements == ("numpy",)

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


@pytest.mark.skipif(ARROW_DTYPE_NOT_SUPPORTED, reason="Arrow Dtype is not supported")
@pytest.mark.parametrize(
    "dtype, fill_value, expected",
    [
        (
            ArrowDtype(pa.list_(pa.string())) if ArrowDtype else None,
            1,
            ["1"],
        ),
        (pa.list_(pa.string()), 1, ["1"]),
        (
            ArrowDtype(pa.map_(pa.int32(), pa.string())) if ArrowDtype else None,
            1,
            [(np.int32(1), "1")],
        ),
        (pa.map_(pa.int32(), pa.string()), 1, [(np.int32(1), "1")]),
        (
            ArrowDtype(
                pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string())])
            )
            if ArrowDtype
            else None,
            1,
            {"a": np.int32(1), "b": "1"},
        ),
        (
            pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string())]),
            1,
            {"a": np.int32(1), "b": "1"},
        ),
        (pa.int32(), 1, np.int32(1)),
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
