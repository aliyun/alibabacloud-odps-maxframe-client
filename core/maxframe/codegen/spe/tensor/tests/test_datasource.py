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
import pytest

from maxframe import tensor as mt
from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.tensor.datasource import (
    TensorArangeAdapter,
    TensorEmptyAdapter,
    TensorFromLiteFrameAdapter,
    TensorFullAdapter,
    TensorOnesZerosAdapter,
    TensorOnesZerosLikeAdapter,
    TensorTriArrayAdapter,
)


def test_arange():
    s1 = mt.arange(10)
    adapter = TensorArangeAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s1.op, context)
    expected_results = [
        "var_0 = np.arange(const_0, const_1, const_2, dtype=np.dtype('int64'))"
    ]
    assert results == expected_results
    assert [context.constants[f"const_{i}"] for i in range(3)] == [0, 10, 1]


def test_empty():
    s1 = mt.empty((100, 100))
    adapter = TensorEmptyAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s1.op, context)
    expected_results = ["var_0 = np.empty((100, 100), dtype=const_0, order='C')"]
    assert results == expected_results


def test_full():
    s1 = mt.full((100, 100), 1)
    adapter = TensorFullAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s1.op, context)
    expected_results = [
        "var_0 = np.full((100, 100), 1, dtype=np.dtype('int64'), order='C')"
    ]
    assert results == expected_results


@pytest.mark.parametrize("method", ["triu", "tril"])
def test_tri_array(method):
    s1 = mt.random.rand(20, 20)
    adapter = TensorTriArrayAdapter()
    s3 = getattr(mt, method)(s1, k=1)
    context = SPECodeContext()
    results = adapter.generate_code(s3.op, context)
    expected_results = [f"var_1 = np.{method}(var_0, k=1)"]
    assert results == expected_results


@pytest.mark.parametrize("method", ["ones", "zeros"])
def test_ones_zeros(method):
    s1 = getattr(mt, method)((1000, 5), order="F")
    adapter = TensorOnesZerosAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s1.op, context)
    expected_results = [f"var_0 = np.{method}((1000, 5), dtype=const_0, order='F')"]
    assert results == expected_results

    s2 = getattr(mt, method)((1000,), order="F")
    adapter = TensorOnesZerosAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s2.op, context)
    expected_results = [f"var_0 = np.{method}((1000,), dtype=const_0, order='F')"]
    assert results == expected_results


@pytest.mark.parametrize("method", ["ones_like", "zeros_like"])
def test_ones_zeros_like(method):
    data = mt.tensor([[2, 0, 0], [0, 0, 2]])
    s1 = getattr(mt, method)(data, order="F")
    adapter = TensorOnesZerosLikeAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s1.op, context)
    expected_results = [
        f"var_1 = np.{method}(var_0, dtype=const_0, order='F', shape=(2, 3))"
    ]
    assert results == expected_results


def test_tensor_from_liteframe_empty_visible_columns():
    """C13: an explicit empty visible_columns list yields a zero-width tensor
    (n, 0) instead of converting every column (n, m)."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # Explicit [] -> zero columns selected -> shape (3, 0).
    arr = TensorFromLiteFrameAdapter._from_liteframe(
        df, visible_columns=[], dtype=np.int64, range_columns=None
    )
    assert arr.shape == (3, 0)

    # None -> all columns selected (unchanged) -> shape (3, 2).
    arr = TensorFromLiteFrameAdapter._from_liteframe(
        df, visible_columns=None, dtype=np.int64, range_columns=None
    )
    assert arr.shape == (3, 2)
