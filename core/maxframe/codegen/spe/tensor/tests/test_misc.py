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

from ..... import tensor as mt
from ...core import SPECodeContext
from ..misc import (
    TensorCopyToAdapter,
    TensorGetShapeAdapter,
    TensorInsertAdapter,
    TensorIsInAdapter,
    TensorSplitAdapter,
    TensorSqueezeAdapter,
    TensorTransposeAdapter,
    TensorUniqueAdapter,
)


def test_isin():
    result = mt.isin(2 * mt.arange(4).reshape((2, 2)), [1, 2, 4, 8])

    adapter = TensorIsInAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = [
        "var_0 = np.isin(var_1, var_2, assume_unique=False, invert=False)"
    ]
    assert results == expected_results


def test_transpose():
    result = mt.transpose(mt.ones((1, 2, 3)), (1, 0, 2))

    adapter = TensorTransposeAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_0 = np.transpose(var_1, axes=[1, 0, 2])"]
    assert results == expected_results


def test_squeeze():
    result = mt.squeeze(mt.ones((1, 2, 3)), axis=0)

    adapter = TensorSqueezeAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_0 = np.squeeze(var_1, axis=(0,))"]
    assert results == expected_results


def test_unique():
    tensor = mt.unique([1, 1])
    adapter = TensorUniqueAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(tensor.op, context)
    expected_results = [
        "var_1 = np.unique(var_0, return_index=False, return_inverse=False, return_counts=False, axis=0)"
    ]
    assert results == expected_results

    tensor = mt.unique([1, 1], return_index=True)
    context = SPECodeContext()
    results = adapter.generate_code(tensor[0].op, context)
    expected_results = [
        "var_1, var_2 = "
        "np.unique(var_0, return_index=True, return_inverse=False, return_counts=False, axis=0)"
    ]
    assert results == expected_results

    tensor = mt.unique([1, 1], return_index=True, return_inverse=True)
    context = SPECodeContext()
    results = adapter.generate_code(tensor[0].op, context)
    expected_results = [
        "var_1, var_2, var_3 = "
        "np.unique(var_0, return_index=True, return_inverse=True, return_counts=False, axis=0)"
    ]
    assert results == expected_results

    tensor = mt.unique(
        [1, 1], return_index=True, return_inverse=True, return_counts=True
    )
    context = SPECodeContext()
    results = adapter.generate_code(tensor[0].op, context)
    expected_results = [
        "var_1, var_2, var_3, var_4 = "
        "np.unique(var_0, return_index=True, return_inverse=True, return_counts=True, axis=0)"
    ]
    assert results == expected_results


def test_insert():
    result = mt.insert(mt.array([[1, 1], [2, 2], [3, 3]]), 1, 5)

    adapter = TensorInsertAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_0 = np.insert(var_1, 1, values=5)"]
    assert results == expected_results


def test_split():
    res_tp = mt.split(mt.array([1, 2, 3, 4, 5, 6]), 3)

    adapter = TensorSplitAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(res_tp[0].op, context)
    expected_results = ["var_1, var_2, var_3 = np.split(var_0, 3 axis=0)"]
    assert results == expected_results


def test_get_shape():
    inp = mt.array([[1, 1], [2, 2], [3, 3]])
    res = mt.shape(inp[:, inp[0, :] == 1])

    adapter = TensorGetShapeAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(res[0].op, context)
    expected_results = ["var_1, var_2 = np.shape(var_0)"]
    assert results == expected_results


def test_copyto():
    dest = mt.array([[1, 1], [2, 2], [3, 3]])
    src = mt.array([[1, 1], [2, 2], [3, 4]])
    mt.copyto(dest, src)

    adapter = TensorCopyToAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(dest.op, context)
    expected_results = [
        "var_2 = var_0.copy()",
        "np.copyto(var_2, var_1, casting='same_kind')",
    ]
    assert results == expected_results
