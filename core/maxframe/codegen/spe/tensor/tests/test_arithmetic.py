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
from ..arithmetic import (
    TensorAngleAdapter,
    TensorAroundAdapter,
    TensorBinOpAdapter,
    TensorIscloseAdapter,
    TensorSetImagAdapter,
    TensorSetRealAdapter,
    TensorUnaryOpAdapter,
)


def test_unary_op():
    s1 = mt.random.rand(1000, 5)
    result = mt.sin(s1)

    adapter = TensorUnaryOpAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_1 = np.sin(var_0, casting='same_kind', order='K')"]
    assert results == expected_results


def test_around():
    s1 = mt.random.rand(1000, 5)
    result = mt.around(s1, 2)

    adapter = TensorAroundAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_1 = np.around(var_0, decimals=2)"]
    assert results == expected_results


def test_angle():
    s1 = mt.random.rand(1000, 5)
    result = mt.angle(s1, deg=True)

    adapter = TensorAngleAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_1 = np.angle(var_0, deg=True)"]
    assert results == expected_results


def test_binary_op():
    s1 = mt.random.rand(1000, 5)
    s2 = mt.random.rand(1000, 5)
    result = s1 + s2

    adapter = TensorBinOpAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_2 = np.add(var_0, var_1, casting='same_kind', order='K')"]
    assert results == expected_results


def test_isclose():
    s1 = mt.random.rand(10, 5)
    s2 = mt.random.rand(10, 5)
    result = mt.isclose(s1, s2, rtol=0.1, atol=0.2)
    adapter = TensorIscloseAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = [
        "var_2 = np.isclose(var_0, var_1, equal_nan=False, rtol=0.1, atol=0.2)"
    ]
    assert results == expected_results


def test_set_imag():
    a = mt.array([1 + 2j, 3 + 4j, 5 + 6j])
    a.imag = mt.array([8, 10, 12])
    adapter = TensorSetImagAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(a.op, context)
    expected_results = ["var_0.imag = var_1", "var_2 = var_0"]
    assert results == expected_results


def test_set_real():
    a = mt.array([1 + 2j, 3 + 4j, 5 + 6j])
    a.real = mt.array([8, 10, 12])
    adapter = TensorSetRealAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(a.op, context)
    expected_results = ["var_0.real = var_1", "var_2 = var_0"]
    assert results == expected_results
