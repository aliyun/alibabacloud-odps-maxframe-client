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
from ..linalg import TensorEinsumAdapter, TensorQRAdapter, TensorTensorDotAdapter


def test_einsum():
    a = mt.arange(25).reshape(5, 5)
    b = mt.arange(5)
    adapter = TensorEinsumAdapter()
    r = mt.einsum("ij,j", a, b)
    context = SPECodeContext()
    results = adapter.generate_code(r.op, context)
    expected_results = [
        "var_2 = np.einsum('ij,j->i', var_0, var_1, dtype=np.dtype('int64'),"
        " order='K', casting='safe')"
    ]
    assert results == expected_results


def test_qr():
    a = mt.random.randn(9, 6)
    adapter = TensorQRAdapter()
    q, r = mt.linalg.qr(a)
    context = SPECodeContext()
    results = adapter.generate_code(q.op, context)
    expected_results = ["var_0, var_1 = spl.qr(var_2)"]
    assert results == expected_results


def test_tensordot():
    s1 = mt.arange(60).reshape(3, 4, 5)
    s2 = mt.arange(24).reshape(4, 3, 2)
    adapter = TensorTensorDotAdapter()
    s3 = mt.tensordot(s1, s2, axes=([1, 0], [0, 1]))
    context = SPECodeContext()
    results = adapter.generate_code(s3.op, context)
    expected_results = ["var_2 = npl.tensordot(var_0, var_1, ((1, 0), (0, 1)))"]
    assert results == expected_results
