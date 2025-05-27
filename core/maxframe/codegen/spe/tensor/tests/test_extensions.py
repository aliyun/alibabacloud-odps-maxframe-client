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
from ..extensions import TensorApplyChunkAdapter


def test_tensor_apply_chunk():
    s1 = mt.random.rand(10, 20, 30)

    adapter = TensorApplyChunkAdapter()

    result = s1.mf.apply_chunk(lambda x: x + 1)
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = [f"var_1 = {context.get_udfs()[0].name}(var_0)"]
    assert results == expected_results

    result = s1.mf.apply_chunk(lambda x, chunk_index: x + 1, with_chunk_index=True)
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = [
        f"var_1 = {context.get_udfs()[0].name}(var_0, chunk_index=(0, 0, 0))"
    ]
    assert results == expected_results
