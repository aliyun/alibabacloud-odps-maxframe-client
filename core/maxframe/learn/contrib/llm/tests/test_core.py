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

from .. import core as llm_core


def test_text_gen_operator_output_dtypes():
    op = llm_core.LLMTextGenOperator()
    dtypes = op.get_output_dtypes()
    assert dtypes["response"] == np.dtype("O")
    assert dtypes["success"] == np.dtype("bool")


def test_text_embedding_operator_output_dtypes_and_defaults():
    op = llm_core.LLMTextEmbeddingOp()
    dtypes = op.get_output_dtypes()
    assert dtypes["response"] == np.dtype("O")
    assert dtypes["success"] == np.dtype("bool")
    assert op.simple_output is False
    assert op.dimensions is None
    assert op.encoding_format is None
