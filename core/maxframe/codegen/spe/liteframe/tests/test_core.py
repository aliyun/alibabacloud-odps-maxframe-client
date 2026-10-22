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

import pandas as pd

from maxframe.codegen.spe.core import SPECodeGenerator
from maxframe.liteframe import LiteFrame


def test_liteframe_codegen_end_to_end():
    """End-to-end test: build a LiteFrame graph and generate SPE code."""
    lf = LiteFrame(pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}))
    result = lf[["a", "b"]]
    dag = result.build_graph()
    codegen = SPECodeGenerator("session_1")
    generated = codegen.generate(dag)
    # Verify code was generated
    assert len(generated.code) > 0
    # Exec the code and verify results
    local_vars = generated.constants.copy()
    exec(generated.code, {}, local_vars)
    result_key = list(generated.output_key_to_variables.values())[0]
    result_df = local_vars[result_key]
    assert list(result_df.columns) == ["a", "b"]
    assert len(result_df) == 4
