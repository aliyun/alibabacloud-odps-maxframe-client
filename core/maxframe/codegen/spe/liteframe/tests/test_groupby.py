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

from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.liteframe.groupby import LiteFrameGroupByOpAdapter
from maxframe.liteframe.operators.groupby import LiteFrameGroupByOp


def test_groupby_adapter_code_generation(make_liteframe):
    """LiteFrameGroupByOpAdapter should generate _run_groupby call."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    data = make_liteframe(df)
    op = LiteFrameGroupByOp(
        groupby_params={"by": "a", "sort": False, "dropna": True},
    )
    out = op(data)
    op = out.op
    adapter = LiteFrameGroupByOpAdapter()
    context = SPECodeContext()
    result = adapter.generate_code(op, context)
    assert len(result) == 1
    assert "LiteFrameGroupByOpAdapter._run_groupby" in result[0]


def test_groupby_run_pass_through():
    """Test _run_groupby returns input unchanged (SPE is single-process)."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = LiteFrameGroupByOpAdapter._run_groupby(df)
    pd.testing.assert_frame_equal(result, df)
