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
from maxframe.codegen.spe.liteframe.merge import (
    LiteFrameConcatAdapter,
    LiteFrameMergeAdapter,
)
from maxframe.liteframe.operators.merge import LiteFrameConcat, LiteFrameMerge


def test_merge_adapter(make_liteframe):
    """LiteFrameMergeAdapter should generate pd.merge() call."""
    left = pd.DataFrame({"key": [1, 2], "a": [3, 4]})
    right = pd.DataFrame({"key": [1, 2], "b": [5, 6]})
    left_lf = make_liteframe(left)
    right_lf = make_liteframe(right)
    op = LiteFrameMerge(how="inner", on="key")
    out = op(left_lf, right_lf)
    op = out.op
    adapter = LiteFrameMergeAdapter()
    context = SPECodeContext()
    result = adapter.generate_code(op, context)
    assert len(result) == 1
    assert "pd.merge(" in result[0]
    assert "how=" in result[0]


def test_concat_adapter(make_liteframe):
    """LiteFrameConcatAdapter should generate pd.concat() call."""
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [3, 4]})
    d1 = make_liteframe(df1)
    d2 = make_liteframe(df2)
    op = LiteFrameConcat(axis=0)
    out = op([d1, d2])
    op = out.op
    adapter = LiteFrameConcatAdapter()
    ctx = SPECodeContext()
    result = adapter.generate_code(op, ctx)
    assert len(result) == 1
    assert "pd.concat(" in result[0]
